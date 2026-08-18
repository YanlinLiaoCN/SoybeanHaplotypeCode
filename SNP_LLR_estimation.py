import pandas as pd
import numpy as np
import pysam
import jax
import jax.numpy as jnp
import haiku as hk
from tqdm import tqdm
from nucleotide_transformer.pretrained import get_pretrained_model

# ============================================================
# Configuration
# ============================================================
# Input and output files
FASTA_PATH = "Glycine_max.V1.0.dna_sm.chromosome.1.fa"  # Reference genome sequence
VCF_PATH = "out_7.gt.vcf.gz"                            # Input VCF file containing SNP variants
OUTPUT_CSV = "llr_scores_64128.csv"                     # Output file for LLR scores
# AgroNT model parameters
MODEL_NAME = "1B_agro_nt"
MAX_SEQ_LEN = 128       # Maximum sequence length used for model input
BATCH_SIZE = 1          # Number of variants processed per batch
CONTEXT_WINDOW = 64     # Number of nucleotides extracted upstream and downstream of each SNP

# ============================================================
# 1. Load AgroNT model
# ============================================================
print("Loading AgroNT model...")
parameters, forward_fn, tokenizer, config = get_pretrained_model(
    model_name=MODEL_NAME,
    embeddings_layers_to_save=(12,),   # Required model configuration for output generation
    max_positions=MAX_SEQ_LEN
)
# Transform Haiku model function for inference
forward_fn = hk.transform(forward_fn)
# Random seed for JAX computation
random_key = jax.random.PRNGKey(42)
print("AgroNT model successfully loaded.")

# ============================================================
# 2. Core functions
# ============================================================
def get_sequence_log_likelihood(logits, token_ids, pad_id=None):
    """
    Calculate sequence log-likelihood based on model output probabilities.
    The log-likelihood represents how well the nucleotide sequence
    is supported by the pretrained language model.
    Parameters
    ----------
    logits : array
        Model output logits with dimensions:
        (batch_size, sequence_length, vocabulary_size)
    token_ids : array
        Tokenized nucleotide sequence IDs.
    pad_id : int, optional
        Token ID used for padding positions.
    Returns
    -------
    array
        Log-likelihood score for each sequence.
    """
    # Convert logits into normalized log probabilities
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    # Extract log probability corresponding to the observed nucleotide token
    target_tokens = token_ids[:, :, None]
    seq_log_probs = jnp.take_along_axis(
        log_probs,
        target_tokens,
        axis=-1
    )
    # Remove contributions from padded positions
    if pad_id is not None:
        mask = (token_ids != pad_id)
        seq_log_probs = seq_log_probs * mask[:, :, None]
    # Sum nucleotide-level log probabilities to obtain sequence-level score
    return jnp.sum(seq_log_probs, axis=(1, 2))
def process_batch(batch_records, fasta_ref):
    """
    Calculate AgroNT log-likelihood ratios (LLRs) for a batch of SNPs.
    For each SNP, a pair of sequences is generated:
    (1) the reference allele sequence;
    (2) the alternative allele sequence.
    The LLR is calculated as:
        LLR = log likelihood(ALT sequence) - log likelihood(REF sequence)
    Parameters
    ----------
    batch_records : list
        SNP information tuples:
        (chromosome, position, reference allele, alternative allele, SNP ID)
    fasta_ref : pysam.FastaFile
        Reference genome sequence.
    Returns
    -------
    metadata : list
        SNP information retained after quality control.
    llr_values : array
        AgroNT-derived LLR scores.
    """
    seqs_ref = []
    seqs_alt = []
    meta_data = []
    for rec in batch_records:
        chrom, pos_0based, ref_base, alt_base, snp_id = rec

# --------------------------------------------------------
# A. Extract local sequence context surrounding SNP
# --------------------------------------------------------
        start = max(0, pos_0based - CONTEXT_WINDOW)
        end = pos_0based + CONTEXT_WINDOW
        try:
            ref_context = fasta_ref.fetch(
                chrom,
                start,
                end
            ).upper()
        except KeyError:
            # Skip variants with unmatched chromosome names
            continue
        except ValueError:
            continue
        # Verify that the reference genome nucleotide matches VCF annotation
        relative_pos = pos_0based - start
        if len(ref_context) <= relative_pos:
            continue
        fetched_base = ref_context[relative_pos]
        if fetched_base != ref_base:
            # Exclude variants inconsistent with reference genome sequence
            Continue

# --------------------------------------------------------
# B. Generate reference and alternative allele sequences
# --------------------------------------------------------
        seq_ref_str = ref_context
        seq_alt_str = (
            ref_context[:relative_pos]
            + alt_base
            + ref_context[relative_pos + 1:]
        )
        seqs_ref.append(seq_ref_str)
        seqs_alt.append(seq_alt_str)
        meta_data.append(rec)
    if not seqs_ref:
        return [], []

# ------------------------------------------------------------
# C. Tokenization of nucleotide sequences
# ------------------------------------------------------------
# Reference and alternative sequences are processed together
    # to enable parallel inference.
    all_seqs = seqs_ref + seqs_alt
    tokens = tokenizer.batch_tokenize(all_seqs)
    token_ids_list = []
    for t in tokens:
        ids = t[1]   # Extract token IDs
        # Truncate sequences exceeding maximum model length
        if len(ids) > MAX_SEQ_LEN:
            ids = ids[:MAX_SEQ_LEN]
        # Padding to fixed sequence length for JAX inference
        padded = ids + [0] * (MAX_SEQ_LEN - len(ids))
        token_ids_list.append(padded)
    token_ids_arr = jnp.array(
        token_ids_list,
        dtype=jnp.int32
)

# ------------------------------------------------------------
# D. AgroNT inference
# ------------------------------------------------------------
    outputs = forward_fn.apply(
        parameters,
        random_key,
        token_ids_arr
    )
logits = outputs["logits"]

# ------------------------------------------------------------
# E. Calculate LLR values
# ------------------------------------------------------------
    scores = get_sequence_log_likelihood(
        logits,
        token_ids_arr
    )
    scores_np = np.array(scores)
    n = len(seqs_ref)
    scores_ref = scores_np[:n]
    scores_alt = scores_np[n:]
    # Log-likelihood ratio:
    # Positive values indicate higher model preference for ALT allele sequence
    llr_values = scores_alt - scores_ref
    return meta_data, llr_values