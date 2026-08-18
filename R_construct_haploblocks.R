############################################################
# Supplementary Code S1
#
# LD-based haplotype block identification and construction
#
# This script defines functions for:
# (1) Encoding unique haplotypes using alphabetical labels;
# (2) Constructing multi-marker haplotypes from phased genotype data;
# (3) Identifying LD-based haplotype blocks;
# (4) Generating haplotype matrices for downstream association analysis.
#
############################################################

############################################################
# Function 1: Generate haplotype labels
############################################################
# Assign alphabetical labels to unique haplotypes.
# Uppercase and lowercase letters are used first, followed by
# two-letter combinations when the number of haplotypes exceeds 52.
get_hap_letters <- function(n){
  base <- c(LETTERS, letters)
  if(n <= length(base)){
    return(base[1:n])
  } else {
    # Generate two-letter combinations
    comb <- as.matrix(expand.grid(LETTERS, LETTERS))
    comb_letters <- c(
      apply(comb, 1, paste0, collapse = ""))
    if(n <= length(comb_letters)){
      return(comb_letters[1:n])
    } else {
      stop("The number of haplotypes exceeds the available coding capacity.")
    }}}

############################################################
# Function 2: Construct multi-marker haplotypes
############################################################

# Convert phased multi-marker genotypes into haplotype
# categories and generate two output formats:
#
# format1:
#   Wide-format haplotype matrix used for association analysis.
#
# format2:
#   Haplotype dosage/count matrix representing the number of
#   copies of each haplotype carried by each individual.
#
# hap_letter:
#   Mapping table between original haplotype sequences and
#   assigned alphabetical labels.

make_multimarkers <- function(df, mrks, haploblock_nme){
  library(plyr)
  # Extract genotype data for markers included in the haplotype block
  df_set <- df[df$marker %in% mrks, ]
  rownames(df_set) <- df_set$marker
  # Remove marker identifier column
  df_set <- df_set[, -1, drop = FALSE]
  # Concatenate alleles across markers to define haplotype sequences
  haplotypes <- apply(df_set,2,paste0,collapse = "")
  # Identify unique haplotype sequences
  unique_hap <- unique(haplotypes)
  # Assign alphabetical labels to haplotypes
  unique_hap_c <- get_hap_letters(length(unique_hap))
  hap_converted <- plyr::mapvalues(haplotypes,
                                   from = unique_hap,
                                   to = unique_hap_c)
  # Extract individual IDs from phased genotype column names
  individuals <- unique(gsub("(.+)_\\d+$","\\1",names(hap_converted)))
  ##########################################################
  # Generate format1: individual haplotype genotype matrix
  ##########################################################
  format1 <- t(
    sapply(individuals,function(i){
      paste0(hap_converted[paste0(i, "_", 1:2)],collapse = "")
    }))
  rownames(format1) <- haploblock_nme
  colnames(format1) <- individuals
  ##########################################################
  # Generate format2: haplotype dosage matrix
  ##########################################################
  format2 <- data.frame(
    haploblock = haploblock_nme,
    Letter = unique_hap_c,
    matrix(0,nrow = length(unique_hap_c),ncol = length(individuals)))
  colnames(format2)[3:ncol(format2)] <- individuals
  # Count haplotype copies for each individual
  for(i in individuals){
    cols <- paste0(i, "_", 1:2)
    cols <- cols[
      cols %in% names(hap_converted)
    ]
    e <- hap_converted[cols]
    # Replace missing or empty haplotypes
    e[is.na(e)] <- "NA"
    e[e == ""] <- "NA"
    e_tab <- table(e)
    for(j in names(e_tab)){
      format2$Letter <- as.character(format2$Letter)
      if(j %in% format2$Letter){
        idx <- which(format2$Letter == j)
        format2[idx, i] <- as.integer(e_tab[j])
      }}}
  
  
  
  ##########################################################
  # Generate haplotype sequence-label correspondence table
  ##########################################################
  hap_record <- data.frame(unique_hap = unique_hap,
                           Letter = unique_hap_c)
  return(
    list(format1 = format1,
         format2 = format2,
         hap_letter = hap_record,
         markers = mrks))
}

############################################################
# LD-based haplotype block identification and haplotype construction
#
# This script identifies haplotype blocks based on pairwise LD
# between markers, physical distance constraints, and LD thresholds.
# Identified blocks are subsequently used for haplotype construction.
#
# Parameters:
# LD_core   : threshold for defining core LD connections
# LD_extend : threshold for extending haplotype blocks
# D_decay   : maximum physical distance allowed within a block
#
############################################################
# Extract LD matrix for the target chromosome
LD_mat <- abs(LD_list[[as.numeric(chr)]])
# Extract marker names and physical positions
markers <- colnames(LD_mat)
pos_vector <- as.numeric(sapply(strsplit(markers, "_"), `[`, 2))
names(pos_vector) <- markers
# Check whether all markers have valid physical positions
if(any(is.na(pos_vector))){
  stop(paste0("Chromosome ", chr, 
              " contains markers with missing physical positions!"))
}
############################################################
# Identify LD haplotype blocks
############################################################
blocks <- list()
block_id <- 1
nSNP <- length(pos_vector)
for(start_idx in 1:nSNP){
  # Initialize candidate block with the starting marker
  candidate_block <- start_idx
  # Flag indicating whether a core LD connection is detected
  has_core <- FALSE
  # Search downstream markers within the defined physical distance
  if(start_idx == nSNP){
    next_indices <- nSNP
  } else {
    next_indices <- (start_idx + 1):nSNP
  }
  for(next_idx in next_indices){
    # Stop searching when physical distance exceeds LD decay distance
    if(pos_vector[next_idx] - pos_vector[start_idx] > D_decay){
      break
    }
    # Limit the maximum number of SNPs included in one block
    if(length(candidate_block) >= 5){
      break
    }
    # Extend block when the candidate SNP shows sufficient LD
    # with at least one SNP already included in the block
    ld_with_block <- LD_mat[next_idx, candidate_block]
    if(any(ld_with_block >= LD_extend)){
      candidate_block <- c(candidate_block, next_idx)
      # Record whether the new SNP has strong LD with the
      # starting SNP (core LD relationship)
      if(LD_mat[next_idx, start_idx] >= LD_core){
        has_core <- TRUE
      }
    }
  }
  # Retain only blocks containing at least one core LD relationship
  if(has_core && length(candidate_block) > 1){
    blocks[[block_id]] <- candidate_block
    block_id <- block_id + 1
  }
}

############################################################
# Generate haplotype block summary table
############################################################
block_table <- data.table(chr = chr,
                          Block_ID = seq_along(blocks),
                          Start_SNP = sapply(blocks, function(idx)markers[idx[1]]),
                          End_SNP = sapply(blocks, function(idx)markers[idx[length(idx)]]),
                          Num_SNPs = sapply(blocks, length),
                          Start_pos = sapply(blocks, function(idx)
                            pos_vector[idx[1]]),
                          End_pos = sapply(blocks, function(idx)
                            pos_vector[idx[length(idx)]]),
                          Block_length = sapply(blocks, function
                                                (idx)pos_vector[idx[length(idx)]] - pos_vector[idx[1]]),
                          SNPs_in_block = sapply(blocks, function
                                                 (idx)paste(markers[idx], collapse = ",")))
# Calculate average LD within each haplotype block
block_table[, mean_LD := sapply(blocks, function(idx){
  if(length(idx) < 2){
    return(NA)
  }
  submat <- LD_mat[idx, idx]
  mean(submat[upper.tri(submat)])
})]
# Calculate the average physical position of each block
block_table[, mean_pos := sapply(blocks, function(idx)
  round(mean(pos_vector[idx])))
]

############################################################
# Construct haplotypes from identified LD blocks
############################################################
for(b in 1:nrow(block_table)){
  # Extract markers included in the current haplotype block
  block_markers <- strsplit(unlist(block_table[b, 'SNPs_in_block']),",")[[1]]
  # Generate haplotype identifier
  hap_name <- paste0("chr", chr,
                     "_Block", block_table$Block_ID[b],
                     "_", block_table$mean_pos[b])
  # Construct haplotypes using phased genotype data
  hap_res <- make_multimarkers(df = phase_chr,
                               mrks = block_markers,
                               haploblock_nme = hap_name)
  # Store haplotype results in different formats
  all_format1_list[[hap_name]] <- hap_res$format1
  all_format2_list[[hap_name]] <- hap_res$format2
}