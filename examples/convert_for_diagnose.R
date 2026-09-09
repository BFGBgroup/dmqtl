#!/usr/bin/env Rscript
# Convert the data for a single gene from
# examples/pas_counts.txt + examples/covariates.txt (MatrixEQTL-style,
# tab-delimited, rows = feature/covariate names, columns = samples)
# into the format expected by diagnose_drimseq_coefs.R / r_null_fit.R:
#   - comma-separated
#   - rows = samples, columns = PAS (counts) or covariates
#   - no row names; the first row is the header
#
# Usage:
#   Rscript convert_for_diagnose.R examples/pas_counts.txt examples/covariates.txt GENE0
#
# The script will generate counts_for_diagnose.csv and
# covariates_for_diagnose.csv in the current directory.
# You can then run:
#   Rscript dmqtl/diagnose_drimseq_coefs.R counts_for_diagnose.csv covariates_for_diagnose.csv

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript convert_for_diagnose.R pas_counts.txt covariates.txt gene_id  (e.g. GENE0)")
}
pas_path   <- args[1]
covar_path <- args[2]
gene_id    <- args[3]

pas   <- read.delim(pas_path, check.names = FALSE, row.names = 1)
covar <- read.delim(covar_path, check.names = FALSE, row.names = 1)

pas_rows <- grep(paste0("^", gene_id, "_PAS"), rownames(pas), value = TRUE)
if (length(pas_rows) == 0) {
  stop("No PAS rows found for gene_id='", gene_id, "'. Example row names: ",
       paste(head(rownames(pas)), collapse = ", "))
}
# Ensure that PAS rows are ordered as PAS0, PAS1, ...
pas_rows <- pas_rows[order(as.integer(sub(".*_PAS", "", pas_rows)))]

counts_mat <- t(as.matrix(pas[pas_rows, , drop = FALSE]))  # samples x K
storage.mode(counts_mat) <- "numeric"
colnames(counts_mat) <- paste0("PAS", seq_len(ncol(counts_mat)) - 1)

covar_mat <- t(as.matrix(covar))  # samples x p
storage.mode(covar_mat) <- "numeric"
covar_mat <- cbind(Intercept = 1, covar_mat)

stopifnot(identical(rownames(counts_mat), rownames(covar_mat)))

write.csv(counts_mat, "counts_for_diagnose.csv", row.names = FALSE)
write.csv(covar_mat, "covariates_for_diagnose.csv", row.names = FALSE)

cat(sprintf("Generated: counts_for_diagnose.csv (%d samples x %d PAS), covariates_for_diagnose.csv (%d samples x %d columns)\n",
            nrow(counts_mat), ncol(counts_mat), nrow(covar_mat), ncol(covar_mat)))
cat("Next step:\n")
cat("  Rscript dmqtl/diagnose_drimseq_coefs.R counts_for_diagnose.csv covariates_for_diagnose.csv\n")
