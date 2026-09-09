#!/usr/bin/env Rscript

# Diagnostic-only script: does not attempt to calculate beta/eta.
# It simply prints the complete structures returned by DRIMSeq's
# coefficients() and proportions() functions in the installed version,
# so that we can determine the actual parameterization used by DRIMSeq
# rather than continuing to make assumptions.
#
# Usage (same input files as r_null_fit.R):
#   Rscript diagnose_drimseq_coefs.R counts.csv covariates.csv
#
# After running the script, copy the complete terminal output and send it back.
# This script does not write any output files or modify your input data.

suppressMessages({
  library(DRIMSeq)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript diagnose_drimseq_coefs.R counts.csv covariates.csv")
}
counts_path <- args[1]
covar_path  <- args[2]

cat("=== DRIMSeq version information ===\n")
cat("DRIMSeq version:", as.character(packageVersion("DRIMSeq")), "\n")
cat("R version:", R.version.string, "\n\n")

counts_mat <- as.matrix(read.csv(counts_path, check.names = FALSE))
covar_mat  <- as.matrix(read.csv(covar_path, check.names = FALSE))

n <- nrow(counts_mat)
K <- ncol(counts_mat)
p <- ncol(covar_mat)
sample_ids <- paste0("S", seq_len(n))
pas_ids <- paste0("PAS", seq_len(K) - 1)

cat(sprintf("n=%d individuals, K=%d PAS (%s), p=%d covariate columns (%s)\n\n",
            n, K, paste(pas_ids, collapse=","), p, paste(colnames(covar_mat), collapse=",")))

counts_df <- data.frame(
  gene_id = rep("GENE", K),
  feature_id = pas_ids,
  t(counts_mat),
  check.names = FALSE
)
colnames(counts_df)[3:(2 + n)] <- sample_ids
samples_df <- data.frame(sample_id = sample_ids, covar_mat, check.names = FALSE)

d <- dmDSdata(counts = counts_df, samples = samples_df)

covariate_names <- colnames(covar_mat)
design_formula <- as.formula(
  paste("~", paste(covariate_names[covariate_names != "Intercept"], collapse = " + "))
)
design <- model.matrix(design_formula, data = samples_df)

cat("=== Design matrix ===\n")
cat("colnames(design):", paste(colnames(design), collapse=", "), "\n")
cat("dim(design):", paste(dim(design), collapse=" x "), "\n")
print(head(design))
cat("\n")

d <- dmPrecision(d, design = design)
d <- dmFit(d, design = design)

coefs <- coefficients(d)
cat("=== Complete structure of coefficients(d) ===\n")
cat("class(coefs):", class(coefs), "\n")
cat("dim(coefs):", paste(dim(coefs), collapse=" x "), "\n")
cat("colnames(coefs):", paste(colnames(coefs), collapse=", "), "\n\n")
cat("Complete contents (there should be K =", K, "rows, one row per PAS, and one column per design coefficient):\n")
print(coefs, digits = 6)
cat("\n")

# Check whether the coefficients for each design column sum to zero across
# the K features (a common sum-to-zero parameterization), or whether there
# is a row whose coefficients are all zero (indicating a reference feature).
# These are two common parameterizations and this diagnostic helps determine
# which one is actually used by DRIMSeq.
cat("=== Diagnostic: do the coefficients sum to zero across features for each design column? ===\n")
coef_only <- coefs[, setdiff(colnames(coefs), c("gene_id", "feature_id")), drop = FALSE]
col_sums <- colSums(as.matrix(coef_only))
print(col_sums)
cat("(If all values are close to 0, the model is likely using a sum-to-zeroparameterization, in which case eta needs to be reconstructed differently.)\n\n")

cat("=== Diagnostic: is there a row whose coefficients are all zero (reference feature)? ===\n")
row_all_zero <- apply(as.matrix(coef_only), 1, function(r) all(abs(r) < 1e-8))
for (i in seq_len(nrow(coefs))) {
  cat(sprintf("  feature_id=%s : all zero? %s\n, coefs$feature_id[i], row_all_zero[i]))
}
cat("\n")

# If the proportions() accessor is available, also print its output.
# These are observed proportions and may not necessarily be model-fitted
# values. If DRIMSeq provides fitted proportions directly, this may provide
# a more robust way to obtain eta without manually reconstructing it from
# the coefficients.
cat("=== Attempting to call proportions(d) (observed proportions; may not be fitted values) ===\n")
tryCatch({
  props <- proportions(d)
  cat("colnames(proportions(d)):", paste(colnames(props), collapse=", "), "\n")
  print(head(props))
}, error = function(e) {
  cat("proportions(d) failed:", conditionMessage(e), "\n")
})
cat("\n")

cat("=== Please copy and send back the complete output above ===\n")
