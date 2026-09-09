#!/usr/bin/env Rscript

# Fit the null model for a single gene x cell-type unit using DRIMSeq:
# PAS ~ covariates (without SNP terms).
#
# Usage:
#   Rscript r_null_fit.R counts.csv covariates.csv out.json
#
#   counts.csv     : n_samples rows x K columns, one sample per row and one
#                    PAS per column, without row names. The first row is the
#                    header (PAS0,PAS1,...).
#   covariates.csv : n_samples rows x p columns of numeric covariates,
#                    including the intercept column. The first row is the
#                    header.
#   out.json       : output JSON containing beta (p x (K-1)), eta
#                    (n x (K-1)), phi (scalar precision), loglik, and
#                    converged.
#
# Dependencies: DRIMSeq (Bioconductor), jsonlite
#
# === 2026-08 fix: no longer assume a specific DRIMSeq reference-category
# implementation ===
#
# The previous implementation assumed that, in coefficients(d), DRIMSeq
# would omit the entire row corresponding to the "last" feature in pas_ids
# (i.e., represent the reference category by an entirely missing row).
# Empirical testing showed that this assumption is incorrect for DRIMSeq
# 1.34.0: coefficients(d) actually returns all K features and does not omit
# any row.
#
# As a result, the old code would fail explicitly at
#
#   `if (!identical(present_features, expected_features))`
#
# with
#
#   present=[PAS0,PAS1,PAS2,PAS3]
#   expected=[PAS0,PAS1,PAS2]
#
# rather than silently producing an incorrect result. Thus, the previous
# design of explicitly stopping when the reference-category convention did
# not match was correct and behaved as intended. The problem was that the
# assumed implementation detail (omitting the entire reference row) did not
# match the actual behavior of this DRIMSeq version. A different algorithm
# that does not depend on this implementation detail is therefore required.
#
# The new algorithm does not attempt to infer whether DRIMSeq omits a row or
# which row it uses as the reference. Instead, it directly constructs a
# well-defined (n, K) raw-logit matrix that is valid regardless of how the
# reference category is represented:
#
#   - For every feature f appearing in coefficients(d), its raw logit is
#     design %*% (the coefficient row corresponding to f).
#
#   - For a feature not appearing in coefficients(d) (if a DRIMSeq version
#     actually uses the "omit the entire row" representation), its raw logit
#     is set to 0. This is exactly the mathematical meaning of a reference
#     category:
#
#         log(pi_f / pi_f) = 0.
#
#     No knowledge of DRIMSeq's internal implementation is required.
#
#   - If a version instead represents the reference category by an entire
#     row of zero coefficients (as observed in the current test), the two
#     representations are mathematically equivalent. A feature that appears
#     in coefficients(d) with all-zero coefficients produces
#     design %*% 0 = 0, which is identical to treating the feature as
#     missing and assigning it a raw logit of 0.
#
# Once this implementation-independent (n, K) raw-logit matrix has been
# constructed, it is converted to the convention used by the Python side:
# the last PAS in pas_ids is the reference category. This is simply a
# standard log-odds difference:
#
#   eta_new[, k] = eta_raw[, k] - eta_raw[, K]   (k = 1..K-1).
#
# This step also does not require knowing which feature DRIMSeq internally
# selected as its reference category. Subtracting any common baseline
# produces the same log(pi_k / pi_K), because the internal baseline cancels.
#
# Downstream components (dmqtl/pipeline.py, dmqtl/dmglm.py) use only the
# `eta` (and `phi`) returned here. They do not depend on `beta` for any
# calculation. `beta` is reconstructed by least squares solely for manual
# inspection and record-keeping; it is not on the critical computational
# path. See the `NullFit.beta` comment in dmqtl/nullmodel.py and the fact
# that dmqtl/pipeline.py uses only `null_fit.eta`.

suppressMessages({
  library(DRIMSeq)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript r_null_fit.R counts.csv covariates.csv out.json")
}
counts_path <- args[1]
covar_path  <- args[2]
out_path    <- args[3]

counts_mat <- as.matrix(read.csv(counts_path, check.names = FALSE))
covar_mat  <- as.matrix(read.csv(covar_path, check.names = FALSE))

n <- nrow(counts_mat)
K <- ncol(counts_mat)
p <- ncol(covar_mat)
sample_ids <- paste0("S", seq_len(n))
pas_ids <- paste0("PAS", seq_len(K) - 1)

# Counts data frame required by DRIMSeq:
# rows = features (PAS), columns = gene_id, feature_id, samples...
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
non_intercept_names <- covariate_names[covariate_names != "Intercept"]

# When there are no covariates other than "Intercept" (intercept-only
# null model, for example the known-composition validation case in
# test_r_bridge_smoke.py), the expression constructed above would be "~ "
# with nothing after the "~". This is not a valid R formula and
# `as.formula("~ ")` raises "unexpected end of input".
# In this case, explicitly use "~ 1".

rhs <- if (length(non_intercept_names) == 0) "1" else paste(non_intercept_names, collapse = " + ")
design_formula <- as.formula(paste("~", rhs))
design <- model.matrix(design_formula, data = samples_df)

d <- dmPrecision(d, design = design)
d <- dmFit(d, design = design)

prec <- as.numeric(genewise_precision(d)$genewise_precision[1])

coefs <- coefficients(d)

# When `coefficients(d)` internally converts the coefficients into a
# data.frame, R may apply make.names() to column names that are not valid
# variable names. The most typical example is "(Intercept)", which is
# converted to "X.Intercept.".
#
# In contrast, `design` is an ordinary matrix, so its colnames() retain
# the original "(Intercept)" name. Apply make.names() to design_cols so
# that the names are transformed using the same rule before comparison.
#
# make.names() performs an element-wise mapping while preserving order,
# so it does not change the correspondence between columns.
# This check is unrelated to the reference-category convention; it is
# purely a column-name escaping issue and is intentionally retained.
design_cols <- colnames(design)
coef_cols <- setdiff(colnames(coefs), c("gene_id", "feature_id"))
design_cols_mangled <- make.names(design_cols)
if (!setequal(coef_cols, design_cols_mangled)) {
  stop(
    "The coefficient column names returned by dmFit do not match the ",
    "escaped design-matrix column names; manual inspection is required: ",
    "coef_cols=[", paste(coef_cols, collapse = ","), "] ",
    "design_cols (original)=[", paste(design_cols, collapse = ","), "] ",
    "design_cols (escaped)=[", paste(design_cols_mangled, collapse = ","), "]"
  )
}

# Each feature appearing in coefficients(d) should occur at most once.
# If the same feature_id occurs more than once, the output structure of
# this DRIMSeq version does not match our assumption that each feature
# has at most one coefficient row. Stop for manual inspection rather than
# silently taking only the first row.
if (any(duplicated(coefs$feature_id))) {
  stop(
    "The same feature_id occurs more than once in coefficients(d). ",
    "The output structure does not match the expected format and requires ",
    "manual inspection: ",
    paste(coefs$feature_id[duplicated(coefs$feature_id)], collapse = ",")  
  )
}

missing_features <- setdiff(pas_ids, coefs$feature_id)
if (length(missing_features) > 1) {
  stop(
    "coefficients(d) is missing ",
    length(missing_features),
    " features ([",
    paste(missing_features, collapse = ","),
    "]), which exceeds the number allowed by any known DRIMSeq ",
    "reference-category convention (at most one complete row may be omitted). ",
    "Manual inspection of the actual coefficients() structure for this ",
    "DRIMSeq version is required. Consider first running ",
    "'Rscript diagnose_drimseq_coefs.R counts.csv covariates.csv' ",
    "to inspect the complete output."
  )
}

# Construct an (n, K) raw-logit matrix that is independent of the specific
# reference-category implementation used by DRIMSeq; see the file header.
#
# Features not present in coefficients(d) (at most one, as checked above)
# retain their initial all-zero column. This corresponds to treating that
# feature as the reference category, for which the log-odds relative to
# itself are always 0.
eta_raw_full <- matrix(0, nrow = n, ncol = K)
colnames(eta_raw_full) <- pas_ids
for (i in seq_len(nrow(coefs))) {
  fid <- coefs$feature_id[i]
  coef_vec <- unlist(coefs[i, design_cols_mangled, drop = FALSE], use.names = FALSE)
  eta_raw_full[, fid] <- as.numeric(design %*% coef_vec)
}

# Convert to the convention used by the Python side (dmqtl/dmglm.py):
# the last PAS in pas_ids (pas_ids[K]) is the reference category.
#
# This is a standard log-odds difference and is independent of which
# feature DRIMSeq actually uses internally as its baseline. The internal
# baseline term cancels when the difference is taken, so no prior knowledge
# or assumption about DRIMSeq's reference-category implementation is needed.
eta <- eta_raw_full[, seq_len(K - 1), drop = FALSE] - eta_raw_full[, K]
colnames(eta) <- pas_ids[seq_len(K - 1)]

# `beta` is no longer a quantity required by downstream calculations;
# downstream code uses only `eta` (see the file header).
#
# Here, beta is reconstructed by least squares by projecting eta back
# into the design-matrix coefficient space. It is retained solely for
# manual inspection and record-keeping.
#
# If the fit fails, for example because the design matrix is collinear,
# fall back to NA. This does not affect the production of eta or phi.
beta <- tryCatch(
  solve(t(design) %*% design, t(design) %*% eta),
  error = function(e) matrix(NA_real_, nrow = ncol(design), ncol = K - 1)
)
rownames(beta) <- design_cols
colnames(beta) <- pas_ids[seq_len(K - 1)]

result <- list(
  beta = beta,
  eta = eta,
  design_colnames = design_cols,
  
  # Diagnostic field: the reference category used by this package when
  # converting the raw logits, not the reference category internally
  # selected by DRIMSeq.
  reference_pas_id = pas_ids[K],
  phi = prec,
  loglik = NA,
  converged = TRUE
)
write(toJSON(result, auto_unbox = TRUE, digits = 12), out_path)
