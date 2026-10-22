#!/usr/bin/env Rscript

# Batch null-model fitting: pass *multiple genes* to a single
# dmPrecision()/dmFit() call so that DRIMSeq's cross-gene precision
# shrinkage can actually take place.
#
# This is the core distinction from the single-gene version
# (r_null_fit.R), rather than merely a performance optimization.
#
# In the single-gene version (r_null_fit.R), each invocation passes only
# the K PAS of one gene to dmDSdata. There are therefore no neighboring
# genes from which dmPrecision can borrow information, so the effective
# behavior is essentially a single-gene estimator and does not realize
# the cross-gene shrinkage described in the documentation.
#
# This version instead passes all genes in a batch (either the complete
# set of genes or a pre-defined chunk) to a single call. dmPrecision then
# estimates a common dispersion-versus-mean trend across the genes in
# that batch and performs empirical-Bayes shrinkage for each gene. This
# is the intended meaning of "borrowing information from neighboring
# genes to stabilize variance/precision estimates" in tools such as
# DRIMSeq, edgeR, and DESeq2.
#
# Usage:
#   Rscript r_null_fit_batch.R counts_long.csv covariates.csv out_prefix [n_cores] [prec_subset]
#
#   counts_long.csv : long-format table with columns
#                     gene_id, feature_id, S1, S2, ..., Sn
#                     (rows for multiple genes are concatenated).
#                     Within each gene, the order of feature_id determines
#                     the PAS order used by the Python side of this package.
#                     The last feature_id within each gene is the reference
#                     category.
#
#   covariates.csv  : n_samples rows x p columns of covariates, including
#                     the intercept column. The first row is the header.
#
#   out_prefix      : output prefix. Two files are generated:
#                     <out_prefix>_phi.csv and <out_prefix>_eta.csv.
#
#   n_cores         : optional number of cores used by BiocParallel inside
#                     dmPrecision/dmFit. Default: 1 (serial).
#                     This controls parallelization over multiple genes
#                     on the R side and is independent of
#                     run_dmqtl(n_workers) on the Python side.
#                     The latter is now responsible only for parallelizing
#                     the score-test + permutation stage; worker processes
#                     no longer launch R subprocesses.
#                     See CHANGES.md for details of this architectural change.
#
#   prec_subset     : optional value passed to dmPrecision as `prec_subset`.
#                     If omitted, DRIMSeq's default value of 0.1 is used.
#
#                     An important issue discovered through empirical
#                     testing is that `dmPrecision` by default uses only
#                     10% of the genes in the batch to estimate the common
#                     cross-gene dispersion trend. In genome-wide analyses,
#                     this reduces computational cost, and the R side prints
#                     a message such as:
#
#                         Using a subset of 0.1 genes to estimate common precision !
#
#                     For realistic data with chunks containing several
#                     thousand genes, 10% still corresponds to hundreds
#                     of genes and is usually sufficient. However, when the
#                     batch contains only a small number of genes (for example,
#                     a smoke test with a dozen genes, or a very small
#                     chunk_size), 10% may leave only a few genes or even
#                     fewer than one effective gene for trend estimation.
#                     In that situation, R may print:
#
#                         Using 0 as a shrinkage factor !
#
#                     This means that the batch effectively receives little
#                     or no meaningful cross-gene shrinkage and may behave
#                     similarly to independent single-gene estimation.
#
#                     For small batches, explicitly setting prec_subset=1.0
#                     is recommended so that all genes in the batch are used
#                     to estimate the trend. For large batches (thousands
#                     of genes or more), the default 0.1 can be retained
#                     to reduce computation time.
#
# The reference-category conversion logic and robustness checks are
# identical to those in r_null_fit.R (see the "2026-08 fix" discussion
# in that file). The same logic is applied independently to each gene
# here; the only structural difference is that the data are now stored
# in a multi-gene data.frame and processed by gene_id.

suppressMessages({
  library(DRIMSeq)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3 || length(args) > 11) {
  stop(
    paste0(
      "Usage: Rscript r_null_fit_batch.R ",
      "counts_long.csv covariates.csv out_prefix ",
      "[n_cores] [prec_subset] [prec_prior_df] ",
      "[prec_moderation] [outlier_log2_threshold] [optim_maxit] [rng_seed] [cr_logdet]"
    )
  )
}

counts_path <- args[1]
covar_path  <- args[2]
out_prefix  <- args[3]
n_cores <- if (length(args) >= 4 && nzchar(args[4])) as.integer(args[4]) else 1L
prec_subset <- if (length(args) >= 5 && nzchar(args[5])) as.numeric(args[5]) else NULL

# prec_prior_df / prec_moderation:
# see the detailed comments at the dmPrecision call below.
#
# *** 2026-09 third correction (confirmed by consulting the official
# `?DRIMSeq::dmPrecision` documentation after two earlier conclusions
# were made without checking the authoritative source):
#
# For the `dmDSdata` method dispatch actually used here, the default
# `prec_moderation` is **"trended"**, not "none".
#
# "none" is the default for a different `dmSQTLdata` method dispatch,
# which had previously been confused with the `dmDSdata` case.
#
# Under normal circumstances, `prec_prior_df` is also not a parameter
# that directly controls the amount of shrinkage. DRIMSeq estimates the
# amount of shrinkage from the data automatically. `prec_prior_df` is
# only used as a fallback when automatic estimation fails (according to
# the official documentation, when fewer than 10 genes are located at
# the boundaries of the precision grid).
#
# Therefore, observing little shrinkage is not necessarily caused by
# `prec_subset`, and simply increasing `prec_prior_df` does not generally
# solve the problem. A more likely explanation for very small batches is
# that `prec_span` (the smoothing-window size, expressed as a fraction
# of the number of genes) becomes effectively degenerate.
#
# With a realistic chunk containing thousands of genes, this issue is
# unlikely to be important.
prec_prior_df <- if (length(args) >= 6 && nzchar(args[6])) as.numeric(args[6]) else NULL
prec_moderation <- if (length(args) >= 7 && nzchar(args[7])) args[7] else NULL

# outlier_log2_threshold:
# see the detailed comments at the second outlier-detection stage below.
#
# Exact grid-boundary hits represent an extreme case. This additional
# threshold performs a more permissive, robust outlier check based on
# the median precision estimated from the current batch.
#
# Default: 5, meaning that a precision estimate more than 2^5 = 32-fold
# away from the batch median is considered unreliable.
outlier_log2_threshold <- if (length(args) >= 8 && nzchar(args[8])) as.numeric(args[8]) else 5

# optim_maxit / rng_seed (2026-10): see the DRIMSeq workarounds just below.
# Empty or 0 / empty keeps DRIMSeq's behaviour unchanged.
optim_maxit <- if (length(args) >= 9 && nzchar(args[9])) as.integer(args[9]) else 0L
rng_seed <- if (length(args) >= 10 && nzchar(args[10])) as.integer(args[10]) else NA_integer_
cr_logdet <- if (length(args) >= 11 && nzchar(args[11])) as.integer(args[11]) == 1L else TRUE

# ---- DRIMSeq workarounds (2026-10) -------------------------------------------
#
# (1) Whole-chunk failure. dmDS_estimateTagwisePrecision() returns
#     rep(NA, length(counts)) -- a *logical* vector -- when the profile
#     likelihood is NA for every gene of the call. new("dmDSprecision") then
#     stops with 'invalid object for slot "genewise_precision" ... got class
#     "logical", should be or extend class "numeric"', and every gene of the
#     chunk is lost. Coercing the result to double keeps those genes NA
#     (converged = FALSE below) and lets the call finish. A chunk that did not
#     hit this case is unaffected. Always on.
#     dmFit() has the same problem one step later: when every gene's fit is NA,
#     dmDS_fit() / bbDS_fit() return an all-NA logical `lik`, and the
#     dmDSfit slots lik_full / lik_full_bb fail the same check. Their `lik`
#     is coerced the same way.
local({
  ns <- asNamespace("DRIMSeq")
  orig_prec <- get("dmDS_estimateTagwisePrecision", envir = ns)
  assignInNamespace("dmDS_estimateTagwisePrecision", function(...) {
    r <- orig_prec(...)
    storage.mode(r) <- "double"
    r
  }, ns = "DRIMSeq")
  for (fname in c("dmDS_fit", "bbDS_fit")) {
    local({
      orig_fit <- get(fname, envir = ns)
      assignInNamespace(fname, function(...) {
        r <- orig_fit(...)
        if (!is.null(r[["lik"]])) storage.mode(r[["lik"]]) <- "double"
        r
      }, ns = "DRIMSeq")
    })
  }
})

# (2) Why the profile likelihood is NA. dm_fitRegression() fits the
#     coefficients with optim(method = "BFGS", reltol = 1e-12) and R's default
#     maxit = 100. With ~20 covariates and K PAS there are 21 x (K - 1)
#     coefficients, and many fits stop at the iteration limit; DRIMSeq then
#     returns NA for that precision grid point, and a gene with any NA grid
#     point gets genewise precision NA. optim_maxit > 0 replaces
#     dm_fitRegression with a copy of the DRIMSeq 1.36.0 code whose only change
#     is control$maxit. A fit that converged within 100 iterations takes the
#     same path and returns the same value. Measured on 30 synthetic genes
#     (418 samples, 21-column design): with workaround (4) alone 6 genes
#     still had NA precision, with maxit = 1000 none. dmqtl passes 1000 by
#     default; 0 / empty keeps R's 100. More genes then enter the
#     common-precision sum, which moves the grid centre (see (4)).
if (optim_maxit > 0) {
  dm_fitRegression_maxit <- function(y, design, prec, coef_mode = "optim", coef_tol = 1e-12) {
    q <- nrow(y)
    p <- ncol(design)
    n <- ncol(y)
    if (q < 2 || is.na(prec)) {
      b <- matrix(NA, nrow = q, ncol = p)
      prop <- matrix(NA, nrow = q, ncol = n)
      rownames(prop) <- rownames(y)
      rownames(b) <- rownames(y)
      return(list(b = b, lik = NA, fit = prop))
    }
    yt <- t(y) + 1
    prop <- yt / rowSums(yt)
    logit_prop <- log(prop / prop[, q])
    b_init <- c(MASS::ginv(design) %*% logit_prop[, -q, drop = FALSE])
    if (coef_mode != "optim") stop("dmqtl optim_maxit patch supports coef_mode = 'optim' only")
    co <- optim(par = b_init, fn = dm_lik_regG, gr = dm_score_regG,
                x = design, prec = prec, y = y, method = "BFGS",
                control = list(fnscale = -1, reltol = coef_tol, maxit = DMQTL_OPTIM_MAXIT))
    if (co$convergence == 0) {
      b <- rbind(t(matrix(co$par, p, q - 1)), rep(0, p))
      lik <- co$value
    } else {
      b <- matrix(NA, nrow = q, ncol = p)
      lik <- NA
    }
    if (!is.na(lik)) {
      z <- exp(design %*% t(b[-q, , drop = FALSE]))
      prop <- z / (1 + rowSums(z))
      prop <- t(cbind(prop, 1 - rowSums(prop)))
    } else {
      prop <- matrix(NA, nrow = q, ncol = n)
    }
    rownames(prop) <- rownames(y)
    rownames(b) <- rownames(y)
    list(b = b, lik = lik, fit = prop)
  }
  environment(dm_fitRegression_maxit) <- list2env(list(DMQTL_OPTIM_MAXIT = optim_maxit),
                                                   parent = asNamespace("DRIMSeq"))
  assignInNamespace("dm_fitRegression", dm_fitRegression_maxit, ns = "DRIMSeq")
  message("[r_null_fit_batch] optim maxit = ", optim_maxit, " (DRIMSeq ", packageVersion("DRIMSeq"), ")")
}

# (4) det() overflow in the Cox-Reid adjustment -- the cause of NA precision
#     seen with ~20 covariates. dm_CRadjustmentRegression() computes
#     log(det(n * (-H))) / 2 for the p(K-1)-dimensional Hessian H. R's det()
#     is sign * exp(log|det|); for p = 21 and K >= 4 (an 84 x 84 matrix scaled
#     by n ~ 418) that exceeds the double range, det() returns Inf, and
#     DRIMSeq turns the infinite adjustment into NA (then also tries a slow
#     optimHess() fallback that overflows the same way). Taking
#     determinant(., logarithm = TRUE)$modulus directly gives the same value
#     up to rounding whenever det() did not overflow, and the same NA for a
#     singular or indefinite matrix. On by default (cr_logdet = 1); 0 restores
#     DRIMSeq's code. Besides rescuing the overflowing genes, this also changes
#     the common precision: DRIMSeq sums the profile likelihood over genes with
#     na.rm = TRUE, so at precisions where many genes overflowed the sum was
#     taken over the few remaining genes, which pulled the common precision
#     (the centre of every gene's grid) upwards.
if (cr_logdet) {
  dm_CRadjustmentRegression_logdet <- function(y, x, prec, prop) {
    if (any(is.na(prop[1, ])))
      return(NA)
    prop <- t(prop)
    n <- nrow(prop)
    q <- ncol(prop)
    half_logdet <- function(M) {
      z <- determinant(M, logarithm = TRUE)
      if (z$sign <= 0) return(if (is.finite(z$modulus)) NaN else -Inf)  # as log(det()) would
      as.numeric(z$modulus) / 2
    }
    H <- dm_Hessian_regG_prop(y = t(y), prec = prec, prop = prop, x = x)
    adj <- half_logdet(n * (-H))
    if (is.na(adj)) {
      logit_prop <- log(prop / prop[, q])
      par <- c(MASS::ginv(x) %*% logit_prop[, -q, drop = FALSE])
      # optimHess() stops with "non-finite value supplied by optim" when the
      # likelihood is not finite at the start values (seen on real Ast data);
      # inside bplapply that error took down every gene of the call. Treat it
      # like a singular Hessian: NA for this grid point.
      H <- tryCatch(optimHess(par = par, fn = dm_lik_regG, gr = dm_score_regG, x = x, prec = prec, y = y),
                    error = function(e) NULL)
      if (is.null(H))
        return(NA)
      adj <- half_logdet(n * (-H))
      if (is.na(adj))
        return(NA)
    }
    if (abs(adj) == Inf)
      return(NA)
    adj
  }
  environment(dm_CRadjustmentRegression_logdet) <- asNamespace("DRIMSeq")
  assignInNamespace("dm_CRadjustmentRegression", dm_CRadjustmentRegression_logdet, ns = "DRIMSeq")
}

# (3) Reproducibility. With prec_subset < 1 (DRIMSeq's default is 0.1),
#     dmPrecision sample()s the genes used for the common precision; the
#     script never set a seed, so two runs on the same data could give
#     different fits. rng_seed fixes the subset.
if (!is.na(rng_seed)) set.seed(rng_seed)

counts_df <- read.csv(counts_path, check.names = FALSE, colClasses = "character")
covar_mat <- as.matrix(read.csv(covar_path, check.names = FALSE))

n <- nrow(covar_mat)
sample_cols <- colnames(counts_df)[-(1:2)]

if (length(sample_cols) != n) {
  stop(
    "The number of sample columns in counts_long.csv (",
    length(sample_cols),
    ") does not match the number of sample rows in covariates.csv (",
    n,
    ")"
  )
}

sample_ids <- paste0("S", seq_len(n))
colnames(counts_df) <- c("gene_id", "feature_id", sample_ids)

# The sample count columns remain character strings after
# read.csv(colClasses="character"). dmDSdata requires numeric counts,
# so convert all sample columns back to numeric while keeping
# gene_id and feature_id as character columns.
for (s in sample_ids) counts_df[[s]] <- as.numeric(counts_df[[s]])

samples_df <- data.frame(sample_id = sample_ids, covar_mat, check.names = FALSE)

d <- dmDSdata(counts = counts_df, samples = samples_df)

covariate_names <- colnames(covar_mat)
non_intercept_names <- covariate_names[covariate_names != "Intercept"]
rhs <- if (length(non_intercept_names) == 0) "1" else paste(non_intercept_names, collapse = " + ")
design_formula <- as.formula(paste("~", rhs))
design <- model.matrix(design_formula, data = samples_df)

bpparam <- if (n_cores > 1) {
  BiocParallel::MulticoreParam(workers = n_cores)
} else {
  BiocParallel::SerialParam()
}


# *** This is the core change: the entire batch (possibly all genes) is
# passed to dmPrecision in a single call so that the common dispersion
# trend is estimated jointly across genes.
#
# This avoids the degenerate single-gene case in r_null_fit.R, where
# there are no neighboring genes from which information can be borrowed.
#
# 2026-09 investigation (fully documented here to avoid repeating the
# same debugging process):
#
# The first investigation ruled out `prec_subset`: it affects the accuracy
# of estimating the common trend itself, but does not determine whether
# that trend is used for shrinkage.
#
# The second investigation supplied a positive `prec_prior_df`, but the
# results were exactly identical to the default. This initially raised
# the suspicion that `prec_moderation` had not been enabled because its
# default was incorrectly assumed to be "none".
#
# The third investigation consulted the official `?DRIMSeq::dmPrecision`
# documentation and established two facts:
#
#   (1) For the `dmDSdata` method dispatch used here, the default
#       `prec_moderation` is actually "trended". The previous assumption
#       of "none" resulted from confusing it with the default for the
#       `dmSQTLdata` method.
#
#   (2) `prec_prior_df` normally has no effect because DRIMSeq automatically
#       estimates the amount of shrinkage from the data. The user-supplied
#       `prec_prior_df` is used only when automatic estimation fails
#       (officially, when fewer than 10 genes are at the grid boundaries).
#       This explains why setting it to 5.0 produced no change.
#
# Conclusion:
#
# The shrinkage strength is not a direct tuning parameter controlled by
# this package (or by the caller of dmPrecision). It is determined by
# DRIMSeq from the data.
#
# The parameters exposed here control only:
#
#   - the direction/type of moderation (`prec_moderation`:
#     "common" vs. "trended"), and
#   - the smoothing-window size (`prec_span`, not exposed here and
#     defaulting to 0.1).
#
# With very few genes, for example a smoke test containing only a dozen
# genes, `prec_span=0.1` corresponds to a smoothing window covering only
# one or two genes. Such a configuration is intrinsically degenerate, so
# little or no shrinkage may be a reasonable result from DRIMSeq rather
# than a configuration error.
#
# With realistic chunks containing thousands of genes, this degeneracy
# is unlikely to occur. The behavior should nevertheless be verified
# empirically on the actual dataset rather than inferred from a small
# synthetic test.
prec_args <- list(x = d, design = design, BPPARAM = bpparam)
if (!is.null(prec_subset)) prec_args$prec_subset <- prec_subset
if (!is.null(prec_prior_df)) prec_args$prec_prior_df <- prec_prior_df
if (!is.null(prec_moderation)) prec_args$prec_moderation <- prec_moderation
d <- do.call(dmPrecision, prec_args)
d <- dmFit(d, design = design, BPPARAM = bpparam)

prec_tab <- genewise_precision(d)  # Columns: gene_id, genewise_precision
prec_by_gene <- setNames(as.numeric(prec_tab$genewise_precision), prec_tab$gene_id)

# *** addition: detect cases where precision estimation reaches
# the upper boundary of the grid search.
#
# This is a real phenomenon observed during testing with 300 genes,
# rather than purely defensive programming.
#
# Internally, dmPrecision uses a grid search for each gene. With the
# default `prec_grid_range=c(-10,10)` and `prec_grid_length=21`, the grid
# is approximately:
#
#     prec_init * 2^seq(-10, 10, length=21)
#
# where prec_init is the common_precision estimated from the batch.
#
# If a gene has extremely low overdispersion, for example when its observed
# composition contains almost no variability beyond multinomial sampling
# noise, its unconstrained MLE for precision may tend toward infinity.
# The grid search can then only return the largest available grid value.
#
# Such a value is numerically valid (not NA/NaN), but it does not represent
# a meaningful finite precision estimate. Passing it directly to the score
# test can substantially overestimate the precision of the null model and
# make downstream p-values unreliable.
#
# This behavior was observed empirically: with common_precision approximately
# 55.73, one gene had genewise_precision exactly equal to
#
#     55.73 * 2^10 ≈ 57065.7,
#
# which is the theoretical upper boundary of the default grid.
#
# Here we handle only the case where the default `prec_grid_range` is used.
# If `prec_grid_range` is exposed as a parameter in the future, this
# hard-coded upper-bound exponent must be updated accordingly.
prec_grid_range_upper_default <- 10
common_prec_val <- tryCatch(common_precision(d), error = function(e) NA_real_)
if (length(common_prec_val) != 1) common_prec_val <- NA_real_
grid_max <- if (!is.na(common_prec_val)) {
  common_prec_val * 2^prec_grid_range_upper_default
} else {
  NA_real_
}

# *** 2026-09 second-level extension:
#
# An exact grid-boundary hit is only the extreme form of this problem.
# Testing with 300 genes confirmed that, even after removing the two genes
# that hit the exact boundary, some remaining genes still had substantially
# inflated precision estimates (for example, max approximately 4845 versus
# a median of approximately 59).
#
# The underlying cause is related to the same grid parameterization:
# all genes in a batch share the same grid center (`common_precision`).
# For genes whose true precision is far from this common center, the grid
# resolution is inherently poorer.
#
# Therefore, the problem is not limited to genes that exactly hit the
# theoretical upper boundary. Some genes can instead have clearly inflated
# estimates without reaching the boundary.
#
# In addition to exact boundary detection, perform a robust outlier check
# based on the median precision of the current batch after excluding
# boundary hits.
#
# A gene is flagged when
#
#     |log2(phi / median_phi)| > outlier_log2_threshold.
#
# The default threshold is 5, corresponding to a deviation greater than
# 32-fold from the batch median.
#
# This is an empirical threshold rather than a theoretically derived
# criterion. Different datasets, sample sizes, or count distributions may
# require a different threshold. The value can be adjusted through the
# `outlier_log2_threshold` CLI parameter.
n_boundary_hit <- 0
boundary_hit_by_gene <- setNames(rep(FALSE, length(prec_by_gene)), names(prec_by_gene))
for (gid in names(prec_by_gene)) {
  phi_g <- prec_by_gene[[gid]]
  hit <- !is.na(phi_g) && !is.na(grid_max) && phi_g >= 0.999 * grid_max
  boundary_hit_by_gene[[gid]] <- hit
  if (hit) n_boundary_hit <- n_boundary_hit + 1
}
non_boundary_phi <- prec_by_gene[!is.na(prec_by_gene) & !boundary_hit_by_gene[names(prec_by_gene)]]
median_phi_ref <- if (length(non_boundary_phi) > 0) median(non_boundary_phi) else NA_real_
n_outlier <- 0
outlier_by_gene <- setNames(rep(FALSE, length(prec_by_gene)), names(prec_by_gene))
if (!is.na(median_phi_ref) && median_phi_ref > 0) {
  for (gid in names(prec_by_gene)) {
    phi_g <- prec_by_gene[[gid]]
    if (is.na(phi_g) || boundary_hit_by_gene[[gid]]) next
    is_outlier <- abs(log2(phi_g / median_phi_ref)) > outlier_log2_threshold
    outlier_by_gene[[gid]] <- is_outlier
    if (is_outlier) n_outlier <- n_outlier + 1
  }
}

coefs <- coefficients(d)  # Columns: gene_id, feature_id, <design coefficient columns>

design_cols <- colnames(design)
coef_cols <- setdiff(colnames(coefs), c("gene_id", "feature_id"))
design_cols_mangled <- make.names(design_cols)
if (!setequal(coef_cols, design_cols_mangled)) {
  stop(
    "The coefficient column names returned by dmFit do not match ",
    "the escaped design-matrix column names; manual inspection is required: ",
    "coef_cols=[", paste(coef_cols, collapse = ","), "] ",
    "design_cols (original)=[", paste(design_cols, collapse = ","), "] ",
    "design_cols (escaped)=[", paste(design_cols_mangled, collapse = ","), "]"
  )
}
if (any(duplicated(coefs[c("gene_id", "feature_id")]))) {
  stop(
    "The same (gene_id, feature_id) pair occurs more than once in ",
    "coefficients(d). The output structure does not match the expected ",
    "format and requires manual inspection."
  )
}

all_gene_ids <- unique(counts_df$gene_id)

# The original feature order for each gene is determined by the first
# occurrence of that gene_id in counts_long.csv. This preserves the same
# convention as the single-gene r_null_fit.R version:
#
#   PAS order follows the input order, and the last PAS is the reference.
gene_feature_order <- split(counts_df$feature_id, counts_df$gene_id)
gene_feature_order <- gene_feature_order[all_gene_ids]  # 保持 all_gene_ids 的顺序

phi_rows <- vector("list", length(all_gene_ids))
eta_rows <- list()
n_missing_prec <- 0

for (gi in seq_along(all_gene_ids)) {
  gid <- all_gene_ids[gi]
  pas_ids <- gene_feature_order[[gid]]
  K <- length(pas_ids)

  phi_g <- if (gid %in% names(prec_by_gene)) prec_by_gene[[gid]] else NA_real_
  if (is.na(phi_g)) n_missing_prec <- n_missing_prec + 1
  boundary_hit <- !is.na(phi_g) && isTRUE(boundary_hit_by_gene[[gid]])
  outlier_flag <- !is.na(phi_g) && isTRUE(outlier_by_gene[[gid]])
  phi_rows[[gi]] <- data.frame(gene_id = gid, phi = phi_g,
                                converged = !is.na(phi_g) && !boundary_hit && !outlier_flag,
                                boundary_hit = boundary_hit, outlier_flag = outlier_flag)

  coefs_g <- coefs[coefs$gene_id == gid, , drop = FALSE]
  missing_features <- setdiff(pas_ids, coefs_g$feature_id)
  if (length(missing_features) > 1) {
    stop("Gene ", gid, ": coefficients(d) is missing ", length(missing_features),
         " features, which exceeds the number allowed by any known DRIMSeq reference-category convention (at most one complete row).")
  }

  eta_raw <- matrix(0, nrow = n, ncol = K)
  colnames(eta_raw) <- pas_ids
  for (i in seq_len(nrow(coefs_g))) {
    fid <- coefs_g$feature_id[i]
    coef_vec <- unlist(coefs_g[i, design_cols_mangled, drop = FALSE], use.names = FALSE)
    eta_raw[, fid] <- as.numeric(design %*% coef_vec)
  }
  # Convert to the convention that the last feature within each gene is
  # the reference category. This is consistent with the single-gene
  # version and the Python-side dmglm.softmax_ref convention.
  #
  # See the derivation in the header of r_null_fit.R.
  eta <- eta_raw[, seq_len(K - 1), drop = FALSE] - eta_raw[, K]

  # eta has shape (n individuals, K-1 features), with individuals as rows.
  #
  # The output CSV convention is the transpose:
  # one row per feature and one column per sample, matching the structure
  # of counts_long.csv. The Python-side fit_null_drimseq_batch() reads the
  # output using this convention.
  #
  # The transpose is therefore essential. Previously, eta was inserted
  # directly into data.frame() together with a feature_id vector of length
  # K-1. When n happened to be an integer multiple of K-1, data.frame()
  # could recycle feature_id to n rows without raising an error, silently
  # producing a data frame with the wrong dimensions and incorrect content.
  # When n was not a multiple of K-1, the problem surfaced later as an
  # error during colnames assignment or rbind.
  #
  # Both cases had the same root cause: eta had the wrong orientation.
  # This must be fixed by explicitly transposing the matrix rather than
  # adding a downstream tolerance.
  block <- data.frame(gene_id = gid, feature_id = pas_ids[seq_len(K - 1)],
                       t(eta), check.names = FALSE)
  colnames(block)[-(1:2)] <- sample_ids
  eta_rows[[gid]] <- block
}

if (n_missing_prec > 0) {
  message(
    "[r_null_fit_batch] WARNING: ",
    n_missing_prec,
    " / ",
    length(all_gene_ids),
    " genes have missing genewise_precision values ",
    "(dmPrecision did not converge for these genes or they were filtered). ",
    "They are marked as converged=FALSE in phi.csv. Downstream code must ",
    "decide explicitly whether to skip them or use a global-median phi as ",
    "a fallback. Do not silently pass 0 or NA into the score test."
  )
}

if (n_boundary_hit > 0) {
  message(
    "[r_null_fit_batch] WARNING: ",
    n_boundary_hit,
    " / ",
    length(all_gene_ids),
    " genes have genewise_precision values at the upper boundary of ",
    "the grid search (>= common_precision * 2^",
    prec_grid_range_upper_default,
    " ~= ",
    round(grid_max, 2),
    "). These genes most likely have true precision tending toward ",
    "infinity, indicating almost no variation beyond multinomial sampling ",
    "noise. They are marked as converged=FALSE in phi.csv. This is a real ",
    "phenomenon confirmed empirically using 300 synthetic genes in 2026-09, ",
    "rather than merely defensive programming. These saturated boundary ",
    "values should not be treated as ordinary precision estimates and ",
    "passed directly into the score test, because doing so can substantially ",
    "overestimate null-model precision and make downstream p-values unreliable."
  )
}

if (n_outlier > 0) {
  message(
    "[r_null_fit_batch] WARNING: an additional ",
    n_outlier,
    " / ",
    length(all_gene_ids),
    " genes have genewise_precision values that deviate from the batch ",
    "median (",
    round(median_phi_ref, 2),
    ") by more than 2^",
    outlier_log2_threshold,
    " ~= ",
    round(2^outlier_log2_threshold, 1),
    "-fold. Although these genes do not hit the exact theoretical grid ",
    "boundary, empirical testing indicates that such estimates can also ",
    "be substantially less stable than independent single-gene fits. ",
    "In a 2026-09 experiment with 300 synthetic genes, after excluding ",
    "exact boundary hits, some genes still had precision estimates more ",
    "than an order of magnitude larger than their independent estimates. ",
    "The underlying cause is that all genes in a batch share the same ",
    "grid center, so genes whose true precision is far from that center ",
    "have intrinsically poorer grid resolution. These genes are marked ",
    "as converged=FALSE in phi.csv. The threshold ",
    "(outlier_log2_threshold, default 5) is empirical rather than ",
    "theoretically derived, and may need to be adjusted for different ",
    "datasets, sample sizes, or count distributions."
  )
}

phi_df <- do.call(rbind, phi_rows)
eta_df <- do.call(rbind, eta_rows)

write.csv(phi_df, paste0(out_prefix, "_phi.csv"), row.names = FALSE)
write.csv(eta_df, paste0(out_prefix, "_eta.csv"), row.names = FALSE)
