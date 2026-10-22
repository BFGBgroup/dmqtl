"""
Null model fitting (PAS ~ covariates, without SNPs).

Two fitting paths are available, with the first one recommended for
production use:

1. `fit_null_drimseq()` — recommended production path. It calls Rscript
   through subprocess to run DRIMSeq (`dmPrecision`/`dmFit`) and takes
   advantage of its mature cross-gene dispersion (precision, i.e. `phi`
   here) shrinkage. This requires R and the DRIMSeq package to be installed
   in the user's environment.

2. `fit_null_python()` — pure NumPy/SciPy fallback MLE. Each gene estimates
   `phi` independently and does not borrow information from other genes.

The difference is not simply that "DRIMSeq is somewhat more sophisticated".
Under sample sizes and sequencing depths commonly encountered in real PAS
count data, independent per-gene MLE can provide highly unreliable estimates
of `phi`. This has been observed empirically rather than inferred only from
theory: when repeatedly simulating data with a true `phi = 60`, independent
MLE estimates with n = 15 or 30 individuals can have means and standard
deviations reaching tens of thousands or even hundreds of thousands, even
after constraining `phi` to [0.01, 1e6] to prevent numerical explosion.
See the documentation of `fit_null_python` and `tests/test_score_math.py`.

This is the fundamental reason why methods such as DRIMSeq, edgeR, and
DESeq2 use cross-gene dispersion/precision shrinkage. The data from an
individual gene are often insufficient to estimate a variance-related
parameter reliably, so information must be borrowed across genes, for
example through empirical Bayes shrinkage.

Severely under- or over-estimated `phi` directly affects the variance of the
score test and therefore distorts p-values and FDR control. Thus, precision
shrinkage is not merely an optional refinement; it can materially affect
the reliability of the final results.

Accordingly, `eqtl.run_dmqtl` defaults to `use_drimseq="auto"`. It
automatically checks whether Rscript and DRIMSeq are available. If they are,
it uses `fit_null_drimseq` and prints a one-time reminder that the
reference-category convention still needs to be manually verified (see the
docstring of `fit_null_drimseq`). If DRIMSeq is unavailable, it falls back
to `fit_null_python` and prints a warning that this is a degraded fallback
whose variance estimation may be unreliable for small-sample genes.

The user can explicitly select `use_drimseq="yes"` or `"no"`, or completely
override the default behavior by passing a custom `null_fit_fn=`.

The current sandbox has no network access and does not have R installed, so
the `fit_null_drimseq` path cannot be executed or validated here. The
accompanying R script (`r_null_fit.R`) follows the standard DRIMSeq calling
pattern. On a machine with a working R environment, first run
`tests/test_r_bridge_smoke.py` to verify that the R bridge works. In
addition, as described by that test and the `fit_null_drimseq` docstring,
the reference-category convention should be manually checked using a small
example with known true coefficients. This cannot be fully automated and
cannot be validated in the current sandbox.
"""
from __future__ import annotations

import csv
import functools
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .dmglm import dm_loglik, softmax_ref

_THIS_DIR = Path(__file__).resolve().parent
R_SCRIPT = _THIS_DIR / "r_null_fit.R"
R_SCRIPT_BATCH = _THIS_DIR / "r_null_fit_batch.R"


@dataclass
class NullFit:
    beta: np.ndarray      # (p, K-1) null covariate coefficients
    phi: float            # dispersion/precision scalar from DRIMSeq shrinkage or fallback MLE
    eta: np.ndarray        # (n, K-1) fitted linear predictors X @ beta
    loglik: float
    n_categories: int
    converged: bool
    method: str            # "drimseq" or "python_fallback"


def _pack(beta: np.ndarray, log_phi: float) -> np.ndarray:
    return np.concatenate([beta.ravel(), [log_phi]])


def _unpack(theta: np.ndarray, p: int, km1: int):
    beta = theta[: p * km1].reshape(p, km1)
    phi = float(np.exp(theta[-1]))
    return beta, phi


def fit_null_python(y: np.ndarray, X: np.ndarray,
                     beta0: np.ndarray | None = None,
                     phi0: float = 50.0,
                     ridge: float = 1e-3,
                     phi_bounds: tuple[float, float] = (1e-2, 1e6)) -> NullFit:
    """Pure NumPy/SciPy MLE without cross-gene shrinkage.

    This function is intended only for testing and as a fallback when the
    DRIMSeq-based implementation is unavailable.

    Parameters
    ----------
    ridge
        A small L2 penalty applied to non-intercept beta coefficients
        (default: 1e-3).

    phi_bounds
        Search range for the precision parameter `phi`, defaulting to
        [0.01, 1e6]. The range is deliberately wide but finite.

    Motivation for the ridge penalty
    ---------------------------------
    A real numerical stability problem has been observed in sparse PAS
    count data rather than merely assumed theoretically: quasi-separation
    can easily occur when a PAS category has nearly zero counts over a
    range of values of a covariate.

    Under unpenalized MLE, the corresponding beta coefficient can be driven
    toward +/- infinity so that the likelihood continues to increase.
    This is the classical separation problem known from logistic and
    multinomial regression, and the same phenomenon can occur in a
    DM-GLM.

    With an unbounded objective, L-BFGS-B can only stop after reaching the
    iteration limit or exhausting numerical precision. The resulting `eta`
    values may become large enough to cause underflow in the softmax
    calculation.

    A small ridge penalty keeps beta finite in such cases. For typical
    non-degenerate data, it has little effect on the fitted values; see
    the calibration/LRT consistency tests in `tests/test_score_math.py`,
    which continue to pass after adding the penalty.

    The intercept (the first row of beta under the default design) is not
    penalized. Only the remaining covariate coefficients are penalized.

    Motivation for bounding phi
    ----------------------------
    Another numerical issue was identified empirically rather than
    theoretically: `phi` was previously completely unconstrained.

    With sample sizes and sequencing depths commonly observed in real PAS
    count data, for example n = 15-30 individuals with only tens of total
    PAS counts per individual, `phi` is weakly identifiable from the
    Dirichlet-Multinomial likelihood.

    In an experiment with n = 30 and 30 repeated simulations, unconstrained
    MLE produced estimates with means and standard deviations on the order
    of 1e80, even though the true value was only 60.

    This is precisely the type of instability that motivates cross-gene
    precision/dispersion shrinkage in methods such as DRIMSeq: the data from
    a single gene may be insufficient to estimate `phi` reliably, so
    information should be borrowed across genes.

    The Python fallback does not perform cross-gene shrinkage (see the
    module-level documentation), but it should at least constrain `phi`
    to a wide yet finite range rather than allowing the optimizer to drive
    it to values such as 1e80, where even float64 precision becomes
    unreliable.

    Extremely large `phi` values make the DM distribution approach a
    nearly deterministic multinomial distribution and can therefore lead
    to poorly behaved or directionally distorted downstream score-test
    variance estimates.

    The L-BFGS-B `bounds` argument constrains `log(phi)` to the logarithmic
    range corresponding to `phi_bounds`. With the default [0.01, 1e6],
    this range is sufficiently wide that normally converged fits should
    remain unaffected.
    """
    n, K = y.shape
    p = X.shape[1]
    km1 = K - 1
    if beta0 is None:
        beta0 = np.zeros((p, km1))
    theta0 = _pack(beta0, np.log(phi0))

    def neg_ll(theta):
        beta, phi = _unpack(theta, p, km1)
        eta = X @ beta
        pi = softmax_ref(eta)
        nll = -dm_loglik(y, pi, phi)
        if p > 1:
            nll = nll + ridge * np.sum(beta[1:] ** 2)
        return nll

    n_theta = theta0.shape[0]
    bounds = [(None, None)] * (n_theta - 1) + \
             [(np.log(phi_bounds[0]), np.log(phi_bounds[1]))]

    res = minimize(neg_ll, theta0, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10})
    beta, phi = _unpack(res.x, p, km1)
    eta = X @ beta
    return NullFit(beta=beta, phi=phi, eta=eta, loglik=-res.fun,
                    n_categories=K, converged=bool(res.success),
                    method="python_fallback")


def fit_null_drimseq(y: np.ndarray, X: np.ndarray, sample_ids: list[str],
                      covariate_names: list[str],
                      rscript_bin: str = "Rscript",
                      timeout: int = 600) -> NullFit:
    """Fit the null model through DRIMSeq via subprocess.

    This is the recommended production path for Option C.

    Requirements
    ------------
    A working R environment with DRIMSeq installed, for example:

        BiocManager::install("DRIMSeq")

    This function writes counts and covariates to a temporary directory,
    invokes `r_null_fit.R`, and reads the resulting JSON output.

    Notes
    -----
    This path cannot be executed in the current sandbox because R is not
    installed. The interface and standard invocation pattern are provided
    here for use on a real R-enabled machine.

    Before using this path for real or simulated data, verify the following:

    1. `eta` (the linear predictor) is now calculated on the R side and
       returned unchanged. Python no longer recomputes `X @ beta` using the
       X matrix passed by the caller.

       This fixes a real robustness issue identified previously: if the
       column order of the Python-side X matrix differed from the design
       matrix reconstructed by R's `model.matrix()`, Python could silently
       compute an incorrect `eta` without raising an error.

       Both sides now use the R-generated design matrix, eliminating this
       source of column-order mismatch.

    2. The reference-category convention (i.e. which PAS is treated as the
       omitted/reference category among the K categories) must agree with
       the Python implementation in `dmglm.softmax_ref` and
       `residual_matrix`, where the last category is treated as the
       reference.

       This still depends on the internal implementation details of DRIMSeq
       and cannot be verified without a working R environment. It should
       therefore be checked end-to-end using a small example with known
       true coefficients.
    """
    n, K = y.shape
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        counts_path = tmp / "counts.csv"
        covar_path = tmp / "covariates.csv"
        out_path = tmp / "null_fit.json"

        np.savetxt(counts_path, y, delimiter=",",
                   header=",".join(f"PAS{i}" for i in range(K)),
                   comments="")
        np.savetxt(covar_path, X, delimiter=",",
                   header=",".join(covariate_names), comments="")

        cmd = [rscript_bin, str(R_SCRIPT),
               str(counts_path), str(covar_path), str(out_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"DRIMSeq null fit failed "
                f"(Rscript exit code {proc.returncode}).\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        with open(out_path) as f:
            result = json.load(f)

    beta = np.array(result["beta"])           # (p, K-1), aligned to R design-matrix columns
    phi = float(result["phi"])

   # Use the eta calculated by R directly instead of recomputing X @ beta
    # in Python using the caller-provided X matrix. See the docstring above
    # for the motivation and robustness issue.

    eta = np.array(result["eta"])              # (n, K-1)

    # `r_null_fit.R` currently returns NA for loglik. `jsonlite::toJSON`
    # serializes NA as JSON null, which `json.load` reads as Python None.
    #
    # Note that dict.get(key, default) uses the default only when the key
    # does not exist. If the key exists but its value is None, None is still
    # returned. Therefore, `float(result.get("loglik", np.nan))` would raise
    # a TypeError. Explicitly handle None here.
    #
    # The same defensive handling is applied to `converged`, although the
    # current R implementation always returns TRUE.

    raw_loglik = result.get("loglik")
    loglik = float(raw_loglik) if raw_loglik is not None else float("nan")
    raw_converged = result.get("converged")
    converged = bool(raw_converged) if raw_converged is not None else True
    return NullFit(beta=beta, phi=phi, eta=eta,
                    loglik=loglik,
                    n_categories=K, converged=converged,
                    method="drimseq")


def fit_null_drimseq_batch(gene_ys: dict[str, np.ndarray], X: np.ndarray,
                            covariate_names: list[str],
                            rscript_bin: str = "Rscript",
                            timeout: int = 3600,
                            n_cores: int = 1,
                            prec_subset: float | None = None,
                            prec_prior_df: float | None = None,
                            prec_moderation: str | None = None,
                            outlier_log2_threshold: float | None = None,
                            verbose: bool = False) -> dict[str, NullFit]:
    """Batch version of `fit_null_drimseq`.

    This is the key function that allows DRIMSeq cross-gene precision
    shrinkage to actually operate. Production analyses should use this
    function rather than calling `fit_null_drimseq` separately for every
    gene.

    Background
    ----------
    A real bug was identified while investigating results from the
    `dmqtl_fixed3` version on real data. `fit_null_drimseq` previously
    received only one gene per call. With a single gene in `dmDSdata`,
    `dmPrecision` has no other genes from which to estimate a
    dispersion-vs-mean trend. In practice, the resulting `phi` is therefore
    effectively a single-gene estimate.

    This means that the previously emphasized motivation for choosing
    DRIMSeq over the Python fallback — cross-gene precision shrinkage —
    was not actually realized by an architecture that launched a separate
    DRIMSeq process for every gene.

    This function instead packages multiple genes into a single
    `dmDSdata` / `dmPrecision` / `dmFit` call (see `r_null_fit_batch.R`).
    The dispersion trend is then estimated jointly across the genes in the
    batch, allowing each gene's precision estimate to be shrunk toward the
    shared trend.

    Parameters
    ----------
    gene_ys
        Mapping `{gene_id: y}`, where `y` is an `(n, K_g)` PAS count matrix.
        `K_g` may differ between genes.

        The dictionary iteration order determines the order of genes in the
        output but does not affect numerical results. Each gene's `eta` and
        `phi` depend only on its own data and the shared dispersion trend
        estimated from the batch, not on dictionary iteration order.

    X
        Null covariate design matrix of shape `(n, p)`, including the
        intercept.

        **All genes must share the same X.** This is a prerequisite of this
        function and of `pipeline.fit_null_models_batched`. Because the null
        model contains covariates but no SNP term, X should not vary across
        genes.

        If the covariate design genuinely differs by gene in a particular
        application, this function is not appropriate; use separate
        `fit_null_drimseq` calls instead.

    n_cores
        Number of BiocParallel cores passed to the R-side
        `dmPrecision`/`dmFit` calls. This parallelizes multiple genes within
        a batch.

        It is independent of Python-side
        `run_dmqtl(n_workers=...)`, which controls parallelism during the
        score-test and permutation stage. These are two separate controls
        and should not be confused.

        For real datasets of approximately 20,000 genes, a reasonable
        starting point is to reserve some cores for the Python-side workers
        and assign the remaining cores to R. The optimal allocation is
        system-dependent and should be benchmarked on the target machine.

    prec_subset
        Value passed to `dmPrecision` as `prec_subset`. If omitted, DRIMSeq's
        default value of 0.1 is used.

        This parameter affects how accurately the shared dispersion trend is
        estimated. It does not determine whether shrinkage is performed;
        that behavior is controlled separately by `prec_moderation`.

    prec_moderation
        Value passed to `dmPrecision` as `prec_moderation`.

        Supported values are:

            "common"
                Shrink toward a single common precision.

            "trended"
                Shrink toward a precision-vs-mean trend.

            "none"
                No precision shrinkage.

        **Correction confirmed by the third empirical check in September
        2026:** for the `dmDSdata` method dispatch used here, the DRIMSeq
        default is `"trended"`, not `"none"`.

        `"none"` is the default for the different `dmSQTLdata` dispatch.
        These two method defaults were previously confused.

        Therefore, when this parameter is omitted for `dmDSdata`, DRIMSeq
        already performs trended precision shrinkage. If shrinkage appears
        absent, the reason is likely related to the behavior of
        `prec_prior_df` or to the data-dependent trend-fitting procedure,
        rather than the default `prec_moderation`.

    prec_prior_df
        Value passed to `dmPrecision` as `prec_prior_df`.

        The parameter does not directly specify shrinkage strength. DRIMSeq
        normally estimates the amount of shrinkage from the data itself,
        analogous in spirit to empirical-Bayes estimation of prior degrees
        of freedom.

        `prec_prior_df` is primarily a fallback value used when this
        automatic estimation fails, for example when fewer than 10 genes
        fall at a boundary of the relevant grid.

        Therefore, changing `prec_prior_df` may have no effect when DRIMSeq's
        automatic estimation succeeds. This explains why empirical tests
        with 15 genes produced identical results with and without
        `prec_prior_df=5.0`.

        In other words, shrinkage strength is not a directly exposed tuning
        parameter controlled by this package. DRIMSeq determines it
        data-dependently. The main controls available here are the direction
        of shrinkage (`prec_moderation`, such as `"common"` versus
        `"trended"`) and the smoothing window used for trend fitting
        (`prec_span` inside DRIMSeq, whose default is 0.1).

        `prec_span` is defined as a fraction of the number of genes. With
        very small gene sets, such as the 15-gene smoke test, 0.1 corresponds
        to only about 1.5 genes and therefore provides a highly degenerate
        smoothing window. In such a case, DRIMSeq may reasonably conclude
        that the trend is unreliable and avoid effective shrinkage.

        With real datasets and batches containing thousands of genes, this
        small-sample degeneration is generally not expected. Nevertheless,
        whether substantial shrinkage is observed remains a data-dependent
        decision made by DRIMSeq and cannot be guaranteed by this package.

    outlier_log2_threshold
        Optional log2-scale threshold passed to the R-side DRIMSeq fitting
        code for outlier handling.

    verbose
        If `True`, always print R stdout/stderr to `sys.stderr`, regardless
        of whether the R process succeeds or fails.

        This was added in September 2026 to fix a diagnostic blind spot.
        Previously, R-side `message()`/`print()` output, including useful
        diagnostics such as
        `Using ... as a shrinkage factor !` and information about whether
        the grid search reached a boundary, was shown only when the Rscript
        exited with a non-zero status.

        Consequently, a successful call with suspicious numerical results,
        such as unexpectedly large precision estimates, could silently
        discard the most useful diagnostic information. `verbose=True`
        makes these messages visible in both successful and failed runs.

    Returns
    -------
    dict[str, NullFit]
        Mapping `{gene_id: NullFit}`.

        If `dmPrecision` cannot provide a valid precision estimate for a
        gene, for example because of an extremely sparse or degenerate count
        pattern, the corresponding `NullFit.converged` is `False` and `phi`
        is `nan`.

        The caller (`pipeline.fit_null_models_batched`) must explicitly
        handle these cases and must not pass a `nan` `phi` into the score
        test.

    Additional convergence condition
    ---------------------------------
    A second type of `converged=False` case was identified empirically in
    September 2026 using 300 synthetic genes.

    For some genes, `genewise_precision` reaches the upper boundary of the
    internal `dmPrecision` grid search, corresponding to
    `common_precision * 2^10` under DRIMSeq's default
    `prec_grid_range=c(-10, 10)`.

    This indicates extremely low over-dispersion, where the true precision
    may effectively tend toward infinity. The reported value is therefore
    only the largest value available within the search range rather than a
    meaningful finite estimate.

    Such genes may have numerically valid-looking, non-NaN `phi` values, but
    the R-side `r_null_fit_batch.R` code marks them as `converged=False`.
    They are subsequently skipped by `pipeline.run_one_gene`, preventing a
    saturated boundary value from being silently treated as a valid
    precision estimate in the score test.
    """
    gene_ids = list(gene_ys.keys())
    n = X.shape[0]
    for gid, y in gene_ys.items():
        if y.shape[0] != n:
            raise ValueError(f"Gene {gid!r} has {y.shape[0]} rows in y, but X has {n} rows. "
                              "fit_null_drimseq_batch requires all genes to share the same individuals and the same X matrix.")

    sample_cols = [f"S{i+1}" for i in range(n)]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        counts_path = tmp / "counts_long.csv"
        covar_path = tmp / "covariates.csv"
        out_prefix = tmp / "batch_out"

        with open(counts_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gene_id", "feature_id", *sample_cols])
            for gid in gene_ids:
                y = gene_ys[gid]
                K = y.shape[1]
                for k in range(K):
                    # `feature_id` must be globally unique across the entire
                    # counts_long.csv file. DRIMSeq's dmDSdata checks
                    # `sum(duplicated(counts$feature_id)) == 0`, which applies
                    # across all genes, not merely within each gene.
                    #
                    # The single-gene implementation naturally avoids this
                    # constraint because each call contains only one gene.
                    # The batch implementation must therefore include the
                    # gene ID in the feature ID. Otherwise, every gene would
                    # start again at PAS0, causing the R-side check to fail
                    # with:
                    # "sum(duplicated(counts$feature_id)) == 0 is not TRUE".
                    w.writerow([gid, f"{gid}__PAS{k}", *[f"{v:.10g}" for v in y[:, k]]])

        with open(covar_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(covariate_names)
            for row in X:
                w.writerow([f"{v:.10g}" for v in row])

        cmd = [rscript_bin, str(R_SCRIPT_BATCH), str(counts_path), str(covar_path),
               str(out_prefix), str(int(n_cores)),
               "" if prec_subset is None else str(prec_subset),
               "" if prec_prior_df is None else str(prec_prior_df),
               "" if prec_moderation is None else str(prec_moderation),
               "" if outlier_log2_threshold is None else str(outlier_log2_threshold)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"DRIMSeq batch null fit failed "
                f"({len(gene_ids)} genes, Rscript exit code "
                f"{proc.returncode}).\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        if verbose:
            print(f"[fit_null_drimseq_batch] R stdout ({len(gene_ids)} 个 gene):\n"
                  f"{proc.stdout}", file=sys.stderr)
            if proc.stderr:
                print(f"[fit_null_drimseq_batch] R stderr:\n{proc.stderr}", file=sys.stderr)

        return _read_batch_outputs(out_prefix, gene_ids, gene_ys, X, sample_cols)


def _csv_float(s: str) -> float:
    """R's write.csv writes missing values as NA."""
    s = s.strip()
    return float("nan") if s in ("", "NA", "NaN") else float(s)


def _read_batch_outputs(out_prefix, gene_ids, gene_ys, X, sample_cols) -> dict[str, NullFit]:
    """Read <out_prefix>_phi.csv / _eta.csv written by r_null_fit_batch.R.

    Genes whose fit is NA (genewise precision NA -> NA coefficients -> NA
    eta) come back with phi = nan, NaN eta and converged=False, so
    run_one_gene skips them. Earlier, float("NA") on such an eta row raised
    ValueError and the whole chunk was lost (2026-10 fix).
    """
    n = X.shape[0]
    phi_by_gene: dict[str, tuple[float, bool]] = {}
    with open(f"{out_prefix}_phi.csv", newline="") as f:
        for row in csv.DictReader(f):
            phi_val = _csv_float(row["phi"])
            converged = row["converged"].strip().upper() in ("TRUE", "T", "1")
            phi_by_gene[row["gene_id"]] = (phi_val, converged)

    eta_blocks: dict[str, list[list[float]]] = {gid: [] for gid in gene_ids}
    with open(f"{out_prefix}_eta.csv", newline="") as f:
        for row in csv.DictReader(f):
            gid = row["gene_id"]
            vals = [_csv_float(row[c]) for c in sample_cols]
            eta_blocks[gid].append(vals)

    results: dict[str, NullFit] = {}
    for gid in gene_ids:
        K = gene_ys[gid].shape[1]
        phi, converged = phi_by_gene.get(gid, (float("nan"), False))
        rows = eta_blocks.get(gid, [])
        if len(rows) != K - 1:
            raise RuntimeError(
                f"Gene {gid!r}: expected {K - 1} rows of eta "
                f"(K-1, K={K}), but the R side returned {len(rows)} rows. "
                "This indicates a mismatch between the batch script and the "
                "K-category assumption here and requires manual inspection; "
                "do not silently pad with zeros."
            )
        eta = np.array(rows, dtype=float).T if rows else np.zeros((n, 0))
        if not np.isfinite(eta).all():
            converged = False  # never pass a non-finite linear predictor to the score test
        results[gid] = NullFit(beta=np.full((X.shape[1], max(K - 1, 0)), np.nan),
                                phi=phi, eta=eta, loglik=float("nan"),
                                n_categories=K, converged=converged, method="drimseq_batch")
    return results


def drimseq_available(rscript_bin: str = "Rscript") -> bool:
    """Check whether the DRIMSeq path is available in the current environment.

    Returns `True` when both Rscript is available and the DRIMSeq package can
    be loaded.

    This function is used by `eqtl.run_dmqtl` for the
    `use_drimseq="auto"` selection logic and by
    `tests/test_r_bridge_smoke.py`.

    The check itself is lightweight and requires only one `Rscript -e`
    invocation. However, it should not be repeated unnecessarily in
    high-throughput workflows, such as repeatedly calling `run_dmqtl` for
    many jobs. The caller should check availability once and reuse the
    result.
    """
    if shutil.which(rscript_bin) is None:
        return False
    check = subprocess.run(
        [rscript_bin, "-e", 'if (!requireNamespace("DRIMSeq", quietly=TRUE)) quit(status=1)'],
        capture_output=True, text=True,
    )
    return check.returncode == 0


def make_drimseq_null_fit_fn(sample_ids: list[str], covariate_names: list[str],
                              rscript_bin: str = "Rscript",
                              timeout: int = 600) -> Callable[[np.ndarray, np.ndarray], NullFit]:
    """Wrap `fit_null_drimseq` into the standard `(y, X) -> NullFit` signature.

    This allows it to be passed directly to
    `pipeline.run_one_gene` / `run_many_genes` in the same way as
    `fit_null_python`.

    Those functions internally call `null_fit_fn(y, X)` and therefore do not
    know that the DRIMSeq implementation additionally requires
    `sample_ids` and `covariate_names`. `functools.partial` is used here to
    bind those arguments in advance.
    """
    return functools.partial(fit_null_drimseq, sample_ids=sample_ids,
                              covariate_names=covariate_names,
                              rscript_bin=rscript_bin, timeout=timeout)
