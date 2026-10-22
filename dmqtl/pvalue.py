"""
Gene-level permutation p-value processing:

1. Adaptive early stopping: perform permutations in batches. Once the number
   of hits, defined as "permuted min(P) <= observed min(P)", reaches the
   specified threshold, stop early to save computation. The actual number of
   permutations performed is recorded and used when fitting the Beta
   approximation.

2. Beta approximation (following the GTEx/FastQTL approach): instead of using
   (hits + 1)/(N + 1) directly as the final p-value, fit a Beta(shape1, shape2)
   distribution to the permutation null distribution that has already been
   generated, and use this continuous distribution to calculate the p-value.
   This avoids strong-signal genes being effectively "pinned" near 1/(N+1),
   which can distort downstream q-value ranking.

3. Storey q-value: perform multiple-testing correction on the final p-values
   across all gene x cell-type tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np
from scipy import optimize, stats


@dataclass
class AdaptivePermResult:
    obs_min_p: float
    n_perm_used: int
    n_hits: int
    perm_min_p: np.ndarray  # min p-values from the permutations actually used (for Beta fitting)


def adaptive_permutation(perm_min_p_fn, obs_min_p: float,
                          batch_size: int = 100, min_perm: int = 1000,
                          max_perm: int = 10000,
                          min_hits_to_stop: int = 15) -> AdaptivePermResult:
    """perm_min_p_fn(n_batch) -> ndarray(n_batch,)

    Each call generates a new batch of permutations and returns the
    corresponding min p-values. The caller determines how permutation indices
    are generated and how the min p-values are calculated. This function only
    handles the adaptive early-stopping logic and accumulation of results.

    A real bug was identified in the previous implementation (rather than a
    deliberate design choice): it stopped as soon as
    `n_hits >= min_hits_to_stop`, without enforcing a minimum number of
    permutations. The methodological description provided previously states:

        "A minimum of 1000 permutations and a maximum of 10,000 permutations
        were performed, with an exit criteria in between this range of having
        at least 15 permuted min(p) values less than the nominal min(p)."

    Thus, the stopping criterion (`n_hits >= 15`) should only become active
    within the interval [1000, 10000]. Before 1000 permutations have been
    completed, early stopping is not allowed even if the hit count has already
    reached the threshold.

    The previous implementation could therefore stop after only tens or
    hundreds of permutations. For example, GENE7/GENE8 in the previous run
    used only about 200-400 permutations because a single batch happened to
    accumulate 15 hits and triggered immediate termination. This means that
    the Beta distribution would be fitted using far fewer than 1000 samples,
    which can substantially reduce fitting stability and precision for genes
    with weaker signals or relatively high hit rates, such as genes whose
    observed min(p) is not below approximately 1e-4.

    The new `min_perm` parameter (default: 1000) implements the lower bound
    described above: even if `n_hits` has already reached
    `min_hits_to_stop`, at least `min_perm` permutations must be completed
    before early stopping is allowed.

    The process always stops after `max_perm` permutations, which was already
    correct in the previous implementation.
    """
    all_min_p = []
    n_hits = 0
    n_used = 0
    while n_used < max_perm:
        this_batch = min(batch_size, max_perm - n_used)
        batch_min_p = perm_min_p_fn(this_batch)
        all_min_p.append(batch_min_p)
        n_used += this_batch
        n_hits += int(np.sum(batch_min_p <= obs_min_p))
        if n_used >= min_perm and n_hits >= min_hits_to_stop:
            break
    perm_min_p = np.concatenate(all_min_p)
    return AdaptivePermResult(obs_min_p=obs_min_p, n_perm_used=n_used,
                               n_hits=n_hits, perm_min_p=perm_min_p)


def _beta_mle_nelder_mead(x: np.ndarray) -> tuple[float, float] | None:
    """Beta(a, b) MLE on (0, 1) by Nelder-Mead in log-parameter space, started from the moments.

    Returns None when the moments give no valid start (no spread) or the optimum is not finite.
    """
    if x.max() <= x.min():  # no spread (rounding can still give a tiny positive variance)
        return None
    m, v = float(x.mean()), float(x.var())
    if not (0 < m < 1) or v <= 0 or m * (1 - m) <= v:
        return None
    a0 = m * (m * (1 - m) / v - 1)
    b0 = a0 * (1 / m - 1)

    def nll(t):
        return -float(stats.beta.logpdf(x, np.exp(t[0]), np.exp(t[1])).sum())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = optimize.minimize(nll, np.log([a0, b0]), method="Nelder-Mead",
                              options=dict(xatol=1e-8, fatol=1e-10, maxiter=20000, maxfev=40000))
    a, b = (float(v) for v in np.exp(r.x))
    if not (np.isfinite(r.fun) and np.isfinite(a) and np.isfinite(b)):
        return None
    return a, b


def beta_approx_pvalue(result: AdaptivePermResult) -> tuple[float, float, float]:
    """Fit a Beta(a, b) distribution to the permuted min-p null distribution
    and return `(p_beta, a, b)`.

    Because the min-p value itself lies in (0, 1), a Beta distribution provides
    a convenient parametric approximation of its null distribution, following
    the approach used in FastQTL/GTEx.

    Maximum-likelihood estimation is performed with `loc=0` and `scale=1`
    fixed. When adaptive early stopping results in a small number of
    permutations (`n_perm_used` is small), the fitted distribution may be
    unstable. The caller should therefore consider `n_perm_used` when deciding
    whether additional permutations or a more conservative reporting strategy
    is needed.
    """
    x = np.clip(result.perm_min_p, 1e-12, 1 - 1e-12)
    try:
        a, b, _, _ = stats.beta.fit(x, floc=0, fscale=1)
    except Exception:  # noqa: BLE001 - scipy raises FitSolverError and others
        # 2026-10, real data: scipy's MLE solver did not converge for a gene
        # whose permutation min-p sat at the 1e-12 floor in half of the
        # permutations, and the uncaught error ended the whole run. Maximise the
        # same log-likelihood with Nelder-Mead from the method-of-moments start
        # (as tensorQTL does); if no finite fit exists, use the empirical
        # permutation p-value. Genes scipy can fit keep scipy's numbers.
        ab = _beta_mle_nelder_mead(x)
        if ab is None:
            return (result.n_hits + 1) / (result.n_perm_used + 1), float("nan"), float("nan")
        a, b = ab
    # Smaller min(P) values are more significant, so the desired quantity is
    # the CDF: the probability that a permuted min-p value is <= the observed
    # value. It is not the survival function. The previous implementation
    # incorrectly used `sf`, which reversed the direction of the p-value.
    # This was identified and corrected using a sanity check: a strong signal
    # should produce a p-value close to 0 rather than close to 1.
    p_beta = float(stats.beta.cdf(result.obs_min_p, a, b))
    return p_beta, float(a), float(b)


def floor_warning(result: AdaptivePermResult, p_beta: float) -> str | None:
    """Message when more than 1% of the permutation min-p values are <= 1e-12, else None.

    Under a well-calibrated null this share is ~0. On real data (2026-10) it
    came from a single outlying individual being the only informative carrier
    of rare SNPs: the score test is far from its chi-square null, and the Beta
    fit, pulled towards 0 by the floor values, gives too small a p-value
    (ADAMTS18: 63% at the floor, Beta p 0.032, empirical p 0.37). The observed
    min-p can be below the floor as well (2.1e-25 there), so it is not used to
    skip genes. The empirical p-value is reported next to the Beta one.
    """
    share = float(np.mean(result.perm_min_p <= 1e-12))
    if share <= 0.01:
        return None
    p_emp = (result.n_hits + 1) / (result.n_perm_used + 1)
    return (f"{share:.0%} of permutation min-p values are <= 1e-12 (observed min-p {result.obs_min_p:.3g}); "
            f"Beta p {p_beta:.3g} vs empirical p {p_emp:.3g} from {result.n_perm_used} permutations")


def storey_qvalue(pvals: np.ndarray, lambdas: np.ndarray | None = None) -> np.ndarray:
    """Simplified Storey q-value calculation.

    pi0 is estimated by scanning a set of lambda values and using the median
    of the resulting estimates. In the degenerate/simple case, this is
    equivalent to using a representative lambda such as 0.5. A more complete
    implementation could replace this with the natural-spline fitting strategy
    used by the `qvalue` R package. The present implementation follows the
    standard textbook formulation and is sufficient for ranking and
    preliminary screening.
    """
    pvals = np.asarray(pvals)
    m = len(pvals)
    if lambdas is None:
        lambdas = np.arange(0.05, 0.96, 0.05)
    pi0_est = np.array([(pvals > lam).sum() / (m * (1 - lam)) for lam in lambdas])
    pi0 = float(np.clip(np.median(pi0_est), 0.0, 1.0))

    order = np.argsort(pvals)
    ranked_p = pvals[order]
    qvals_sorted = pi0 * m * ranked_p / (np.arange(1, m + 1))
    # Monotonize from largest to smallest by taking the cumulative minimum,
    # ensuring that the q-value does not decrease as the p-value increases.
    qvals_sorted = np.minimum.accumulate(qvals_sorted[::-1])[::-1]
    qvals_sorted = np.clip(qvals_sorted, 0, 1)

    qvals = np.empty(m)
    qvals[order] = qvals_sorted
    return qvals


@dataclass
class SignificantPair:
    gene_id: str
    variant_id: str
    nominal_pvalue: float
    gene_nominal_threshold: float


def significant_snp_gene_pairs(gene_results, q_threshold: float = 0.05):
    """Generate significant SNP-gene (eSNP) pairs using the
    permutation-threshold approach.

    The method follows the procedure described in the requested methodology,
    which is consistent with the standard strategy used to define significant
    variant-gene pairs in GTEx-style eQTL calling pipelines:

        1. In addition to the eGene list, generate a list of all significant
           SNP-gene pairs.

        2. Determine a permutation-based significance threshold and map it
           back to a gene-specific nominal min(p) threshold.

        3. The permutation threshold is defined as the Beta-approximation
           p-value of the eGene at the q-value 0.05 boundary. Because the
           q-value is a monotonic transformation of the ranked p-values,
           the Beta p-value of the eGene at this boundary defines the global
           significance cutoff:

               p_threshold = max(beta_p) among all eGenes with q <= 0.05.

        4. For each eGene, use its own fitted Beta(a, b) permutation null
           distribution to map `p_threshold` back to the nominal-p scale:

               nominal_threshold_gene =
                   Beta_ppf(p_threshold; a_gene, b_gene)

           Because

               beta_p = Beta_cdf(nominal_min_p; a, b),

           this operation is simply the inverse transformation.

        5. All SNPs from the observed scan stage (not from the permutation
           stage) whose nominal p-values satisfy

               nominal_p <= nominal_threshold_gene

           are included in the final significant SNP-gene pair list.

    Intuitively, different genes can have different numbers of cis-SNPs and
    different LD structures. Therefore, their null distributions of min(p)
    can also differ, resulting in different Beta(a, b) distributions.

    Applying the same multiple-testing significance criterion and mapping it
    back to each gene's own nominal-p scale therefore produces a
    gene-specific nominal threshold. A single nominal threshold shared by all
    genes (for example, 5e-8) would not account for differences in the number
    of cis-SNPs or their correlation structure and could systematically favor
    genes with fewer cis-SNPs.

    Parameters
    ----------
    gene_results
        A list of `pipeline.GeneResult` objects. The results must have been
        generated by `run_one_gene`/`run_many_genes` with `variant_ids`
        retained, meaning that both `nominal_pvalues` and `variant_ids` are
        available. Otherwise, SNP-level results cannot be generated and the
        function raises an error.

        This is an explicit failure rather than silently skipping the affected
        gene. Silently ignoring missing fields during batch processing could
        produce an apparently complete significant-pair result while actually
        omitting part of the analysis, which is harder to detect than an
        immediate error.

    q_threshold
        Storey q-value threshold for defining eGenes.

    Returns
    -------
    egene_ids
        List of gene IDs passing `q_threshold`, in input order.

    qvals
        Array of q-values with the same length as `gene_results`.

    pairs
        List of `SignificantPair` objects containing significant
        SNP-gene pairs for each eGene.

    If no gene passes `q_threshold`, returns `([], qvals, [])`.
    """
    if len(gene_results) == 0:
        return [], np.array([]), []

    beta_p = np.array([r.beta_p for r in gene_results])
    qvals = storey_qvalue(beta_p)
    passing_mask = qvals <= q_threshold

    if not passing_mask.any():
        return [], qvals, []

    p_threshold = float(beta_p[passing_mask].max())

    egene_ids: list[str] = []
    pairs: list[SignificantPair] = []
    for r, ok in zip(gene_results, passing_mask):
        if not ok:
            continue
        egene_ids.append(r.gene_id)
        if r.nominal_pvalues is None or r.variant_ids is None:
            raise ValueError(
                f"Gene {r.gene_id} has a GeneResult without "
                "nominal_pvalues/variant_ids, so SNP-gene pairs cannot be "
                "generated. Please ensure that `variant_ids` is passed to "
                "pipeline.run_one_gene/run_many_genes."
            )
        nominal_threshold = float(stats.beta.ppf(p_threshold, r.beta_a, r.beta_b))
        nominal_p = np.asarray(r.nominal_pvalues)
        # Real floating-point boundary bug:
        # For the eGene that exactly defines `p_threshold`, its own beta_p is
        # equal to `p_threshold`. Therefore, in exact arithmetic, its lead SNP
        # must satisfy
        #
        #     nominal_p <= nominal_threshold
        #
        # because `nominal_threshold` is the inverse Beta-CDF of its own
        # observed min-p value.
        #
        # However, `beta.ppf` is computed numerically, and the floating-point
        # result of the inverse operation can differ slightly from the value
        # used by `beta.cdf`. In practice, a relative error on the order of
        # 1e-5 was observed. This can cause the defining eGene's own lead SNP
        # to fall just outside its threshold, producing an eGene with zero
        # significant SNP-gene pairs.
        #
        # A very small relative tolerance is therefore used to absorb this
        # numerical round-trip error.
        sig_mask = np.isfinite(nominal_p) & (nominal_p <= nominal_threshold * (1 + 1e-6))
        for vid, p in zip(np.asarray(r.variant_ids)[sig_mask], nominal_p[sig_mask]):
            pairs.append(SignificantPair(
                gene_id=r.gene_id, variant_id=str(vid),
                nominal_pvalue=float(p), gene_nominal_threshold=nominal_threshold,
            ))
    return egene_ids, qvals, pairs
