"""
Batch-vectorized score test (with Schur-complement adjustment for nuisance
parameters) and batched permutation testing.

Notation:
    n       number of individuals
    K       number of PAS categories; K-1 free parameters
            (the last category is the reference)
    p       number of null-model covariates, including the intercept
    n_snp   number of cis-SNPs for the gene

Core quantities (all depend only on the null fit and do not change across
SNPs or permutations; computed only once):
    R      (n, K-1)          : generalized residuals from
                               dmglm.residual_matrix
    W      (n, K-1, K-1)     : IRLS weight matrices from
                               dmglm.info_weight_matrix
    X      (n, p)            : null-model design matrix, including intercept
    Ibb    (p*(K-1), p*(K-1)): nuisance-nuisance information matrix
                               (computed and inverted only once)

For each SNP m (genotype column g_m, length n):
    U_m      = g_m^T @ R                                  , (K-1,)
    Idd_m    = sum_i g_im^2 * W_i                          , (K-1, K-1)
    Idb_m    = sum_i g_im * X_i (outer) W_i, reshaped       , (K-1, p*(K-1))
    Schur_m  = Idd_m - Idb_m @ Ibb^{-1} @ Idb_m^T
    T_m      = U_m^T @ pinv(Schur_m) @ U_m
               ~ chi2_{K-1} under H0

Permutation testing only requires reordering the genotype matrix by
permuted individual indices. R / W / X remain unchanged because they
represent the null-model fit for the original individual ordering.

A real performance bug was identified (this is not a case of an unavoidable
FLOP-bound matrix multiplication): an earlier development benchmark
incorrectly concluded that the main bottleneck was a necessary matrix
multiplication and therefore could not be further optimized. The actual
problem was that several multi-operand `np.einsum` calls (for example,
`"im,ib,ijk->mjbk"`) did not specify `optimize=True`.

With the default `optimize=False`, NumPy can use a naive contraction path
for `einsum` expressions with three or more operands instead of decomposing
the computation into efficient BLAS-backed matrix multiplications. Enabling
`optimize=True` allows NumPy to select a much better contraction path. For
the same computational workload, this was measured to improve runtime by
one to two orders of magnitude (in the local single-core sandbox, for
n=500, n_snp=2000, p=4, and K-1=5, runtime decreased from approximately
30 seconds to approximately 1 second).

This change does not alter the statistical calculation or numerical result;
it only changes the order in which mathematically equivalent operations are
performed. Therefore, no new statistical calibration validation is required
solely because of this optimization. However, it substantially changes the
estimate of whether the full pipeline can finish within a practical time
for 96,551 gene x cell-type units. Previous runtime estimates should
therefore be reconsidered and reduced by roughly one to two orders of
magnitude.

Regarding whether permutation testing should also be fully vectorized into
a single `einsum` call (previously listed in the README as the "next
optimization"): this module retains a `score_test_batch_perm` /
`score_pvalues_perm` implementation that processes an entire batch of
permutations at once. Instead of repeatedly reordering the large genotype
matrix, it reorders the much smaller R/W/X arrays and keeps the genotype
matrix read-only and shared across permutations. The implementation has
been cross-validated against a brute-force permutation loop and produces
numerically identical results.

However, honest benchmarking shows that after fixing `optimize=True`, the
fully batched permutation strategy is actually slower and more memory
intensive at realistic scales. Its intermediate tensors have complexity
O(n_perm * n_snp * p * (K-1)^2), which can quickly reach gigabyte-scale
memory usage when thousands of cis-SNPs are tested. For example, with
n_perm=300 and n_snp=5000, the process was directly killed by OOM in the
local sandbox.

In contrast, calling the already optimized `score_pvalues` once per
permutation is sufficiently fast that the Python loop overhead becomes
small, while memory usage remains stable and low. Therefore,
`pipeline.run_one_gene` continues to use the simple permutation loop as
the default path.

The `score_test_batch_perm` implementation is retained as a validated
correctness-preserving optional method, but is recommended only when
`n_snp` is small, such as during fine-mapping with tens to a few hundred
candidate SNPs. It should not be enabled by default for routine cis-QTL
scans involving thousands or tens of thousands of SNPs.

This is also an important optimization lesson for the project: an apparently
reasonable additional vectorization strategy can move the implementation
in the wrong direction if it is not benchmarked honestly at realistic
problem sizes.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from .dmglm import residual_matrix, info_weight_matrix


class GeneNullContext:
    """Precompute null-model quantities for one gene x cell-type unit.

    All subsequent SNP-level and permutation score tests reuse the
    precomputed R, W, X, and Ibb_inv matrices rather than recomputing them.
    """

    def __init__(self, y: np.ndarray, X: np.ndarray, eta_null: np.ndarray, phi: float):
        self.n, self.K = y.shape
        self.km1 = self.K - 1
        self.p = X.shape[1]
        self.X = X
        self.R = residual_matrix(y, eta_null, phi)          # (n, K-1)
        self.W = info_weight_matrix(y, eta_null, phi)        # (n, K-1, K-1)

        # Ibb: (p, K-1, p, K-1) -> reshape (p*(K-1), p*(K-1))
        Ibb_full = np.einsum("ia,ib,ijk->ajbk", X, X, self.W, optimize=True)
        self.Ibb = Ibb_full.reshape(self.p * self.km1, self.p * self.km1)
        # 用 pinv 而非 inv: null 设计矩阵在极端情形下可能接近共线, pinv 更稳健
        self.Ibb_inv = np.linalg.pinv(self.Ibb)

    def score_test_batch(self, G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute score tests for all SNPs in a genotype matrix.

        Parameters
        ----------
        G : np.ndarray
            Genotype matrix of shape (n, n_snp), or the same matrix with
            individual rows reordered by a permutation.

        Returns
        -------
        T : np.ndarray
            Score-test statistic for each SNP, shape (n_snp,).

        U : np.ndarray
            Score vector for each SNP, shape (n_snp, K-1).
            Returned primarily for debugging and diagnostics.
        """
        n, n_snp = G.shape
        assert n == self.n

        U = G.T @ self.R  # (n_snp, K-1)

        Idd = np.einsum("im,ijk->mjk", G ** 2, self.W, optimize=True)  # (n_snp, K-1, K-1)
        Idb = np.einsum("im,ib,ijk->mjbk", G, self.X, self.W, optimize=True)  # (n_snp, K-1, p, K-1)
        Idb = Idb.reshape(n_snp, self.km1, self.p * self.km1)

        # Schur = Idd - Idb @ Ibb_inv @ Idb^T, computed in batch for all SNPs.
        tmp = Idb @ self.Ibb_inv  # (n_snp, K-1, p*(K-1))
        correction = np.einsum("mij,mkj->mik", tmp, Idb, optimize=True)  # (n_snp, K-1, K-1)
        Schur = Idd - correction

        # T_m = U_m^T @ pinv(Schur_m) @ U_m.
        # NumPy supports batched pseudoinverse for stacked matrices,
        # eliminating the remaining per-SNP Python-level loop.
        Schur_inv = np.linalg.pinv(Schur)  # (n_snp, K-1, K-1)
        T = np.einsum("mi,mij,mj->m", U, Schur_inv, U, optimize=True)
        return T, U

    def score_pvalues(self, G: np.ndarray) -> np.ndarray:
        T, _ = self.score_test_batch(G)
        return _t_to_pvalue(T, self.km1)

    def score_test_batch_perm(self, G: np.ndarray, perm_indices: np.ndarray) -> np.ndarray:
        """Compute score-test statistics for all SNPs across a batch of permutations.

        The key optimization, corresponding to the "next vectorization
        optimization" documented in the README, is to avoid constructing a
        separately reordered (n, n_snp) genotype matrix for every permutation.

        Instead, reordering the rows of G is mathematically equivalent to
        reordering R/W/X using the inverse permutation. Because these arrays
        are much smaller than the genotype matrix, the permutation dimension
        can be processed in batch while the original G matrix remains
        unchanged and is shared across all permutations.

        Let idx be one permutation and inv its inverse permutation, such that

            inv[idx[i]] = i.

        If

            Gp[i] = G[idx[i]],

        then

            U = Gp^T @ R
              = sum_i G[idx[i]] R[i]
              = sum_j G[j] R[inv[j]]
              = G^T @ R[inv].

        The same transformation applies to Idd and Idb by reordering W
        (and X in the Idb calculation) with the inverse permutation.

        Parameters
        ----------
        G : np.ndarray
            Genotype matrix of shape (n, n_snp).

        perm_indices : np.ndarray
            Array of shape (n_perm, n). Each row is a permutation of
            0..n-1 representing one permutation of individuals.

        Returns
        -------
        T : np.ndarray
            Score-test statistics with shape (n_perm, n_snp).
        """
        n, n_snp = G.shape
        assert n == self.n
        n_perm = perm_indices.shape[0]

        # inv[b] is the inverse permutation of perm_indices[b].
        # This is pure index manipulation and has negligible cost.
        inv = np.argsort(perm_indices, axis=1)  # (n_perm, n)

        R_perm = self.R[inv]        # (n_perm, n, km1)
        W_perm = self.W[inv]        # (n_perm, n, km1, km1)
        X_perm = self.X[inv]        # (n_perm, n, p)

        G2 = G ** 2

        U = np.einsum("im,bik->bmk", G, R_perm, optimize=True)              # (n_perm, n_snp, km1)
        Idd = np.einsum("im,bijk->bmjk", G2, W_perm, optimize=True)          # (n_perm, n_snp, km1, km1)
        Idb = np.einsum("im,bia,bijk->bmjak", G, X_perm, W_perm, optimize=True)
        Idb = Idb.reshape(n_perm, n_snp, self.km1, self.p * self.km1)

        tmp = Idb @ self.Ibb_inv                              # (n_perm, n_snp, km1, p*km1)
        correction = np.einsum("bmij,bmkj->bmik", tmp, Idb, optimize=True)   # (n_perm, n_snp, km1, km1)
        Schur = Idd - correction

        Schur_inv = np.linalg.pinv(Schur)                     # Batched pseudoinverse supports arbitrary leading batch dimensions.
        T = np.einsum("bmi,bmij,bmj->bm", U, Schur_inv, U, optimize=True)
        return T

    def score_pvalues_perm(self, G: np.ndarray, perm_indices: np.ndarray) -> np.ndarray:
        T = self.score_test_batch_perm(G, perm_indices)
        return _t_to_pvalue(T, self.km1)


def _t_to_pvalue(T: np.ndarray, df: int) -> np.ndarray:
    """Convert score statistics to chi-square p-values.

    Non-finite values and negative statistics caused by numerical
    degeneracy or floating-point noise are handled defensively.

    A Schur matrix can become numerically degenerate under extreme
    collinearity, for example when MAF filtering has not been applied or
    when very-low-frequency variants are present. In such cases, the
    batched `np.linalg.pinv` may produce slightly negative test statistics
    due to floating-point noise, even though the theoretical statistic
    cannot be negative. In more extreme cases, NaN or Inf may also occur.

    `chi2.sf` silently maps negative statistics to p-values close to 1,
    which is conservative in this context. However, NaN values propagate
    through downstream operations: `np.argmin` on an array containing NaN
    is unreliable and may lead to an incorrect lead SNP or minimum p-value,
    potentially contaminating the gene-level result and subsequent
    multiple-testing correction.

    We therefore explicitly map non-finite or negative test statistics to
    the most conservative value, p=1.0, rather than allowing NaNs to
    propagate downstream.
    """
    T = np.asarray(T, dtype=float)
    bad = ~np.isfinite(T) | (T < 0)
    T_clean = np.where(bad, 0.0, T)
    p = chi2.sf(T_clean, df=df)
    p = np.where(bad, 1.0, p)
    return p


def permute_min_p(ctx: GeneNullContext, G: np.ndarray, perm_indices: np.ndarray,
                   method: str = "loop") -> np.ndarray:
    """Compute the minimum score-test p-value for each permutation.

    Parameters
    ----------
    ctx : GeneNullContext
        Precomputed null-model context for the gene.

    G : np.ndarray
        Genotype matrix of shape (n, n_snp).

    perm_indices : np.ndarray
        Array of shape (n_perm, n). Each row is a permutation of
        0..n-1 representing one permutation of individuals.

    method : {"loop", "batched"}, default="loop"
        Permutation implementation.

        ``"loop"`` (default, recommended):
            For each permutation, reorder G and call `score_pvalues`.
            The underlying `score_test_batch` already uses
            `optimize=True`, which fixes the actual performance bottleneck
            described at the top of this module.

            Memory usage is O(n * n_snp), making this approach suitable
            for routine QTL scans with thousands to tens of thousands of
            cis-SNPs.

        ``"batched"``:
            Use `score_test_batch_perm` to process all permutations in a
            single batched einsum calculation.

            The results are numerically identical to the loop method,
            but memory usage is approximately
            O(n_perm * n_snp * p * (K-1)^2).

            This method is therefore recommended only when `n_snp` is
            small, such as during fine-mapping with tens to a few hundred
            candidate SNPs. At routine cis-QTL scale, it can be slower
            and may be terminated by the operating system due to excessive
            memory usage.

    Returns
    -------
    np.ndarray
        Minimum p-value across all cis-SNPs for each permutation,
        with shape (n_perm,).
    """
    if method == "batched":
        p = ctx.score_pvalues_perm(G, perm_indices)  # (n_perm, n_snp)
        return p.min(axis=1)
    if method != "loop":
        raise ValueError(f"Unknown method={method!r}; supported methods are 'loop' and 'batched'")
    n_perm = perm_indices.shape[0]
    min_p = np.empty(n_perm)
    for b in range(n_perm):
        G_perm = G[perm_indices[b], :]
        min_p[b] = ctx.score_pvalues(G_perm).min()
    return min_p
