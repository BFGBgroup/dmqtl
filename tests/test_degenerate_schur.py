"""Degenerate score tests (found on real data, 2026-10).

Individuals with zero total count for a gene carry no information: their R_i and
W_i are exactly 0. A SNP whose genotype is constant over the individuals that do
carry information lies in the span of the intercept there, so U = 0 and the
Schur complement is 0 in exact arithmetic and T = U' Schur^+ U is 0/0. In
floating point both are rounding noise; v1.0.0 returned an arbitrary T for such
a SNP (on real data: fake permutation p-values down to 1e-8, and different noise
from the loop and the fast engine). These SNPs are untestable and must get
p = 1, in the observed scan and in every permutation, under both engines.
"""
import numpy as np

from dmqtl.dmglm import softmax_ref
from dmqtl.scoretest import GeneNullContext, PermScanner, _t_to_pvalue, permute_min_p


def _sparse_gene(rng, n=60, K=3, p=3, n_snp=120, n_zero=40, phi=20.0):
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    eta = X @ rng.normal(scale=0.3, size=(p, K - 1))
    pi = softmax_ref(eta)
    N = np.maximum(rng.poisson(40, size=n), 3)
    N[:n_zero] = 0  # no reads for this gene -> R_i = 0, W_i = 0
    y = np.array([rng.multinomial(N[i], rng.dirichlet(phi * pi[i])) for i in range(n)], dtype=float)
    G = np.zeros((n, n_snp))
    for j in range(n_snp):  # rare SNPs: 2-4 carriers, so permutations often put all of them among the zeros
        G[rng.choice(n, size=rng.integers(2, 5), replace=False), j] = 1.0
    for j in range(0, n_snp, 10):  # observed data: every tenth SNP carried only by zero-count individuals
        G[:, j] = 0.0
        G[rng.choice(n_zero, size=3, replace=False), j] = 1.0
    G = G - G.mean(axis=0, keepdims=True)
    return y, X, eta, phi, G


def _constant_on_informative(y, G):
    inf = y.sum(axis=1) > 0
    return np.array([len(np.unique(np.round(G[inf, j], 12))) == 1 for j in range(G.shape[1])])


def _v100_pvalues(ctx, G):
    """v1.0.0 formula, one SNP at a time (no masking)."""
    p = np.empty(G.shape[1])
    for j in range(G.shape[1]):
        g = G[:, j]
        U = g @ ctx.R
        Idd = np.einsum("i,ijk->jk", g * g, ctx.W)
        Idb = np.einsum("i,ib,ijk->jbk", g, ctx.X, ctx.W).reshape(ctx.km1, ctx.p * ctx.km1)
        S = Idd - Idb @ ctx.Ibb_inv @ Idb.T
        p[j] = _t_to_pvalue(np.array([U @ np.linalg.pinv(S) @ U]), ctx.km1)[0]
    return p


def test_degenerate_snps_get_p_one():
    for K in (2, 3, 5):
        rng = np.random.default_rng(K)
        y, X, eta, phi, G = _sparse_gene(rng, K=K)
        ctx = GeneNullContext(y, X, eta, phi)
        p = ctx.score_pvalues(G)
        const = _constant_on_informative(y, G)
        assert const[::10].all() and 0 < const.sum() < len(const)
        assert np.all(p[const] == 1.0), f"K={K}: degenerate SNPs got p {np.sort(p[const])[:3]}"
        # every other SNP keeps the v1.0.0 value
        np.testing.assert_allclose(p[~const], _v100_pvalues(ctx, G[:, ~const]), rtol=1e-8, err_msg=f"K={K}")


def test_engines_agree_when_permutations_hit_degenerate_snps():
    for K in (2, 3, 5):
        rng = np.random.default_rng(10 + K)
        y, X, eta, phi, G = _sparse_gene(rng, K=K)
        ctx = GeneNullContext(y, X, eta, phi)
        perms = np.array([rng.permutation(ctx.n) for _ in range(100)])
        loop = permute_min_p(ctx, G, perms, method="loop")
        for fold in (True, False):
            fast = PermScanner(ctx, G, max_block_bytes=200_000, fold_ibb=fold).min_pvalues(perms)
            np.testing.assert_allclose(fast, loop, rtol=1e-9, atol=0, err_msg=f"K={K} fold_ibb={fold}")
