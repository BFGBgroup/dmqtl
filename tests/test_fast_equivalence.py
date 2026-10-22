"""Equivalence tests for the fast permutation engine (2026-10 optimization).

The fast engine must reproduce the v1.0.0 loop path (reorder G, call
`score_pvalues`, take the minimum) up to floating-point rounding. These are
refactoring tests, not statistical validation: any difference beyond
rounding means the optimization changed the result and must not be merged.
"""
import numpy as np

from dmqtl.dmglm import softmax_ref
from dmqtl.pipeline import run_one_gene
from dmqtl.scoretest import GeneNullContext, PermScanner, dedup_columns, permute_min_p


def _gene(rng, n, K, p, n_snp, phi=60.0):
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    eta = X @ rng.normal(scale=0.3, size=(p, K - 1))
    pi = softmax_ref(eta)
    N = np.maximum(rng.poisson(150, size=n), 5)
    y = np.array([rng.multinomial(N[i], rng.dirichlet(phi * pi[i])) for i in range(n)], dtype=float)
    n_uniq = n_snp // 2
    G0 = rng.binomial(2, rng.uniform(0.05, 0.5, size=n_uniq), size=(n, n_uniq)).astype(float)
    idx = np.concatenate([np.arange(n_uniq), rng.integers(0, n_uniq, size=n_snp - n_uniq)])
    G = G0[:, idx]
    flip = np.zeros(n_snp, bool)
    flip[n_uniq:] = rng.random(n_snp - n_uniq) < 0.5
    G[:, flip] = 2.0 - G[:, flip]
    G[:, 3] = 1.0  # monomorphic column -> Schur is all zeros -> p = 1
    G = G - G.mean(axis=0, keepdims=True)
    return y, X, eta, phi, G


def test_dedup_columns_reconstructs_exactly():
    rng = np.random.default_rng(0)
    *_, G = _gene(rng, 50, 3, 3, 40)
    uniq, col, sign = dedup_columns(G)
    assert len(uniq) < G.shape[1]
    rebuilt = G[:, uniq][:, col] * sign
    assert np.array_equal(rebuilt + 0.0, G + 0.0)


def test_scanner_matches_loop_for_each_km1():
    for K in (2, 3, 4, 6):
        rng = np.random.default_rng(K)
        y, X, eta, phi, G = _gene(rng, 80, K, 4, 150)
        ctx = GeneNullContext(y, X, eta, phi)
        perms = np.array([rng.permutation(80) for _ in range(25)])
        loop = permute_min_p(ctx, G, perms, method="loop")
        # tiny block budget forces several blocks, exercising the chunking
        fast = PermScanner(ctx, G, max_block_bytes=200_000).min_pvalues(perms)
        np.testing.assert_allclose(fast, loop, rtol=1e-9, atol=0, err_msg=f"K={K}")


def test_scanner_observed_order_matches_score_pvalues():
    rng = np.random.default_rng(5)
    y, X, eta, phi, G = _gene(rng, 70, 3, 3, 60)
    ctx = GeneNullContext(y, X, eta, phi)
    ident = np.arange(70)[None, :]
    np.testing.assert_allclose(PermScanner(ctx, G).min_pvalues(ident),
                               [ctx.score_pvalues(G).min()], rtol=1e-9)


def test_run_one_gene_fast_equals_loop():
    for K in (2, 4):
        rng = np.random.default_rng(10 + K)
        y, X, eta, phi, G = _gene(rng, 90, K, 3, 80)

        class _NF:  # minimal stand-in for a converged NullFit
            converged = True
        nf = _NF(); nf.eta = eta; nf.phi = phi
        kw = dict(perm_batch_size=100, min_perm=300, max_perm=600, min_hits_to_stop=15)
        a = run_one_gene("g", "c", y, X, G, np.random.default_rng(1), null_fit=nf, perm_engine="loop", **kw)
        b = run_one_gene("g", "c", y, X, G, np.random.default_rng(1), null_fit=nf, perm_engine="fast", **kw)
        assert a.lead_variant_id == b.lead_variant_id
        assert a.obs_min_p == b.obs_min_p
        assert np.array_equal(a.nominal_pvalues, b.nominal_pvalues)
        assert (a.n_perm_used, a.n_hits) == (b.n_perm_used, b.n_hits)
        np.testing.assert_allclose(b.beta_p, a.beta_p, rtol=1e-7)
