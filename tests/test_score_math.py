"""
Core correctness tests.

Design principles (following the previous lesson that an analytical Hessian
derivation can accidentally omit curvature terms):
    - Do not trust analytical derivations blindly; always cross-validate with
      numerical or brute-force methods.
    - test_residual_matches_numerical_gradient / test_info_weight_matches_numerical_hessian
      are the lowest-level validation tests. If both pass, the mathematical
      correctness of all subsequent vectorized score tests is well supported.
    - test_score_test_matches_brute_force_lrt_alternative provides an end-to-end
      validation by comparing the significance conclusions of the score test
      and LRT on simulated data. These methods should be asymptotically
      equivalent for sufficiently large samples.
    - test_null_calibration repeatedly simulates data under H0 (delta=0) and
      checks whether score-test p-values are approximately Uniform(0,1).
      This is a fundamental requirement for applying the method to real QTL
      scans and is more important than numerical agreement with the LRT.
      Even if the approximation is otherwise reasonable, poor calibration
      would invalidate downstream FDR control for the entire pipeline.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, kstest

from dmqtl.dmglm import dm_loglik, softmax_ref, residual_matrix, info_weight_matrix
from dmqtl.nullmodel import fit_null_python
from dmqtl.pvalue import adaptive_permutation
from dmqtl.scoretest import GeneNullContext

from .simulate import simulate_dm


def _numerical_grad(f, x, step=1e-5):
    g = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        xp = x.copy(); xp[idx] += step
        xm = x.copy(); xm[idx] -= step
        g[idx] = (f(xp) - f(xm)) / (2 * step)
    return g


def test_residual_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    n, K, phi = 40, 5, 80.0
    y, X, g, beta = simulate_dm(n, K, p=3, phi=phi, delta=None, rng=rng)
    eta = X @ beta

    def ll_of_eta(eta_flat):
        eta_ = eta_flat.reshape(n, K - 1)
        pi_ = softmax_ref(eta_)
        return dm_loglik(y, pi_, phi)

    analytic = residual_matrix(y, eta, phi)  # d loglik / d eta (before summing across individuals)
    # The numerical gradient is the derivative of the total log-likelihood
    # with respect to each element of eta. Since individuals are independent,
    # residual_matrix returns the per-individual d ll_i / d eta_i, so the two
    # quantities should agree element-wise.
    num_grad_flat = _numerical_grad(ll_of_eta, eta.ravel())
    num_grad = num_grad_flat.reshape(n, K - 1)

    np.testing.assert_allclose(analytic, num_grad, atol=1e-4, rtol=1e-3)


def test_info_weight_matches_numerical_hessian():
    rng = np.random.default_rng(1)
    n, K, phi = 25, 4, 60.0
    y, X, g, beta = simulate_dm(n, K, p=2, phi=phi, delta=None, rng=rng)
    eta = X @ beta

    W = info_weight_matrix(y, eta, phi)  # (n, K-1, K-1), W_i = -d r_i/d eta_i

    # For several randomly selected individuals, compute the second-order
    # derivatives using brute-force finite differences. Only eta_i for the
    # selected individual is perturbed because the log-likelihood is additive
    # across independent individuals, so cross-individual second derivatives
    # are zero.
    for i in rng.choice(n, size=5, replace=False):
        km1 = K - 1

        def ll_i(eta_i):
            pi_i = softmax_ref(eta_i.reshape(1, km1))[0]
            return dm_loglik(y[i:i+1], pi_i.reshape(1, K), phi)

        H_num = np.zeros((km1, km1))
        step = 1e-4
        base = eta[i].copy()
        for a in range(km1):
            for b in range(km1):
                e1 = base.copy(); e1[a] += step; e1[b] += step
                e2 = base.copy(); e2[a] += step; e2[b] -= step
                e3 = base.copy(); e3[a] -= step; e3[b] += step
                e4 = base.copy(); e4[a] -= step; e4[b] -= step
                H_num[a, b] = (ll_i(e1) - ll_i(e2) - ll_i(e3) + ll_i(e4)) / (4 * step ** 2)
        W_num = -H_num
        np.testing.assert_allclose(W[i], W_num, atol=2e-2, rtol=5e-2)


def _fit_full_model(y, X, g, beta0, phi0, delta0):
    n, K = y.shape
    p = X.shape[1]
    km1 = K - 1

    def unpack(theta):
        beta = theta[: p * km1].reshape(p, km1)
        delta = theta[p * km1: p * km1 + km1]
        phi = np.exp(theta[-1])
        return beta, delta, phi

    def neg_ll(theta):
        beta, delta, phi = unpack(theta)
        eta = X @ beta + np.outer(g, delta)
        pi = softmax_ref(eta)
        return -dm_loglik(y, pi, phi)

    theta0 = np.concatenate([beta0.ravel(), delta0, [np.log(phi0)]])
    res = minimize(neg_ll, theta0, method="L-BFGS-B",
                    options={"maxiter": 3000, "ftol": 1e-13, "gtol": 1e-11})
    beta, delta, phi = unpack(res.x)
    return -res.fun, beta, delta, phi


def test_score_test_matches_brute_force_lrt_alternative():
    """When a true signal (delta != 0) is present, the score-test statistic
    should lead to the same significance conclusion as the LRT.

    The numerical values do not need to be identical, but they should be of
    the same order of magnitude and both should indicate strong significance.
    """
    rng = np.random.default_rng(42)
    n, K, p, phi = 400, 4, 3, 100.0
    delta_true = np.array([0.9, -0.7, 0.5])  # Strong signal, K-1=3

    y, X, g, beta_true = simulate_dm(n, K, p, phi, delta_true, rng)

    null_fit = fit_null_python(y, X)
    ctx = GeneNullContext(y, X, null_fit.eta, null_fit.phi)
    T, U = ctx.score_test_batch(g.reshape(n, 1))
    score_p = chi2.sf(T[0], df=K - 1)

    ll_null = null_fit.loglik
    ll_full, *_ = _fit_full_model(y, X, g, null_fit.beta, null_fit.phi,
                                   np.zeros(K - 1))
    lrt_stat = 2 * (ll_full - ll_null)
    lrt_p = chi2.sf(lrt_stat, df=K - 1)

    assert score_p < 1e-6, f"Score test should be highly significant, got p={score_p}"
    assert lrt_p < 1e-6, f"LRT should be highly significant, got p={lrt_p}"

    # Under a true signal, the two statistics should be of the same order
    # of magnitude and both should be much larger than the mean of chi-square
    # with K-1 degrees of freedom.
    assert T[0] > (K - 1) * 3
    assert lrt_stat > (K - 1) * 3


def test_null_calibration():
    """Under H0, repeatedly simulate data and verify that score-test p-values
    are approximately Uniform(0,1) using a relaxed KS test.

    Perfect uniformity is not required, but the p-values should not show a
    systematic deviation, such as being predominantly below 0.5, which would
    indicate obvious anti-conservative behavior.
    """
    rng = np.random.default_rng(7)
    n, K, p, phi = 150, 3, 2, 60.0
    n_sim = 300
    pvals = np.empty(n_sim)
    for s in range(n_sim):
        y, X, g, beta_true = simulate_dm(n, K, p, phi, delta=None, rng=rng)
        null_fit = fit_null_python(y, X)
        ctx = GeneNullContext(y, X, null_fit.eta, null_fit.phi)
        T, U = ctx.score_test_batch(g.reshape(n, 1))
        pvals[s] = chi2.sf(T[0], df=K - 1)

    stat, ks_p = kstest(pvals, "uniform")
    assert ks_p > 0.01, (
        f"Score-test p-values under H0 show a clear deviation from Uniform(0,1) "
        f"(KS p={ks_p:.4g}), indicating poor calibration; "
        f"mean(p)={pvals.mean():.3f} (expected to be close to 0.5)"
    )


def test_permutation_batch_matches_loop():
    """The batched implementation of permute_min_p should be exactly equivalent
    to manually running score_test_batch for each permutation.

    This is a numerical refactoring test rather than a statistical validation.
    """
    from dmqtl.scoretest import permute_min_p

    rng = np.random.default_rng(3)
    n, K, p, phi = 60, 3, 2, 50.0
    n_snp = 8
    y, X, g0, beta_true = simulate_dm(n, K, p, phi, delta=None, rng=rng)
    G = rng.binomial(2, 0.3, size=(n, n_snp)).astype(float)
    G = G - G.mean(axis=0, keepdims=True)

    null_fit = fit_null_python(y, X)
    ctx = GeneNullContext(y, X, null_fit.eta, null_fit.phi)

    n_perm = 5
    perm_idx = np.array([rng.permutation(n) for _ in range(n_perm)])

    batched = permute_min_p(ctx, G, perm_idx)

    manual = np.empty(n_perm)
    for b in range(n_perm):
        Gp = G[perm_idx[b], :]
        p_each = ctx.score_pvalues(Gp)
        manual[b] = p_each.min()

    np.testing.assert_allclose(batched, manual, rtol=1e-10)

    # method="batched" packs all permutations into a single einsum operation.
    # See the module-level documentation in scoretest.py for when this method
    # is worthwhile and when it should be avoided.
    # Numerically, it must still be exactly equivalent to the loop
    # implementation; this is purely a refactoring-equivalence test.
    batched_method = permute_min_p(ctx, G, perm_idx, method="batched")
    np.testing.assert_allclose(batched_method, manual, rtol=1e-8)


def test_score_pvalues_handles_degenerate_snp_without_nan():
    """A monomorphic SNP (zero variance) is a known degenerate case that can
    occur in real data.

    Even after upstream MAF filtering, a SNP may become degenerate in a
    particular subset after permutation. Here we explicitly construct an
    all-zero (after centering) genotype column and verify that it does not
    produce NaN/Inf values or contaminate argmin. This corresponds to the
    defensive handling added to scoretest._t_to_pvalue.
    """
    rng = np.random.default_rng(11)
    n, K, p, phi = 100, 4, 2, 60.0
    y, X, g0, beta_true = simulate_dm(n, K, p, phi, delta=None, rng=rng)
    null_fit = fit_null_python(y, X)
    ctx = GeneNullContext(y, X, null_fit.eta, null_fit.phi)

    G = np.column_stack([
        np.zeros(n),                       # Degenerate: completely invariant "SNP"
        g0,                                 # Normal, variable SNP for comparison
    ])
    pvals = ctx.score_pvalues(G)
    assert np.isfinite(pvals).all(), f"Non-finite values found in score p-values: {pvals}"
    assert pvals[0] == 1.0, "A zero-variance SNP should conservatively be assigned p=1"
    # argmin should not be contaminated; the degenerate column must not be
    # incorrectly selected as the lead SNP.
    assert np.argmin(pvals) != 0


def test_adaptive_permutation_respects_min_perm_floor():
    """Regression test for a real bug: the early-stopping logic previously had
    no lower bound on the number of permutations. As long as enough hits were
    observed (15 by default), the first batch (e.g., 100-200 permutations)
    could terminate immediately, violating the methodological requirement of
    a "minimum of 1000 permutations".

    This test constructs a high hit-rate scenario in which the number of hits
    is almost certainly >= min_hits_to_stop within the first 100-permutation
    batch, and verifies:

      1. Without min_perm (using the default value of 1000), n_perm_used must
         be >= 1000.
      2. When min_perm is explicitly reduced, early stopping should occur
         before 1000 permutations, confirming that the lower-bound parameter
         is actually honored rather than being hard-coded.
    """
    rng = np.random.default_rng(7)

    def perm_min_p_fn(n_batch):
        # In this uniform scenario with obs_min_p=0.5, each permutation has
        # approximately a 50% chance of being a hit (min_p <= 0.5).
        # Therefore, the first batch of 100 permutations will almost certainly
        # accumulate the required 15 hits.
        return rng.uniform(0, 1, size=n_batch)

    obs_min_p = 0.5

    res_default = adaptive_permutation(perm_min_p_fn, obs_min_p,
                                        batch_size=100, max_perm=10000,
                                        min_hits_to_stop=15)
    assert res_default.n_perm_used >= 1000, (
        f"With the default min_perm=1000, early stopping should not occur at "
        f"{res_default.n_perm_used} permutations"
    )

    res_low_floor = adaptive_permutation(perm_min_p_fn, obs_min_p,
                                          batch_size=100, min_perm=100,
                                          max_perm=10000, min_hits_to_stop=15)
    assert res_low_floor.n_perm_used < 1000, (
        "After explicitly reducing min_perm, early stopping should occur "
        "within 1000 permutations, confirming that the parameter is honored "
        "rather than ignored"
    )
    assert res_low_floor.n_perm_used >= 100
