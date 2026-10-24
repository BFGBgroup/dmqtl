"""The Beta approximation must not take down the run when scipy's MLE does not converge (2026-10, real data).

On Han's Ast data, gene GPR157 had 523 of 1000 permutation min-p values at the 1e-12 floor; scipy's
beta.fit(floc=0, fscale=1) raised FitSolverError and, uncaught, ended the whole run (v1.0.0 and dev alike).
beta_approx_pvalue now keeps scipy's answer whenever scipy converges (same numbers as before), falls back to
Nelder-Mead on the same log-likelihood from the method-of-moments start (as tensorQTL does), and only if that
fails too returns the empirical permutation p-value. It never returns NaN.
"""
import warnings

import numpy as np
import pytest
from scipy import stats

from dmqtl.pvalue import AdaptivePermResult, beta_approx_pvalue


def _degenerate(seed=0):
    rng = np.random.default_rng(seed)
    x = np.concatenate([np.full(523, 1e-12), rng.beta(0.6, 600, size=477)])
    rng.shuffle(x)
    return x


def test_fixture_breaks_scipy():
    with warnings.catch_warnings(), pytest.raises(Exception):
        warnings.simplefilter("ignore")
        stats.beta.fit(np.clip(_degenerate(), 1e-12, 1 - 1e-12), floc=0, fscale=1)


def test_scipy_failure_falls_back_to_nelder_mead():
    x = _degenerate()
    res = AdaptivePermResult(obs_min_p=1.1e-3, n_perm_used=1000, n_hits=int((x <= 1.1e-3).sum()), perm_min_p=x)
    p, a, b = beta_approx_pvalue(res)
    assert np.isfinite(p) and 0 <= p <= 1 and np.isfinite(a) and np.isfinite(b)
    # it is the MLE: no nearby point has a higher log-likelihood
    xc = np.clip(x, 1e-12, 1 - 1e-12)
    ll = lambda aa, bb: stats.beta.logpdf(xc, aa, bb).sum()  # noqa: E731
    for fa in (0.98, 1.02):
        for fb in (0.98, 1.02):
            assert ll(a, b) >= ll(a * fa, b * fb) - 1e-6
    assert p == pytest.approx(float(stats.beta.cdf(1.1e-3, a, b)))


def test_scipy_answer_kept_when_it_converges():
    rng = np.random.default_rng(3)
    x = rng.beta(0.9, 1200, size=1000)
    res = AdaptivePermResult(obs_min_p=2e-4, n_perm_used=1000, n_hits=int((x <= 2e-4).sum()), perm_min_p=x)
    a0, b0, _, _ = stats.beta.fit(np.clip(x, 1e-12, 1 - 1e-12), floc=0, fscale=1)
    p, a, b = beta_approx_pvalue(res)
    assert (a, b) == (float(a0), float(b0))
    assert p == float(stats.beta.cdf(2e-4, a0, b0))


def test_empirical_p_when_no_fit_is_possible():
    x = np.full(1000, 1e-12)  # no spread at all: no Beta fit exists
    res = AdaptivePermResult(obs_min_p=0.5, n_perm_used=1000, n_hits=1000, perm_min_p=x)
    p, a, b = beta_approx_pvalue(res)
    assert p == pytest.approx(1001 / 1001)
    assert np.isnan(a) and np.isnan(b)


# scipy can also return an invalid fit without raising. Full Ast run, 2026-10-24 (scipy 1.15.3 on the server):
# 11 genes with 25-96% of their min-p values at the floor got Beta p = nan, one nan made every Storey q-value
# nan, and run_dmqtl's q-value check stopped the run after the 20 h permutation stage, before any output was
# written. scipy 1.15.3 returns b = -7.5e6 for the vector below (84% at the floor, the rest log-uniform on
# 1e-12..1e-6); newer scipy may fit it fine, so the invalid fit is also forced with a stand-in.
def _invalid_on_scipy_115(seed=1):
    rng = np.random.default_rng(seed)
    x = np.concatenate([np.full(840, 1e-12), 10 ** rng.uniform(-12, -6, size=160)])
    return np.clip(x, 1e-12, 1 - 1e-12)


def _check_fallback(res, p, a, b):
    assert np.isfinite(p) and 0 <= p <= 1
    if np.isnan(a):  # no fit at all: empirical permutation p-value
        assert np.isnan(b) and p == pytest.approx((res.n_hits + 1) / (res.n_perm_used + 1))
    else:
        assert a > 0 and b > 0 and np.isfinite(a) and np.isfinite(b)
        assert p == pytest.approx(float(stats.beta.cdf(res.obs_min_p, a, b)))


def test_invalid_scipy_fit_is_treated_as_failed(monkeypatch):
    x = _invalid_on_scipy_115()
    monkeypatch.setattr(stats.beta, "fit", lambda *args, **kw: (0.038, -7.5e6, 0.0, 1.0))
    res = AdaptivePermResult(obs_min_p=8.1e-14, n_perm_used=1000, n_hits=int((x <= 8.1e-14).sum()), perm_min_p=x)
    _check_fallback(res, *beta_approx_pvalue(res))


def test_server_scipy_shape_gives_finite_p():
    x = _invalid_on_scipy_115()
    res = AdaptivePermResult(obs_min_p=8.1e-14, n_perm_used=1000, n_hits=int((x <= 8.1e-14).sum()), perm_min_p=x)
    _check_fallback(res, *beta_approx_pvalue(res))
