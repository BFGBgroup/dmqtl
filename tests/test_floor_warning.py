"""Warn when the Beta approximation rests on permutation min-p values piled up at the 1e-12 floor (2026-10).

On Han's Ast data ADAMTS18 had 63% of 1000 permutation min-p values <= 1e-12 (a single outlying individual
was the only informative carrier of rare SNPs). Its observed min-p, 2.1e-25, is below the floor too, so the
earlier rule (warn only when the observed min-p is above the floor) stayed silent, although its Beta p
(0.032) was far below the empirical p (0.37). The warning must fire whatever the observed min-p, and show
both p-values.
"""
import numpy as np

import dmqtl.pipeline as pl
from dmqtl.nullmodel import NullFit
from dmqtl.pvalue import AdaptivePermResult, floor_warning


def _adamts18_like(seed=0):
    rng = np.random.default_rng(seed)
    x = np.concatenate([np.full(370, 1e-30), np.full(262, 1e-14), rng.beta(0.5, 500, size=368)])
    rng.shuffle(x)
    return AdaptivePermResult(obs_min_p=2.1e-25, n_perm_used=1000, n_hits=370, perm_min_p=x)


def test_warns_when_observed_min_p_is_below_the_floor():
    msg = floor_warning(_adamts18_like(), p_beta=0.032)
    assert msg is not None
    assert "63%" in msg and "0.032" in msg and "0.37" in msg  # empirical p = 371 / 1001


def test_warns_when_observed_min_p_is_above_the_floor():
    rng = np.random.default_rng(1)
    x = np.concatenate([np.full(350, 1e-15), rng.beta(0.5, 500, size=650)])
    res = AdaptivePermResult(obs_min_p=6.5e-8, n_perm_used=1000, n_hits=584, perm_min_p=x)
    assert "35%" in floor_warning(res, p_beta=0.48)


def test_silent_when_at_most_one_percent_at_the_floor():
    rng = np.random.default_rng(2)
    x = np.concatenate([np.full(10, 1e-15), rng.beta(0.9, 1000, size=990)])
    res = AdaptivePermResult(obs_min_p=1e-4, n_perm_used=1000, n_hits=int((x <= 1e-4).sum()), perm_min_p=x)
    assert floor_warning(res, p_beta=0.1) is None


def test_run_one_gene_prints_the_warning(monkeypatch, capsys):
    res = _adamts18_like()
    monkeypatch.setattr(pl, "adaptive_permutation", lambda *a, **k: res)
    monkeypatch.setattr(pl, "beta_approx_pvalue", lambda r: (0.032, 0.07, 5560.0))
    rng = np.random.default_rng(3)
    n = 40
    y = rng.integers(0, 20, size=(n, 2)).astype(float)
    X = np.ones((n, 1))
    G = rng.integers(0, 3, size=(n, 5)).astype(float)
    G -= G.mean(axis=0)
    fit = NullFit(beta=np.zeros((1, 1)), phi=50.0, eta=np.zeros((n, 1)), loglik=0.0, n_categories=2,
                  converged=True, method="test")
    out = pl.run_one_gene("ADAMTS18", "Ast", y, X, G, rng, null_fit=fit)
    assert out is not None and out.beta_p == 0.032
    err = capsys.readouterr().err
    assert "WARNING: gene ADAMTS18" in err and "63%" in err and "0.37" in err
