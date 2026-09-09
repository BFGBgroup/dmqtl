from __future__ import annotations

import numpy as np

from dmqtl.dmglm import softmax_ref


def simulate_dm(n: int, K: int, p: int, phi: float, delta: np.ndarray | None,
                 rng: np.random.Generator, mean_total_count: float = 200.0):
    """Simulate DM data for a single gene.

    Parameters
    ----------
    delta : (K-1,) array-like or None
        SNP effects. If None or all zeros, the null hypothesis holds.

    Returns
    -------
    y : ndarray
        Count matrix of shape (n, K).
    X : ndarray
        Covariate matrix of shape (n, p), including the intercept.
    g : ndarray
        Genotype vector of shape (n,), with values 0/1/2 under an
        approximate HWE distribution.
    beta : ndarray
        Null-model coefficient matrix of shape (p, K-1).
    """
    X = np.column_stack([
        np.ones(n),
        rng.normal(size=(n, p - 1)) if p > 1 else np.zeros((n, 0)),
    ])
    beta = rng.normal(scale=0.3, size=(p, K - 1))
    maf = 0.3
    g = rng.binomial(2, maf, size=n).astype(float)
    g = g - g.mean()  # Center the genotype to reduce collinearity with the intercept.

    eta = X @ beta
    if delta is not None:
        eta = eta + np.outer(g, delta)

    pi = softmax_ref(eta)
    N = rng.poisson(mean_total_count, size=n).astype(float)
    N[N < 5] = 5

    y = np.zeros((n, K))
    for i in range(n):
        alpha_i = phi * pi[i]
        p_i = rng.dirichlet(alpha_i)
        y[i] = rng.multinomial(int(N[i]), p_i)
    return y, X, g, beta
