"""
Core mathematical implementation of the Dirichlet-Multinomial GLM.

Parameterization:
    - There are K PAS categories, with the last category (index K-1)
      treated as the reference category.
    - The mean composition pi_i for individual i (length K, summing to 1)
      is specified by a multinomial-logit link:
          eta_i = X_i @ beta            (n, K-1)   beta: (p, K-1)
          pi_i[:K-1] = softmax probabilities for the non-reference categories
          pi_i[K-1]  = probability of the reference category
    - Concentration:
          alpha_i = phi * pi_i
      where phi is a scalar precision parameter for each gene × cell type.
      It is obtained from the DRIMSeq null-model fit and treated as a known
      plug-in parameter during the score test rather than being re-estimated.
      This is a standard approach, analogous to using a shrunken dispersion
      estimate as a fixed quantity in downstream testing in frameworks such
      as edgeR/DESeq2.

Key design decisions:
    - The score (first derivative) uses an analytical formula because the
      derivation is straightforward and numerically reliable.
    - The Fisher/observed information (second derivative), i.e. the IRLS
      weight matrix W_i, is obtained by numerically differentiating the
      analytical score residual function using central differences, rather
      than deriving the full second-order analytical expression manually.
      This is only a small (K-1) x (K-1) matrix for each individual and needs
      to be computed only once because it does not change across SNPs or
      permutations. Therefore, the computational cost is negligible, while
      this approach avoids reintroducing errors from omitted curvature terms.
      `test_score_math.py` cross-validates the result against a numerical
      Hessian computed directly from the brute-force log-likelihood.
"""
from __future__ import annotations

import numpy as np
from scipy.special import digamma, gammaln

# Lower bound to prevent pi from becoming exactly zero under extreme eta
# values due to numerical underflow.
#
# If alpha_ik = phi * pi_ik becomes exactly zero and the corresponding count
# is also zero (a common degenerate case), digamma(0) is singular (-inf).
# Then d_ik = digamma(0) - digamma(0) becomes (-inf) - (-inf) = NaN,
# which can contaminate the entire score-test pipeline for the corresponding
# SNP/gene through U/W/Schur/T. Since NaNs can propagate through subsequent
# matrix operations and produce undefined behavior in operations such as
# np.argmin, this is not merely a theoretical edge case.
#
# In practice, especially with an unregularized `fit_null_python` model,
# quasi-separation can occur in real data when a PAS has nearly zero counts
# at certain covariate levels. The fitted beta/eta can then become sufficiently
# extreme to cause float64 underflow (exp(eta) underflows to zero at roughly
# eta < -745).
#
# Clipping probabilities to a value far below any realistic signal magnitude
# but well above the float64 underflow limit, followed by re-normalization,
# prevents this NaN propagation path while having a negligible numerical
# effect in non-degenerate cases.
_PI_FLOOR = 1e-10


def softmax_ref(eta: np.ndarray) -> np.ndarray:
    """Compute the reference-category multinomial softmax.

    Parameters
    ----------
    eta : np.ndarray
        Shape (n, K-1), containing logits for the non-reference categories.
        The reference category has an implicit logit of 0.

    Returns
    -------
    np.ndarray
        Shape (n, K), containing the full composition. Columns 0..K-2
        correspond to non-reference categories, and column K-1 corresponds
        to the reference category.

        Each probability is clipped to at least `_PI_FLOOR` and the full
        composition is then re-normalized to avoid exact zeros.
    """
    n, km1 = eta.shape
    # Numerical stability: subtract the maximum logit in each row,
    # including the implicit reference logit of 0.
    m = np.maximum(eta.max(axis=1, keepdims=True), 0.0)
    exp_eta = np.exp(eta - m)
    exp_ref = np.exp(-m)  # 参照类 logit=0 => exp(0-m)
    denom = exp_eta.sum(axis=1, keepdims=True) + exp_ref
    pi_nonref = exp_eta / denom
    pi_ref = exp_ref / denom
    pi = np.concatenate([pi_nonref, pi_ref], axis=1)
    pi = np.clip(pi, _PI_FLOOR, None)
    pi = pi / pi.sum(axis=1, keepdims=True)
    return pi


def dm_loglik(y: np.ndarray, pi: np.ndarray, phi: float) -> float:
    """Compute the summed Dirichlet-Multinomial log-likelihood.

    The combinatorial constant that does not depend on the model parameters
    is omitted.

    Parameters
    ----------
    y : np.ndarray
        Shape (n, K), containing category counts.
    pi : np.ndarray
        Shape (n, K), containing the mean composition.
    phi : float
        Scalar concentration/precision parameter.
    """
    alpha = phi * pi
    N = y.sum(axis=1)
    ll = gammaln(phi) - gammaln(N + phi)
    ll = ll + (gammaln(y + alpha) - gammaln(alpha)).sum(axis=1)
    return float(ll.sum())


def residual_matrix(y: np.ndarray, eta: np.ndarray, phi: float) -> np.ndarray:
    """Compute the (K-1)-dimensional generalized residual for each individual.

    Analytical derivation (valid for any covariate entering the likelihood
    through the same linear predictor eta):

        d_ik    = digamma(y_ik + phi*pi_ik) - digamma(phi*pi_ik)
        dbar_i  = sum_k pi_ik * d_ik
        r_ij    = phi * pi_ij * (d_ij - dbar_i),
                  j = 0..K-2 (non-reference categories)

    For any covariate matrix Z with shape (n, q), where the same coefficient
    is applied to each category j, the corresponding score is:

        Z^T @ r

    with shape (q, K-1).
    """
    pi = softmax_ref(eta)
    alpha = phi * pi
    d = digamma(y + alpha) - digamma(alpha)  # (n, K)
    dbar = (pi * d).sum(axis=1, keepdims=True)  # (n, 1)
    r_full = phi * pi * (d - dbar)  # (n, K)
    return r_full[:, :-1]  # Drop the reference-category column (delta_K = 0)


def info_weight_matrix(y: np.ndarray, eta: np.ndarray, phi: float,
                        step: float = 1e-4) -> np.ndarray:
    """Numerically compute the (K-1) x (K-1) IRLS weight matrix W_i.

    For each individual:

        W_i = -d r_i / d eta_i

    Central differences are used to perturb each column of eta. The
    computation is vectorized across all individuals.

    Returns
    -------
    np.ndarray
        Array of shape (n, K-1, K-1).
    """
    n, km1 = eta.shape
    W = np.zeros((n, km1, km1))
    for j in range(km1):
        eta_p = eta.copy()
        eta_m = eta.copy()
        eta_p[:, j] += step
        eta_m[:, j] -= step
        r_p = residual_matrix(y, eta_p, phi)
        r_m = residual_matrix(y, eta_m, phi)
        # d r / d eta_j -> column j; W = -(d r / d eta)
        W[:, :, j] = -(r_p - r_m) / (2 * step)
    # The result should be symmetric up to numerical precision because it
    # corresponds to a Hessian/information matrix. Symmetrize it to remove
    # small finite-difference errors.
    W = 0.5 * (W + np.transpose(W, (0, 2, 1)))
    return W
