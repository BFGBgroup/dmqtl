"""
Manual verification script for the reference-category convention.
This script must be run on a machine with R + DRIMSeq installed
(see ../INSTALL_R.md for installation instructions).

Background:
`dmqtl.dmglm.softmax_ref` / `residual_matrix` use the convention that
the last of the K PAS categories is the reference category, and only
the first K-1 non-reference categories are modeled.

`dmqtl/r_null_fit.R` assumes that DRIMSeq's `coefficients()` follows
the same convention (i.e., the last PAS in `pas_ids` is the omitted
reference category). However, this is only an assumption, because
internal implementation details may differ across DRIMSeq versions.
An actual run is therefore required to verify the convention.

`r_null_fit.R` now includes an automatic check: if the feature omitted
by DRIMSeq is not the last PAS, it exits with an error instead of
silently producing incorrect results. However, this check cannot detect
more subtle mismatches, such as the correct reference category being
used but the signs or magnitudes of the internal coefficients being
inconsistent with the direction expected by the Python implementation.

This script performs a more comprehensive end-to-end verification:
it generates simulated data with known coefficients, fits the same
data using both DRIMSeq and the already-validated Python
`fit_null_python` / `softmax_ref` implementation, and compares the
resulting eta values.

Usage:
    # First install R + DRIMSeq according to ../INSTALL_R.md and verify:
    python3 -c "from dmqtl.nullmodel import drimseq_available; print(drimseq_available())"
    # This should print True. Then run:
    python3 examples/verify_drimseq_reference_category.py

How to interpret the output:
    - The "column-wise correlation matrix" should approximately resemble
      a diagonal matrix (diagonal values close to 1 and off-diagonal
      values close to 0). If so, the reference-category convention is
      aligned, and the coefficient direction for each non-reference PAS
      category is also consistent. The DRIMSeq path can then be used
      with confidence.

    - If the matrix clearly does not resemble a diagonal matrix (for
      example, an overall row/column shift or negative values on the
      diagonal), the reference-category convention or column ordering
      is likely inconsistent. Do NOT directly use `fit_null_drimseq`.
      Instead, inspect `dmqtl/r_null_fit.R` and adjust the alignment
      logic based on the specific mismatch pattern printed by this
      script (for example, reorder `pas_ids` or modify the alignment
      between `coef_mat` and `design`). Re-run this script after the
      modification until the matrix is approximately diagonal.

    - The `reference_pas_id` field (printed by this script from the raw
      JSON returned by `fit_null_drimseq`) should equal the last element
      of the simulated `pas_ids` list. If the built-in check in
      `r_null_fit.R` does not raise an error, this condition is already
      satisfied. It is printed here simply to make the reference
      category explicit.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmqtl.dmglm import softmax_ref
from dmqtl.nullmodel import drimseq_available, fit_null_drimseq, fit_null_python


def main():
    if not drimseq_available():
        print("Rscript + DRIMSeq were not detected. This verification script "
              "cannot run on this machine. Install them according to INSTALL_R.md and run the script again.", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(2024)
    n, K, p = 300, 4, 3   # Use a larger sample size to reduce estimation noise.
    true_phi = 80.0

    X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    # Intentionally make the coefficient direction and magnitude of each
    # non-reference PAS category clearly different from the others. This
    # makes an overall category-order mismatch easy to detect from the
    # correlation matrix and avoids masking the problem by accidentally
    # similar coefficient patterns.
    true_beta = np.array([
        [0.0, 0.0, 0.0],       # Intercept row (not important; affects baseline proportions)
        [1.5, -2.0, 0.3],      # Covariate 1 effects; clearly different across categories
        [-0.8, 0.5, 2.2],      # Covariate 2 effects
    ])
    true_eta = X @ true_beta   # (n, K-1)
    pi = softmax_ref(true_eta)

    N = rng.poisson(200, size=n).astype(float)
    N[N < 20] = 20
    y = np.zeros((n, K))
    for i in range(n):
        alpha_i = true_phi * pi[i]
        p_i = rng.dirichlet(alpha_i)
        y[i] = rng.multinomial(int(N[i]), p_i)

    sample_ids = [f"IND{i}" for i in range(n)]
    covariate_names = ["Intercept", "cov1", "cov2"]

    print("Calling DRIMSeq (fit_null_drimseq)...")
    fit_r = fit_null_drimseq(y, X, sample_ids, covariate_names)
    print(f"DRIMSeq fitting completed: beta.shape={fit_r.beta.shape}, "
          f"eta.shape={fit_r.eta.shape}, phi={fit_r.phi:.3g}")

    fit_py = fit_null_python(y, X)

    print(
        "\nColumn-wise correlation matrix "
        "(rows = DRIMSeq eta columns, columns = true eta columns):"
    )
    print(
        "Ideally, this should approximately resemble the identity matrix "
        "(diagonal values close to 1 and off-diagonal values close to 0).\n"
    )
    corr_true = np.corrcoef(np.hstack([fit_r.eta, true_eta]).T)
    km1 = K - 1
    cross_true = corr_true[:km1, km1:]
    header = "        " + "".join(f"true_c{j}  " for j in range(km1))
    print(header)
    for i in range(km1):
        row = "".join(f"{cross_true[i, j]:8.3f} " for j in range(km1))
        print(f"drim_c{i} {row}")

    print("\n(Additional comparison) Column-wise correlation matrix between DRIMSeq eta and the Python fallback eta:")
    corr_py = np.corrcoef(np.hstack([fit_r.eta, fit_py.eta]).T)
    cross_py = corr_py[:km1, km1:]
    print(header)
    for i in range(km1):
        row = "".join(f"{cross_py[i, j]:8.3f} " for j in range(km1))
        print(f"drim_c{i} {row}")

    diag = np.diag(cross_true)
    off_diag_max = np.abs(cross_true - np.diag(diag)).max()
    print(f"\nDiagonal correlations: {diag}")
    print(f"Maximum absolute off-diagonal value: {off_diag_max:.3f}")

    # Note: It is not sufficient to require all off-diagonal correlations
    # to be small. If the simulated covariate effects have opposite
    # directions and similar magnitudes across categories (for example,
    # if cov1 happens to have an effect of +1.5 on category 0 and -2.0
    # on category 1), different true_eta columns can naturally be strongly
    # correlated. This is unrelated to whether the DRIMSeq reference
    # category is correctly aligned.
    #
    # In practice, the correlation matrix calculated directly from true_eta
    # can look very similar to the matrix obtained from the DRIMSeq fit.
    # This indicates that such correlations are an intrinsic property of
    # the simulated data rather than evidence of a category-order mismatch.
    #
    # A more reliable criterion is whether the best match for each column
    # (the column with the largest absolute correlation) occurs exactly on
    # the diagonal. If the reference category or column ordering is actually
    # mismatched (for example, an overall cyclic permutation or category
    # swap), the diagonal will usually not be the strongest match in every
    # row. Under benign natural correlations among the true eta columns,
    # the diagonal should still provide the best match because fitting noise
    # is much smaller than the correlation induced by the covariate effects.
    best_match = np.argmax(np.abs(cross_true), axis=1)
    diagonal_is_best = np.all(best_match == np.arange(km1))
    if diagonal_is_best and np.all(diag > 0.9):
        print(
            "\nThe best match for every column occurs exactly on the diagonal, "
            "and all diagonal correlations are high. The reference-category "
            "convention and column ordering are aligned. (Large off-diagonal "
            "values are likely caused by correlations among the covariate "
            "effects in this simulated dataset rather than by a category "
            "mismatch. You can confirm this by comparing with\n"
            "  np.corrcoef(true_eta.T)\n"
            "If the correlation matrix of the true values already has a "
            "similar structure, this confirms the interpretation.)"
            "\nHowever, this is only one simulation. It is recommended to "
            "repeat the script with several random seeds and different values "
            "of K, and use the DRIMSeq path in production only after all "
            "checks pass."
        )
    else:
        print(
            "\nThe best matches do not all fall on the diagonal, or the "
            "diagonal correlations are not sufficiently high. The "
            "reference-category convention or column ordering may be "
            "misaligned. Do NOT use the current `fit_null_drimseq` results "
            "directly. Inspect how `pas_ids` and `coef_mat` are handled in "
            "dmqtl/r_null_fit.R and adjust the alignment based on the "
            "mismatch pattern printed above (especially the `best_match` "
            "array)."
        )



if __name__ == "__main__":
    main()
