"""
Smoke tests for the R/DRIMSeq bridge.

This test is intentionally separate from `test_score_math.py` because it
depends on an external R environment with the DRIMSeq package installed.
The sandbox used for developing this code has neither network access nor R,
so these tests cannot be run here.

The module documentation in `dmqtl/nullmodel.py` has always instructed users
to run `tests/test_r_bridge_smoke.py` once to verify that the R/DRIMSeq bridge
is available, but this file did not actually exist before this round of
changes. It was only mentioned in the documentation. This missing test was
identified and added as part of the current update.

Usage:
    On a machine with R + DRIMSeq (`BiocManager::install("DRIMSeq")`):

        python3 -m pytest tests/test_r_bridge_smoke.py -v

    Or directly:

        python3 tests/test_r_bridge_smoke.py

In an environment without R/DRIMSeq, the tests are automatically skipped
with an informational message and do not fail.

This test performs only the following checks:

    1. Whether `fit_null_drimseq` can be called successfully without raising
       an exception, and whether the returned beta/eta/phi have the expected
       shapes and values.

    2. Whether `fit_null_drimseq` and `fit_null_python`, when run on the same
       simulated data, produce eta (linear predictor) values that are broadly
       consistent, as measured by a sufficiently high correlation.

       This is only a coarse-grained sanity check. The two fitting methods are
       not expected to produce exactly identical numerical results because
       DRIMSeq performs cross-gene precision shrinkage whereas the Python
       fallback does not.

       This test therefore CANNOT replace the more important and more
       fine-grained verification emphasized repeatedly in the `nullmodel.py`
       documentation: whether the PAS category treated internally as the
       reference category by DRIMSeq is exactly consistent with the Python-side
       convention used by `dmqtl.dmglm.softmax_ref` / `residual_matrix`, where
       the last category is the reference.

       That convention must be manually verified in a real R environment
       using a small, self-constructed example with known true coefficients
       and analytically specified eta values, followed by a direct
       category-by-category numerical comparison with the DRIMSeq output.

       A high correlation alone does not prove that the reference category is
       correctly aligned. A category mismatch may appear as a sign reversal
       or an overall cyclic permutation of categories, and a coarse
       correlation measure may not reliably distinguish this from the
       ordinary numerical differences between two fitting methods.

       This manual verification must therefore be performed once in a real
       R environment. This smoke test cannot replace it.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dmqtl.nullmodel import drimseq_available


def test_drimseq_bridge_smoke():
    if not drimseq_available():
        print(
            "[SKIP] Rscript or the DRIMSeq package was not found. "
            "Skipping the R bridge smoke test. Install R + DRIMSeq on "
            "your machine and run this test again.",
            file=sys.stderr,
        )
        return

    from dmqtl.nullmodel import fit_null_drimseq, fit_null_python
    from tests.simulate import simulate_dm

    rng = np.random.default_rng(0)
    n, K, p, phi = 150, 4, 3, 60.0
    y, X, g, beta_true = simulate_dm(n, K, p, phi, delta=None, rng=rng)
    sample_ids = [f"IND{i}" for i in range(n)]
    covariate_names = ["Intercept", "cov1", "cov2"]

    fit_r = fit_null_drimseq(y, X, sample_ids, covariate_names)
    assert fit_r.beta.shape == (p, K - 1)
    assert fit_r.eta.shape == (n, K - 1)
    assert np.isfinite(fit_r.eta).all()
    assert fit_r.phi > 0

    fit_py = fit_null_python(y, X)

    corr = np.corrcoef(fit_r.eta.ravel(), fit_py.eta.ravel())[0, 1]
    print(
        f"[R bridge smoke test] eta correlation "
        f"(DRIMSeq vs Python fallback): {corr:.4f}"
    )
    assert corr > 0.8, (
        f"The correlation between the DRIMSeq and Python fallback eta "
        f"estimates is only {corr:.4f}, which is unexpectedly low. "
        "The reference-category convention may not be aligned "
        "(see the explanation at the top of this file). "
        "This failure should not be ignored; manual verification is required."
    )
    print(
        "[R bridge smoke test] PASS "
        "(coarse-grained sanity check only; reference-category alignment "
        "still requires manual verification; see the explanation above)"
    )


def test_drimseq_recovers_known_composition():
    """A more stringent check than `test_drimseq_bridge_smoke`: instead of
    requiring only a high correlation with the Python fallback, directly
    construct an intercept-only dataset with a known true composition and
    verify that the composition fitted by DRIMSeq, interpreted through the
    Python-side `dmqtl.dmglm.softmax_ref`, matches the known composition
    category by category within the expected sampling error.

    Why this is more effective at detecting reference-category mismatches:

    If the reference category or coefficient ordering is globally misaligned
    (for example, the category order is cyclically permuted, or the sign or
    interpretation of the coefficient for category k is opposite to the
    Python-side convention), the fitted composition will typically be a
    permutation of the true composition or have the wrong direction.

    In such a situation, the correlation with an independently fitted
    Python fallback model may not necessarily be obviously low. Both methods
    are fitting the same data, and even when the categories are misaligned,
    the numerical magnitudes can still look similar. A coarse correlation
    measure may therefore fail to distinguish a genuine category mismatch
    from ordinary numerical differences between the two fitting methods.

    By directly comparing the fitted composition with a simple, explicitly
    specified composition whose values can be checked by inspection, a
    category mismatch becomes much easier to detect: the fitted proportions
    will either have incorrect numerical values or appear in the wrong
    category order.

    This is also closer to the intended meaning of the repeated instruction
    in the `nullmodel.py` documentation to manually verify the DRIMSeq
    reference-category convention using a small example with known
    coefficients.

    Usage (same as `test_drimseq_bridge_smoke`):
        python3 tests/test_r_bridge_smoke.py

    If this test fails, do not bypass it or simply relax the tolerance.
    Instead, inspect the specific error reported by `r_null_fit.R`, or
    manually print `fitted_pi` versus `true_pi` category by category to
    identify exactly which category is misaligned.
    """
    if not drimseq_available():
        print(
            "[SKIP] Rscript or the DRIMSeq package was not found. "
            "Skipping the known-composition verification.",
            file=sys.stderr,
        )
        return

    from dmqtl.dmglm import softmax_ref
    from dmqtl.nullmodel import fit_null_drimseq

    # Manually specify a simple, easy-to-interpret true composition
    # (K=3, with no covariate effects and the same true pi for every
    # individual), then derive the corresponding eta values, which are
    # the inputs expected by softmax_ref.
    #
    # The data are simulated with a sufficiently large sample size
    # (n=1500) so that the MLE has a good chance of approaching the true
    # values. With a small sample size, even a perfectly aligned reference
    # category would produce estimation noise. That is not what this test
    # is intended to measure, so n is deliberately made large to separate
    # ordinary MLE noise from a reference-category mismatch.
    true_pi = np.array([0.50, 0.30, 0.20])   # Categories 0/1/2; category 2 is the reference.
    true_eta = np.log(true_pi[:2] / true_pi[2])  # (K-1,), softmax_ref convention.
    K = 3
    n = 1500
    phi = 80.0

    rng = np.random.default_rng(123)
    X = np.ones((n, 1))  # Intercept only; no covariates.
    eta_all = np.tile(true_eta, (n, 1))
    pi_check = softmax_ref(eta_all)
    assert np.allclose(pi_check[0], true_pi, atol=1e-6), "The derived eta is incorrect; check this calculation first."

    N = rng.poisson(200, size=n).astype(float)
    N[N < 20] = 20
    y = np.zeros((n, K))
    for i in range(n):
        p_i = rng.dirichlet(phi * true_pi)
        y[i] = rng.multinomial(int(N[i]), p_i)

    sample_ids = [f"IND{i}" for i in range(n)]
    fit_r = fit_null_drimseq(y, X, sample_ids, covariate_names=["Intercept"])
    fitted_pi = softmax_ref(fit_r.eta).mean(axis=0)  # Intercept-only model: all rows should be nearly identical.

    print(f"[Known-composition check] True pi = {true_pi}")
    print(
        "[Known-composition check] DRIMSeq-fitted pi "
        f"(interpreted using Python softmax_ref) = "
        f"{np.round(fitted_pi, 4)}"
    )

    max_abs_err = np.max(np.abs(fitted_pi - true_pi))
    assert max_abs_err < 0.03, (
        f"The maximum absolute error in the category-by-category comparison "
        f"is {max_abs_err:.4f}, which is substantially larger than the "
        f"expected sampling noise at n=1500. This is likely a "
        "reference-category or coefficient-ordering mismatch rather than "
        f"ordinary fitting noise. True pi={true_pi}, "
        f"fitted pi={np.round(fitted_pi, 4)}. "
        "Check whether r_null_fit.R reported a reference-category mismatch. "
        "If no error was reported but the values still disagree, the omitted "
        "feature may be correct while the mathematical interpretation or "
        "sign convention of the returned coefficients differs from the "
        "Python-side convention. The dmFit/coefficients() convention must "
        "then be re-verified before integrating the DRIMSeq path into the "
        "main pipeline."
    )
    print(
        "[Known-composition check] PASS: the DRIMSeq-fitted composition "
        "matches the true values category by category, indicating that the "
        "reference-category convention is aligned for this example."
    )


def test_drimseq_batch_recovers_known_composition_multi_gene():
    """Batch version of `test_drimseq_recovers_known_composition`: fit three
    genes with different K values and different known true compositions
    in a single `fit_null_drimseq_batch` call, and verify that the
    reference-category transformation is correct for each gene.

    This requires a dedicated test because the batch implementation
    (`r_null_fit_batch.R`) changes the original single-gene logic into
    grouped processing by `gene_id`. If there is a grouping or ordering
    bug (for example, coefficients from gene A are accidentally assigned
    to gene B), such a bug may not be detected by the single-gene test.
    It must be tested using multiple genes in the same batch.
    """
    if not drimseq_available():
        print(
            "[SKIP] Rscript or the DRIMSeq package was not found. "
            "Skipping the batch known-composition verification.",
            file=sys.stderr,
        )
        return

    from dmqtl.dmglm import softmax_ref
    from dmqtl.nullmodel import fit_null_drimseq_batch

    rng = np.random.default_rng(7)
    n = 1500
    X = np.ones((n, 1))

    gene_specs = {
        "GENE_A": (np.array([0.50, 0.30, 0.20]), 80.0),
        "GENE_B": (np.array([0.60, 0.25, 0.15]), 60.0),
        "GENE_C": (np.array([0.40, 0.35, 0.15, 0.10]), 100.0),
    }

    gene_ys = {}
    for gid, (true_pi, phi) in gene_specs.items():
        N = rng.poisson(200, size=n).astype(float)
        N[N < 20] = 20
        K = len(true_pi)
        y = np.zeros((n, K))
        for i in range(n):
            p_i = rng.dirichlet(phi * true_pi)
            y[i] = rng.multinomial(int(N[i]), p_i)
        gene_ys[gid] = y

    fits = fit_null_drimseq_batch(gene_ys, X, covariate_names=["Intercept"])

    for gid, (true_pi, _phi) in gene_specs.items():
        assert gid in fits, f"{gid} is missing from the batch-fitting results."
        fit = fits[gid]
        assert fit.converged, f"{gid} did not converge."
        fitted_pi = softmax_ref(fit.eta).mean(axis=0)
        max_abs_err = np.max(np.abs(fitted_pi - true_pi))
        print(
            f"[Batch known-composition check] {gid}: "
            f"true pi={true_pi}, fitted pi={np.round(fitted_pi, 4)}, "
            f"maximum absolute error={max_abs_err:.4f}"
        )
        assert max_abs_err < 0.03, (
            f"{gid} has a relatively large category-by-category error "
            f"of {max_abs_err:.4f}. The batch implementation may be "
            "mixing coefficients between genes, or the reference-category "
            "transformation may contain a bug. Inspect "
            "r_null_fit_batch.R manually."
        )
    print(
        "[Batch known-composition check] PASS: all three genes with "
        "different K values and different true compositions recovered "
        "their respective true values correctly in the same batch call."
    )


def test_drimseq_batch_shrinks_precision_across_genes():
    """Core validation: determine whether batch fitting actually performs
    cross-gene precision shrinkage rather than merely exposing a different
    interface while internally fitting each gene independently.

    === Complete investigation log from 2026-09
    (for future maintainers; avoid repeating the same debugging steps) ===

    The first version used the assertion
    "batch_phi.std() < independent_phi.std()" as the success criterion.
    This produced a technically true but practically meaningless false
    positive: the standard deviation decreased by only 0.5%.

    The second round suspected that `prec_subset` was too small. After
    explicitly setting `prec_subset=1.0`, the results were almost unchanged,
    ruling this out.

    The third round suspected that `prec_prior_df` defaulted to 0. Passing
    a positive value produced results that were numerically identical to
    the default, initially leading to the suspicion that
    `prec_moderation` also needed to be explicitly set because its default
    was assumed to be `"none"`.

    The fourth round checked the official `?DRIMSeq::dmPrecision`
    documentation and confirmed that for the `dmDSdata` dispatch used here,
    `prec_moderation` actually defaults to `"trended"`. The previous
    assumption had confused it with the default for the `dmSQTLdata`
    dispatch. It was also confirmed that `prec_prior_df` is not normally a
    user-controlled shrinkage-strength parameter; DRIMSeq estimates the
    shrinkage level automatically.

    The fifth round increased the number of genes from 15 to 300 and used
    the complete DRIMSeq defaults. The result was worse than simply
    "shrinkage not working": the batch-fitted phi values had mean/std
    one to two orders of magnitude larger than the independent
    single-gene fits (for example, the standard deviation increased from
    approximately 47 to more than 9000). This is no longer a question of
    "shrinkage not taking effect"; the batch fit itself was producing
    implausible numerical values.

    This version makes three changes:

    1. `fit_null_drimseq_batch(..., verbose=True)` is used. Previously,
       diagnostic output from the R-side `message()` / `print()` calls was
       only surfaced when an error occurred. A call that completed
       successfully but produced suspicious numerical values could therefore
       fail silently. This is exactly the situation that required further
       investigation in the fifth round.

    2. An additional comparison with `prec_moderation="common"` is performed.
       These synthetic data are deliberately designed so that phi is
       unrelated to mean expression. For the default `"trended"` mode,
       which shrinks toward a precision-versus-mean trend, this means fitting
       a trend that does not actually exist. Sampling noise across only
       18 individuals may be mistaken for a trend, and extrapolation for
       genes at the tails of the expression distribution may produce
       unstable extreme values. `"common"` shrinks toward a single common
       value and does not use mean expression, so it is theoretically more
       consistent with the structure of these synthetic data. Stability must
       still be determined empirically rather than assumed in advance.

    3. Summary statistics now include min/max/median in addition to mean/std.
       Mean and standard deviation are highly sensitive to outliers and cannot
       distinguish between a generally problematic fit and a small number of
       genes with extreme values. The median helps distinguish these cases.

    Independent single-gene fitting (`fit_null_drimseq`) launches one Rscript
    subprocess per gene. Running all 300 genes independently would therefore
    be very slow, so only 100 genes are sampled as the independent baseline.
    The batch side still fits all 300 genes jointly.

    === Additional note from 2026-09
    After excluding four unreliable genes, the remaining difference was likely
    a sample-size effect rather than a new bug.

    In the ninth-round patch, after excluding the four genes flagged as
    boundary hits or outliers, the median was already close to the true
    phi=60, and the standard deviation difference dropped from approximately
    six-fold to 1.6-fold. However, the maximum value (690.89) was still nearly
    three times the maximum of 244.76 observed in the independent sample,
    which at that time contained only 30 genes.

    This difference is likely a natural sampling effect: when drawing 296
    observations from the same right-skewed distribution instead of 30,
    the expected maximum is naturally larger. Comparing standard deviation
    and maximum between groups with different sample sizes is therefore not
    a fair comparison.

    The independent baseline was increased from 30 to 100 genes so that the
    two groups have more comparable sample sizes. If the maximum among the
    100 independently fitted genes is also substantially larger than the
    previous maximum of 244.76, this supports the sampling-effect explanation.
    If the independent maximum remains around 250 while the filtered batch
    maximum is still substantially larger, then a batch-specific problem
    remains and requires further investigation.

    No hard conclusion is made here. The exact numerical values depend on
    the data and DRIMSeq version. Two previous attempts to use intuitive
    hard thresholds were shown to be flawed test designs. This test therefore
    reports the observed values without imposing a rigid pass/fail criterion;
    use the printed R stdout/stderr for further investigation if necessary.

    This test is relatively slow because it launches more than 100 Rscript
    subprocesses. It only needs to be run once on a machine with R and is
    not recommended for routine CI execution.
    """
    if not drimseq_available():
        print("[SKIP] Rscript or the DRIMSeq package was not found. Skipping the cross-gene shrinkage verification.",
              file=sys.stderr)
        return

    from dmqtl.nullmodel import fit_null_drimseq, fit_null_drimseq_batch

    rng = np.random.default_rng(2026)
    n_genes = 300
    n_independent_check = 100  # Number of independently fitted genes used as the baseline.
    n, K, true_phi = 18, 3, 60.0
    X = np.ones((n, 1))
    true_pi = np.array([0.5, 0.3, 0.2])

    gene_ys = {}
    for gi in range(n_genes):
        N = rng.poisson(60, size=n).astype(float)
        N[N < 10] = 10
        y = np.zeros((n, K))
        for i in range(n):
            p_i = rng.dirichlet(true_phi * true_pi)
            y[i] = rng.multinomial(int(N[i]), p_i)
        gene_ys[f"GENE{gi}"] = y

    sample_ids = [f"IND{i}" for i in range(n)]
    check_gene_ids = list(gene_ys.keys())[:n_independent_check]
    independent_phi = []
    for gid in check_gene_ids:
        fit = fit_null_drimseq(gene_ys[gid], X, sample_ids, covariate_names=["Intercept"])
        independent_phi.append(fit.phi)
    independent_phi = np.array(independent_phi)

    # Do not pass either prec_subset or prec_prior_df. Use the exact DRIMSeq
    # defaults. This corresponds to the path taken by
    # run_dmqtl(use_drimseq="yes") when no null_fit_prec_* parameters are
    # explicitly configured.
    #
    # For the dmDSdata dispatch, prec_moderation already defaults to
    # "trended"; see the docstring above.
    #
    # verbose=True ensures that R stdout/stderr is printed regardless of
    # whether the numerical result appears reasonable.
    batch_fits_trended = fit_null_drimseq_batch(
        gene_ys, X, covariate_names=["Intercept"], verbose=True)
    batch_phi_trended_checked = np.array([batch_fits_trended[g].phi for g in check_gene_ids])
    batch_phi_trended_all = np.array([batch_fits_trended[g].phi for g in gene_ys])

    batch_fits_common = fit_null_drimseq_batch(
        gene_ys, X, covariate_names=["Intercept"], prec_moderation="common", verbose=True)
    batch_phi_common_all = np.array([batch_fits_common[g].phi for g in gene_ys])

    def summarize(label, phi):
        print(
            f"[Cross-gene shrinkage validation] {label}: "
            f"mean={phi.mean():.2f}, std={phi.std():.2f}, "
            f"min={phi.min():.2f}, median={np.median(phi):.2f}, "
            f"max={phi.max():.2f}"
        )

    def summarize_converged_only(label, fits, gene_ids):
        # Added in 2026-09: r_null_fit_batch.R now marks genes whose precision
        # estimates hit the upper boundary of the grid search as
        # converged=False. The phi value itself is still numerical rather
        # than NaN, but it should not be treated as a normal estimate.
        #
        # Here we summarize only genes with converged=True to determine
        # whether the remaining batch-fitted values become reasonable after
        # excluding these saturated boundary cases. If so, the batch
        # architecture itself is likely working correctly and the problematic
        # values are isolated to individual genes that need this safety filter.
        converged_ids = [g for g in gene_ids if fits[g].converged]
        n_flagged = len(gene_ids) - len(converged_ids)
        phi = np.array([fits[g].phi for g in converged_ids])
        print(
            f"[Cross-gene shrinkage validation] {label} "
            f"(excluding {n_flagged} genes marked converged=False; "
            f"{len(converged_ids)} remaining): "
            f"mean={phi.mean():.2f}, std={phi.std():.2f}, "
            f"min={phi.min():.2f}, median={np.median(phi):.2f}, "
            f"max={phi.max():.2f}"
        )

    print(
        f"[Cross-gene shrinkage validation] True phi = {true_phi}, "
        f"number of batch genes = {n_genes}"
    )
    summarize(
        f"Independent single-gene fits (sampled {n_independent_check})",
        independent_phi,
    )
    summarize(
        f"Batch fit (trended, default), same {n_independent_check} genes",
        batch_phi_trended_checked,
    )
    summarize(
        f"Batch fit (trended, default), all {n_genes} genes "
        "(including unfiltered raw values)",
        batch_phi_trended_all,
    )
    summarize(
        f"Batch fit (common), all {n_genes} genes "
        "(including unfiltered raw values)",
        batch_phi_common_all,
    )
    summarize_converged_only(
        "Batch fit (trended, default), all genes",
        batch_fits_trended,
        list(gene_ys),
    )
    summarize_converged_only(
        "Batch fit (common), all genes",
        batch_fits_common,
        list(gene_ys),
    )
    print(
        "[Cross-gene shrinkage validation] Use your own judgment: if, after "
        "excluding genes marked converged=False, the std/max values in both "
        "batch-fit summaries become comparable to the independent "
        "single-gene fits and the median approaches the true phi=60, "
        "the batch architecture itself is likely sound. The extreme values "
        "observed previously were then likely caused by individual genes "
        "whose precision estimates saturated at the grid boundary and are "
        "now automatically flagged and excluded. If std/max remain much "
        "larger after exclusion, some boundary case is still not covered. "
        "Inspect the R stdout/stderr printed above with verbose=True and "
        "continue the investigation."
    )


if __name__ == "__main__":
    test_drimseq_bridge_smoke()
    test_drimseq_recovers_known_composition()
    test_drimseq_batch_recovers_known_composition_multi_gene()
    test_drimseq_batch_shrinks_precision_across_genes()
