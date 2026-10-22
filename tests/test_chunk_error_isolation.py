"""An R error caused by one gene must cost that gene only, not its whole chunk (2026-10, real data).

On Han's Ast data DRIMSeq's Cox-Reid fallback (optimHess) stopped with 'non-finite value supplied by optim' for
one gene, BiocParallel turned that into an error of the whole Rscript call, and the chunk of 10 genes -- then
the whole run -- was lost. fit_null_models_batched now refits a failed chunk in halves until the failing gene
is alone, marks it converged=False (method 'r_error'), and keeps every other gene. An error that no sub-chunk
escapes (Rscript missing, DRIMSeq not installed) is still raised.
"""
import numpy as np
import pytest

from dmqtl.nullmodel import NullFit
from dmqtl.pipeline import fit_null_models_batched


def _tasks(n_genes=10, n=30):
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    return [dict(gene_id=f"g{i}", y=rng.integers(0, 50, size=(n, 3)).astype(float), X=X) for i in range(n_genes)]


class _Fit:
    def __init__(self, bad=(), always_fail=False):
        self.bad, self.always_fail, self.calls = set(bad), always_fail, []

    def __call__(self, gene_ys, X):
        self.calls.append(sorted(gene_ys))
        if self.always_fail or self.bad & set(gene_ys):
            raise RuntimeError("DRIMSeq batch null fit failed (Rscript exit code 1): optimHess non-finite")
        return {gid: NullFit(beta=np.zeros((X.shape[1], 2)), phi=10.0 + len(gene_ys), eta=np.zeros((len(y), 2)),
                             loglik=float("nan"), n_categories=3, converged=True, method="fake")
                for gid, y in gene_ys.items()}


@pytest.mark.parametrize("parallel", [1, 2])
def test_one_bad_gene_costs_only_itself(tmp_path, parallel):
    tasks = _tasks()
    fn = _Fit(bad={"g7"})
    fits = fit_null_models_batched(tasks, fn, chunk_size=10, cache_dir=tmp_path, parallel_chunks=parallel)
    assert sorted(fits) == sorted(t["gene_id"] for t in tasks)
    assert not fits["g7"].converged and fits["g7"].method == "r_error" and np.isnan(fits["g7"].phi)
    assert all(fits[g].converged for g in fits if g != "g7")
    assert ["g7"] in fn.calls  # isolated down to the single gene
    # the cache keeps the failure, so a resume does not refit and still reports r_error
    again = _Fit()
    resumed = fit_null_models_batched(tasks, again, chunk_size=10, cache_dir=tmp_path, parallel_chunks=parallel)
    assert again.calls == []
    assert resumed["g7"].method == "r_error" and not resumed["g7"].converged
    assert all(resumed[g].converged for g in resumed if g != "g7")


def test_chunks_without_errors_are_untouched():
    tasks = _tasks(n_genes=20)
    fn = _Fit(bad={"g13"})
    fits = fit_null_models_batched(tasks, fn, chunk_size=10)
    first_chunk = [t["gene_id"] for t in tasks[:10]]
    assert sorted(first_chunk) in fn.calls  # fitted as one call, as before
    assert all(fits[g].phi == 10.0 + 10 for g in first_chunk)


def test_error_that_every_subchunk_hits_is_raised():
    with pytest.raises(RuntimeError):
        fit_null_models_batched(_tasks(), _Fit(always_fail=True), chunk_size=10)
