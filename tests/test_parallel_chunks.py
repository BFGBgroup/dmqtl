"""Running null-model chunks concurrently must give the same fits and cache files as running them in order."""
import threading
import time

import numpy as np

from dmqtl.nullmodel import NullFit
from dmqtl.pipeline import fit_null_models_batched


def _tasks(n_genes=23, n=30):
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    return [dict(gene_id=f"g{i}", y=rng.integers(0, 50, size=(n, 3)).astype(float), X=X) for i in range(n_genes)]


class _FakeBatchFit:
    """Deterministic per-gene 'fit' that also records how many chunks ran at once."""

    def __init__(self):
        self.lock, self.active, self.peak = threading.Lock(), 0, 0

    def __call__(self, gene_ys, X):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.05)
        out = {gid: NullFit(beta=np.zeros((X.shape[1], 2)), phi=float(y.sum() % 97) + 1.0,
                            eta=y[:, :2] / (1.0 + y.sum()), loglik=float("nan"), n_categories=3,
                            converged=bool(int(y.sum()) % 5), method="fake")
               for gid, y in gene_ys.items()}
        with self.lock:
            self.active -= 1
        return out


def test_parallel_chunks_same_fits_and_cache(tmp_path):
    tasks = _tasks()
    seq_fn, par_fn = _FakeBatchFit(), _FakeBatchFit()
    a = fit_null_models_batched(tasks, seq_fn, chunk_size=4, cache_dir=tmp_path / "a")
    b = fit_null_models_batched(tasks, par_fn, chunk_size=4, cache_dir=tmp_path / "b", parallel_chunks=3)
    assert seq_fn.peak == 1 and par_fn.peak > 1
    assert list(a) == list(b)
    for gid in a:
        assert a[gid].phi == b[gid].phi and a[gid].converged == b[gid].converged
        assert np.array_equal(a[gid].eta, b[gid].eta)
    files = sorted(p.name for p in (tmp_path / "a").iterdir())
    assert files == sorted(p.name for p in (tmp_path / "b").iterdir())
    for name in files:
        assert (tmp_path / "a" / name).read_text() == (tmp_path / "b" / name).read_text()
    # resume from cache: nothing is refitted
    again = _FakeBatchFit()
    c = fit_null_models_batched(tasks, again, chunk_size=4, cache_dir=tmp_path / "b", parallel_chunks=3)
    assert again.peak == 0 and list(c) == list(a)
