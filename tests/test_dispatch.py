"""Genes go to the worker pool one at a time (2026-10).

Per-gene cost spans four orders of magnitude on real data (Ast: K=2 genes take seconds; PCM1, K=13 with
10,000 permutations, took 3,809 s on one core). pool.map's default chunksize hands each worker ~n/(4*workers)
consecutive genes at once (~61 for 13,126 genes on 54 workers), so a chunk holding a few heavy genes keeps one
worker busy while the rest of the node, still billed, sits idle. Results do not depend on scheduling: each
gene's RNG is seeded with seed + its task index (see test_genostore.test_run_dmqtl_store_matches_text).
"""
import multiprocessing as mp

from dmqtl.pipeline import run_many_genes


def test_genes_are_dispatched_one_at_a_time(monkeypatch):
    seen = {}

    class FakePool:
        def __init__(self, n):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, items, chunksize=None):
            seen["chunksize"] = chunksize
            return [fn(x) for x in items]

    monkeypatch.setattr(mp, "Pool", FakePool)
    tasks = [{"gene_id": f"g{i}"} for i in range(500)]
    assert run_many_genes(tasks, null_fits={}, n_workers=4) == []  # no null fits: every gene is skipped
    assert seen["chunksize"] == 1
