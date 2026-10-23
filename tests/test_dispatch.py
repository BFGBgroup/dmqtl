"""Genes go to the worker pool one at a time (2026-10).

Per-gene cost spans four orders of magnitude on real data (Ast: K=2 genes take seconds; PCM1, K=13 with
10,000 permutations, took 3,809 s on one core). pool.map's default chunksize hands each worker ~n/(4*workers)
consecutive genes at once (~61 for 13,126 genes on 54 workers), so a chunk holding a few heavy genes keeps one
worker busy while the rest of the node, still billed, sits idle. Results do not depend on scheduling: each
gene's RNG is seeded with seed + its task index (see test_genostore.test_run_dmqtl_store_matches_text).

Heaviest genes go first. In task (genome) order the heaviest genes start wherever they happen to sit; one
that starts near the end runs alone for hours after the rest of the pool has finished (full Ast replay:
~2 h of the 8-14 h permutation stage). The cost key is the per-permutation cost, which scales as
(K-1)^1.83 * n_snp^1.02 (x6 bench, R^2 0.98); the number of permutations a gene will need is not known
in advance.
"""
import multiprocessing as mp
from pathlib import Path

import numpy as np

from dmqtl.eqtl import build_gene_tasks, run_dmqtl
from dmqtl.pipeline import _dispatch_order, run_many_genes

EX = Path(__file__).resolve().parents[1] / "examples"
INPUTS = dict(phenotype_path=EX / "pas_counts.txt", pas_gene_map_path=EX / "pas_to_gene.txt",
              covariates_path=EX / "covariates.txt", snp_pos_path=EX / "snp_pos.txt",
              gene_pos_path=EX / "gene_pos.txt")


def _task(gene_id, K, n_snp, lazy=True):
    t = {"gene_id": gene_id, "y": np.zeros((4, K))}
    if lazy:
        t["variant_ids"] = [f"{gene_id}_v{j}" for j in range(n_snp)]
    else:
        t["G"] = np.zeros((4, n_snp))
    return t


def test_heaviest_genes_go_first_and_results_keep_task_order(monkeypatch):
    sent = []

    class FakePool:
        def __init__(self, n):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, items, chunksize=None):
            sent.extend(item[1]["gene_id"] for item in items)
            return [item[1]["gene_id"] for item in items]  # stand-in result per gene

    monkeypatch.setattr(mp, "Pool", FakePool)
    tasks = [_task("light", 2, 50), _task("wide", 3, 5000), _task("deep", 30, 800, lazy=False),
             _task("mid", 6, 2000), _task("tie_a", 4, 100), _task("tie_b", 4, 100)]
    out = run_many_genes(tasks, null_fits={}, n_workers=4)
    assert sent == ["deep", "mid", "wide", "tie_a", "tie_b", "light"]  # ties keep task order
    assert out == [t["gene_id"] for t in tasks]


def test_reordered_dispatch_gives_identical_output(tmp_path):
    tasks, _ = build_gene_tasks(EX / "genotype.txt", **INPUTS)
    order = _dispatch_order(tasks)
    assert sorted(order) == list(range(len(tasks)))
    assert order != list(range(len(tasks)))  # the example genes really are reordered
    kw = dict(INPUTS, genotype_path=EX / "genotype.txt", use_drimseq="no", perm_batch_size=100,
              min_perm=200, max_perm=400, min_hits_to_stop=10, seed=5)
    run_dmqtl(output_prefix=str(tmp_path / "serial" / "o"), n_workers=1, **kw)
    run_dmqtl(output_prefix=str(tmp_path / "pool" / "o"), n_workers=3, **kw)
    for suf in ("egenes", "cis_qtl_pairs", "significant_pairs"):
        a = (tmp_path / "serial" / f"o.{suf}.txt").read_text()
        b = (tmp_path / "pool" / f"o.{suf}.txt").read_text()
        assert a == b, suf


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
