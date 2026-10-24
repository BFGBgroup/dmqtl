"""The permutation results are saved before the q-value stage, and the outputs can be rebuilt from them.

Full Ast run, 2026-10-24: the permutation stage took 20 h (~28 yuan), then the q-value stage stopped the run
(11 genes had Beta p = nan, see test_beta_fallback) before any output was written, and everything was lost.
run_dmqtl now writes <prefix>.gene_results.pkl as soon as the permutations finish; `dmqtl-finish` (cli.finish_main)
writes the three output files from it, the same files run_dmqtl writes.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import dmqtl.eqtl as eqtl
from dmqtl.cli import finish_main
from dmqtl.eqtl import run_dmqtl

EX = Path(__file__).resolve().parents[1] / "examples"
KW = dict(genotype_path=EX / "genotype.txt", phenotype_path=EX / "pas_counts.txt",
          pas_gene_map_path=EX / "pas_to_gene.txt", covariates_path=EX / "covariates.txt",
          snp_pos_path=EX / "snp_pos.txt", gene_pos_path=EX / "gene_pos.txt", use_drimseq="no",
          perm_batch_size=100, min_perm=200, max_perm=400, min_hits_to_stop=10, seed=5, n_workers=1)
SUFFIXES = ("egenes", "cis_qtl_pairs", "significant_pairs")


def test_outputs_rebuilt_from_saved_gene_results(tmp_path):
    run_dmqtl(output_prefix=str(tmp_path / "run" / "o"), **KW)
    pkl = tmp_path / "run" / "o.gene_results.pkl"
    assert pkl.exists()
    assert finish_main([str(pkl), str(tmp_path / "fin" / "o")]) == 0
    for suf in SUFFIXES:
        assert (tmp_path / "run" / f"o.{suf}.txt").read_text() == (tmp_path / "fin" / f"o.{suf}.txt").read_text(), suf


def test_gene_results_saved_when_post_processing_fails(tmp_path, monkeypatch):
    def broken(*args, **kw):
        raise RuntimeError("post-processing failed")

    monkeypatch.setattr(eqtl, "significant_snp_gene_pairs", broken)
    with pytest.raises(RuntimeError):
        run_dmqtl(output_prefix=str(tmp_path / "o"), **KW)
    assert (tmp_path / "o.gene_results.pkl").exists()


def test_run_finishes_when_one_gene_gets_an_invalid_beta_fit(tmp_path, monkeypatch):
    real_fit = stats.beta.fit
    calls = []

    def fit(*args, **kw):  # first gene: what scipy 1.15.3 returned for the 11 Ast genes
        calls.append(1)
        return (0.038, -7.5e6, 0.0, 1.0) if len(calls) == 1 else real_fit(*args, **kw)

    monkeypatch.setattr(stats.beta, "fit", fit)
    run_dmqtl(output_prefix=str(tmp_path / "o"), **KW)
    rows = [ln.split("\t") for ln in (tmp_path / "o.egenes.txt").read_text().splitlines()[1:]]
    assert rows and all(np.isfinite(float(r[9])) and np.isfinite(float(r[10])) for r in rows)
