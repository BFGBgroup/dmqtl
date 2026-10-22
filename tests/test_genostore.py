"""Equivalence tests for the binary genotype store and lazy per-gene genotypes (2026-10 optimization).

Reading the genotype text once into a memory-mapped binary file, and building
each gene's G only when it is scanned, must give bit-identical genotype
matrices, the same gene/variant lists and the same output files as the
in-memory text path.
"""
from pathlib import Path

import numpy as np

from dmqtl.eqtl import build_gene_tasks, materialize_genotype, run_dmqtl
from dmqtl.genostore import convert_genotype_text, is_store, open_store
from dmqtl.io import read_matrix_text

EX = Path(__file__).resolve().parents[1] / "examples"
INPUTS = dict(phenotype_path=EX / "pas_counts.txt", pas_gene_map_path=EX / "pas_to_gene.txt",
              covariates_path=EX / "covariates.txt", snp_pos_path=EX / "snp_pos.txt",
              gene_pos_path=EX / "gene_pos.txt")


def _messy_text(path):
    rng = np.random.default_rng(3)
    rows = ["id\tA\tB\tC\tD\tE"]
    for i in range(60):
        vals = []
        for _ in range(5):
            r = rng.random()
            if r < 0.1:
                vals.append(rng.choice(["NA", "nan", "NaN", ""]))
            elif r < 0.4:
                vals.append(f"{rng.uniform(0, 2):.3f}")
            elif r < 0.5:
                vals.append(f" {rng.integers(0, 3)} ")
            else:
                vals.append(str(rng.integers(0, 3)))
        rows.append(f"v{i}\t" + "\t".join(vals))
        if i == 30:
            rows.append("   ")  # blank line, skipped by read_matrix_text
    path.write_text("\n".join(rows) + "\n")


def test_store_values_match_text_reader(tmp_path):
    txt = tmp_path / "g.txt"
    _messy_text(txt)
    prefix = convert_genotype_text(txt, tmp_path / "g", chunk_rows=7)
    assert is_store(prefix)
    ref = read_matrix_text(txt)
    st = open_store(prefix)
    assert st.row_ids == ref.row_ids and st.sample_ids == ref.sample_ids
    a, b = np.asarray(st.values), ref.values
    assert a.dtype == b.dtype and a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.array_equal(np.nan_to_num(a), np.nan_to_num(b))


def test_lazy_tasks_match_eager(tmp_path):
    prefix = convert_genotype_text(EX / "genotype.txt", tmp_path / "geno")
    eager, info_e = build_gene_tasks(EX / "genotype.txt", **INPUTS)
    lazy, info_l = build_gene_tasks(prefix, **INPUTS)
    assert [t["gene_id"] for t in eager] == [t["gene_id"] for t in lazy]
    assert info_e.skipped_genes == info_l.skipped_genes
    assert info_e.matched_sample_ids == info_l.matched_sample_ids
    for te, tl in zip(eager, lazy):
        assert "G" not in tl
        assert te["variant_ids"] == tl["variant_ids"]
        assert np.array_equal(te["y"], tl["y"]) and np.array_equal(te["X"], tl["X"])
        G = materialize_genotype(tl)
        assert G.shape == te["G"].shape and np.array_equal(G, te["G"])


def test_run_dmqtl_store_matches_text(tmp_path):
    prefix = convert_genotype_text(EX / "genotype.txt", tmp_path / "geno")
    kw = dict(INPUTS, use_drimseq="no", perm_batch_size=100, min_perm=200, max_perm=400,
              min_hits_to_stop=10, seed=3)
    run_dmqtl(genotype_path=EX / "genotype.txt", output_prefix=str(tmp_path / "a" / "o"), n_workers=1, **kw)
    run_dmqtl(genotype_path=prefix, output_prefix=str(tmp_path / "b" / "o"), n_workers=2, **kw)
    for suf in ("egenes", "cis_qtl_pairs", "significant_pairs"):
        a = (tmp_path / "a" / f"o.{suf}.txt").read_text()
        b = (tmp_path / "b" / f"o.{suf}.txt").read_text()
        assert a == b, suf
