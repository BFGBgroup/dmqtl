"""MAC threshold among individuals with reads (--min-mac; 2026-10-24, option c for the floor genes).

A cis-SNP is tested for a gene only if its minor allele occurs at least `min_mac` times among the
individuals that have reads for that gene. On real data the 1e-12 floor comes from SNPs whose only
informative carriers are one or two individuals; MAF over all individuals does not see this, because
most carriers can sit among individuals with no reads.

min_mac=0 (default) is the current behaviour. A gene left without SNPs keeps its task, so the null-fit
chunks, the null-fit cache and every gene's seed stay where they are; it is just not tested.
"""
from pathlib import Path

import numpy as np

from dmqtl.cis import mac_among_readers
from dmqtl.cli import build_parser, main
from dmqtl.eqtl import build_gene_tasks, materialize_genotype
from dmqtl.genostore import convert_genotype_text


def test_mac_counts_only_individuals_with_reads():
    y = np.array([[0, 0], [0, 0], [3, 1], [2, 2], [1, 0], [0, 5]])  # first two have no reads
    G = np.array([
        [2, 1, 0, 2, 0],
        [2, 1, 0, 2, 0],
        [0, 0, 1, 2, np.nan],
        [0, 0, 0, 2, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 2, 2],
    ], dtype=float)
    # col 0/1: carriers only among individuals without reads; col 2: one het among readers;
    # col 3: alt is the major allele among readers (7 of 8) -> minor count 1; col 4: missing not counted
    assert mac_among_readers(G, y).tolist() == [0, 0, 1, 1, 2]


N = 40
NO_READS = range(10)  # individuals 0-9 have no reads for gene A


def _write_inputs(d: Path) -> dict:
    rng = np.random.default_rng(11)
    ids = [f"I{i:02d}" for i in range(N)]

    def common():
        return rng.binomial(2, 0.4, N).astype(float)

    snps = {}
    g = np.zeros(N); g[[0, 1, 2, 3, 4]] = 1; snps["A_hidden"] = ("A", g)          # carriers only without reads
    g = np.zeros(N); g[[5, 6, 20]] = 1; snps["A_rare1"] = ("A", g)                 # MAC 1 among readers
    snps["A_common1"] = ("A", common())
    g = common(); g[3] = np.nan; snps["A_common2"] = ("A", g)
    g = np.zeros(N); g[[30, 31]] = 1; snps["B_rare2"] = ("B", g)                   # MAC 2
    g = np.full(N, 2.0); g[[12, 33]] = 1; snps["B_rare2_alt"] = ("B", g)           # alt major, MAC 2
    for k in range(3):
        snps[f"C_common{k}"] = ("C", common())

    with open(d / "geno.txt", "w") as f:
        f.write("id\t" + "\t".join(ids) + "\n")
        for vid, (_, g) in snps.items():
            f.write(vid + "\t" + "\t".join("NA" if np.isnan(x) else str(int(x)) for x in g) + "\n")
    with open(d / "snp_pos.txt", "w") as f:
        f.write("variant_id\tchr\tpos\n")
        start = {"A": 1_000_000, "B": 5_000_000, "C": 9_000_000}
        for j, (vid, (gene, _)) in enumerate(snps.items()):
            f.write(f"{vid}\tchr1\t{start[gene] + 1000 * j}\n")
    with open(d / "gene_pos.txt", "w") as f:
        f.write("gene_id\tchr\tstart\tend\n")
        for gene, s in (("A", 1_000_000), ("B", 5_000_000), ("C", 9_000_000)):
            f.write(f"{gene}\tchr1\t{s}\t{s + 1000}\n")

    pas = {"A": 2, "B": 2, "C": 3}
    with open(d / "pas.txt", "w") as f, open(d / "pas_map.txt", "w") as m:
        f.write("id\t" + "\t".join(ids) + "\n")
        m.write("pas_id\tgene_id\n")
        for gene, K in pas.items():
            for k in range(K):
                c = rng.poisson(30 + 10 * k, N)
                if gene == "A":
                    c[list(NO_READS)] = 0
                f.write(f"{gene}_P{k}\t" + "\t".join(map(str, c)) + "\n")
                m.write(f"{gene}_P{k}\t{gene}\n")
    with open(d / "cov.txt", "w") as f:
        f.write("id\t" + "\t".join(ids) + "\n")
        f.write("cov1\t" + "\t".join(f"{x:.4f}" for x in rng.normal(size=N)) + "\n")
    return dict(phenotype_path=d / "pas.txt", pas_gene_map_path=d / "pas_map.txt",
                covariates_path=d / "cov.txt", snp_pos_path=d / "snp_pos.txt", gene_pos_path=d / "gene_pos.txt")


def _by_gene(tasks):
    return {t["gene_id"]: t for t in tasks}


def test_min_mac_zero_is_current_behaviour(tmp_path):
    inp = _write_inputs(tmp_path)
    ref, _ = build_gene_tasks(tmp_path / "geno.txt", **inp)
    new, _ = build_gene_tasks(tmp_path / "geno.txt", min_mac=0, **inp)
    assert [t["gene_id"] for t in ref] == [t["gene_id"] for t in new] == ["A", "B", "C"]
    for a, b in zip(ref, new):
        assert a["variant_ids"] == b["variant_ids"] and np.array_equal(a["G"], b["G"])


def test_min_mac_drops_rare_snps_and_keeps_every_gene_task(tmp_path):
    inp = _write_inputs(tmp_path)
    ref = _by_gene(build_gene_tasks(tmp_path / "geno.txt", **inp)[0])
    tasks, info = build_gene_tasks(tmp_path / "geno.txt", min_mac=3, **inp)
    assert [t["gene_id"] for t in tasks] == ["A", "B", "C"]  # same layout -> same null-fit chunks and seeds
    assert info.skipped_genes == []
    t = _by_gene(tasks)
    assert t["A"]["variant_ids"] == ["A_common1", "A_common2"]
    cols = [ref["A"]["variant_ids"].index(v) for v in t["A"]["variant_ids"]]
    assert np.array_equal(t["A"]["G"], ref["A"]["G"][:, cols])
    assert t["B"]["variant_ids"] == [] and t["B"]["G"].shape == (N, 0)
    assert t["C"]["variant_ids"] == ref["C"]["variant_ids"] and np.array_equal(t["C"]["G"], ref["C"]["G"])
    # MAC 1 and 2 both pass at min_mac=1 / 2 respectively; the hidden one never does
    a1 = _by_gene(build_gene_tasks(tmp_path / "geno.txt", min_mac=1, **inp)[0])["A"]["variant_ids"]
    assert a1 == ["A_rare1", "A_common1", "A_common2"]


def test_store_path_matches_text_path_with_min_mac(tmp_path):
    inp = _write_inputs(tmp_path)
    prefix = convert_genotype_text(tmp_path / "geno.txt", tmp_path / "geno")
    eager, _ = build_gene_tasks(tmp_path / "geno.txt", min_mac=3, **inp)
    lazy, _ = build_gene_tasks(prefix, min_mac=3, **inp)
    for te, tl in zip(eager, lazy):
        assert "G" not in tl and te["variant_ids"] == tl["variant_ids"]
        G = materialize_genotype(tl)
        assert G.shape == te["G"].shape and np.array_equal(G, te["G"])


def _egenes(prefix):
    rows = Path(f"{prefix}.egenes.txt").read_text().splitlines()
    head = rows[0].split("\t")
    return {r.split("\t")[0]: dict(zip(head, r.split("\t"))) for r in rows[1:]}


def test_run_with_min_mac_skips_empty_genes_and_leaves_others_alone(tmp_path):
    inp = _write_inputs(tmp_path)
    common = ["--genotype", str(tmp_path / "geno.txt"), "--phenotype", str(inp["phenotype_path"]),
              "--pas-gene-map", str(inp["pas_gene_map_path"]), "--covariates", str(inp["covariates_path"]),
              "--snp-pos", str(inp["snp_pos_path"]), "--gene-pos", str(inp["gene_pos_path"]),
              "--use-drimseq", "no", "--min-perm", "200", "--max-perm", "400", "--perm-batch-size", "100",
              "--seed", "5"]
    assert main(common + ["--output-prefix", str(tmp_path / "m0" / "o")]) == 0
    assert main(common + ["--output-prefix", str(tmp_path / "m3" / "o"), "--min-mac", "3"]) == 0
    e0, e3 = _egenes(tmp_path / "m0" / "o"), _egenes(tmp_path / "m3" / "o")
    assert set(e0) == {"A", "B", "C"} and set(e3) == {"A", "C"}
    assert e3["A"]["n_cis_snp"] == "2"
    # C's SNP set did not change: same seed, same scan, same permutations (q-value depends on all genes)
    for col in ("n_cis_snp", "lead_variant_id", "obs_min_p", "n_perm_used", "n_hits", "beta_shape1",
                "beta_shape2", "beta_pval"):
        assert e0["C"][col] == e3["C"][col], col
    pairs3 = Path(tmp_path / "m3" / "o.cis_qtl_pairs.txt").read_text().splitlines()[1:]
    assert sorted(r.split("\t")[1] for r in pairs3 if r.startswith("A\t")) == ["A_common1", "A_common2"]


def test_cli_min_mac_default_is_zero():
    req = ["--genotype", "g", "--phenotype", "p", "--pas-gene-map", "m", "--covariates", "c",
           "--output-prefix", "o"]
    assert build_parser().parse_args(req).min_mac == 0
    assert build_parser().parse_args(req + ["--min-mac", "5"]).min_mac == 5
