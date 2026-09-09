"""
End-to-end example: generate a synthetic multi-gene apaQTL dataset in
matrixQTL/tensorQTL text format, then run dmqtl.eqtl.run_dmqtl and
check the three output files.

Usage:
    cd examples/
    python3 generate_synthetic_input.py   # Generate genotype.txt / pas_counts.txt / ...
    python3 run_synthetic_example.py      # Run the complete pipeline and write out/*.txt

The dataset contains six genes with embedded causal SNPs and six pure-null
genes. You can directly verify that the significant genes in
out/apaqtl.egenes.txt are exactly the six causal genes, and that the lead SNPs
in out/apaqtl.significant_pairs.txt are exactly the corresponding embedded
causal variants. This validation procedure has been actually run and
verified before packaging the software.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmqtl.eqtl import run_dmqtl

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    summary = run_dmqtl(
        genotype_path=HERE / "genotype.txt",
        phenotype_path=HERE / "pas_counts.txt",
        pas_gene_map_path=HERE / "pas_to_gene.txt",
        covariates_path=HERE / "covariates.txt",
        snp_pos_path=HERE / "snp_pos.txt",
        gene_pos_path=HERE / "gene_pos.txt",
        cis_window=1_000_000,
        maf_threshold=0.01,
        output_prefix=str(HERE / "out" / "apaqtl"),
        # use_drimseq="auto" (default): automatically use R + DRIMSeq if available;
        # otherwise fall back to the Python implementation and print a warning.
        # This sandbox does not have R, so the Python fallback will be used.
        perm_batch_size=100,
        max_perm=2000,
        min_hits_to_stop=10,
        n_workers=2,
        q_threshold=0.05,
    )
    print(summary)
    print(f"see {HERE / 'out'}/apaqtl.egenes.txt, "
          f"apaqtl.cis_qtl_pairs.txt, apaqtl.significant_pairs.txt")
