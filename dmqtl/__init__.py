"""
dmqtl: Dirichlet-Multinomial GLM score test for APA-QTL / dmQTL mapping.

The main entry point is `run_dmqtl`, which provides an end-to-end workflow:
reading input files -> fitting the null model -> scanning cis-SNPs with the
score test -> permutation testing -> writing three result files.

For more fine-grained control, such as constructing the gene task list
manually or selecting a specific null-model fitting method, see
`dmqtl.eqtl.build_gene_tasks`, `dmqtl.pipeline.run_many_genes`, and the
`fit_null_*` functions in `dmqtl.nullmodel`.

Quick start:
    >>> from dmqtl import run_dmqtl
    >>> summary = run_dmqtl(
    ...     genotype_path="genotype.txt",
    ...     phenotype_path="pas_counts.txt",
    ...     pas_gene_map_path="pas_to_gene.txt",
    ...     covariates_path="covariates.txt",
    ...     snp_pos_path="snp_pos.txt",
    ...     gene_pos_path="gene_pos.txt",
    ...     output_prefix="out/apaqtl",
    ...     use_drimseq="auto",
    ... )

Known issues and usage limitations, including manual verification of the
DRIMSeq reference-category convention and data-dependent behavior of
cross-gene shrinkage, are documented in the repository root at
`docs/DEVELOPMENT_NOTES.md`.
"""
from .eqtl import MatchedInfo, RunSummary, build_gene_tasks, run_dmqtl
from .nullmodel import (
    NullFit,
    drimseq_available,
    fit_null_drimseq,
    fit_null_drimseq_batch,
    fit_null_python,
)
from .pipeline import GeneResult, fit_null_models_batched, run_many_genes, run_one_gene

__version__ = "1.1.0"

__all__ = [
    "__version__",
    "run_dmqtl",
    "build_gene_tasks",
    "RunSummary",
    "MatchedInfo",
    "NullFit",
    "fit_null_drimseq",
    "fit_null_drimseq_batch",
    "fit_null_python",
    "drimseq_available",
    "GeneResult",
    "run_one_gene",
    "run_many_genes",
    "fit_null_models_batched",
]
