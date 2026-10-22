"""
Command-line entry point: ``dmqtl-run`` (available directly in the shell
after package installation; see ``[project.scripts]`` in pyproject.toml).

This module is only a thin argument-parsing wrapper around
`dmqtl.eqtl.run_dmqtl`. All core logic is implemented in `eqtl.py`,
`pipeline.py`, and `nullmodel.py`; no statistical logic is duplicated here.

BLAS thread control (see the "thread oversubscription" entry in CHANGES.md):
when `n_workers > 1`, BLAS threads must be restricted to avoid competition
between process-level multiprocessing and BLAS-level multithreading.
In practice, thread oversubscription can cause performance differences of
one or even two orders of magnitude.

The BLAS-related environment variables are therefore configured as soon as
this module is imported. This is also why `numpy` and other heavy imports are
deferred until inside the function body rather than placed at module scope:
the environment variables must be set before the BLAS library is loaded for
the first time. If `numpy` or `scipy` has already been imported by another
module before these `os.environ` statements are executed, setting these
variables is too late and will have no effect.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dmqtl-run",
        description="Dirichlet-Multinomial GLM score test for composition/APA-QTL mapping.",
    )
    req = p.add_argument_group("Required inputs")
    req.add_argument("--genotype", required=True, help="Path to the genotype matrix text file")
    req.add_argument("--phenotype", required=True, help="Path to the PAS/feature count matrix text file")
    req.add_argument("--pas-gene-map", required=True, help="Two-column file containing: pas_id, gene_id")
    req.add_argument("--covariates", required=True, help="Path to the null-model covariate file (without an intercept column)")
    req.add_argument("--output-prefix", required=True,
                      help="Output file prefix. Generates <prefix>.egenes.txt /  "
                           ".cis_qtl_pairs.txt / .significant_pairs.txt")

    opt = p.add_argument_group("Optional inputs")
    opt.add_argument("--snp-pos", default=None, help="Variant position file (if omitted, no cis-window filtering is performed)")
    opt.add_argument("--gene-pos", default=None, help="Gene position file (used together with --snp-pos)")
    opt.add_argument("--cis-window", type=int, default=1_000_000, help="Distance from the TSS in bp (default: 1e6)")
    opt.add_argument("--maf-threshold", type=float, default=0.01)
    opt.add_argument("--celltype", default="celltype", help="Group/cell-type label written to the output files")

    drm = p.add_argument_group("DRIMSeq / null model")
    drm.add_argument("--use-drimseq", choices=["auto", "yes", "no"], default="auto")
    drm.add_argument("--rscript-bin", default="Rscript")
    drm.add_argument("--null-fit-chunk-size", type=int, default=None,
                      help="Number of genes passed to each dmPrecision call; if omitted, all genes are processed in a single call")
    drm.add_argument("--null-fit-n-cores", type=int, default=1,
                      help="Number of parallel cores used by BiocParallel on the R side, independent of --n-workers")
    drm.add_argument("--null-fit-cache-dir", default=None, help="Cache directory for checkpointing and resuming null-model fitting")
    drm.add_argument("--null-fit-timeout", type=int, default=3600)
    drm.add_argument("--null-fit-prec-subset", type=float, default=None)
    drm.add_argument("--null-fit-prec-prior-df", type=float, default=None)
    drm.add_argument("--null-fit-prec-moderation", choices=["none", "common", "trended"], default=None)

    perm = p.add_argument_group("Permutation / significance")
    perm.add_argument("--perm-batch-size", type=int, default=200)
    perm.add_argument("--min-perm", type=int, default=1000)
    perm.add_argument("--max-perm", type=int, default=10000)
    perm.add_argument("--min-hits-to-stop", type=int, default=15)
    perm.add_argument("--q-threshold", type=float, default=0.05)

    run = p.add_argument_group("Run")
    run.add_argument("--n-workers", type=int, default=1, help="Number of parallel processes for the score-test + permutation stage")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--perm-engine", choices=["fast", "loop"], default="fast",
                     help="Permutation implementation; 'loop' is the v1.0.0 reference path (same results, slower)")
    drm.add_argument("--null-fit-parallel-chunks", type=int, default=1,
                     help="Null-fit chunks run at the same time (each is one Rscript using --null-fit-n-cores)")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Import here rather than at module scope so that the BLAS thread
    # environment variables above are set before numpy is loaded.
    from .eqtl import run_dmqtl

    summary = run_dmqtl(
        genotype_path=args.genotype,
        phenotype_path=args.phenotype,
        pas_gene_map_path=args.pas_gene_map,
        covariates_path=args.covariates,
        output_prefix=args.output_prefix,
        snp_pos_path=args.snp_pos,
        gene_pos_path=args.gene_pos,
        cis_window=args.cis_window,
        maf_threshold=args.maf_threshold,
        use_drimseq=args.use_drimseq,
        rscript_bin=args.rscript_bin,
        null_fit_chunk_size=args.null_fit_chunk_size,
        null_fit_n_cores=args.null_fit_n_cores,
        null_fit_cache_dir=args.null_fit_cache_dir,
        null_fit_timeout=args.null_fit_timeout,
        null_fit_prec_subset=args.null_fit_prec_subset,
        null_fit_prec_prior_df=args.null_fit_prec_prior_df,
        null_fit_prec_moderation=args.null_fit_prec_moderation,
        perm_batch_size=args.perm_batch_size,
        min_perm=args.min_perm,
        max_perm=args.max_perm,
        min_hits_to_stop=args.min_hits_to_stop,
        q_threshold=args.q_threshold,
        n_workers=args.n_workers,
        seed=args.seed,
        celltype=args.celltype,
        perm_engine=args.perm_engine,
        null_fit_parallel_chunks=args.null_fit_parallel_chunks,
    )
    print(summary, file=sys.stderr)
    return 0


def convert_main(argv: list[str] | None = None) -> int:
    """``dmqtl-convert-genotype``: parse a genotype text matrix once into a binary store.

    Pass the printed prefix to ``dmqtl-run --genotype``.
    """
    p = argparse.ArgumentParser(prog="dmqtl-convert-genotype",
                                description="Convert a genotype text matrix into a memory-mapped binary store.")
    p.add_argument("genotype_txt")
    p.add_argument("out_prefix")
    p.add_argument("--chunk-rows", type=int, default=20000)
    args = p.parse_args(argv)
    from .genostore import convert_genotype_text
    print(convert_genotype_text(args.genotype_txt, args.out_prefix, chunk_rows=args.chunk_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
