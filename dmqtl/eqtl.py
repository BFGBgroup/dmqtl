"""
End-to-end driver: runs the complete workflow from text input files
(genotype / PAS counts / PAS-gene mapping / covariates; see `io.py` for
format conventions) to three output files:

    <prefix>.egenes.txt
        Gene-level results (one row per gene).

    <prefix>.cis_qtl_pairs.txt
        Nominal p-values for all tested SNP-gene pairs
        (corresponding to the requested per-variant information).

    <prefix>.significant_pairs.txt
        Significant SNP-gene (eSNP) pairs identified using the
        permutation-based significance threshold.

See the `run_dmqtl` docstring near the end of this file or
`examples/run_synthetic_example.py` for usage examples.
"""
from __future__ import annotations

import functools
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cis import build_cis_map, center_columns, impute_genotype_mean, maf_filter
from .genostore import is_store, materialize_genotype, open_store, raw_gene_genotype
from .io import (
    match_individuals,
    read_gene_positions,
    read_matrix_text,
    read_pas_gene_map,
    read_variant_positions,
    reindex_matrix,
)
from .nullmodel import drimseq_available, fit_null_drimseq_batch, fit_null_python
from .pipeline import GeneResult, NullFitFn, fit_null_models_batched, run_many_genes
from .pvalue import significant_snp_gene_pairs, storey_qvalue


@dataclass
class RunSummary:
    n_genes_total: int
    n_genes_tested: int
    n_genes_skipped: int
    n_egenes: int
    n_significant_pairs: int
    elapsed_seconds: float


def build_gene_tasks(genotype_path, phenotype_path, pas_gene_map_path, covariates_path,
                      snp_pos_path=None, gene_pos_path=None, cis_window: int = 1_000_000,
                      maf_threshold: float = 0.01, min_pas_per_gene: int = 2,
                      celltype: str = "celltype") -> tuple[list[dict], MatchedInfo]:
    """Read all text inputs, match individual IDs, and construct one task
    per gene containing `(y, X, G, variant_ids)`.

    Default assumptions are explicitly stated here rather than hidden in
    the implementation:

    - The covariate file should not contain an intercept row supplied by
      the user. An all-ones intercept column is automatically added as the
      first column of X. If the covariate file already contains a manually
      specified intercept column, the resulting design matrix will contain
      two constant columns and therefore be collinear. Do not provide the
      intercept twice.

    - Genotypes are encoded as 0/1/2, or as dosage values in the range [0, 2].
      MAF is calculated as `min(mean/2, 1 - mean/2)`. Missing values (NA)
      are imputed using the sample mean of the corresponding variant.
      MAF is calculated before imputation, so missing values are not included
      in the MAF denominator.

    - If both `snp_pos_path` and `gene_pos_path` are provided, candidate SNPs
      for each gene are restricted to the region defined by
      TSS +/- `cis_window`. Otherwise, the candidate set for every gene
      falls back to all variants in the genotype file. This fallback should
      only be used when the number of variants is already small (for example,
      when the input is a pre-filtered cis-region dataset).

      For genome-wide genotype data, do NOT run without position files.
      Otherwise, the analysis becomes an n_gene x n_variant_total scan,
      which is statistically inappropriate because trans- and cis-testing
      do not use the same significance framework, and is also computationally
      infeasible at genome-wide scale.

    - A gene is skipped if it has fewer than `min_pas_per_gene` matched PAS
      (default: 2, because a Dirichlet-Multinomial composition requires at
      least two categories) or if it has no candidate cis-SNPs. Skipped genes
      are recorded in `MatchedInfo.skipped_genes` and do not cause the
      entire workflow to fail.

    Genotype input (2026-10):
        `genotype_path` may also be the prefix of a binary store written by
        `genostore.convert_genotype_text`. The genome-wide matrix is then
        memory-mapped instead of parsed into RAM, and tasks carry the gene's
        candidate row indices (`task["geno"]`) instead of a G matrix; the
        worker rebuilds G with `genostore.materialize_genotype` just before
        scanning the gene. Which genes are skipped, the variant lists and G
        itself are identical to the text path (tests/test_genostore.py).
    """
    store = open_store(genotype_path) if is_store(genotype_path) else None
    if store is None:
        geno = read_matrix_text(genotype_path)
        geno_sample_ids, geno_row_ids = geno.sample_ids, geno.row_ids
    else:
        geno_sample_ids, geno_row_ids = store.sample_ids, store.row_ids
    pheno = read_matrix_text(phenotype_path)
    covar = read_matrix_text(covariates_path)
    gene_to_pas = read_pas_gene_map(pas_gene_map_path)

    matched = match_individuals(geno_sample_ids, pheno.sample_ids, covar.sample_ids)
    if store is None:
        geno = reindex_matrix(geno, matched.sample_ids)
    else:
        sample_pos = {s: i for i, s in enumerate(store.sample_ids)}
        col_idx = np.array([sample_pos[s] for s in matched.sample_ids], dtype=np.intp)
    pheno = reindex_matrix(pheno, matched.sample_ids)
    covar = reindex_matrix(covar, matched.sample_ids)
    n = len(matched.sample_ids)

    # Null-model design matrix:
    # intercept (all ones) + covariates
    # (transposed to individual x covariate).
    X = np.column_stack([np.ones(n), covar.values.T])

    pheno_row_index = {pid: i for i, pid in enumerate(pheno.row_ids)}

    cis_map = None
    if snp_pos_path is not None and gene_pos_path is not None:
        gene_pos = read_gene_positions(gene_pos_path)
        snp_pos = read_variant_positions(snp_pos_path)
        cis_map = build_cis_map(gene_pos, snp_pos, window=cis_window)
    else:
        print("[dmqtl] WARNING: SNP/gene position files were not provided. "
            "The candidate SNP set for each gene will fall back to all "
            "variants in the genotype file. This should only be used when "
            "the number of variants is already small (e.g., the input is "
            "a pre-filtered cis-region dataset).",
              file=sys.stderr)

    geno_row_index = {vid: i for i, vid in enumerate(geno_row_ids)}

    tasks = []
    skipped = []
    for gene_id, pas_ids in gene_to_pas.items():
        pas_rows = [pheno_row_index[p] for p in pas_ids if p in pheno_row_index]
        missing_pas = [p for p in pas_ids if p not in pheno_row_index]
        if missing_pas:
            print(f"[dmqtl] WARNING: PAS {missing_pas} for gene {gene_id} "
                  f"were not found in the phenotype matrix and will be ignored.", file=sys.stderr)
        if len(pas_rows) < min_pas_per_gene:
            skipped.append((gene_id, f"available PAS count {len(pas_rows)} < {min_pas_per_gene}"))
            continue
        y = pheno.values[pas_rows, :].T  # (n, K)

        if cis_map is not None:
            candidate_vids = [v for v in cis_map.get(gene_id, []) if v in geno_row_index]
        else:
            candidate_vids = geno_row_ids
        if len(candidate_vids) == 0:
            skipped.append((gene_id, "no candidate cis-SNPs"))
            continue
        rows = [geno_row_index[v] for v in candidate_vids]
        if store is None:
            G_raw = geno.values[rows, :].T  # (n, n_snp)
        else:
            rows = np.array(rows, dtype=np.int32 if len(geno_row_ids) < 2**31 else np.int64)
            G_raw = raw_gene_genotype(store, rows, col_idx)

        G_imputed = impute_genotype_mean(G_raw)
        kept_vids, G_kept = maf_filter(candidate_vids, G_imputed, maf_threshold=maf_threshold)
        if len(kept_vids) == 0:
            skipped.append((gene_id, f"no candidate SNPs remain after MAF >= {maf_threshold} filtering"))
            continue
        if store is not None:
            # keep only what is needed to rebuild G later; the matrices are dropped here
            del G_raw, G_imputed, G_kept
            tasks.append(dict(gene_id=gene_id, celltype=celltype, y=y, X=X, variant_ids=kept_vids,
                              geno=dict(prefix=store.prefix, rows=rows, col_idx=col_idx,
                                        maf_threshold=maf_threshold)))
            continue
        G_centered = center_columns(G_kept)

        tasks.append(dict(gene_id=gene_id, celltype=celltype, y=y, X=X,
                           G=G_centered, variant_ids=kept_vids))

    return tasks, MatchedInfo(matched_sample_ids=matched.sample_ids,
                               dropped=matched.dropped, skipped_genes=skipped,
                               n_genes_total=len(gene_to_pas),
                               covariate_names=["Intercept"] + covar.row_ids)


@dataclass
class MatchedInfo:
    matched_sample_ids: list[str]
    dropped: dict
    skipped_genes: list[tuple[str, str]]
    n_genes_total: int
    covariate_names: list[str]


def write_egenes(path, gene_results: list[GeneResult], qvals: np.ndarray):
    with open(path, "w") as f:
        f.write("gene_id\tcelltype\tn_cis_snp\tlead_variant_id\tobs_min_p\t"
                "n_perm_used\tn_hits\tbeta_shape1\tbeta_shape2\tbeta_pval\tqvalue\n")
        for r, q in zip(gene_results, qvals):
            f.write(f"{r.gene_id}\t{r.celltype}\t{r.n_snp}\t{r.lead_variant_id}\t"
                     f"{r.obs_min_p:.6g}\t{r.n_perm_used}\t{r.n_hits}\t"
                     f"{r.beta_a:.6g}\t{r.beta_b:.6g}\t{r.beta_p:.6g}\t{q:.6g}\n")


def write_all_pairs(path, gene_results: list[GeneResult]):
    with open(path, "w") as f:
        f.write("gene_id\tvariant_id\tnominal_pvalue\n")
        for r in gene_results:
            for vid, p in zip(r.variant_ids, r.nominal_pvalues):
                f.write(f"{r.gene_id}\t{vid}\t{p:.6g}\n")


def write_significant_pairs(path, pairs):
    with open(path, "w") as f:
        f.write("gene_id\tvariant_id\tnominal_pvalue\tgene_nominal_threshold\n")
        for pr in pairs:
            f.write(f"{pr.gene_id}\t{pr.variant_id}\t{pr.nominal_pvalue:.6g}\t"
                     f"{pr.gene_nominal_threshold:.6g}\n")


def run_dmqtl(genotype_path, phenotype_path, pas_gene_map_path, covariates_path,
              output_prefix, snp_pos_path=None, gene_pos_path=None,
              cis_window: int = 1_000_000, maf_threshold: float = 0.01,
              null_fit_fn: NullFitFn | None = None,
              use_drimseq: str = "auto", rscript_bin: str = "Rscript",
              null_fit_chunk_size: int | None = None,
              null_fit_n_cores: int = 1,
              null_fit_cache_dir: str | None = None,
              null_fit_timeout: int = 3600,
              null_fit_prec_subset: float | None = None,
              null_fit_prec_prior_df: float | None = None,
              null_fit_prec_moderation: str | None = None,
              perm_batch_size: int = 200, min_perm: int = 1000,
              max_perm: int = 10000,
              min_hits_to_stop: int = 15, q_threshold: float = 0.05,
              n_workers: int = 1, seed: int = 0, celltype: str = "celltype") -> RunSummary:
    """Run the complete dmqtl workflow.

    Workflow:
        read inputs
        -> assemble gene-level tasks
        -> null-model fitting
        -> cis-SNP score-test scanning
        -> permutation testing (independently for each gene)
        -> Storey q-value calculation
        -> write three output files

    Null-model fitting path (`use_drimseq`, default: "auto"):

        "auto" (default):
            Automatically detect whether Rscript + DRIMSeq are available.
            If available, use DRIMSeq, which is the recommended production
            path. The batched architecture makes cross-gene precision
            shrinkage mathematically possible, but it is NOT enabled by
            default. DRIMSeq's `prec_moderation` defaults to `"none"`;
            see the `null_fit_prec_moderation` description below.
            To actually enable shrinkage, it must be explicitly configured.

            Even without shrinkage, DRIMSeq is generally more robust than
            pure Python MLE for genes with small sample sizes or low counts
            (see the empirical evaluation documented in `nullmodel.py`).
            If R is unavailable, the workflow falls back to the pure Python
            MLE implementation and prints a warning.

        "yes":
            Force the use of DRIMSeq. If Rscript or DRIMSeq cannot be detected,
            raise an error rather than silently falling back to Python. This
            prevents users from believing that DRIMSeq was used when it was
            not. The error message also suggests checking `rscript_bin` or
            using `"auto"`/`"no"` instead.

        "no":
            Force the Python fallback (`fit_null_python`) without probing for R.

        All of the above can be completely overridden by `null_fit_fn=`.
        If provided, the user-supplied function is used directly and
        `use_drimseq` is ignored. This is useful when using a custom
        `rscript_bin`/timeout configuration or wrapping the null-model fitting
        step with custom caching or retry logic.

    Important:
        When DRIMSeq is selected, the reference-category convention used by
        `fit_null_drimseq_batch` / `r_null_fit_batch.R` still needs to be
        manually checked against the Python implementation on a machine with
        a working R environment. The current sandbox does not provide R, so
        this consistency check cannot be performed here.

        The automatic selection only avoids manually switching between
        DRIMSeq and Python depending on the environment. It does not mean
        that the reference-category convention has already been validated.

    === 2026-08: DRIMSeq now uses batched null-model fitting ===

        The previous implementation started a separate Rscript process for
        each gene and supplied only that gene's data to DRIMSeq. Therefore,
        `dmPrecision` had no information from other genes, making cross-gene
        precision shrinkage architecturally impossible.

        The current implementation first uses
        `pipeline.fit_null_models_batched` to package a batch of genes and
        pass them to `nullmodel.fit_null_drimseq_batch`. Internally, a single
        `dmPrecision`/`dmFit` call processes the entire batch.

        The resulting `NullFit` objects are then passed to
        `run_many_genes` for score-test scanning and permutation testing.
        This stage no longer requires R and can safely use `n_workers`
        for parallelization.

        However, "architecturally possible" does not mean that shrinkage
        occurs by default. DRIMSeq's `dmPrecision` uses
        `prec_moderation="none"` by default. Therefore, even when a whole
        batch of genes is supplied, the precision estimates are not
        automatically shrunk toward a common trend.

        To enable actual cross-gene shrinkage, explicitly configure
        `null_fit_prec_moderation` as described below.

    null_fit_chunk_size:
        Number of genes supplied to R in each `dmPrecision` call; see the
        detailed documentation of `pipeline.fit_null_models_batched`.

        `None` (default) means that all genes are supplied in a single call.
        This is generally the least memory- and time-friendly option.

        For real datasets (~20,000 genes), it is recommended to first
        benchmark a smaller number of genes per chunk, such as 50-100 genes,
        to measure runtime and memory usage, and then choose an appropriate
        chunk size, potentially in the range of 1,000-5,000 genes.

        This parameter controls how many genes are used to estimate the
        common precision trend. It does NOT determine whether shrinkage
        occurs; that is controlled by `null_fit_prec_moderation`.

    null_fit_n_cores:
        Number of parallel cores passed to BiocParallel on the R side for
        `dmPrecision`/`dmFit`. This controls parallelization across genes
        within a batch.

        It is independent of `n_workers`, which controls parallelization
        across genes during the score-test/permutation stage. These two
        parameters should not be treated as the same setting.

    null_fit_cache_dir:
        If provided, null-model fitting results are cached by chunk in this
        directory. Completed chunks are skipped when the analysis is resumed.

        For real datasets, long-running jobs may be interrupted by scheduling
        limits, preemption, or timeouts. Therefore, using a cache directory
        is strongly recommended before starting a full-scale analysis.

    null_fit_timeout:
        Timeout in seconds for a single `fit_null_drimseq_batch` call,
        corresponding to one chunk. The default is 3600 seconds (1 hour).

        Larger chunks may require longer timeouts, so this parameter should
        be benchmarked together with `null_fit_chunk_size`.

    null_fit_prec_subset:
        Passed to the R-side `dmPrecision` function as `prec_subset`.
        This parameter affects how accurately the common precision trend
        itself is fitted, but does NOT determine whether shrinkage occurs.

        This distinction was confirmed empirically in the September 2026
        debugging process; see the "third patch" entry in CHANGES.md.

    null_fit_prec_moderation / null_fit_prec_prior_df:
        These two parameters determine whether cross-gene precision shrinkage
        is actually enabled, and both must be configured appropriately.

        This behavior was established through three rounds of empirical
        testing. Two alternative explanations were ruled out:
        first, `null_fit_prec_subset` was suspected but was shown not to
        control shrinkage; second, setting `null_fit_prec_prior_df` alone was
        tested and produced results indistinguishable from the unconfigured
        case.

        The final conclusion was confirmed by inspecting the complete
        parameter list of DRIMSeq's `dmPrecision`.

        `null_fit_prec_moderation` accepts:

            None
                Use DRIMSeq's default `"none"`; no shrinkage.

            `"common"`
                Shrink toward a single common precision.

            `"trended"`
                Shrink toward a precision-versus-mean trend curve.

        `null_fit_prec_prior_df` is only used when
        `null_fit_prec_moderation` is not `None`/`"none"` and controls the
        strength of the shrinkage.

        If actual cross-gene shrinkage is desired, both parameters must be
        configured. Setting only one of them does not enable the intended
        shrinkage behavior.

        DRIMSeq does not provide an automatic recommendation for which
        moderation method to use or what value of `prec_prior_df` is
        appropriate. These choices should be made based on the documentation
        of `DRIMSeq::dmPrecision`, particularly the descriptions of
        `prec_moderation`, `prec_prior_df`, and `prec_span`.

        After configuration, use
        `tests/test_r_bridge_smoke.py`,
        specifically `test_drimseq_batch_shrinks_precision_across_genes`,
        to inspect the three reported precision values and confirm that
        shrinkage is actually taking effect.

        Do not assume that the behavior described here necessarily applies
        identically to every DRIMSeq version or dataset without verification.

    Example:
        from dmqtl.eqtl import run_dmqtl

        summary = run_dmqtl(
            genotype_path="genotype.txt",
            phenotype_path="pas_counts.txt",
            pas_gene_map_path="pas_to_gene.txt",
            covariates_path="covariates.txt",
            snp_pos_path="snp_pos.txt",       # optional; enables cis filtering
            gene_pos_path="gene_pos.txt",      # optional
            output_prefix="out/apaqtl",
            n_workers=8,
            # use_drimseq="auto" is the default and does not need to be specified
        )
        print(summary)

    Output files:
        out/apaqtl.egenes.txt
        out/apaqtl.cis_qtl_pairs.txt
        out/apaqtl.significant_pairs.txt
    """
    t0 = time.time()
    tasks, info = build_gene_tasks(
        genotype_path, phenotype_path, pas_gene_map_path, covariates_path,
        snp_pos_path=snp_pos_path, gene_pos_path=gene_pos_path,
        cis_window=cis_window, maf_threshold=maf_threshold, celltype=celltype,
    )
    for pool_name, extra in info.dropped.items():
        print(f"[dmqtl] WARNING: {len(extra)} individual IDs in {pool_name} were not found in the intersection and were dropped "
              f"(example: {extra[:5]})", file=sys.stderr)
    for gene_id, reason in info.skipped_genes:
        print(f"[dmqtl] Skipping gene {gene_id}: {reason}", file=sys.stderr)

    print(f"[dmqtl] {len(info.matched_sample_ids)} individuals; "
          f"{len(tasks)}/{info.n_genes_total} genes entered the analysis "
          f"({len(info.skipped_genes)} skipped)", file=sys.stderr)

    use_batched_drimseq = False
    if null_fit_fn is None:
        if use_drimseq not in ("auto", "yes", "no"):
            raise ValueError(f"use_drimseq must be 'auto', 'yes', or 'no', received: {use_drimseq!r}")
        has_drimseq = drimseq_available(rscript_bin) if use_drimseq != "no" else False
        if use_drimseq == "yes" and not has_drimseq:
            raise RuntimeError(
                f"use_drimseq='yes' but Rscript ('{rscript_bin}') + DRIMSeq "
                "could not be detected. Please verify that DRIMSeq is installed "
                "(BiocManager::install(\"DRIMSeq\")), or set rscript_bin to "
                "the correct Rscript path. Alternatively, use "
                "use_drimseq='auto'/'no'。"
            )
        if use_drimseq != "no" and has_drimseq:
            print("[dmqtl] Rscript + DRIMSeq detected. Null-model fitting will use batched DRIMSeq "
                  f"(chunk_size={null_fit_chunk_size!r}). Cross-gene precision "
                  "shrinkage, if enabled, occurs at this stage; see pipeline.fit_null_models_batched. "
                  "WARNING: The reference-category convention has not been automatically validated against the Python implementation. "
                  "Please perform the manual check described in the nullmodel.fit_null_drimseq_batch docstring.",
                  file=sys.stderr)
            use_batched_drimseq = True
        else:
            if use_drimseq == "auto":
                print("[dmqtl] WARNING: Rscript/DRIMSeq could not be detected. "
                      "Falling back to pure Python MLE (fit_null_python). "
                      "This is a fallback mode: for genes with small sample "
                      "sizes or low counts, the precision parameter phi is "
                      "estimated without cross-gene shrinkage and may therefore "
                      "be substantially less stable. For production analyses, "
                      "installing R + DRIMSeq and using use_drimseq='auto' "
                      "(default) or 'yes' is strongly recommended.", file=sys.stderr)
            null_fit_fn = fit_null_python

    if use_batched_drimseq:
        batch_fn = functools.partial(
            fit_null_drimseq_batch, covariate_names=info.covariate_names,
            rscript_bin=rscript_bin, timeout=null_fit_timeout,
            n_cores=null_fit_n_cores, prec_subset=null_fit_prec_subset,
            prec_prior_df=null_fit_prec_prior_df,
            prec_moderation=null_fit_prec_moderation)
        t_null = time.time()
        null_fits = fit_null_models_batched(
            tasks, batch_fn, chunk_size=null_fit_chunk_size,
            cache_dir=null_fit_cache_dir)
        n_not_converged = sum(
            1 for t in tasks
            if t["gene_id"] not in null_fits or not null_fits[t["gene_id"]].converged
        )
        print(f"[dmqtl] Batched null-model fitting completed in {time.time() - t_null:.1f}s; "
              f"{n_not_converged}/{len(tasks)} genes failed to converge or were missing and will be skipped.",
              file=sys.stderr)
        gene_results = run_many_genes(
            tasks, null_fits=null_fits, n_workers=n_workers, seed=seed,
            perm_batch_size=perm_batch_size, min_perm=min_perm, max_perm=max_perm,
            min_hits_to_stop=min_hits_to_stop,
        )
    else:
        gene_results = run_many_genes(
            tasks, null_fit_fn=null_fit_fn, n_workers=n_workers, seed=seed,
            perm_batch_size=perm_batch_size, min_perm=min_perm, max_perm=max_perm,
            min_hits_to_stop=min_hits_to_stop,
        )

    if gene_results:
        beta_p = np.array([r.beta_p for r in gene_results])
        qvals = storey_qvalue(beta_p)
    else:
        qvals = np.array([])

    egene_ids, qvals_check, pairs = significant_snp_gene_pairs(
        gene_results, q_threshold=q_threshold)
    assert np.array_equal(qvals, qvals_check) or len(qvals) == 0

    out_prefix = Path(output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_egenes(f"{output_prefix}.egenes.txt", gene_results, qvals)
    write_all_pairs(f"{output_prefix}.cis_qtl_pairs.txt", gene_results)
    write_significant_pairs(f"{output_prefix}.significant_pairs.txt", pairs)

    elapsed = time.time() - t0
    summary = RunSummary(
        n_genes_total=info.n_genes_total, n_genes_tested=len(gene_results),
        n_genes_skipped=len(info.skipped_genes) + (len(tasks) - len(gene_results)),
        n_egenes=len(egene_ids), n_significant_pairs=len(pairs),
        elapsed_seconds=elapsed,
    )
    print(f"[dmqtl] Completed: {summary}", file=sys.stderr)
    return summary
