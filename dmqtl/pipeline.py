"""
End-to-end processing for a single gene x cell type unit, and parallel
orchestration across genes.

=== 2026-08 architecture update: null fitting and score testing are now
separated into two independent stages ===

In earlier versions, `null_fit_fn(y, X)` was called directly inside
`run_one_gene`. When `n_workers > 1`, each worker process independently
launched an Rscript subprocess running DRIMSeq and supplied only the single
gene assigned to that worker. This caused two problems:

1. **DRIMSeq's cross-gene precision shrinkage did not actually occur**:
   each call contained only one gene, so `dmPrecision` had no other genes
   from which to borrow information. The effective behavior therefore
   degraded to a per-gene estimator, which was inconsistent with the
   core motivation for using DRIMSeq.

2. At realistic data scales (tens of thousands of genes), launching one
   R subprocess per gene introduced an unacceptable cumulative overhead,
   because every subprocess had to initialize R and load the Bioconductor
   packages.

The workflow is now divided into two stages:

    Stage 1 (`fit_null_models_batched`, see below):
        Multiple genes are packaged together and passed to DRIMSeq through
        `nullmodel.fit_null_drimseq_batch`, allowing cross-gene precision
        shrinkage to occur within the same `dmPrecision`/`dmFit` call.
        Chunking and on-disk caching are supported for large-scale data.
        This stage runs only in the main process and does not consume the
        `n_workers` used by `run_many_genes`.

    Stage 2 (`run_one_gene` / `run_many_genes`):
        The precomputed `NullFit` objects from Stage 1 are used to perform
        score-test scanning and permutation testing independently for each
        gene. This stage remains embarrassingly parallel and uses
        `n_workers` for multiprocessing. The worker processes now perform
        pure NumPy calculations and no longer launch R subprocesses.

The processing workflow for a single gene x cell type remains unchanged:

    1. Build a `GeneNullContext` from the null fit computed in the batch
       stage, including precomputed R, W, and Ibb_inv matrices, which are
       calculated only once.
    2. Observed scan: calculate score-test p-values for all cis-SNPs of
       the gene in batch and take the minimum p-value.
    3. Permutation testing: keep the genotype matrix fixed and randomly
       permute individual indices (equivalent to breaking the pairing
       between phenotypes/covariates and genotypes), calculate the minimum
       p-value for each permutation, and use adaptive early stopping.
    4. Use a Beta approximation to obtain the final gene-level p-value.

For even larger analyses, `run_one_gene` can be wrapped in a command-line
script and distributed using a SLURM job array, with each task processing
one gene subset and writing its results to disk before merging them.
The batched null-model fitting in Stage 1 should preferably be run as a
separate prerequisite job, with its results passed to the subsequent
job-array stage through `cache_dir`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .nullmodel import NullFit
from .pvalue import adaptive_permutation, beta_approx_pvalue
from .scoretest import GeneNullContext, PermScanner


@dataclass
class GeneResult:
    gene_id: str
    celltype: str
    n_snp: int
    lead_snp_index: int
    obs_min_p: float
    n_perm_used: int
    n_hits: int
    beta_p: float
    beta_a: float
    beta_b: float
    # These two fields support output of eSNP / significant SNP-gene pair
    # lists. Earlier versions retained only gene-level summaries and did not
    # preserve nominal p-values for individual SNPs. The permutation-threshold
    # approach requires these values to generate SNP-gene pair results.
    # `repr=False` prevents the complete SNP list and p-value array from
    # flooding the representation when a GeneResult object is printed.
    lead_variant_id: str | None = field(default=None)
    variant_ids: list[str] | None = field(default=None, repr=False)
    nominal_pvalues: np.ndarray | None = field(default=None, repr=False)


NullFitFn = Callable[[np.ndarray, np.ndarray], NullFit]
# Batch null-fit function signature:
# {gene_id: y} x X -> {gene_id: NullFit}.
# The current implementation is `nullmodel.fit_null_drimseq_batch`;
# see `fit_null_models_batched` below.
BatchNullFitFn = Callable[[dict, np.ndarray], dict]


def fit_null_models_batched(tasks: list[dict], batch_null_fit_fn: BatchNullFitFn,
                             chunk_size: int | None = None,
                             cache_dir: str | None = None) -> dict[str, NullFit]:
    """Fit null models for multiple genes as an independent step before
    score testing, rather than calling `null_fit_fn(y, X)` separately inside
    each `run_one_gene` worker.

    This architecture is required for DRIMSeq cross-gene precision shrinkage
    to actually take effect. The shrinkage itself is performed internally by
    `batch_null_fit_fn` through a single `dmPrecision` call that receives all
    genes in the current chunk. This function is responsible only for
    chunking the workload and handling large-scale execution.

    tasks: Same format as accepted by `run_many_genes`
        (dict(gene_id=, y=, X=, ...)).
        **All tasks must share the same X**. Because the null model does not
        contain SNPs, the covariate matrix should normally be identical across
        genes. If inconsistent X matrices are detected, an error is raised
        immediately rather than silently using the X matrix from the first
        gene.

    chunk_size: Number of genes supplied to each `batch_null_fit_fn` call.

        - `None` (default): all genes are passed to a single dmPrecision call.
          This is the most direct DRIMSeq usage because the dispersion trend
          is estimated jointly across the full set of genes. It can provide
          the strongest borrowing of information, but requires a potentially
          large R process for realistic data (for example, 500 individuals
          x 20,000 genes). Memory and runtime should therefore be benchmarked
          on the target machine using a subset of the data.

          A single large chunk also has a practical disadvantage: if the
          process fails because of a timeout, out-of-memory condition, or an
          internal DRIMSeq error triggered by one gene, all progress from that
          chunk is lost and there is no intermediate result to reuse.

        - An integer value (for example, 2000): split the genes into multiple
          chunks and invoke R once for each chunk. This is still substantially
          better than the previous one-R-call-per-gene architecture: thousands
          of genes are jointly used to estimate the dispersion trend, and the
          resulting shrinkage is generally expected to be sufficiently stable
          in practice. This approach trades some global information sharing
          for lower memory requirements, failure isolation, and checkpoint/
          resume support through `cache_dir`.

          For real datasets (~20,000 genes), a reasonable starting point is
          to benchmark a chunk of 50-100 genes for runtime and memory usage
          and then scale upward. The exact value should be determined on the
          target machine rather than assumed universally.

    cache_dir: If provided, write the results of each completed chunk
        (`phi`, `eta`, and `converged`) to a JSON file. The corresponding
        cache file is reused on subsequent runs, allowing completed chunks
        to be skipped without rerunning R.

        This provides checkpoint/resume functionality for realistic analyses
        where a large job may run for several hours and may be interrupted
        by scheduler limits, preemption, or other failures.

        If `cache_dir` is not provided, no caching is performed.

        **Important:** cache validity is determined only by the chunk index
        and the task layout. The cached file does not verify whether the
        underlying `y` or `X` data have changed. If the input data are
        changed while reusing the same cache directory, the cache directory
        must be cleared manually. Automatic cache invalidation is not
        implemented here.

    Returns:
        A `{gene_id: NullFit}` dictionary containing genes for which the X
        consistency check passed and a result was successfully obtained from
        a chunk.

        If an entire chunk fails, an exception is raised rather than silently
        skipping the affected genes. A null-model fitting failure requires
        manual intervention and is fundamentally different from a structurally
        untestable gene that can safely be skipped by `run_one_gene`.
    """
    if not tasks:
        return {}

    X_ref = tasks[0]["X"]
    for t in tasks:
        if t["X"].shape != X_ref.shape or not np.allclose(t["X"], X_ref):
            raise ValueError(
                f"X for gene {t['gene_id']!r} differs from other genes — "
                "fit_null_models_batched requires all genes to share the same null covariate matrix."
            )

    if chunk_size is None:
        chunks = [tasks]
    else:
        chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]

    cache_path = Path(cache_dir) if cache_dir is not None else None
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, NullFit] = {}
    for ci, chunk in enumerate(chunks):
        gene_ids_in_chunk = [t["gene_id"] for t in chunk]
        cache_file = (cache_path / f"chunk_{ci:05d}.json") if cache_path is not None else None

        if cache_file is not None and cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)
            for gid in gene_ids_in_chunk:
                rec = cached[gid]
                results[gid] = NullFit(
                    beta=np.full((X_ref.shape[1], 0), np.nan), phi=rec["phi"],
                    eta=np.array(rec["eta"]), loglik=float("nan"),
                    n_categories=rec["n_categories"], converged=rec["converged"],
                    method="drimseq_batch_cached")
            continue

        gene_ys = {t["gene_id"]: t["y"] for t in chunk}
        chunk_fits = batch_null_fit_fn(gene_ys, X_ref)
        results.update(chunk_fits)

        if cache_file is not None:
            serializable = {
                gid: dict(phi=nf.phi, eta=nf.eta.tolist(),
                          n_categories=nf.n_categories, converged=nf.converged)
                for gid, nf in chunk_fits.items()
            }
            with open(cache_file, "w") as f:
                json.dump(serializable, f)

    return results


def run_one_gene(gene_id: str, celltype: str, y: np.ndarray, X: np.ndarray,
                  G: np.ndarray, rng: np.random.Generator,
                  null_fit_fn: NullFitFn | None = None,
                  null_fit: NullFit | None = None,
                  variant_ids: Sequence[str] | None = None,
                  perm_batch_size: int = 200, min_perm: int = 1000,
                  max_perm: int = 10000,
                  min_hits_to_stop: int = 15,
                  perm_engine: str = "fast") -> GeneResult | None:
    """Process a single gene x cell type unit.

    perm_engine:
        "fast" (default): `scoretest.PermScanner`, the rearranged permutation
        scan (same statistic up to rounding, see its docstring).
        "loop": the v1.0.0 path, reorder G and call `score_pvalues` once per
        permutation. Kept as the reference for equivalence tests.
        Both draw permutations from `rng` in the same order, so with the same
        seed they see identical permutations.

    y:
        `(n, K)` count matrix.

    X:
        `(n, p)` null-model covariate matrix, including the intercept.

    G:
        `(n, n_snp)` genotype matrix for the gene's cis-SNPs. It is
        recommended to center the columns in advance to reduce collinearity
        with the intercept.

    The null model can be supplied in one of two ways (exactly one is
    required):

    - `null_fit`: A precomputed `NullFit` object (recommended). This is
      intended for use together with `fit_null_models_batched`, so that
      worker processes do not need to invoke R/DRIMSeq. The workers perform
      only NumPy computations, making multiprocessing substantially cleaner
      and more scalable.

    - `null_fit_fn`: A `(y, X) -> NullFit` function that fits the null model
      on demand. This is retained for backward compatibility, primarily for
      inexpensive methods such as `fit_null_python`.

      **Do not use this mode to wrap the single-gene DRIMSeq implementation.**
      Calling DRIMSeq independently for each gene does not provide cross-gene
      precision shrinkage. See `nullmodel.fit_null_drimseq_batch`.

    variant_ids:
        Sequence of variant IDs of length `n_snp`. These IDs are stored in
        `GeneResult` so that the lead SNP and the complete nominal p-value
        table can be reported. The latter is required for generating eSNP
        and significant SNP-gene pair lists.

        If not provided, stringified column indices are used.

    Robustness considerations:

    - If `K < 2` (the gene contains only one PAS) or `G` contains no candidate
      cis-SNPs (`n_snp == 0`, for example because all variants were removed
      by MAF filtering), the gene cannot be tested with the score-test
      framework. The function therefore returns `None`.

      Earlier versions could fail on an empty array in `np.argmin` or produce
      undefined behavior when `softmax_ref` received an `(n, 0)` matrix.
      In a genome-scale analysis involving thousands of genes, one such gene
      could otherwise terminate the pipeline or crash a worker process.
      The invalid gene is now explicitly skipped and the caller
      (`run_many_genes` / the I/O layer) is responsible for recording it.

    - If `null_fit` exists but `converged=False` (for example, because
      genewise precision was unavailable for a gene during batched DRIMSeq
      fitting; see `r_null_fit_batch.R`), the gene is also skipped.

      An invalid or missing `phi` must not be passed into the score test,
      because this could produce a seemingly valid but scientifically
      meaningless p-value.
    """
    n, n_snp = G.shape
    K = y.shape[1]
    if K < 2 or n_snp == 0:
        return None

    if null_fit is None:
        if null_fit_fn is None:
            raise ValueError("run_one_gene requires either null_fit or null_fit_fn, neither was provided.")
        null_fit = null_fit_fn(y, X)
    if not null_fit.converged:
        return None

    if variant_ids is None:
        variant_ids = [str(i) for i in range(n_snp)]
    variant_ids = list(variant_ids)
    assert len(variant_ids) == n_snp

    ctx = GeneNullContext(y, X, null_fit.eta, null_fit.phi)

    obs_p = ctx.score_pvalues(G)
    lead_idx = int(np.argmin(obs_p))
    obs_min_p = float(obs_p[lead_idx])

    if perm_engine not in ("fast", "loop"):
        raise ValueError(f"perm_engine must be 'fast' or 'loop', received: {perm_engine!r}")
    scanner = PermScanner(ctx, G) if perm_engine == "fast" else None

    def gen_batch(n_batch: int) -> np.ndarray:
        idx = np.array([rng.permutation(n) for _ in range(n_batch)])
        if scanner is not None:
            return scanner.min_pvalues(idx)
        min_p = np.empty(n_batch)
        for b in range(n_batch):
            Gp = G[idx[b], :]
            min_p[b] = ctx.score_pvalues(Gp).min()
        return min_p

    perm_res = adaptive_permutation(gen_batch, obs_min_p,
                                     batch_size=perm_batch_size,
                                     min_perm=min_perm,
                                     max_perm=max_perm,
                                     min_hits_to_stop=min_hits_to_stop)
    p_beta, a, b = beta_approx_pvalue(perm_res)

    return GeneResult(gene_id=gene_id, celltype=celltype, n_snp=n_snp,
                       lead_snp_index=lead_idx, obs_min_p=obs_min_p,
                       n_perm_used=perm_res.n_perm_used, n_hits=perm_res.n_hits,
                       beta_p=p_beta, beta_a=a, beta_b=b,
                       lead_variant_id=variant_ids[lead_idx],
                       variant_ids=variant_ids, nominal_pvalues=obs_p)


def run_many_genes(tasks: list[dict], null_fit_fn: NullFitFn | None = None,
                    null_fits: dict[str, NullFit] | None = None,
                    n_workers: int = 1, seed: int = 0, **kwargs) -> list[GeneResult]:
    """Run score-test scanning and permutation testing across multiple genes.

    tasks:
        Each element is a dictionary containing
        `gene_id=...`, `celltype=...`, `y=...`, `X=...`, `G=...`, and
        optionally `variant_ids=...`.

    `null_fit_fn` / `null_fits` are mutually exclusive and exactly one must
    be provided. Their semantics correspond to the `null_fit_fn` /
    `null_fit` arguments of `run_one_gene`, but at the batch level:

    - `null_fits`: Precomputed `{gene_id: NullFit}` objects (recommended).
      These are typically produced by `fit_null_models_batched`, where
      DRIMSeq cross-gene precision shrinkage has already been performed.

      The worker processes then perform only NumPy computations and do not
      launch additional R subprocesses. `n_workers` can therefore be scaled
      according to available CPU resources without also having to manage
      concurrent R processes.

    - `null_fit_fn`: Fit the null model independently inside each worker.
      This is retained mainly for `fit_null_python`.

      If the function wraps the single-gene DRIMSeq implementation, multiple
      workers may launch multiple Rscript subprocesses simultaneously, and
      cross-gene precision shrinkage will not occur. This mode is therefore
      not recommended for DRIMSeq.

    n_workers:
        When `n_workers=1`, processing is serial and is convenient for
        debugging. When `n_workers>1`, `multiprocessing.Pool` is used.

    Returns:
        A list of `GeneResult` objects. Genes for which `run_one_gene`
        returns `None` are automatically skipped, including structurally
        untestable genes and genes whose null models did not converge.

    Multiprocessing implementation note:

        An actual bug was found in an earlier implementation: the worker
        function was defined as a local closure `_worker` inside
        `run_many_genes`.

        Regardless of whether `multiprocessing.Pool` uses `fork`, tasks are
        serialized with pickle when submitted through the multiprocessing
        task queue. Python's standard pickle cannot serialize a function
        defined inside another function because there is no module-level
        name by which the child process can resolve it.

        Consequently, whenever `n_workers > 1`, the previous implementation
        failed at `pool.map` with an error such as:

            AttributeError: Can't pickle local object

        The `n_workers=1` path did not exercise this branch, including all
        previous tests, so the bug remained hidden.

        The fix is to define the worker at module scope
        (`_run_one_gene_worker`) and explicitly pass all required state,
        including `null_fit_fn`, `null_fit`, `seed`, and `kwargs`, instead
        of relying on closure capture.
    """
    if (null_fit_fn is None) == (null_fits is None):
        raise ValueError("run_many_genes requires exactly one of null_fit_fn or null_fits; "
                          "both cannot be provided and neither cannot be provided.")
    indexed = [(i, task, null_fit_fn,
                null_fits.get(task["gene_id"]) if null_fits is not None else None,
                seed, kwargs)
               for i, task in enumerate(tasks)]
    if n_workers <= 1:
        results = [_run_one_gene_worker(item) for item in indexed]
    else:
        import multiprocessing as mp
        with mp.Pool(n_workers) as pool:
            results = pool.map(_run_one_gene_worker, indexed)
    return [r for r in results if r is not None]


def _run_one_gene_worker(args):
    """Worker used by `run_many_genes`.

    This function must be defined at module scope so that it can be serialized
    by `multiprocessing.Pool`. See the implementation note in
    `run_many_genes`.
    """
    i, task, null_fit_fn, null_fit, seed, kwargs = args
    if null_fit_fn is None and null_fit is None:
        # The caller supplied a null_fits dictionary, but this gene_id is not
        # present in it (for example, the gene failed or was not reached
        # during `fit_null_models_batched`).
        #
        # Treat this as an untestable gene and skip it rather than allowing
        # the worker process to fail.
        return None
    rng = np.random.default_rng(seed + i)
    return run_one_gene(task["gene_id"], task["celltype"], task["y"],
                         task["X"], task["G"], rng,
                         null_fit_fn=null_fit_fn, null_fit=null_fit,
                         variant_ids=task.get("variant_ids"), **kwargs)
