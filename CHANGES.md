# Changes

## Unreleased (branch `opt`, based on v1.0.0 = 57701af)

Three goals: lower memory, shorter run time, more genes with a usable null fit.
Every change was checked against v1.0.0 on the same inputs; changes that do
not reproduce v1.0.0's results are listed separately below, with the reason.

### Same results, less memory and time

- **Fast permutation engine** (`--perm-engine fast`, the default; `loop` is the
  v1.0.0 code path). Permutes the per-individual quantities (R, W, X⊗W) instead
  of the genotype matrix, scores all SNPs of a block with stacked matrix
  products, scores each distinct genotype column once, uses closed forms for the
  (K−1)×(K−1) Schur inverse, folds Ibb⁻¹ = LLᵀ into X⊗W, and evaluates the
  χ² tail only at the maximum T. Per permutation, one core, 5,000 SNPs:
  K = 2 15×, K = 3 6.5×, K = 5 3.2× faster; largest relative difference to the
  loop path 6.4e-14 (`tests/test_fast_equivalence.py`).
- **Binary genotype store.** `dmqtl-convert-genotype genotype.txt geno` parses
  the text matrix once (constant memory, ~33 s per 200,000 variants); pass the
  printed prefix to `dmqtl-run --genotype`. Genes then read only their cis
  columns, on demand, instead of holding every gene's genotype matrix in
  memory (`tests/test_genostore.py`: values, tasks and output files identical).
- **Null-fit chunks in parallel** (`--null-fit-parallel-chunks N`, each chunk
  one Rscript with `--null-fit-n-cores` cores).
- **Worker pool: one gene per dispatch, heaviest gene first.** v1.0.0 hands
  each worker ~n/(4×workers) consecutive genes at once, in genome order. Now
  genes go one at a time, sorted by per-permutation cost, (K−1)^1.83 × SNP
  count (fit on full Ast, R² 0.98), so no heavy gene starts near the end and
  runs alone while the rest of the node waits. Replay of full Ast on 55
  workers: ~2 h shorter permutation stage. Each gene keeps its task index as
  its RNG seed and results return to task order, so the output files are
  identical to serial processing (`tests/test_dispatch.py`).

Real data (Ast pseudobulk, 100 genes, 432 individuals, 609,505 cis pairs, the
same DRIMSeq null fits on both sides, 16 workers, 1,000 permutations, the same
node, Xeon Gold 6348):

| | v1.0.0 (`loop`, text genotype) | this branch (`fast`, store) |
|---|---|---|
| wall time | 3,981 s | 624 s |
| real memory of the whole run, peak | 15.4 GB | 6.7 GB (+ 2.1 GB mapped genotype file) |
| private memory, main process | 11.8 GB | 0.5 GB |
| private memory, largest worker | 1.1 GB | 1.0 GB |

The three output files (`egenes`, `cis_qtl_pairs`, `significant_pairs`) are
identical line by line (both sides with the two bug fixes below).
Memory is the anonymous PSS summed over the main process and its workers
(pages shared after fork counted once; `tools/mem_sampler.sh`, sampled every
3 s, same inputs with 100 permutations). Per-process RSS and Slurm's MaxRSS
count the shared pages again in every worker (127 GB and 15 GB here) and
overstate both versions.
Full cohort (9.2 M variants × 436 individuals, Ast, 13,126 genes): building
all gene tasks takes 7.5 GB of private memory in the main process (+ 2.9 GB
of the mapped genotype file); workers forked afterwards hold at most 1.8 GB of
private memory each (8 workers: 21.5 GB in total), i.e. they do not copy the
main process's data. v1.0.0's text reader extrapolates to about 170–190 GB
for reading the genotype alone.

### More genes with a usable null fit (DRIMSeq)

- **Cox–Reid adjustment overflow.** With ~20 covariates the adjustment's
  `det(n * (-H))` overflows for K ≥ 4, DRIMSeq reads the precision as NA, and
  whole chunks fail with `genewise_precision ... logical`. The adjustment now
  uses `determinant(..., logarithm = TRUE)`; values that were finite before are
  unchanged to the last digit. `--null-fit-cr-logdet 0` restores v1.0.0.
- **optim iteration limit** 1,000 instead of R's 100
  (`--null-fit-optim-maxit 0` restores v1.0.0).
- An `NA` from R is read as NaN instead of failing the chunk; an all-NA chunk no
  longer stops `dmFit`; an R error caused by one gene is isolated by splitting
  the chunk in halves, so it costs that gene (`converged=False`,
  method `r_error`), not its chunk.
- `--null-fit-seed` makes `--null-fit-prec-subset < 1` reproducible.

Real data (Ast, 100 genes, `prec_subset` 1.0): genes with a usable null fit
18/100 → 84/100. The remaining 16 are flagged grid-boundary/outlier (14) or
precision NA (2); for these genes a median 173 of 432 individuals have zero
total count, so they stay marked as failed.

**Behaviour change:** the common precision moves (real data: 101–995 →
21–292), because in v1.0.0 it was computed mostly from the genes whose
likelihood did not overflow. Genes that already had a null fit therefore get a
somewhat different one. On the real data the 18 genes that had a null fit
with DRIMSeq as shipped were rescored with both fits (same code, same
permutations; `prec_subset` 1.0): for 17 the gene-wise precision moved by at
most 2%, the SNP set and lead SNP are the same, nominal p-values moved by at
most 0.0073 on the log10 scale (Spearman ≥ 0.99999), and permutation hits by
at most 1 in 1,000. The 18th, DIP2A-IT1 (precision 2.6; 286 of 432 individuals
with zero total count), is now flagged by the existing outlier rule
(|log2(phi / median phi)| > 5, median over the genes of its chunk). Its own
precision did not change (2.585 vs 2.583); in v1.0.0 it passed only because 7
of the 10 genes in its chunk had NA precision, so the median came from 3 genes
(55.0 instead of 160.5). The outlier reference therefore depends on which
genes share a chunk; with the recommended chunks of 1,000+ genes this is much
less of an issue than with the 10-gene chunks used in this test.

### Bug fixes that change v1.0.0 output

- **Beta approximation.** When scipy's `beta.fit` fails (it did on real data:
  `FitSolverError`, maxfev 600), v1.0.0 ended the whole run. Now: Nelder–Mead
  on the same likelihood (as in tensorQTL), then the empirical p-value; never
  NaN. Genes where scipy succeeds are unchanged.
- **Degenerate score tests.** Individuals with zero total count carry no
  information (R_i = W_i = 0). A SNP that is constant over the informative
  individuals then has U = 0 and a zero Schur complement, and T = U'S⁺U is 0/0;
  v1.0.0 returned rounding noise as a p-value (down to 1e-8 under permutation,
  and different noise on each engine). Such SNPs now get p = 1 in the observed
  scan and in every permutation (‖S‖_F ≤ 1e-10 ‖Idd‖_F; on real data the ratio
  is ≤ 1e-12 for these SNPs and ≥ 1e-6 for all others). Footprint on the real
  data above: 27 cis pairs in 4 genes go to p = 1 (19 had p < 0.05); in 6 genes
  the permutation hits or Beta parameters change; lead SNPs, significant pairs
  and eGenes are unchanged.

### New warning

- A gene with more than 1% of permutation min-p values at the 1e-12 floor is
  reported on stderr with its Beta p and its empirical p: its Beta
  approximation is unreliable. On real data this happens when a single
  outlying individual is the only informative carrier of a rare SNP; 9 of 84
  tested genes, e.g. ADAMTS18 with 63% at the floor, Beta p 0.032 and
  empirical p 0.37 (its observed min-p, 2.1e-25, is below the floor too). The
  p-values themselves are not changed here; a MAC threshold or the empirical
  p-value for such genes are options.

### Reproducing v1.0.0

`--perm-engine loop --null-fit-cr-logdet 0 --null-fit-optim-maxit 0`
reproduces v1.0.0 apart from the two bug fixes above.
