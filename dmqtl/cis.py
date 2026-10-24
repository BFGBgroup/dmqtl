"""
Build cis-window candidate SNP sets and perform MAF filtering.

This module implements two steps that were explicitly discussed previously
but had not been implemented in earlier versions:

    - Construction of the cis-SNP set:
      Previously, `pipeline.run_one_gene` assumed that the caller had already
      prepared G as the cis-SNP genotype matrix for each gene. However, when
      starting from a matrixQTL/tensorQTL-style input consisting of a genome-
      wide genotype matrix plus variant and gene position files, this step
      must be performed explicitly. Candidate variants are defined by the
      gene TSS +/- the specified cis-window. Otherwise, every gene would be
      tested against all genome-wide variants, which is both incorrect and
      computationally infeasible.

    - MAF filtering:
      The README previously mentioned low-frequency variants as a known
      limitation/optimization, noting that low-frequency cis-SNPs may cause
      unstable DM-GLM convergence and underestimated variance, and therefore
      recommended MAF filtering before scanning. However, this filtering was
      never actually implemented in the code. This module now provides it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import PositionTable


def build_cis_map(gene_pos: PositionTable, variant_pos: PositionTable,
                   window: int = 1_000_000) -> dict[str, list[str]]:
    """For each gene, identify variants on the same chromosome whose distance
    from the gene TSS is within the specified window, and use them as the
    candidate cis-SNP set.

    Implementation details:
    Variants are first grouped by chromosome and sorted by genomic position.
    Binary search (`np.searchsorted`) is then used to query the relevant
    interval for each gene, rather than scanning all variants for every gene.

    This is important at genome-wide scale (tens of thousands of genes and
    millions of variants), where a naive nested loop would require
    O(n_gene * n_variant) operations. The implementation here requires
    approximately O(n_variant log n_variant + n_gene log n_variant).
    """
    by_chrom: dict[str, list[tuple[int, str]]] = {}
    for vid in variant_pos.ids:
        by_chrom.setdefault(variant_pos.chrom[vid], []).append((variant_pos.anchor[vid], vid))
    sorted_by_chrom: dict[str, tuple[np.ndarray, list[str]]] = {}
    for chrom, pairs in by_chrom.items():
        pairs.sort(key=lambda t: t[0])
        positions = np.array([p for p, _ in pairs])
        vids = [v for _, v in pairs]
        sorted_by_chrom[chrom] = (positions, vids)

    gene_to_snps: dict[str, list[str]] = {}
    for gid in gene_pos.ids:
        chrom = gene_pos.chrom[gid]
        if chrom not in sorted_by_chrom:
            gene_to_snps[gid] = []
            continue
        tss = gene_pos.anchor[gid]
        positions, vids = sorted_by_chrom[chrom]
        lo = np.searchsorted(positions, tss - window, side="left")
        hi = np.searchsorted(positions, tss + window, side="right")
        gene_to_snps[gid] = vids[lo:hi]
    return gene_to_snps


def compute_maf(G: np.ndarray) -> np.ndarray:
    """Compute the minor allele frequency (MAF) for each variant.

    Genotypes are assumed to be encoded as 0/1/2, or as general dosage
    values in the range [0, 2]. Under the standard convention:

        MAF = min(mean(g) / 2, 1 - mean(g) / 2)

    For columns containing missing values (NaN), the mean is computed using
    only non-missing genotypes. In other words, NaN values are not treated
    as zero.

    This is an important detail: using `mean` instead of `nanmean` would
    systematically underestimate the MAF of variants with substantial
    missingness.
    """
    freq = np.nanmean(G, axis=0) / 2.0
    return np.minimum(freq, 1.0 - freq)


def maf_keep_mask(G: np.ndarray, maf_threshold: float = 0.01) -> np.ndarray:
    """Boolean mask of the variants that pass the MAF threshold (NaN MAF fails)."""
    maf = compute_maf(G)
    return np.isfinite(maf) & (maf >= maf_threshold)


def maf_filter(variant_ids: list[str], G: np.ndarray,
                maf_threshold: float = 0.01) -> tuple[list[str], np.ndarray]:
    """Filter variants by MAF threshold.

    Variants with all or nearly all genotypes missing are also removed because
    their MAF is NaN.

    Returns the filtered `(variant_ids, G)` pair.
    """
    keep = maf_keep_mask(G, maf_threshold)
    kept_ids = [vid for vid, k in zip(variant_ids, keep) if k]
    return kept_ids, G[:, keep]


def mac_among_readers(G: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Minor allele count of each variant among the individuals with reads for the gene (2026-10).

    G: raw genotypes (n, n_snp), 0/1/2 or dosage, NaN = missing (not counted).
    y: the gene's counts (n, K); an individual has reads if its row sum is > 0.

    MAF over all individuals can pass while every carrier but one or two has no
    reads for the gene. Those individuals carry no information in the score
    test, so the SNP is in effect tested on one or two people; on real data
    this is where the 1e-12 permutation floor comes from. The count is
    min(alt, 2 * n_called - alt) over the individuals with reads.
    """
    g = G[np.asarray(y).sum(axis=1) > 0]
    alt = np.nansum(g, axis=0)
    total = 2.0 * np.sum(~np.isnan(g), axis=0)
    return np.minimum(alt, total - alt)


def impute_genotype_mean(G: np.ndarray) -> np.ndarray:
    """Impute missing genotypes (NaN) using the sample mean of each variant.

    Mean imputation is commonly used in the matrixQTL/tensorQTL ecosystem.
    Under a linear-model framework, mean imputation is a reasonable default
    and avoids allowing NaN values to propagate into downstream matrix
    operations.

    For the DM-GLM score test, genotype is likewise treated as a linear
    predictor, so using the same convention is a reasonable default.

    Without imputation, a single missing genotype can cause the downstream
    calculation for an entire SNP to become NaN. This results in silently
    undefined test statistics rather than an explicit error, which is more
    difficult to detect and therefore potentially more problematic.
    """
    G = G.copy()
    col_mean = np.nanmean(G, axis=0)
    nan_mask = np.isnan(G)
    if nan_mask.any():
        rows, cols = np.where(nan_mask)
        G[rows, cols] = col_mean[cols]
    return G


def center_columns(G: np.ndarray) -> np.ndarray:
    """Center each genotype column by subtracting its mean.

    Column centering reduces collinearity between genotype predictors and the
    intercept term in the null model.

    This convention has consistently been used in `simulate.py` and the test
    suite, but the IO layer needs to apply it explicitly because real genotype
    data are not automatically centered."""
    return G - np.nanmean(G, axis=0, keepdims=True)
