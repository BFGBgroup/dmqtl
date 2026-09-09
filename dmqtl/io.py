"""
Text-based input layer. File formats follow the conventions used by
MatrixEQTL / tensorQTL:

    genotype file   : rows = variants, columns = individuals (the first
                      column is the variant ID; the first row is the header;
                      columns after header[0] contain individual IDs).

    phenotype file  : rows = PAS, columns = individuals, following the same
                      row/column convention. Values are raw counts, not ratios.
                      This is intentional: as discussed in the README and
                      previously, the DM model operates directly on counts and
                      does not require explicit calculation of ratios, thereby
                      naturally avoiding the "zero denominator" problem.

    PAS-gene mapping: two columns, (pas_id, gene_id), specifying which gene
                      each PAS belongs to. All PAS belonging to the same gene
                      form the K categories of one Dirichlet-Multinomial
                      composition test.

    covariates file : rows = covariates, columns = individuals (the same
                      convention as MatrixEQTL covariate files). Users do not
                      need to provide an intercept row. A column of ones is
                      automatically added as the first column of the null
                      design matrix X. This is an explicit default assumption;
                      see the documentation of `eqtl.build_gene_tasks`.

    variant/gene position files (optional, used to define the cis-window):
        variant: (variant_id, chr, pos)
        gene   : (gene_id, chr, start, end[, strand])

        Cis filtering is performed only when both files are provided. If they
        are not provided, all variants in the genotype file are used as the
        candidate set for every gene. This is appropriate only when the
        number of variants is already small, for example, when the input has
        been pre-filtered to cis variants. It is not recommended for
        genome-wide genotype files; see the explicit warning in
        `eqtl.build_gene_tasks`.

All input readers match individual IDs by name and do not assume that columns
appear in the same order across files. This is a standard convention for
MatrixEQTL-style tools, but it is also a common source of real-world errors.
If the columns in one file are silently out of order relative to another file,
and the program does not explicitly realign them by sample ID, it could pair
one individual's genotype with another individual's phenotype. This is more
dangerous than a program crash because no error is raised and incorrect
results are silently produced. `match_individuals` is designed to prevent
this class of error.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _sniff_delimiter(header_line: str) -> str:
    if "\t" in header_line:
        return "\t"
    if "," in header_line:
        return ","
    return None  # Let split() handle arbitrary whitespace


@dataclass
class TextMatrix:
    """Text matrix with rows = IDs (variant_id / pas_id / covariate name) and columns = individual IDs."""
    row_ids: list[str]
    sample_ids: list[str]
    values: np.ndarray  # (n_rows, n_samples)

    def as_frame_like(self):
        return self.row_ids, self.sample_ids, self.values


def read_matrix_text(path: str | Path) -> TextMatrix:
    """Read a MatrixEQTL/tensorQTL-style matrix text file.

    First line:
        <arbitrary column name> <sample_id_1> <sample_id_2> ...

    Subsequent lines:
        <row_id> <value_1> <value_2> ...

    Supports tab, comma, and arbitrary whitespace delimiters, which are
    detected automatically. Empty strings, NA, and NaN in numeric columns
    are converted to np.nan. The caller is responsible for deciding how
    missing values should be handled. The current strategy for missing
    genotype values in `eqtl.build_gene_tasks` is documented there.
    """
    path = Path(path)
    with open(path) as f:
        header_line = f.readline().rstrip("\n")
        delim = _sniff_delimiter(header_line)
        header = header_line.split(delim) if delim else header_line.split()
        sample_ids = [h.strip() for h in header[1:]]

        row_ids: list[str] = []
        rows: list[list[float]] = []
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(delim) if delim else line.split()
            if len(parts) != len(header):
                raise ValueError(
                    f"{path}: line {len(row_ids) + 2} has {len(parts)} columns, "
                    f"but the header has {len(header)} columns (row_id={parts[0] if parts else '?'})"
                )
            row_ids.append(parts[0].strip())
            vals = []
            for x in parts[1:]:
                x = x.strip()
                vals.append(np.nan if x == "" or x.upper() in ("NA", "NAN") else float(x))
            rows.append(vals)

    if len(row_ids) != len(set(row_ids)):
        dup = [r for r in set(row_ids) if row_ids.count(r) > 1]
        raise ValueError(f"{path}: duplicate row IDs found: {dup[:10]}{'...' if len(dup) > 10 else ''}")
    if len(sample_ids) != len(set(sample_ids)):
        dup = [s for s in set(sample_ids) if sample_ids.count(s) > 1]
        raise ValueError(f"{path}: duplicate sample IDs found in the header: {dup[:10]}")

    values = np.array(rows, dtype=float) if rows else np.zeros((0, len(sample_ids)))
    return TextMatrix(row_ids=row_ids, sample_ids=sample_ids, values=values)


def read_pas_gene_map(path: str | Path) -> dict[str, list[str]]:
    """Read a two-column file `(pas_id, gene_id)`, with or without a header.

    Header detection is heuristic. If the first row contains terms such as
    "pas", "feature", "gene", or "id" in the expected columns, it is treated
    as a header and skipped.

    The parser intentionally uses a simple heuristic rather than requiring
    a header, because both headered and headerless mapping files are commonly
    used in MatrixEQTL-style workflows.

    Returns:
        A mapping from gene_id to a list of PAS IDs, preserving the order
        in which PAS entries appear in the input file.
    """
    path = Path(path)
    rows: list[tuple[str, str]] = []
    with open(path) as f:
        header_line = f.readline().rstrip("\n")
        delim = _sniff_delimiter(header_line)
        first = header_line.split(delim) if delim else header_line.split()
        looks_like_header = False
        if len(first) >= 2:
            lower = [c.strip().lower() for c in first[:2]]
            if any(tok in lower[0] for tok in ("pas", "feature", "id")) or \
               any(tok in lower[1] for tok in ("gene", "id")):
                looks_like_header = True
        if not looks_like_header:
            rows.append((first[0].strip(), first[1].strip()))

        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(delim) if delim else line.split()
            if len(parts) < 2:
                raise ValueError(f"{path}: line '{line}' has fewer than 2 columns")
            rows.append((parts[0].strip(), parts[1].strip()))

    gene_to_pas: dict[str, list[str]] = {}
    for pas_id, gene_id in rows:
        gene_to_pas.setdefault(gene_id, []).append(pas_id)
    return gene_to_pas


@dataclass
class PositionTable:
    ids: list[str]
    chrom: dict[str, str]
    anchor: dict[str, int]  # variant: pos; gene: TSS, depending on strand


def read_variant_positions(path: str | Path) -> PositionTable:
    """Read a variant position file: variant_id, chr, pos. At least three columns are required. A header is optional."""
    ids, chrom, anchor = [], {}, {}
    with open(path) as f:
        header_line = f.readline().rstrip("\n")
        delim = _sniff_delimiter(header_line)
        first = header_line.split(delim) if delim else header_line.split()
        lines = [first] + [
            (l.rstrip("\n").split(delim) if delim else l.rstrip("\n").split())
            for l in f if l.strip()
        ]
    # If the third column of the first row is not numeric, treat the row as a header.
    start = 0
    try:
        float(lines[0][2])
    except (ValueError, IndexError):
        start = 1
    for parts in lines[start:]:
        if len(parts) < 3 or not parts[0].strip():
            continue
        vid = parts[0].strip()
        ids.append(vid)
        chrom[vid] = parts[1].strip()
        anchor[vid] = int(float(parts[2]))
    return PositionTable(ids=ids, chrom=chrom, anchor=anchor)


def read_gene_positions(path: str | Path) -> PositionTable:
    """Read a gene position file:
        gene_id, chr, start, end[, strand]

    At least four columns are required. A header is optional.

    The TSS is used as the anchor point for cis-window filtering. When a
    strand column is available, '+' uses `start` and '-' uses `end`.
    Without strand information, `start` is used for all genes. This is an
    explicit assumption because the correct TSS orientation cannot be
    inferred from genomic coordinates alone without strand information.
    """
    ids, chrom, anchor = [], {}, {}
    with open(path) as f:
        header_line = f.readline().rstrip("\n")
        delim = _sniff_delimiter(header_line)
        first = header_line.split(delim) if delim else header_line.split()
        lines = [first] + [
            (l.rstrip("\n").split(delim) if delim else l.rstrip("\n").split())
            for l in f if l.strip()
        ]
    start_row = 0
    try:
        float(lines[0][2])
    except (ValueError, IndexError):
        start_row = 1
    for parts in lines[start_row:]:
        if len(parts) < 4 or not parts[0].strip():
            continue
        gid = parts[0].strip()
        ids.append(gid)
        chrom[gid] = parts[1].strip()
        start_pos, end_pos = int(float(parts[2])), int(float(parts[3]))
        strand = parts[4].strip() if len(parts) > 4 else None
        if strand == "-":
            anchor[gid] = end_pos
        else:
            anchor[gid] = start_pos
    return PositionTable(ids=ids, chrom=chrom, anchor=anchor)


@dataclass
class MatchedIndividuals:
    sample_ids: list[str]          # Final order of the matched individuals
    dropped: dict[str, list[str]]  # IDs absent from the intersection in each input


def match_individuals(*sample_id_lists: list[str], reference: int = 0) -> MatchedIndividuals:
    """Match individual IDs across multiple inputs (genotype / phenotype /
    covariates) by intersection.

    The final order follows the order in the selected reference input
    (default: the first input, typically genotype).

    In real-world datasets, some individuals may be present in one file but
    absent from another, for example because of sample QC differences or
    sequencing failures. The function therefore explicitly takes the
    intersection of all sample sets and reports which IDs were dropped,
    rather than assuming that all files contain exactly the same individuals.

    Aligning samples by position rather than by name is one of the most
    dangerous and difficult-to-detect errors in this type of analysis:
    the program may continue running normally while silently pairing the
    wrong genotype and phenotype data.
    """
    sets = [set(s) for s in sample_id_lists]
    common = set.intersection(*sets) if sets else set()
    if not common:
        raise ValueError("No overlapping individual IDs were found among genotype/phenotype/covariates")
    ref_order = sample_id_lists[reference]
    final_order = [s for s in ref_order if s in common]
    dropped = {}
    for i, ids in enumerate(sample_id_lists):
        extra = [s for s in ids if s not in common]
        if extra:
            dropped[f"input_{i}"] = extra
    return MatchedIndividuals(sample_ids=final_order, dropped=dropped)


def reindex_matrix(mat: TextMatrix, sample_order: list[str]) -> TextMatrix:
    """Reorder and subset the matrix columns according to `sample_order`.
    `sample_order` must be a subset of `mat.sample_ids`; this condition is
    normally guaranteed by `match_individuals`."""
    idx = {s: i for i, s in enumerate(mat.sample_ids)}
    missing = [s for s in sample_order if s not in idx]
    if missing:
        raise ValueError(f"The following individual IDs were not found in the matrix: {missing[:10]}")
    col_idx = [idx[s] for s in sample_order]
    return TextMatrix(row_ids=mat.row_ids, sample_ids=list(sample_order),
                       values=mat.values[:, col_idx])
