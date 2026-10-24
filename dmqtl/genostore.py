"""
Binary genotype store: parse the genotype text matrix once, then memory-map it.

`io.read_matrix_text` builds the whole matrix as Python lists of floats
before converting it to a NumPy array. For a genome-wide file (about 8.3
million variants x 419 individuals, 13 GB of text) that is roughly 3.5e9
Python float objects, i.e. on the order of 100+ GB of peak memory, plus the
final float64 array (about 28 GB) and a second copy made by
`reindex_matrix`. Every run pays this again.

`convert_genotype_text` streams the text file in chunks and writes the
values as a raw float64 file (row-major, variants x individuals) next to a
small JSON header and the variant ID list. Parsing follows
`read_matrix_text` exactly (same delimiter detection, same NA rules, same
duplicate-ID checks), so the stored numbers are bit-identical to what the
text reader returns. `open_store` memory-maps the values read-only: pages
are loaded on demand, shared between worker processes, and released by the
operating system under memory pressure, so only the rows a gene actually
uses need to be in RAM.

Store layout for a prefix P:
    P.meta.json   format, n_rows, n_cols, dtype, sample_ids, source file
    P.rows.txt    variant IDs, one per line, in file order
    P.values.f8   float64, n_rows x n_cols, C order
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cis import center_columns, compute_maf, impute_genotype_mean, mac_among_readers
from .io import _sniff_delimiter

FORMAT = "dmqtl-genotype-store-v1"


def _paths(prefix) -> tuple[Path, Path, Path]:
    p = str(prefix)
    return Path(p + ".meta.json"), Path(p + ".rows.txt"), Path(p + ".values.f8")


def is_store(path) -> bool:
    """True if `path` is the prefix of a genotype store written by `convert_genotype_text`."""
    return _paths(path)[0].exists()


def _parse_values(parts: list[str]) -> np.ndarray:
    """Numeric fields of one line, with the same rules as `read_matrix_text`:
    stripped; "", "NA", "NaN" (any case) -> nan; otherwise float()."""
    try:
        return np.array(parts, dtype=np.float64)
    except ValueError:
        out = np.empty(len(parts))
        for k, x in enumerate(parts):
            x = x.strip()
            out[k] = np.nan if x == "" or x.upper() in ("NA", "NAN") else float(x)
        return out


def convert_genotype_text(txt_path, out_prefix, chunk_rows: int = 20000) -> str:
    """Convert a MatrixEQTL-style genotype text file into a binary store.

    Memory use is bounded by `chunk_rows` x n_individuals values. Returns the
    store prefix (pass it as `genotype_path` to `run_dmqtl` /
    `build_gene_tasks`).
    """
    txt_path = Path(txt_path)
    meta_p, rows_p, vals_p = _paths(out_prefix)
    vals_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_vals = vals_p.with_name(vals_p.name + ".part")
    row_ids: list[str] = []
    with open(txt_path) as f, open(tmp_vals, "wb") as out:
        header_line = f.readline().rstrip("\n")
        delim = _sniff_delimiter(header_line)
        header = header_line.split(delim) if delim else header_line.split()
        sample_ids = [h.strip() for h in header[1:]]
        n_cols = len(sample_ids)
        buf = np.empty((chunk_rows, n_cols))
        k = 0
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(delim) if delim else line.split()
            if len(parts) != len(header):
                raise ValueError(
                    f"{txt_path}: line {len(row_ids) + 2} has {len(parts)} columns, "
                    f"but the header has {len(header)} columns (row_id={parts[0] if parts else '?'})"
                )
            row_ids.append(parts[0].strip())
            buf[k] = _parse_values(parts[1:])
            k += 1
            if k == chunk_rows:
                out.write(buf.tobytes())
                k = 0
        if k:
            out.write(buf[:k].tobytes())

    if len(row_ids) != len(set(row_ids)):
        seen, dup = set(), []
        for r in row_ids:
            if r in seen:
                dup.append(r)
            seen.add(r)
        tmp_vals.unlink()
        raise ValueError(f"{txt_path}: duplicate row IDs found: {dup[:10]}{'...' if len(dup) > 10 else ''}")
    if len(sample_ids) != len(set(sample_ids)):
        dup = [s for s in set(sample_ids) if sample_ids.count(s) > 1]
        tmp_vals.unlink()
        raise ValueError(f"{txt_path}: duplicate sample IDs found in the header: {dup[:10]}")

    rows_p.write_text("".join(r + "\n" for r in row_ids))
    tmp_vals.replace(vals_p)
    st = txt_path.stat()
    meta_p.write_text(json.dumps(dict(
        format=FORMAT, n_rows=len(row_ids), n_cols=n_cols, dtype="float64", order="C",
        sample_ids=sample_ids, source=str(txt_path.resolve()), source_size=st.st_size,
        source_mtime=st.st_mtime), indent=1))
    return str(out_prefix)


@dataclass
class GenotypeStore:
    prefix: str
    row_ids: list[str]
    sample_ids: list[str]
    values: np.ndarray  # read-only memmap, (n_rows, n_cols)


_open_stores: dict[str, GenotypeStore] = {}


def open_store(prefix) -> GenotypeStore:
    """Open (and cache per process) a genotype store."""
    key = str(prefix)
    if key in _open_stores:
        return _open_stores[key]
    meta_p, rows_p, vals_p = _paths(prefix)
    meta = json.loads(meta_p.read_text())
    if meta.get("format") != FORMAT:
        raise ValueError(f"{meta_p}: not a {FORMAT} store")
    row_ids = rows_p.read_text().splitlines()
    if len(row_ids) != meta["n_rows"]:
        raise ValueError(f"{rows_p}: {len(row_ids)} IDs, header says {meta['n_rows']}")
    shape = (meta["n_rows"], meta["n_cols"])
    if shape[0] == 0:
        values = np.zeros(shape)
    else:
        values = np.memmap(vals_p, dtype=np.float64, mode="r", shape=shape, order="C")
    st = GenotypeStore(prefix=key, row_ids=row_ids, sample_ids=meta["sample_ids"], values=values)
    _open_stores[key] = st
    return st


def raw_gene_genotype(store: GenotypeStore, rows: np.ndarray, col_idx: np.ndarray) -> np.ndarray:
    """(n_individuals, n_candidate) raw genotypes for one gene.

    Same values and memory layout as ``reindex_matrix(geno, ...).values[rows, :].T``
    in the text path, so the imputation / MAF / centering arithmetic that
    follows is bit-identical.
    """
    return np.asarray(store.values[rows])[:, col_idx].T


def materialize_genotype(task: dict) -> np.ndarray:
    """Build a lazy task's centered genotype matrix G (n, n_snp) on demand.

    Repeats exactly the steps `build_gene_tasks` applies on the text path
    (mean imputation -> MAF filter -> MAC filter if min_mac > 0 -> column centering) on the gene's
    candidate rows, so G equals the eager task's G bit for bit. Called by the
    worker right before the gene is scanned; G is dropped afterwards.
    """
    spec = task["geno"]
    st = open_store(spec["prefix"])
    G_raw = raw_gene_genotype(st, spec["rows"], spec["col_idx"])
    G_imputed = impute_genotype_mean(G_raw)
    maf = compute_maf(G_imputed)
    keep = np.isfinite(maf) & (maf >= spec["maf_threshold"])
    min_mac = spec.get("min_mac", 0)
    if min_mac > 0:
        keep[keep] = mac_among_readers(G_raw[:, keep], task["y"]) >= min_mac
    G = center_columns(G_imputed[:, keep])
    if G.shape[1] != len(task["variant_ids"]):
        raise RuntimeError(
            f"gene {task['gene_id']}: {G.shape[1]} SNPs pass MAF / MAC now, {len(task['variant_ids'])} "
            "when the task was built; the genotype store changed underneath the run")
    return G
