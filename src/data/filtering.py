from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.constraints import DataConstraints


_UNIMOD_RE = re.compile(r"\[UNIMOD:(\d+)\]", re.IGNORECASE)
_MOD_BRACKET_RE = re.compile(r"\[[^\]]*\]")


def _list_parquet_files(parquet_path: str | Path) -> List[Path]:
    p = Path(parquet_path)
    if p.is_dir():
        return sorted([x for x in p.iterdir() if x.suffix == ".parquet"])
    return [p]


def _detect_columns(names: Iterable[str]) -> Tuple[str, str]:
    cols = set(names)
    if "normalized_sequence" in cols:
        seq_col = "normalized_sequence"
    else:
        seq_col = "modified_sequence" if "modified_sequence" in cols else "sequence"
    charge_col = "precursor_charge" if "precursor_charge" in cols else "charge"
    return seq_col, charge_col


def _naked_len(seq: Any) -> int:
    if not isinstance(seq, str):
        return 0
    return len(_MOD_BRACKET_RE.sub("", seq))


def _unimod_ids(seq: Any) -> Tuple[int, ...]:
    if not isinstance(seq, str):
        return tuple()
    ids = []
    for m in _UNIMOD_RE.finditer(seq):
        try:
            ids.append(int(m.group(1)))
        except Exception:
            continue
    return tuple(sorted(set(ids)))


def constraints_fingerprint(constraints: DataConstraints) -> str:
    s = f"min{constraints.min_len}_max{constraints.max_len}_z{constraints.max_charge}_u{','.join(map(str,constraints.allowed_unimod_ids))}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def filter_parquet_dataset(
    input_path: str | Path,
    output_path: str | Path,
    constraints: DataConstraints,
    max_rows: Optional[int] = None,
    batch_size: int = 50_000,
) -> Dict[str, Any]:
    in_files = _list_parquet_files(input_path)
    out_base = Path(output_path)
    out_base.mkdir(parents=True, exist_ok=True)

    kept = 0
    scanned = 0
    written_files: List[str] = []

    allowed_set = constraints.allowed_unimod_set()

    for fp in in_files:
        pf = pq.ParquetFile(str(fp))
        seq_col, charge_col = _detect_columns(pf.schema.names)

        out_fp = out_base / fp.name
        writer: Optional[pq.ParquetWriter] = None

        for batch in pf.iter_batches(batch_size=batch_size):
            if max_rows is not None and scanned >= int(max_rows):
                break

            df = batch.to_pandas()

            mask = []
            for _, row in df.iterrows():
                if max_rows is not None and scanned >= int(max_rows):
                    break
                scanned += 1

                seq = row.get(seq_col)
                charge = row.get(charge_col)

                nlen = _naked_len(seq)
                if nlen < int(constraints.min_len) or nlen > int(constraints.max_len):
                    mask.append(False)
                    continue

                try:
                    ch = int(charge)
                except Exception:
                    mask.append(False)
                    continue

                if ch < 1 or ch > int(constraints.max_charge):
                    mask.append(False)
                    continue

                mods = _unimod_ids(seq)
                if any(m not in allowed_set for m in mods):
                    mask.append(False)
                    continue

                mask.append(True)

            df2 = df.iloc[: len(mask)].loc[pd.Series(mask).values]
            if len(df2) == 0:
                continue

            table = pa.Table.from_pandas(df2, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out_fp), table.schema)
                written_files.append(str(out_fp))
            writer.write_table(table)

            kept += int(len(df2))

        if writer is not None:
            writer.close()

        if max_rows is not None and scanned >= int(max_rows):
            break

    return {
        "input": str(input_path),
        "output": str(out_base),
        "constraints": {
            "min_len": constraints.min_len,
            "max_len": constraints.max_len,
            "max_charge": constraints.max_charge,
            "allowed_unimod_ids": list(constraints.allowed_unimod_ids),
        },
        "scanned_rows": int(scanned),
        "kept_rows": int(kept),
        "written_files": written_files,
    }
