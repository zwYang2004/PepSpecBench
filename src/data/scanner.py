from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow.parquet as pq

from src.constraints import DataConstraints


_UNIMOD_RE = re.compile(r"\[UNIMOD:(\d+)\]", re.IGNORECASE)
_MOD_BRACKET_RE = re.compile(r"\[[^\]]*\]")


def _list_parquet_files(parquet_path: str | Path) -> List[Path]:
    p = Path(parquet_path)
    if p.is_dir():
        return sorted([x for x in p.iterdir() if x.suffix == ".parquet"])
    return [p]


def _detect_columns(schema: pq.ParquetSchema) -> Tuple[str, str]:
    cols = set(schema.names)
    seq_col = "modified_sequence" if "modified_sequence" in cols else "sequence"
    charge_col = "precursor_charge" if "precursor_charge" in cols else "charge"
    return seq_col, charge_col


def _naked_len(seq: Any) -> int:
    if not isinstance(seq, str):
        return 0
    s = _MOD_BRACKET_RE.sub("", seq)
    return len(s)


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


@dataclass
class CompatibilityReport:
    total_rows: int
    scanned_rows: int
    compatible_rows: int
    dropped_rows: int
    drop_reasons: Dict[str, int]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "scanned_rows": self.scanned_rows,
            "compatible_rows": self.compatible_rows,
            "dropped_rows": self.dropped_rows,
            "drop_reasons": dict(self.drop_reasons),
            "compatible_ratio": (float(self.compatible_rows) / float(self.scanned_rows)) if self.scanned_rows else 0.0,
        }


def scan_parquet_compatibility(
    parquet_path: str | Path,
    constraints: DataConstraints,
    max_rows: int = 10_000,
    batch_size: int = 20_000,
) -> CompatibilityReport:
    files = _list_parquet_files(parquet_path)

    total_rows = 0
    scanned = 0
    ok = 0
    drop_reasons: Dict[str, int] = {}

    allowed_set = constraints.allowed_unimod_set()

    for fp in files:
        pf = pq.ParquetFile(str(fp))
        total_rows += int(pf.metadata.num_rows) if pf.metadata is not None else 0

        seq_col, charge_col = _detect_columns(pf.schema)
        columns = [seq_col, charge_col]

        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            if scanned >= int(max_rows):
                break

            bdf = batch.to_pandas()
            for _, row in bdf.iterrows():
                if scanned >= int(max_rows):
                    break
                scanned += 1

                seq = row.get(seq_col)
                charge = row.get(charge_col)

                nlen = _naked_len(seq)
                if nlen < int(constraints.min_len) or nlen > int(constraints.max_len):
                    drop_reasons["len"] = drop_reasons.get("len", 0) + 1
                    continue

                try:
                    ch = int(charge)
                except Exception:
                    drop_reasons["charge_parse"] = drop_reasons.get("charge_parse", 0) + 1
                    continue

                if ch < 1 or ch > int(constraints.max_charge):
                    drop_reasons["charge"] = drop_reasons.get("charge", 0) + 1
                    continue

                mods = _unimod_ids(seq)
                bad_mods = [m for m in mods if m not in allowed_set]
                if bad_mods:
                    drop_reasons["ptm"] = drop_reasons.get("ptm", 0) + 1
                    continue

                ok += 1

        if scanned >= int(max_rows):
            break

    return CompatibilityReport(
        total_rows=int(total_rows),
        scanned_rows=int(scanned),
        compatible_rows=int(ok),
        dropped_rows=int(scanned - ok),
        drop_reasons=drop_reasons,
    )


def save_report_json(report: CompatibilityReport, output_path: str | Path) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(report.as_dict(), f, indent=2)
