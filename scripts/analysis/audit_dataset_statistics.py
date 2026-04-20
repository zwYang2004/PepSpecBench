#!/usr/bin/env python3
"""Audit final benchmark datasets and export exact paper-ready statistics."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
import pyarrow.parquet as pq


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


MOD_RE = re.compile(r"\[([^\]]+)\]")
AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_NCE_BIN_WIDTH = 0.05


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    group: str
    split: str
    paths: Tuple[Path, ...]


def _classify_mod_token(tok: str) -> Optional[int]:
    t = str(tok).strip().lower()
    if not t:
        return None
    if "unimod:35" in t or "oxidation" in t or "+15.99" in t or "+15.994" in t:
        return 35
    if "unimod:4" in t or "carbamidomethyl" in t or "+57.02" in t or "+57.021" in t:
        return 4
    if "unimod:1" in t or "acetyl" in t or "+42.01" in t or "+42.011" in t:
        return 1
    return None


def _canonical_ptm_name(unimod: int) -> str:
    return {
        1: "UNIMOD:1_Acetyl",
        4: "UNIMOD:4_CAM",
        35: "UNIMOD:35_Oxidation",
    }.get(unimod, f"UNIMOD:{unimod}")


def _iter_parquet_frames(path: Path, batch_size: int = 100_000) -> Iterator[pd.DataFrame]:
    parquet = pq.ParquetFile(str(path))
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()


def _detect_seq_col(df: pd.DataFrame) -> str:
    for candidate in ("modified_sequence", "sequence", "normalized_sequence", "naked_sequence"):
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Sequence column not found in columns={df.columns.tolist()}")


def _detect_charge_col(df: pd.DataFrame) -> Optional[str]:
    for candidate in ("precursor_charge", "charge"):
        if candidate in df.columns:
            return candidate
    return None


def _detect_nce_col(df: pd.DataFrame) -> Optional[str]:
    for candidate in ("collision_energy", "nce", "ce"):
        if candidate in df.columns:
            return candidate
    return None


def _extract_naked_and_ptms(seq: object) -> Tuple[str, List[int]]:
    text = "" if seq is None else str(seq)
    if "[" not in text:
        return text, []

    ptms: List[int] = []
    for match in MOD_RE.finditer(text):
        unimod = _classify_mod_token(match.group(1))
        if unimod is not None:
            ptms.append(unimod)
    naked = MOD_RE.sub("", text)
    return naked, ptms


def _nce_bin_label(value: float, width: float) -> str:
    start = math.floor(value / width) * width
    end = start + width
    return f"[{start:.2f}, {end:.2f})"


class StreamingStats:
    def __init__(self, nce_bin_width: float) -> None:
        self.nce_bin_width = nce_bin_width
        self.total_rows = 0
        self.unique_backbones: set[str] = set()
        self.ptm_spectrum_rows = 0
        self.ptm_event_total = 0
        self.aa_counter: Counter[str] = Counter()
        self.length_counter: Counter[int] = Counter()
        self.charge_counter: Counter[int] = Counter()
        self.nce_exact_counter: Counter[str] = Counter()
        self.nce_bin_counter: Counter[str] = Counter()
        self.ptm_counter: Counter[str] = Counter()
        self.ptm_combo_counter: Counter[str] = Counter()

    def update(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        seq_col = _detect_seq_col(df)
        charge_col = _detect_charge_col(df)
        nce_col = _detect_nce_col(df)

        self.total_rows += int(len(df))

        seq_values = df[seq_col].astype(str).tolist()
        for seq in seq_values:
            naked, ptms = _extract_naked_and_ptms(seq)
            self.unique_backbones.add(naked)
            self.length_counter[len(naked)] += 1
            self.aa_counter.update([aa for aa in naked if aa in AA_ALPHABET])

            if ptms:
                self.ptm_spectrum_rows += 1
                self.ptm_event_total += len(ptms)
                unique_ptms = sorted(set(ptms))
                combo_name = "+".join(_canonical_ptm_name(x) for x in unique_ptms)
                self.ptm_combo_counter[combo_name] += 1
                for unimod in ptms:
                    self.ptm_counter[_canonical_ptm_name(unimod)] += 1
            else:
                self.ptm_combo_counter["UNMOD"] += 1

        if charge_col is not None:
            values = pd.to_numeric(df[charge_col], errors="coerce").dropna().astype(int)
            self.charge_counter.update(values.tolist())

        if nce_col is not None:
            values = pd.to_numeric(df[nce_col], errors="coerce").dropna().tolist()
            for value in values:
                self.nce_exact_counter[f"{float(value):.4f}"] += 1
                self.nce_bin_counter[_nce_bin_label(float(value), self.nce_bin_width)] += 1

    def to_summary(self) -> Dict[str, object]:
        total = self.total_rows or 1
        return {
            "total_rows": self.total_rows,
            "unique_backbones": len(self.unique_backbones),
            "ptm_spectrum_rows": self.ptm_spectrum_rows,
            "ptm_spectrum_pct": self.ptm_spectrum_rows / total,
            "ptm_event_total": self.ptm_event_total,
            "unique_ptm_types": len(self.ptm_counter),
            "ptm_type_counts": dict(self.ptm_counter.most_common()),
            "ptm_combination_counts": dict(self.ptm_combo_counter.most_common()),
            "aa_counts": dict(sorted(self.aa_counter.items())),
            "length_counts": dict(sorted(self.length_counter.items())),
            "charge_counts": dict(sorted(self.charge_counter.items())),
            "nce_exact_counts": dict(sorted(self.nce_exact_counter.items(), key=lambda x: float(x[0]))),
            "nce_bin_counts": dict(sorted(self.nce_bin_counter.items())),
        }


def _dataset_specs(include_full: bool) -> List[DatasetSpec]:
    specs: List[DatasetSpec] = []

    for dataset_name, root in (
        ("PROSPECT-M", _ROOT / "data" / "reconstructed" / "prospect" / "all"),
        ("MassIVE-KB-M", _ROOT / "data" / "reconstructed" / "massive_kb" / "all"),
    ):
        for split in ("train", "val", "test"):
            specs.append(
                DatasetSpec(
                    name=dataset_name,
                    group="mini_in_domain",
                    split=split,
                    paths=(root / f"{split}.parquet",),
                )
            )

    for path in sorted((_ROOT / "data" / "ood_reference").glob("*.parquet")):
        specs.append(
            DatasetSpec(
                name=f"OOD-Reference::{path.stem}",
                group="mini_ood",
                split="ood",
                paths=(path,),
            )
        )

    if include_full:
        prospect_root = _ROOT / "data" / "Prospect_parquet" / "Prospect_merged_unimod135"
        for split in ("train", "val", "test"):
            candidate = prospect_root / f"{split}.parquet"
            if candidate.exists():
                specs.append(
                    DatasetSpec(
                        name="PROSPECT-FullFiltered",
                        group="full_filtered",
                        split=split,
                        paths=(candidate,),
                    )
                )

        massive_root = _ROOT / "data" / "MassIVE-KB" / "processed_charge_le6_unimod135"
        for split in ("train", "val", "test"):
            paths = tuple(sorted((massive_root / split).glob("*.charge_le6.unimod135.parquet")))
            if paths:
                specs.append(
                    DatasetSpec(
                        name="MassIVE-KB-FullFiltered",
                        group="full_filtered",
                        split=split,
                        paths=paths,
                    )
                )

    return specs


def audit_one_dataset(spec: DatasetSpec, nce_bin_width: float) -> Dict[str, object]:
    stats = StreamingStats(nce_bin_width=nce_bin_width)
    missing = [str(path) for path in spec.paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{spec.name}::{spec.split} missing files: {missing}")

    for path in spec.paths:
        for frame in _iter_parquet_frames(path):
            stats.update(frame)

    result = stats.to_summary()
    result.update(
        {
            "dataset": spec.name,
            "group": spec.group,
            "split": spec.split,
            "source_files": [str(path) for path in spec.paths],
            "source_file_count": len(spec.paths),
        }
    )
    return result


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _flatten_summary_rows(results: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in results:
        rows.append(
            {
                "group": item["group"],
                "dataset": item["dataset"],
                "split": item["split"],
                "total_rows": item["total_rows"],
                "unique_backbones": item["unique_backbones"],
                "ptm_spectrum_rows": item["ptm_spectrum_rows"],
                "ptm_spectrum_pct": item["ptm_spectrum_pct"],
                "ptm_event_total": item["ptm_event_total"],
                "unique_ptm_types": item["unique_ptm_types"],
                "source_file_count": item["source_file_count"],
            }
        )
    return rows


def _flatten_counter_rows(results: Sequence[Dict[str, object]], key: str, value_name: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in results:
        counter = item.get(key, {})
        for counter_key, counter_value in counter.items():
            rows.append(
                {
                    "group": item["group"],
                    "dataset": item["dataset"],
                    "split": item["split"],
                    key.removesuffix("_counts"): counter_key,
                    value_name: counter_value,
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit mini and pre-mini benchmark statistics.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_ROOT / "output" / "dataset_audit_20260318",
    )
    parser.add_argument(
        "--nce-bin-width",
        type=float,
        default=DEFAULT_NCE_BIN_WIDTH,
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Only audit mini datasets.",
    )
    args = parser.parse_args()

    specs = _dataset_specs(include_full=not args.skip_full)
    results: List[Dict[str, object]] = []
    for spec in specs:
        print(f"[audit] {spec.group} {spec.name} {spec.split} ({len(spec.paths)} files)")
        result = audit_one_dataset(spec, nce_bin_width=float(args.nce_bin_width))
        results.append(result)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    _write_json(output_root / "audit_results.json", {"results": results})
    _write_csv(output_root / "summary.csv", _flatten_summary_rows(results))
    _write_csv(output_root / "ptm_type_counts.csv", _flatten_counter_rows(results, "ptm_type_counts", "count"))
    _write_csv(output_root / "ptm_combination_counts.csv", _flatten_counter_rows(results, "ptm_combination_counts", "count"))
    _write_csv(output_root / "aa_counts.csv", _flatten_counter_rows(results, "aa_counts", "count"))
    _write_csv(output_root / "length_counts.csv", _flatten_counter_rows(results, "length_counts", "count"))
    _write_csv(output_root / "charge_counts.csv", _flatten_counter_rows(results, "charge_counts", "count"))
    _write_csv(output_root / "nce_exact_counts.csv", _flatten_counter_rows(results, "nce_exact_counts", "count"))
    _write_csv(output_root / "nce_bin_counts.csv", _flatten_counter_rows(results, "nce_bin_counts", "count"))

    print(f"[done] wrote audit outputs to {output_root}")


if __name__ == "__main__":
    main()
