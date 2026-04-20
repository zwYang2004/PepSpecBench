#!/usr/bin/env python3
"""Verify that MassIVE-KB official split ensures zero backbone overlap.

Reads data/reconstructed/massive_kb/all/{train,val,test}.parquet,
extracts naked_sequence (or strips PTMs from modified_sequence),
and checks that train/val/test have zero intersection at the backbone level.

Usage:
    python scripts/data_process/verify_massivekb_split_overlap.py

Exit code 0 if verified; 1 if overlap detected or files missing.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
MASSIVE_ALL = _ROOT / "data" / "reconstructed" / "massive_kb" / "all"


def _strip_mods(seq: str) -> str:
    """Strip [UNIMOD:x] and similar brackets to get naked sequence."""
    if not isinstance(seq, str):
        return ""
    return re.sub(r"\[[^\]]+\]", "", seq)


def _get_backbones(df: pd.DataFrame) -> set[str]:
    """Extract unique naked sequences (backbones) from dataframe."""
    if "naked_sequence" in df.columns:
        return set(df["naked_sequence"].dropna().astype(str).unique())
    seq_col = "modified_sequence" if "modified_sequence" in df.columns else "sequence"
    if seq_col not in df.columns:
        raise KeyError(f"No sequence column in {df.columns.tolist()}")
    return set(df[seq_col].apply(_strip_mods).dropna().astype(str).unique())


def main() -> int:
    print("MassIVE-KB Split Overlap Verification")
    print("=" * 50)
    print(f"Data dir: {MASSIVE_ALL}")

    if not MASSIVE_ALL.exists():
        print(f"ERROR: Directory not found: {MASSIVE_ALL}")
        return 1

    splits = {}
    for name in ("train", "val", "test"):
        p = MASSIVE_ALL / f"{name}.parquet"
        if not p.exists():
            print(f"ERROR: Missing {p}")
            return 1
        df = pd.read_parquet(p)
        splits[name] = _get_backbones(df)
        print(f"  {name}: {len(df):,} rows, {len(splits[name]):,} unique backbones")

    train_b, val_b, test_b = splits["train"], splits["val"], splits["test"]

    train_val = len(train_b & val_b)
    train_test = len(train_b & test_b)
    val_test = len(val_b & test_b)

    print()
    if train_val == 0 and train_test == 0 and val_test == 0:
        print("PASS: Zero backbone overlap between train/val/test.")
        print("You may add to the paper:")
        print('  "We empirically verified that the official MassIVE-KB split')
        print('   ensures strict zero-overlap at the backbone level."')
        return 0
    else:
        print("FAIL: Backbone overlap detected!")
        if train_val > 0:
            print(f"  train ∩ val: {train_val:,} backbones")
        if train_test > 0:
            print(f"  train ∩ test: {train_test:,} backbones")
        if val_test > 0:
            print(f"  val ∩ test: {val_test:,} backbones")
        return 1


if __name__ == "__main__":
    sys.exit(main())
