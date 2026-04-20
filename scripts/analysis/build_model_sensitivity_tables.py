#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "New" / "tables" / "model_sensitivity_smoke-dry_table.csv"


def _load_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["source_csv"] = str(p)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Build model sensitivity summary table (supports smoke-dry inputs).")
    p.add_argument("--inputs", type=Path, nargs="+", required=True, help="Input summary CSV files.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--label", type=str, default="smoke-dry")
    args = p.parse_args()

    df = _load_csvs(args.inputs)
    if df.empty:
        raise RuntimeError("No input rows found.")

    df["result_label"] = args.label
    # Keep compact columns for paper handoff.
    keep = [c for c in [
        "result_label",
        "run_label",
        "experiment",
        "model",
        "target_true_nce",
        "n_target_subset_all",
        "nce_override_raw",
        "median_sa",
        "median_sas",
        "median_pcc",
        "n_eval_used",
        "source_csv",
    ] if c in df.columns]
    out = df[keep].copy()
    out = out.sort_values([c for c in ["model", "nce_override_raw"] if c in out.columns])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
