#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from trusted_figure_data import FIG_DIR, apply_paper_style, ensure_fig_dir, finalize_figure, set_panel_title, style_axis

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description="Plot charge mode-collapse summary (supports smoke-dry).")
    p.add_argument("--input", type=Path, required=True, help="CSV from charge summary.")
    p.add_argument("--output", type=str, default="charge_mode_collapse_smoke-dry.pdf")
    p.add_argument("--threshold-col", type=str, default="high_sas_threshold")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    req = {"model", "high_sas_fraction"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    colors = apply_paper_style((6.8, 4.0))
    ensure_fig_dir()
    fig, ax = plt.subplots()

    sub = df.sort_values("high_sas_fraction", ascending=False)
    ax.bar(sub["model"], sub["high_sas_fraction"], color=colors["prospect"], alpha=0.9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("High SAS Fraction")
    ax.set_xlabel("Model")
    set_panel_title(ax, "Charge Mode Collapse (smoke-dry)")
    style_axis(ax)
    ax.tick_params(axis="x", rotation=20)

    threshold_text = None
    if args.threshold_col in sub.columns and not sub[args.threshold_col].isna().all():
        threshold_text = float(sub[args.threshold_col].iloc[0])
    if threshold_text is not None:
        ax.text(
            0.01,
            0.96,
            f"High SAS threshold = {threshold_text:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color=colors["reference"],
        )

    finalize_figure(fig, args.output)
    print(f"[OK] wrote {FIG_DIR / args.output}")


if __name__ == "__main__":
    main()
