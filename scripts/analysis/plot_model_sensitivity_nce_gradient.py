#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_ORDER = [
    "prosit",
    "prosit_transformer",
    "predfull_torch",
    "alphapeptdeep",
    "unispec",
    "fastspel",
]

MODEL_LABELS = {
    "prosit": "Prosit",
    "prosit_transformer": "Prosit Transformer",
    "predfull_torch": "PredFull",
    "alphapeptdeep": "AlphaPeptDeep",
    "unispec": "UniSpec",
    "fastspel": "FastSpel",
}

MODEL_COLORS = {
    "prosit": "#355C9A",
    "prosit_transformer": "#D17C28",
    "predfull_torch": "#4E9F6D",
    "alphapeptdeep": "#8C6BB1",
    "unispec": "#8B6F47",
    "fastspel": "#3E8E8C",
}

MODEL_MARKERS = {
    "prosit": "o",
    "prosit_transformer": "s",
    "predfull_torch": "^",
    "alphapeptdeep": "D",
    "unispec": "v",
    "fastspel": "P",
}


def _style_axis(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D0D5DD", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A8B0BC")
    ax.spines["bottom"].set_color("#A8B0BC")
    ax.tick_params(colors="#38424D")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot clean A1 NCE gradient figure without legend-line overlap.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {"model", "nce_override_raw", "median_sa"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    handles = []
    for model in MODEL_ORDER:
        sub = df[df["model"].astype(str) == model].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("nce_override_raw")
        line, = ax.plot(
            sub["nce_override_raw"],
            sub["median_sa"],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linewidth=2.0,
            markersize=6.0,
            label=MODEL_LABELS[model],
        )
        handles.append(line)

    ax.axvline(30, color="#7A7F87", linestyle="--", linewidth=1.4)
    ax.text(30.3, ax.get_ylim()[1] * 0.98, "True NCE = 30", color="#7A7F87", va="top", ha="left")
    ax.set_xlabel("Input NCE Override")
    ax.set_ylabel("Median SA")
    ax.set_xticks([20, 25, 30, 35, 40])
    _style_axis(ax)

    fig.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
