#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from trusted_figure_data import FIG_DIR, apply_paper_style, ensure_fig_dir, finalize_figure, set_panel_title, style_axis


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "output" / "dataset_audit_20260318"


def _load_csv(name: str) -> pd.DataFrame:
    path = AUDIT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing audit CSV: {path}")
    return pd.read_csv(path)


def _dataset_color_map(colors: dict[str, str]) -> dict[str, str]:
    return {
        "PROSPECT-M": colors["prospect"],
        "MassIVE-KB-M": colors["massive"],
    }


def _in_domain(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["group"] == "mini_in_domain"].copy()


def _dataset_split_order(dataset: str) -> int:
    return {"PROSPECT-M": 0, "MassIVE-KB-M": 1}.get(dataset, 99)


def plot_overview() -> None:
    colors = apply_paper_style((10.2, 4.1))
    color_map = _dataset_color_map(colors)
    summary = _in_domain(_load_csv("summary.csv"))
    summary["unmod_pct"] = 100.0 * (1.0 - summary["ptm_spectrum_pct"])
    summary["ptm_pct"] = 100.0 * summary["ptm_spectrum_pct"]
    summary["spectra_k"] = summary["total_rows"] / 1000.0

    fig, axes = plt.subplots(1, 2)
    x = np.arange(3)
    width = 0.34
    split_order = ["train", "val", "test"]
    dataset_order = ["PROSPECT-M", "MassIVE-KB-M"]

    for offset, dataset in zip((-width / 2, width / 2), dataset_order):
        sub = summary[summary["dataset"] == dataset].set_index("split").loc[split_order]
        axes[0].bar(
            x + offset,
            sub["spectra_k"].to_numpy(),
            width,
            label=dataset,
            color=color_map[dataset],
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Train", "Val", "Test"])
    axes[0].set_ylabel("Spectra (k)")
    set_panel_title(axes[0], "Split Sizes")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    style_axis(axes[0], grid_axis="y")

    for offset, dataset in zip((-width / 2, width / 2), dataset_order):
        sub = summary[summary["dataset"] == dataset].set_index("split").loc[split_order]
        axes[1].bar(x + offset, sub["unmod_pct"], width, color=color_map[dataset], alpha=0.9)
        axes[1].bar(
            x + offset,
            sub["ptm_pct"],
            width,
            bottom=sub["unmod_pct"],
            color=color_map[dataset],
            alpha=0.35,
        )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Train", "Val", "Test"])
    axes[1].set_ylabel("Percentage (%)")
    set_panel_title(axes[1], "Unmodified vs PTM")
    axes[1].legend(
        [
            plt.Rectangle((0, 0), 1, 1, color=color_map["PROSPECT-M"], alpha=0.9),
            plt.Rectangle((0, 0), 1, 1, color=color_map["PROSPECT-M"], alpha=0.35),
            plt.Rectangle((0, 0), 1, 1, color=color_map["MassIVE-KB-M"], alpha=0.9),
            plt.Rectangle((0, 0), 1, 1, color=color_map["MassIVE-KB-M"], alpha=0.35),
        ],
        ["PROSPECT Unmod", "PROSPECT PTM", "MassIVE Unmod", "MassIVE PTM"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        frameon=False,
    )
    style_axis(axes[1], grid_axis="y")
    fig.subplots_adjust(bottom=0.27)
    finalize_figure(fig, "dataset_overview.pdf", use_tight_layout=False)


def plot_distribution_compare() -> None:
    colors = apply_paper_style((12.4, 3.9))
    color_map = _dataset_color_map(colors)
    length_df = _in_domain(_load_csv("length_counts.csv"))
    charge_df = _in_domain(_load_csv("charge_counts.csv"))
    nce_df = _in_domain(_load_csv("nce_exact_counts.csv"))

    fig, axes = plt.subplots(1, 3)
    dataset_order = ["PROSPECT-M", "MassIVE-KB-M"]

    for dataset in dataset_order:
        sub = length_df[length_df["dataset"] == dataset].copy()
        sub["length"] = pd.to_numeric(sub["length"], errors="coerce")
        sub = sub.groupby("length", as_index=False)["count"].sum()
        sns.kdeplot(
            data=sub,
            x="length",
            weights="count",
            bw_adjust=0.9,
            fill=False,
            common_norm=False,
            ax=axes[0],
            label=dataset,
            color=color_map[dataset],
            linewidth=2.4,
        )
    axes[0].set_xlabel("Peptide Length")
    axes[0].set_ylabel("Density")
    set_panel_title(axes[0], "Length")
    style_axis(axes[0], grid_axis="y")
    axes[0].legend(loc="upper right", frameon=False)

    charge_plot = charge_df.groupby(["dataset", "charge"], as_index=False)["count"].sum()
    charge_plot["fraction"] = charge_plot.groupby("dataset")["count"].transform(lambda x: x / x.sum())
    sns.barplot(
        data=charge_plot,
        x="charge",
        y="fraction",
        hue="dataset",
        palette=color_map,
        ax=axes[1],
    )
    axes[1].set_xlabel("Precursor Charge")
    axes[1].set_ylabel("Fraction")
    set_panel_title(axes[1], "Charge")
    style_axis(axes[1], grid_axis="y")
    axes[1].legend(loc="upper right", frameon=False, title=None)

    for dataset in dataset_order:
        sub = nce_df[nce_df["dataset"] == dataset].copy()
        sub["nce_exact"] = pd.to_numeric(sub["nce_exact"], errors="coerce")
        sub = sub.groupby("nce_exact", as_index=False)["count"].sum()
        axes[2].hist(
            sub["nce_exact"].to_numpy(),
            bins=np.arange(19.5, 50.6, 1.0),
            weights=sub["count"].to_numpy(),
            density=True,
            alpha=0.55,
            color=color_map[dataset],
            label=dataset,
        )
    axes[2].set_xlabel("NCE")
    axes[2].set_ylabel("Density")
    set_panel_title(axes[2], "NCE")
    style_axis(axes[2], grid_axis="y")
    axes[2].legend(loc="upper right", frameon=False)

    fig.subplots_adjust(bottom=0.2, wspace=0.28)
    fig.savefig(FIG_DIR / "dataset_length_charge.pdf")
    finalize_figure(fig, "dataset_distribution_compare.pdf", use_tight_layout=False)


def plot_ptm_breakdown() -> None:
    colors = apply_paper_style((7.0, 4.1))
    ptm = _in_domain(_load_csv("ptm_type_counts.csv"))
    train_summary = _in_domain(_load_csv("summary.csv"))
    train_summary = train_summary[train_summary["split"] == "train"][["dataset", "total_rows"]]
    ptm = ptm[ptm["split"] == "train"].merge(train_summary, on="dataset", how="left")
    ptm["pct_of_split"] = 100.0 * ptm["count"] / ptm["total_rows"]

    order = ["UNIMOD:35_Oxidation", "UNIMOD:4_CAM", "UNIMOD:1_Acetyl"]
    pretty = {
        "UNIMOD:35_Oxidation": "Ox (UNIMOD:35)",
        "UNIMOD:4_CAM": "CAM (UNIMOD:4)",
        "UNIMOD:1_Acetyl": "Ace (UNIMOD:1)",
    }
    ptm["ptm_label"] = ptm["ptm_type"].map(pretty)

    fig, ax = plt.subplots()
    sns.barplot(
        data=ptm,
        x="dataset",
        y="pct_of_split",
        hue="ptm_label",
        hue_order=[pretty[x] for x in order],
        palette=[colors["prospect"], colors["massive"], colors["predfull_torch"]],
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Percent of spectra")
    set_panel_title(ax, "PTM Type Breakdown (Train)")
    style_axis(ax, grid_axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, title=None)
    fig.subplots_adjust(bottom=0.24)
    finalize_figure(fig, "dataset_ptm_breakdown.pdf", use_tight_layout=False)


def main() -> None:
    ensure_fig_dir()
    plot_overview()
    plot_distribution_compare()
    plot_ptm_breakdown()


if __name__ == "__main__":
    main()
