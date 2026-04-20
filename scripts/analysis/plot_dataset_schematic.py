#!/usr/bin/env python3
from __future__ import annotations

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt

from trusted_figure_data import FIG_DIR, apply_paper_style, ensure_fig_dir


def _box(ax, xy, width, height, text, fc, ec="#2f2f2f", fontsize=10, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.03",
        linewidth=1.35,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
    )
    return patch


def _arrow(ax, start, end, color="#4a4a4a"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.3,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def main() -> None:
    ensure_fig_dir()
    colors = apply_paper_style((11.8, 6.2))

    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    c_prospect = colors["prospect"]
    c_massive = colors["massive"]
    c_ood = "#dcefd8"
    c_filter = "#f6e7b8"
    c_mini = "#e8e1f3"
    c_note = "#f4f4f4"
    c_scope = "#ebf1fb"

    ax.text(0.05, 0.91, "Data Sources", fontsize=12, weight="bold", ha="left", va="center")
    ax.text(0.39, 0.91, "Unified Processing", fontsize=12, weight="bold", ha="left", va="center")
    ax.text(0.72, 0.91, "Released Evaluation Assets", fontsize=12, weight="bold", ha="left", va="center")

    _box(ax, (0.05, 0.70), 0.23, 0.14, "PROSPECT\nsynthetic benchmark\nPTM-rich branch", c_prospect, fontsize=11, weight="bold")
    _box(ax, (0.05, 0.49), 0.23, 0.14, "MassIVE-KB\ncommunity-scale human\nreal-world branch", c_massive, fontsize=11, weight="bold")
    _box(ax, (0.05, 0.16), 0.23, 0.22, "OOD reference sources\n7 subsets / 5 species\n\nH. sapiens, E. coli,\nC. elegans,\nA. thaliana, yeast", c_ood, fontsize=10, weight="bold")

    _box(
        ax,
        (0.39, 0.66),
        0.22,
        0.16,
        "Common physical scope\n\nLength 6-40\nCharge 1-6",
        c_scope,
        fontsize=10,
        weight="bold",
    )
    _box(
        ax,
        (0.39, 0.44),
        0.22,
        0.16,
        "Semantic alignment\n\nPTM normalization\nUNIMOD 1 / 4 / 35\ncanonical parquet schema",
        c_filter,
        fontsize=10,
        weight="bold",
    )
    _box(
        ax,
        (0.39, 0.14),
        0.22,
        0.18,
        "Split-aware reconstruction\n\nbackbone-aware split policy\ndeterministic ranking\nOOD top-20k per source",
        c_filter,
        fontsize=10,
        weight="bold",
    )

    _box(
        ax,
        (0.72, 0.66),
        0.22,
        0.15,
        "PepSpecBench-Mini\nin-domain benchmark\n\nPROSPECT-M\n500k / 50k / 50k\nbalanced PTM",
        c_mini,
        fontsize=10,
        weight="bold",
    )
    _box(
        ax,
        (0.72, 0.45),
        0.22,
        0.15,
        "MassIVE-KB-M\nin-domain benchmark\n\n500k / 50k / 50k\nnatural PTM profile",
        c_mini,
        fontsize=10,
        weight="bold",
    )
    _box(
        ax,
        (0.72, 0.17),
        0.22,
        0.16,
        "PepSpecBench-OOD\n7 mini subsets\n20k each\ncross-species evaluation",
        c_ood,
        fontsize=10,
        weight="bold",
    )

    _box(
        ax,
        (0.36, 0.86),
        0.30,
        0.08,
        "Dual-source in-domain benchmark with a matched physical scope, plus a separate cross-species OOD branch",
        c_note,
        fontsize=10,
        weight="bold",
    )

    _arrow(ax, (0.28, 0.77), (0.39, 0.74))
    _arrow(ax, (0.28, 0.56), (0.39, 0.74))
    _arrow(ax, (0.28, 0.27), (0.39, 0.23))
    _arrow(ax, (0.50, 0.66), (0.50, 0.60))
    _arrow(ax, (0.50, 0.44), (0.50, 0.32))
    _arrow(ax, (0.61, 0.73), (0.72, 0.73))
    _arrow(ax, (0.61, 0.52), (0.72, 0.52))
    _arrow(ax, (0.61, 0.23), (0.72, 0.25))

    ax.text(0.83, 0.63, "benchmark training + ID test", ha="center", va="center", fontsize=9, color="#444444")
    ax.text(0.83, 0.26, "OOD evaluation only", ha="center", va="center", fontsize=9, color="#444444")

    fig.savefig(FIG_DIR / "dataset_schematic.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
