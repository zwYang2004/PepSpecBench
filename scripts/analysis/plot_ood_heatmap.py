#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from trusted_figure_data import (
    FIG_DIR,
    MASSIVE_OOD,
    MASSIVE_RUN,
    MODEL_LABELS,
    MODELS,
    PROSPECT_OOD,
    PROSPECT_RUN,
    SPECIES_WITH_ID,
    draw_annotated_heatmap,
    finalize_figure,
    apply_paper_style,
    ensure_fig_dir,
    load_heatmap_matrix,
)


def plot_heatmap(matrix: np.ndarray, title: str, output_name: str) -> None:
    apply_paper_style((6.5, 4.35))
    fig, ax = plt.subplots()
    im = draw_annotated_heatmap(
        ax,
        matrix,
        xlabels=SPECIES_WITH_ID,
        ylabels=[MODEL_LABELS[key] for key in MODELS],
        title="",
    )
    ax.set_title(title, pad=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label("Median SA (Level-1)")
    cbar.outline.set_linewidth(0.6)
    finalize_figure(fig, output_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prospect", action="store_true", help="Generate the PROSPECT-trained OOD heatmap.")
    args = parser.parse_args()

    ensure_fig_dir()
    if args.prospect:
        matrix = load_heatmap_matrix(PROSPECT_RUN, PROSPECT_OOD)
        plot_heatmap(matrix, "Cross-Species OOD Generalization (PROSPECT trained)", "ood_heatmap_prospect.pdf")
    else:
        matrix = load_heatmap_matrix(MASSIVE_RUN, MASSIVE_OOD)
        plot_heatmap(matrix, "Cross-Species OOD Generalization (MassIVE-KB trained)", "ood_heatmap.pdf")


if __name__ == "__main__":
    main()
