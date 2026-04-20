#!/usr/bin/env python3
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from trusted_figure_data import (
    FIG_DIR,
    MODEL_LABELS,
    apply_paper_style,
    ensure_fig_dir,
    finalize_figure,
    load_length_stratification,
    set_panel_title,
    style_axis,
)

MODELS = ["prosit", "prosit_transformer", "predfull_torch", "alphapeptdeep"]
LENGTH_LABELS = ["[6,10)", "[10,15)", "[15,20)", "[20,25)", "[25,31)", "[31,41)"]


def main() -> None:
    ensure_fig_dir()
    colors = apply_paper_style((10.4, 4.2))
    data = load_length_stratification()
    fig, axes = plt.subplots(1, 2, sharey=True)
    width = 0.2
    x = np.arange(len(LENGTH_LABELS))
    for ax, dataset in zip(axes, ["MassIVE-KB", "PROSPECT"]):
        for idx, model in enumerate(MODELS):
            offset = (idx - 1.5) * width
            ax.bar(x + offset, data[dataset][model], width, label=MODEL_LABELS[model], color=colors[model])
        ax.set_xticks(x)
        ax.set_xticklabels(LENGTH_LABELS)
        ax.set_xlabel("Peptide length")
        set_panel_title(ax, dataset)
        style_axis(ax)
    axes[0].set_ylabel("Med. SA (shared canonical space)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.18), ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.24)
    finalize_figure(fig, "stratification_by_length.pdf", use_tight_layout=False)


if __name__ == "__main__":
    main()
