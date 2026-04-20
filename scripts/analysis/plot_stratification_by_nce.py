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
    load_nce_stratification,
    set_panel_title,
    style_axis,
)

MODELS = ["prosit", "prosit_transformer", "predfull_torch", "alphapeptdeep"]
NCE_LABELS = ["NCE 0.2-0.3", "NCE 0.3-0.4"]


def main() -> None:
    ensure_fig_dir()
    colors = apply_paper_style((5.2, 4.0))
    data = load_nce_stratification()
    fig, ax = plt.subplots()
    width = 0.2
    x = np.arange(len(NCE_LABELS))
    for idx, model in enumerate(MODELS):
        offset = (idx - 1.5) * width
        values = data[model]
        ax.bar(x + offset, values, width, label=MODEL_LABELS[model], color=colors[model])
    ax.set_xticks(x)
    ax.set_xticklabels(NCE_LABELS)
    ax.set_xlabel("Normalized collision energy")
    ax.set_ylabel("Level-1 Med. SA")
    set_panel_title(ax, "PROSPECT: Med. SA by NCE")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    ax.set_ylim(0, 0.32)
    style_axis(ax)
    fig.subplots_adjust(bottom=0.24)
    finalize_figure(fig, "stratification_by_nce.pdf", use_tight_layout=False)


if __name__ == "__main__":
    main()
