#!/usr/bin/env python3
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from trusted_figure_data import (
    FIG_DIR,
    MASSIVE_RUN,
    MODEL_LABELS,
    MODELS,
    PROSPECT_RUN,
    apply_paper_style,
    draw_annotated_heatmap,
    ensure_fig_dir,
    finalize_figure,
    load_eval_metrics,
    set_panel_title,
    style_axis,
)


def load_metric_matrix() -> tuple[dict[str, dict], dict[str, dict]]:
    massive = {model: load_eval_metrics(MASSIVE_RUN, model) or {} for model in MODELS}
    prospect = {model: load_eval_metrics(PROSPECT_RUN, model) or {} for model in MODELS}
    return massive, prospect


def plot_heatmap(massive: dict[str, dict], prospect: dict[str, dict]) -> None:
    apply_paper_style((4.2, 4.8))
    matrix = np.full((len(MODELS), 2), np.nan)
    for i, model in enumerate(MODELS):
        matrix[i, 0] = massive[model].get("level1_median_sa", np.nan)
        matrix[i, 1] = prospect[model].get("level1_median_sa", np.nan)
    fig, ax = plt.subplots()
    im = draw_annotated_heatmap(
        ax,
        matrix,
        xlabels=["MassIVE-KB", "PROSPECT"],
        ylabels=[MODEL_LABELS[model] for model in MODELS],
        title="",
    )
    set_panel_title(ax, "In-Domain Test: MassIVE-KB vs PROSPECT")
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label("Median SA (Level-1)")
    cbar.outline.set_linewidth(0.6)
    finalize_figure(fig, "benchmark_comparison_heatmap.pdf")


def plot_bars(massive: dict[str, dict], prospect: dict[str, dict]) -> None:
    colors = apply_paper_style((12.2, 4.0))
    fig, axes = plt.subplots(1, 3)
    x = np.arange(len(MODELS))
    width = 0.36
    metrics = [
        ("level1_median_sas", "Med. SAS", 0.4, 1.0),
        ("level1_median_sa", "Med. SA", 0.0, 0.52),
        ("level1_median_pcc", "Med. PCC", 0.0, 1.0),
    ]
    for ax, (metric, ylabel, ymin, ymax) in zip(axes, metrics):
        massive_vals = [float(massive[model].get(metric, np.nan)) for model in MODELS]
        prospect_vals = [float(prospect[model].get(metric, np.nan)) for model in MODELS]
        ax.bar(x - width / 2, massive_vals, width, label="MassIVE-KB", color=colors["massive"])
        ax.bar(x + width / 2, prospect_vals, width, label="PROSPECT", color=colors["prospect"])
        ax.set_ylabel(ylabel)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[model] for model in MODELS], rotation=30, ha="right")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
        style_axis(ax)
    fig.subplots_adjust(top=0.84, bottom=0.28)
    fig.suptitle("In-Domain Test Comparison", y=0.97)
    finalize_figure(fig, "benchmark_comparison_bars.pdf", use_tight_layout=False)


def main() -> None:
    ensure_fig_dir()
    massive, prospect = load_metric_matrix()
    plot_heatmap(massive, prospect)
    plot_bars(massive, prospect)


if __name__ == "__main__":
    main()
