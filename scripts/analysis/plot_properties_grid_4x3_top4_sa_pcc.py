#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from trusted_figure_data import apply_paper_style, ensure_fig_dir, finalize_figure, MODEL_LABELS, style_axis, set_panel_title

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "PROSPECT": ROOT / "output" / "benchmark_prospect_mini_234d_20260211_212742",
    "MassIVE-KB": ROOT / "output" / "latest_run_massive_kb_mini",
}
MODELS = ["prosit", "prosit_transformer", "predfull_torch", "alphapeptdeep"]

L_BINS = [(6, 10), (10, 15), (15, 20), (20, 25), (25, 31), (31, 41)]
L_LABELS = ["6-10", "10-15", "15-20", "20-25", "25-31", "31-41"]
C_BINS = [2, 3, 4]
NCE_EDGES = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
NCE_LABELS = ["0.15-0.2", "0.2-0.25", "0.25-0.3", "0.3-0.35", "0.35-0.4", "0.4-0.45"]


def _csv_path(dataset: str, model: str) -> Path:
    return RUNS[dataset] / model / "per_sample_test.csv"


def _load(dataset: str, model: str) -> pd.DataFrame:
    p = _csv_path(dataset, model)
    return pd.read_csv(p, usecols=["length", "precursor_charge", "input_collision_energy", "level1_sa", "level1_pcc"])


def _bootstrap_ci_median(x: np.ndarray, n_boot: int = 200, alpha: float = 0.95) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    med = float(np.median(x))
    if len(x) < 20:
        return med, np.nan, np.nan
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    meds = np.median(x[idx], axis=1)
    lo = float(np.quantile(meds, (1.0 - alpha) / 2.0))
    hi = float(np.quantile(meds, 1.0 - (1.0 - alpha) / 2.0))
    return med, lo, hi


def _binned_stats(df: pd.DataFrame, metric: str, bins: list[tuple[float, float]], col: str):
    meds, los, his, counts = [], [], [], []
    for lo, hi in bins:
        vals = df[(df[col] >= lo) & (df[col] < hi)][metric].to_numpy(dtype=float)
        counts.append(int(len(vals)))
        m, l, h = _bootstrap_ci_median(vals)
        meds.append(m)
        los.append(l)
        his.append(h)
    return np.array(meds), np.array(los), np.array(his), np.array(counts)


def _group_positions(n_groups: int, n_models: int, width: float = 0.16):
    out = []
    for i in range(n_groups):
        base = i + 1
        out.append([base + (j - (n_models - 1) / 2) * width for j in range(n_models)])
    return out


def _sample_for_violin(x: np.ndarray, max_n: int = 4000) -> np.ndarray:
    if len(x) <= max_n:
        return x
    rng = np.random.default_rng(123)
    sel = rng.choice(len(x), size=max_n, replace=False)
    return x[sel]


def _metric_ylim(data, metric: str):
    arr = []
    for ds in RUNS:
        for m in MODELS:
            arr.append(data[(ds, m)][metric].to_numpy(dtype=float))
    v = np.concatenate(arr)
    if metric == "level1_sa":
        lo = max(0.04, float(np.nanquantile(v, 0.01) - 0.02))
        hi = min(0.55, float(np.nanquantile(v, 0.99) + 0.02))
    else:
        lo = max(-0.05, float(np.nanquantile(v, 0.01) - 0.03))
        hi = min(1.00, float(np.nanquantile(v, 0.99) + 0.03))
    return lo, hi


def _plot_row(axA, axB, axC, data, ds: str, metric: str, y_min: float, y_max: float, colors):
    metric_label = "SA" if metric == "level1_sa" else "PCC"
    axis_label_size = plt.rcParams["axes.labelsize"] * 1.12

    # A) Length
    x = np.arange(len(L_LABELS)) * 0.88
    for m in MODELS:
        med, lo, hi, _ = _binned_stats(data[(ds, m)], metric, L_BINS, "length")
        axA.plot(x, med, marker="o", linewidth=2.0, color=colors[m])
        if np.isfinite(lo).any() and np.isfinite(hi).any():
            axA.fill_between(x, lo, hi, color=colors[m], alpha=0.14)
    axA.set_xticks(x)
    axA.set_xlim(x[0]-0.25, x[-1]+0.25)
    axA.set_xticklabels(L_LABELS, rotation=16, ha="right")
    axA.set_ylim(y_min, y_max)
    axA.set_xlabel("Peptide length", fontsize=axis_label_size)
    axA.set_ylabel(f"{metric_label} (median)", fontsize=axis_label_size)
    set_panel_title(axA, f"{ds}: A. Length ({metric_label})")
    style_axis(axA)

    # B) Charge
    pos_groups = _group_positions(len(C_BINS), len(MODELS), width=0.14)
    for i, z in enumerate(C_BINS):
        for j, m in enumerate(MODELS):
            p = pos_groups[i][j]
            vals = data[(ds, m)]
            vals = vals[vals["precursor_charge"] == z][metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) < 10:
                continue
            vals_plot = _sample_for_violin(vals)
            vio = axB.violinplot([vals_plot], positions=[p], widths=0.14, showmeans=False, showmedians=False, showextrema=False)
            for b in vio['bodies']:
                b.set_facecolor(colors[m])
                b.set_alpha(0.28)
                b.set_edgecolor(colors[m])
            q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
            axB.vlines(p, q1, q3, color=colors[m], linewidth=3.0, alpha=0.9)
            axB.hlines(med, p - 0.055, p + 0.055, color="#2f2f2f", linewidth=1.2)
    axB.set_xticks([1, 2, 3])
    axB.set_xlim(0.62, 3.38)
    axB.set_xticklabels(["z=2", "z=3", "z=4"])
    axB.set_ylim(y_min, y_max)
    axB.set_xlabel("Precursor charge", fontsize=axis_label_size)
    axB.set_ylabel(f"{metric_label} distribution", fontsize=axis_label_size)
    set_panel_title(axB, f"{ds}: B. Charge ({metric_label})")
    style_axis(axB)

    # C) NCE
    x = np.arange(len(NCE_LABELS)) * 0.88
    _, _, _, counts = _binned_stats(data[(ds, "prosit")], metric, list(zip(NCE_EDGES[:-1], NCE_EDGES[1:])), "input_collision_energy")
    ax_count = axC.twinx()
    ax_count.bar(x, counts, color="#C9CED6", alpha=0.35, width=0.75, zorder=0)
    ax_count.set_ylabel("N (bin)", color="#7A7F87")
    ax_count.tick_params(axis='y', colors="#7A7F87", labelsize=8)
    ax_count.grid(False)

    nonzero_bins = np.where(counts > 0)[0]
    if ds == "MassIVE-KB" and len(nonzero_bins) == 1:
        b = int(nonzero_bins[0])
        center = 0.0
        offsets = np.linspace(-0.24, 0.24, len(MODELS))
        for off, m in zip(offsets, MODELS):
            med, lo, hi, _ = _binned_stats(data[(ds, m)], metric, list(zip(NCE_EDGES[:-1], NCE_EDGES[1:])), "input_collision_energy")
            y = med[b]
            yerr_lo = y - lo[b] if np.isfinite(lo[b]) else 0.0
            yerr_hi = hi[b] - y if np.isfinite(hi[b]) else 0.0
            axC.errorbar([center + off], [y], yerr=[[yerr_lo], [yerr_hi]], fmt="o", color=colors[m], capsize=3, linewidth=1.4, markersize=5.5, zorder=3)
        axC.set_xticks([center])
        axC.set_xticklabels([NCE_LABELS[b]])
    else:
        for m in MODELS:
            med, lo, hi, _ = _binned_stats(data[(ds, m)], metric, list(zip(NCE_EDGES[:-1], NCE_EDGES[1:])), "input_collision_energy")
            mask = np.isfinite(med)
            if mask.sum() == 0:
                continue
            axC.plot(x[mask], med[mask], marker="o", linewidth=2.0, color=colors[m])
            if np.isfinite(lo).any() and np.isfinite(hi).any():
                axC.fill_between(x, lo, hi, color=colors[m], alpha=0.14)
        axC.set_xticks(x)
        axC.set_xlim(x[0]-0.25, x[-1]+0.25)
        axC.set_xticklabels(NCE_LABELS, rotation=18, ha="right")
    axC.set_ylim(y_min, y_max)
    axC.set_xlabel("Input NCE", fontsize=axis_label_size)
    axC.set_ylabel(f"{metric_label} (median)", fontsize=axis_label_size)
    set_panel_title(axC, f"{ds}: C. NCE ({metric_label})")
    style_axis(axC)


def main() -> None:
    ensure_fig_dir()
    colors = apply_paper_style((15.2, 14.6))
    # Increase readability for figure* rendering in paper.
    plt.rcParams["font.size"] = plt.rcParams["font.size"] * 1.42
    plt.rcParams["axes.labelsize"] = plt.rcParams["axes.labelsize"] * 1.38
    plt.rcParams["axes.titlesize"] = plt.rcParams["axes.titlesize"] * 1.36
    plt.rcParams["xtick.labelsize"] = plt.rcParams["xtick.labelsize"] * 1.30
    plt.rcParams["ytick.labelsize"] = plt.rcParams["ytick.labelsize"] * 1.30
    plt.rcParams["legend.fontsize"] = plt.rcParams["legend.fontsize"] * 1.30
    data = {(ds, m): _load(ds, m) for ds in RUNS for m in MODELS}

    sa_ylim = _metric_ylim(data, "level1_sa")
    pcc_ylim = _metric_ylim(data, "level1_pcc")

    fig, axes = plt.subplots(4, 3)
    _plot_row(axes[0, 0], axes[0, 1], axes[0, 2], data, "PROSPECT", "level1_sa", sa_ylim[0], sa_ylim[1], colors)
    _plot_row(axes[1, 0], axes[1, 1], axes[1, 2], data, "MassIVE-KB", "level1_sa", sa_ylim[0], sa_ylim[1], colors)
    _plot_row(axes[2, 0], axes[2, 1], axes[2, 2], data, "PROSPECT", "level1_pcc", pcc_ylim[0], pcc_ylim[1], colors)
    _plot_row(axes[3, 0], axes[3, 1], axes[3, 2], data, "MassIVE-KB", "level1_pcc", pcc_ylim[0], pcc_ylim[1], colors)

    # Replace long subplot titles with compact panel labels.
    panel_labels = [f"({chr(ord('a') + i)})" for i in range(12)]
    panel_title_size = plt.rcParams["axes.titlesize"] * 1.25
    for i, ax in enumerate(axes.flatten()):
        ax.set_title(panel_labels[i], pad=16, fontsize=panel_title_size, fontweight="bold")
    handles = []
    labels = []
    for m in MODELS:
        h, = axes[0, 0].plot([], [], color=colors[m], marker="o", linewidth=2.2)
        handles.append(h)
        labels.append(MODEL_LABELS[m])
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005))

    fig.subplots_adjust(left=0.062, right=0.994, bottom=0.090, top=0.984, wspace=0.22, hspace=0.52)
    finalize_figure(fig, "properties_grid_4x3_top4_sa_pcc.pdf", use_tight_layout=False)


if __name__ == "__main__":
    main()
