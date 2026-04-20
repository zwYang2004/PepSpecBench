#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"
SNAPSHOT_DIR = ROOT / "scripts" / "analysis" / "real_figure_data_snapshot"

MASSIVE_RUN = ROOT / "output" / "latest_run_massive_kb_mini"
PROSPECT_RUN_MAIN = ROOT / "output" / "benchmark_prospect_mini_234d_20260211_212742"
PROSPECT_RUN_UNISPEC_FINAL = ROOT / "output" / "benchmark_prospect_mini_234d_20260308_202544"
PROSPECT_RUN = PROSPECT_RUN_MAIN
MASSIVE_OOD = ROOT / "output"
PROSPECT_OOD_MAIN = PROSPECT_RUN_MAIN / "ood_eval"
PROSPECT_OOD_UNISPEC_FINAL = PROSPECT_RUN_UNISPEC_FINAL / "ood_eval"
PROSPECT_OOD = PROSPECT_OOD_MAIN
POSTHOC_DIR = ROOT / "output" / "posthoc"

MODELS = [
    "prosit",
    "prosit_transformer",
    "predfull_torch",
    "unispec",
    "fastspel",
    "alphapeptdeep",
]

MODEL_LABELS = {
    "prosit": "Prosit",
    "prosit_transformer": "Prosit Trans.",
    "predfull_torch": "PredFull",
    "unispec": "UniSpec",
    "fastspel": "FastSpel",
    "alphapeptdeep": "AlphaPeptDeep",
}

SPECIES = ["Yeast", "E. coli", "C. elegans", "A. thaliana"]
SPECIES_WITH_ID = ["Human (ID)", *SPECIES]
FILE_TO_SPECIES = {
    "Yeast": "Yeast",
    "Ecoli": "E. coli",
    "coli_with_seq": "E. coli",
    "c_elegans_with_seq": "C. elegans",
    "a_lia_with_seq": "A. thaliana",
}


def _load_neurips_style_module():
    style_path = ROOT / "style.py"
    if not style_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("release_style", style_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_paper_style(figsize: tuple[float, float]) -> dict[str, str]:
    style = _load_neurips_style_module()
    mpl.rcParams.update(mpl.rcParamsDefault)
    if style is not None and hasattr(style, "get_scaled_rcparams"):
        scaled = dict(style.get_scaled_rcparams())
    else:
        scaled = {}
    for key in ["font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize"]:
        if key in scaled:
            scaled[key] = scaled[key] * 1.28
    if scaled:
        mpl.rcParams.update(scaled)
    else:
        mpl.rcParams.update(
            {
                "font.size": 10.0,
                "axes.labelsize": 10.5,
                "axes.titlesize": 11.0,
                "xtick.labelsize": 9.5,
                "ytick.labelsize": 9.5,
                "legend.fontsize": 9.5,
                "legend.frameon": False,
            }
        )
    mpl.rcParams["figure.figsize"] = figsize
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.pad_inches"] = 0.05
    return {
        # Dataset semantics
        "massive": "#355C9A",
        "prospect": "#D17C28",
        # Model semantics
        "prosit": "#355C9A",
        "prosit_transformer": "#D17C28",
        "predfull_torch": "#4E9F6D",
        "alphapeptdeep": "#8C6BB1",
        "unispec": "#8B6F47",
        "fastspel": "#3E8E8C",
        # Ablation semantics
        "backbone": "#355C9A",
        "sequence": "#C98E2B",
        "random": "#7A7F87",
        # Neutral accents
        "grid": "#D9DEE7",
        "reference": "#7A7F87",
    }


def get_heatmap_cmap() -> LinearSegmentedColormap:
    # Lower SA is better; use a restrained blue-green academic colormap.
    return LinearSegmentedColormap.from_list(
        "neurips_bluegreen",
        ["#F4F6F8", "#CFE4E8", "#8EBFCA", "#4E8EA0", "#225B73"],
    )


def style_axis(ax, *, grid_axis: str | None = "y") -> None:
    ax.set_axisbelow(True)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, alpha=0.9, linewidth=0.8, color="#D0D5DD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A8B0BC")
    ax.spines["bottom"].set_color("#A8B0BC")
    ax.tick_params(colors="#38424D")


def set_panel_title(ax, title: str) -> None:
    ax.set_title(title, pad=12)


def finalize_figure(fig, output_name: str, *, use_tight_layout: bool = True) -> None:
    if use_tight_layout:
        fig.tight_layout()
    fig.savefig(FIG_DIR / output_name)
    plt.close(fig)


def draw_annotated_heatmap(
    ax,
    matrix: np.ndarray,
    *,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    cmap=None,
    vmin: float = 0.08,
    vmax: float = 0.52,
):
    if cmap is None:
        cmap = get_heatmap_cmap()
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=24, ha="right")
    ax.set_yticklabels(ylabels)
    ax.set_title(title, pad=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                color = "white" if value > (vmin + vmax) / 2 else "#22313F"
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=color, fontsize=7.6)
    return im


def ensure_fig_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def load_eval_metrics(run_dir: Path, model_key: str) -> dict | None:
    candidates = [
        run_dir / model_key / "eval_test.json",
        run_dir / model_key / f"eval_{model_key}_test.json",
        run_dir / model_key / "eval_prosit_transformer_test.json",
    ]
    if model_key == "fastspel":
        candidates.append(run_dir / model_key / "predictions" / "summary.json")
    for path in candidates:
        if not path.exists():
            continue
        payload = _read_json(path)
        if model_key == "fastspel":
            ev = payload.get("evaluation", {})
            return ev.get("unified", ev)
        unified = payload.get("unified")
        return unified if isinstance(unified, dict) else payload
    return None


def load_id_sa(run_dir: Path, model_key: str) -> float | None:
    if model_key == "alphapeptdeep" and run_dir == MASSIVE_RUN:
        path = ROOT / "output" / "alphapeptdeep_human_id.json"
        if path.exists():
            return _read_json(path).get("level1_median_sa")
    metrics = load_eval_metrics(run_dir, model_key)
    if not metrics:
        return None
    value = metrics.get("level1_median_sa")
    return None if value is None else float(value)


def load_ood_by_species(ood_dir: Path, model_key: str) -> dict[str, float]:
    path = ood_dir / f"ood_results_{model_key}.json"
    if not path.exists():
        return {}
    payload = _read_json(path)
    by_species: dict[str, list[float]] = {}
    for item in payload.get("files", []):
        stem = Path(item.get("file", "")).stem
        if stem in ("HeLa_trypsin", "homo_sapiens_with_seq"):
            continue
        unified = item.get("unified", {})
        sa = unified.get("level1_median_sa")
        species = FILE_TO_SPECIES.get(stem)
        if species is None or sa is None:
            continue
        by_species.setdefault(species, []).append(float(sa))
    return {key: float(np.median(vals)) for key, vals in by_species.items()}


def load_heatmap_matrix(run_dir: Path, ood_dir: Path) -> np.ndarray:
    matrix = np.full((len(MODELS), len(SPECIES_WITH_ID)), np.nan)
    for row, model_key in enumerate(MODELS):
        model_run_dir = run_dir
        model_ood_dir = ood_dir
        if run_dir == PROSPECT_RUN_MAIN and model_key == "unispec":
            model_run_dir = PROSPECT_RUN_UNISPEC_FINAL
            model_ood_dir = PROSPECT_OOD_UNISPEC_FINAL
        id_sa = load_id_sa(model_run_dir, model_key)
        if id_sa is not None:
            matrix[row, 0] = id_sa
        ood = load_ood_by_species(model_ood_dir, model_key)
        for col, species in enumerate(SPECIES, start=1):
            if species in ood:
                matrix[row, col] = ood[species]
    return matrix


def load_posthoc_json(run_name: str) -> dict:
    return _read_json(POSTHOC_DIR / f"posthoc_{run_name}.json")


def load_length_stratification() -> dict[str, dict[str, list[float]]]:
    bins = ["[6,10)", "[10,15)", "[15,20)", "[20,25)", "[25,31)", "[31,41)"]
    runs = {
        "MassIVE-KB": load_posthoc_json("latest_run_massive_kb_mini"),
        "PROSPECT": load_posthoc_json("benchmark_prospect_mini_234d_20260211_212742"),
    }
    out: dict[str, dict[str, list[float]]] = {}
    for label, payload in runs.items():
        out[label] = {}
        for model_key in ["prosit", "prosit_transformer", "predfull_torch", "alphapeptdeep"]:
            stratified = payload["models"][model_key]["subsets"]["test"]["stratified"]["by_length"]
            out[label][model_key] = [float(stratified.get(bin_key, {}).get("level1_sa_median", np.nan)) for bin_key in bins]
    return out


def load_nce_stratification() -> dict[str, list[float]]:
    # Use normalized input_collision_energy when available.
    # This restores models whose raw collision_energy is unavailable or stored on a different scale.
    bin_specs = [
        ("[0.2,0.3)", 0.2, 0.3),
        ("[0.3,0.4)", 0.3, 0.4),
    ]
    out: dict[str, list[float]] = {}
    for model_key in ["prosit", "prosit_transformer", "predfull_torch", "alphapeptdeep"]:
        csv_path = PROSPECT_RUN / model_key / "per_sample_test.csv"
        bins: list[list[float]] = [[] for _ in bin_specs]
        with open(csv_path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                ce_raw = row.get("input_collision_energy") or row.get("collision_energy") or ""
                sa_raw = row.get("level1_sa") or ""
                if ce_raw in ("", "nan", "NaN", "None") or sa_raw in ("", "nan", "NaN", "None"):
                    continue
                ce = float(ce_raw)
                if ce > 1.5:
                    continue
                sa = float(sa_raw)
                for idx, (_, lo, hi) in enumerate(bin_specs):
                    if lo <= ce < hi:
                        bins[idx].append(sa)
                        break
        out[model_key] = [float(np.median(vals)) if vals else np.nan for vals in bins]
    return out
