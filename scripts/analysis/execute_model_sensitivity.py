#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUN_BENCHMARK = ROOT / "scripts" / "run_benchmark.py"
DEFAULT_OUT = ROOT / "output" / "posthoc" / "sensitivity"


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(obj: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=False)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _find_eval_json(model_dir: Path, model: str) -> Path:
    candidates = [
        model_dir / "eval_test.json",
        model_dir / f"eval_{model}_test.json",
        model_dir / "eval_prosit_transformer_test.json",
        model_dir / "predictions" / "summary.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    wildcard = sorted(model_dir.glob("eval_*.json"))
    if wildcard:
        return wildcard[0]
    raise FileNotFoundError(f"Cannot find eval file for model={model} under {model_dir}")


def _extract_unified_metrics(eval_json: Path, model: str) -> dict[str, float]:
    payload = _read_json(eval_json)
    if model == "fastspel":
        uni = payload.get("evaluation", {}).get("unified", payload.get("evaluation", {}))
    else:
        uni = payload.get("unified", payload)
    return {
        "median_sa": float(uni.get("level1_median_sa")),
        "median_sas": float(uni.get("level1_median_sas")),
        "median_pcc": float(uni.get("level1_median_pcc")),
        "n": float(uni.get("n", 0)),
    }


def _normalize_nce(raw_nce: float) -> float:
    # run_benchmark override_nce is normalized to [0,1]
    return float(raw_nce) / 100.0


def _copy_model_checkpoint(src_model_dir: Path, dst_model_dir: Path, model: str) -> None:
    _ensure_dir(dst_model_dir)
    copied = False
    patterns = {
        "prosit": ["best.ckpt", "last.ckpt"],
        "prosit_transformer": ["best.ckpt", "last.ckpt"],
        "predfull_torch": ["predfull_torch.pt", "model.pt"],
        "alphapeptdeep": ["alphapeptdeep_best.pt", "alphapeptdeep.pt"],
        "unispec": ["unispec.pt", "best.ckpt", "last.ckpt"],
        "fastspel": ["checkpoints"],
    }
    for name in patterns.get(model, []):
        src = src_model_dir / name
        dst = dst_model_dir / name
        if not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        copied = True
    if not copied:
        # fallback, keep robust for unknown runner expectations
        shutil.copytree(src_model_dir, dst_model_dir, dirs_exist_ok=True)


def _detect_nce_col(df: pd.DataFrame) -> str:
    for col in ("collision_energy", "nce", "orig_collision_energy"):
        if col in df.columns:
            return col
    raise KeyError("No collision-energy column found in parquet.")


def _detect_charge_col(df: pd.DataFrame) -> str:
    for col in ("precursor_charge", "charge", "z"):
        if col in df.columns:
            return col
    raise KeyError("No precursor-charge column found in parquet.")


@dataclass
class RunContext:
    source_run_dir: Path
    dataset_name: str
    source_config: Path
    out_root: Path
    run_tag: str
    conda_env: str


def _build_run_context(source_run_dir: Path, out_root: Path, run_tag: str, conda_env: str) -> RunContext:
    bench = _read_json(source_run_dir / "benchmark_summary.json")
    dataset_name = str(bench["dataset"])
    source_config = source_run_dir / "config_snapshot.yaml"
    if not source_config.exists():
        raise FileNotFoundError(f"Missing config snapshot: {source_config}")
    return RunContext(
        source_run_dir=source_run_dir,
        dataset_name=dataset_name,
        source_config=source_config,
        out_root=out_root,
        run_tag=run_tag,
        conda_env=conda_env,
    )


def _build_filtered_test_parquet(cfg: dict[str, Any], dataset_name: str, out_dir: Path, target_nce: int) -> tuple[Path, int]:
    ds = cfg["datasets"][dataset_name]
    test_path = Path(ds["test"]["path"])
    df = pd.read_parquet(test_path)
    ce_col = _detect_nce_col(df)
    sub = df[pd.to_numeric(df[ce_col], errors="coerce") == float(target_nce)].copy()
    out = out_dir / f"fullrun_prospect_nce{target_nce}_subset.parquet"
    sub.to_parquet(out, index=False)
    return out, int(len(sub))


def _build_filtered_charge_parquet(
    cfg: dict[str, Any],
    dataset_name: str,
    out_dir: Path,
    charge_from: int,
) -> tuple[Path, int]:
    ds = cfg["datasets"][dataset_name]
    test_path = Path(ds["test"]["path"])
    df = pd.read_parquet(test_path)
    ch_col = _detect_charge_col(df)
    sub = df[pd.to_numeric(df[ch_col], errors="coerce") == float(charge_from)].copy()
    out = out_dir / f"fullrun_charge{charge_from}_subset.parquet"
    sub.to_parquet(out, index=False)
    return out, int(len(sub))


def _run_eval_once(
    model: str,
    config_path: Path,
    dataset_name: str,
    run_dir: Path,
    conda_env: str,
    override_nce_norm: float | None = None,
    override_charge: int | None = None,
) -> Path:
    _ensure_dir(run_dir)
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "python",
        str(RUN_BENCHMARK),
        "--config",
        str(config_path),
        "--dataset",
        dataset_name,
        "--models",
        model,
        "--run-dir",
        str(run_dir),
        "--eval-only",
    ]
    if override_nce_norm is not None:
        cmd += ["--override-nce", str(override_nce_norm)]
    if override_charge is not None:
        cmd += ["--override-charge", str(override_charge)]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return _find_eval_json(run_dir / model, model)


def run_a1_gradient_full(ctx: RunContext, model: str, target_nce: int, nce_grid: list[int]) -> tuple[Path, Path]:
    base_cfg = _read_yaml(ctx.source_config)
    exp_root = ctx.out_root / f"{ctx.run_tag}_a1_gradient_{model}_{_ts()}"
    _ensure_dir(exp_root)

    # Isolated run tree to avoid touching frozen source run.
    src_model_dir = ctx.source_run_dir / model
    dst_model_dir = exp_root / "base_run" / model
    _copy_model_checkpoint(src_model_dir, dst_model_dir, model)

    subset_path, subset_n = _build_filtered_test_parquet(base_cfg, ctx.dataset_name, exp_root, target_nce)
    cfg = _read_yaml(ctx.source_config)
    cfg["datasets"][ctx.dataset_name]["test"]["path"] = str(subset_path)
    if "subsets" in cfg["datasets"][ctx.dataset_name]["test"]:
        cfg["datasets"][ctx.dataset_name]["test"]["subsets"]["all"]["path"] = str(subset_path)
    cfg_path = exp_root / "fullrun_config_a1.yaml"
    _write_yaml(cfg, cfg_path)

    rows: list[dict[str, Any]] = []
    for raw_nce in nce_grid:
        norm_nce = _normalize_nce(raw_nce)
        run_dir = exp_root / f"eval_nce{raw_nce}"
        _copy_model_checkpoint(src_model_dir, run_dir / model, model)
        eval_json = _run_eval_once(
            model=model,
            config_path=cfg_path,
            dataset_name=ctx.dataset_name,
            run_dir=run_dir,
            conda_env=ctx.conda_env,
            override_nce_norm=norm_nce,
        )
        m = _extract_unified_metrics(eval_json, model)
        rows.append(
            {
                "run_label": ctx.run_tag,
                "experiment": "A1_PROSPECT_NCE_GRADIENT",
                "model": model,
                "target_true_nce": target_nce,
                "n_target_subset_all": subset_n,
                "nce_override_raw": raw_nce,
                "nce_override_norm": norm_nce,
                "median_sa": m["median_sa"],
                "median_sas": m["median_sas"],
                "median_pcc": m["median_pcc"],
                "n_eval_used": m["n"],
                "eval_json": str(eval_json),
                "run_dir": str(run_dir),
            }
        )

    detail_csv = exp_root / f"fullrun_prospect_nce_gradient_{model}.csv"
    pd.DataFrame(rows).to_csv(detail_csv, index=False)

    summary = (
        pd.DataFrame(rows)
        .groupby(["run_label", "experiment", "model", "target_true_nce", "n_target_subset_all", "nce_override_raw"], as_index=False)
        .agg(
            median_sa=("median_sa", "median"),
            median_sas=("median_sas", "median"),
            median_pcc=("median_pcc", "median"),
            n_eval_used=("n_eval_used", "max"),
        )
        .sort_values("nce_override_raw")
    )
    summary_csv = exp_root / f"fullrun_prospect_nce_gradient_summary_{model}.csv"
    summary.to_csv(summary_csv, index=False)
    return detail_csv, summary_csv


def run_a2_blind_full(ctx: RunContext, model: str) -> tuple[Path, Path]:
    exp_root = ctx.out_root / f"{ctx.run_tag}_a2_blind_{model}_{_ts()}"
    _ensure_dir(exp_root)
    src_model_dir = ctx.source_run_dir / model
    cfg_path = ctx.source_config

    rows: list[dict[str, Any]] = []
    for raw_nce in (25, 30):
        run_dir = exp_root / f"eval_nce{raw_nce}"
        _copy_model_checkpoint(src_model_dir, run_dir / model, model)
        eval_json = _run_eval_once(
            model=model,
            config_path=cfg_path,
            dataset_name=ctx.dataset_name,
            run_dir=run_dir,
            conda_env=ctx.conda_env,
            override_nce_norm=_normalize_nce(raw_nce),
        )
        m = _extract_unified_metrics(eval_json, model)
        rows.append(
            {
                "run_label": ctx.run_tag,
                "experiment": "A2_MASSIVE_NCE_BLIND",
                "model": model,
                "nce_override_raw": raw_nce,
                "nce_override_norm": _normalize_nce(raw_nce),
                "median_sa": m["median_sa"],
                "median_sas": m["median_sas"],
                "median_pcc": m["median_pcc"],
                "n_eval_used": m["n"],
                "eval_json": str(eval_json),
                "run_dir": str(run_dir),
            }
        )

    detail_df = pd.DataFrame(rows).sort_values("nce_override_raw")
    detail_csv = exp_root / f"fullrun_massive_nce_blind_{model}.csv"
    detail_df.to_csv(detail_csv, index=False)

    r25 = detail_df[detail_df["nce_override_raw"] == 25].iloc[0]
    r30 = detail_df[detail_df["nce_override_raw"] == 30].iloc[0]
    summary_df = pd.DataFrame(
        [
            {
                "run_label": ctx.run_tag,
                "experiment": "A2_MASSIVE_NCE_BLIND",
                "model": model,
                "sa_at_25": r25["median_sa"],
                "sa_at_30": r30["median_sa"],
                "delta_sa": float(r30["median_sa"]) - float(r25["median_sa"]),
                "sas_at_25": r25["median_sas"],
                "sas_at_30": r30["median_sas"],
                "delta_sas": float(r30["median_sas"]) - float(r25["median_sas"]),
                "pcc_at_25": r25["median_pcc"],
                "pcc_at_30": r30["median_pcc"],
                "delta_pcc": float(r30["median_pcc"]) - float(r25["median_pcc"]),
                "n_eval_used": r25["n_eval_used"],
                "fastspel_snap_suspected": bool(
                    model == "fastspel"
                    and abs(float(r30["median_sa"]) - float(r25["median_sa"])) < 1e-12
                    and abs(float(r30["median_sas"]) - float(r25["median_sas"])) < 1e-12
                    and abs(float(r30["median_pcc"]) - float(r25["median_pcc"])) < 1e-12
                ),
            }
        ]
    )
    summary_csv = exp_root / f"fullrun_massive_nce_blind_summary_{model}.csv"
    summary_df.to_csv(summary_csv, index=False)
    return detail_csv, summary_csv


def _rowwise_sas(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    eps = 1e-12
    dot = np.einsum("ij,ij->i", x, y)
    xn = np.linalg.norm(x, axis=1)
    yn = np.linalg.norm(y, axis=1)
    cos = dot / np.maximum(xn * yn, eps)
    cos = np.clip(cos, -1.0, 1.0)
    sa = np.arccos(cos) / np.pi
    return 1.0 - sa


def _rowwise_pcc(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xv = x - x.mean(axis=1, keepdims=True)
    yv = y - y.mean(axis=1, keepdims=True)
    num = np.einsum("ij,ij->i", xv, yv)
    den = np.linalg.norm(xv, axis=1) * np.linalg.norm(yv, axis=1)
    out = np.zeros((x.shape[0],), dtype=np.float64)
    nz = den > 1e-12
    out[nz] = num[nz] / den[nz]
    return out


def run_b_charge_mode_collapse_full(
    ctx: RunContext,
    model: str,
    charge_from: int,
    charge_to: int,
    high_sas_threshold: float,
) -> tuple[Path, Path]:
    vector_supported = {"prosit", "prosit_transformer", "alphapeptdeep", "unispec"}
    if model not in vector_supported:
        raise ValueError(
            f"Model '{model}' does not currently support vector export for charge mode-collapse. "
            f"Supported: {sorted(vector_supported)}"
        )

    exp_root = ctx.out_root / f"{ctx.run_tag}_b_charge_{model}_{_ts()}"
    _ensure_dir(exp_root)
    src_model_dir = ctx.source_run_dir / model

    cfg0 = _read_yaml(ctx.source_config)
    subset_path, subset_n = _build_filtered_charge_parquet(cfg0, ctx.dataset_name, exp_root, charge_from)
    cfg_base = _read_yaml(ctx.source_config)
    cfg_base["datasets"][ctx.dataset_name]["test"]["path"] = str(subset_path)
    if "subsets" in cfg_base["datasets"][ctx.dataset_name]["test"]:
        cfg_base["datasets"][ctx.dataset_name]["test"]["subsets"]["all"]["path"] = str(subset_path)

    vec_paths: dict[int, Path] = {}
    for input_charge in (charge_from, charge_to):
        run_dir = exp_root / f"eval_charge{input_charge}"
        _copy_model_checkpoint(src_model_dir, run_dir / model, model)

        cfg_i = json.loads(json.dumps(cfg_base))
        protocol = cfg_i.setdefault("protocol", {})
        analysis = protocol.setdefault("analysis", {})
        analysis["export_vectors_path"] = str(run_dir / model / f"vectors_charge{input_charge}.npz")
        cfg_path = exp_root / f"fullrun_config_charge{input_charge}.yaml"
        _write_yaml(cfg_i, cfg_path)

        _run_eval_once(
            model=model,
            config_path=cfg_path,
            dataset_name=ctx.dataset_name,
            run_dir=run_dir,
            conda_env=ctx.conda_env,
            override_charge=input_charge,
        )
        vec_paths[input_charge] = Path(analysis["export_vectors_path"])

    z2 = np.load(vec_paths[charge_from], allow_pickle=True)
    z3 = np.load(vec_paths[charge_to], allow_pickle=True)
    pred2 = np.asarray(z2["pred_level1"], dtype=np.float64)
    pred3 = np.asarray(z3["pred_level1"], dtype=np.float64)
    seq2 = np.asarray(z2["modified_sequence"], dtype=object)
    seq3 = np.asarray(z3["modified_sequence"], dtype=object)
    if pred2.shape != pred3.shape:
        raise ValueError(f"Prediction shape mismatch: {pred2.shape} vs {pred3.shape}")
    if len(seq2) != len(seq3) or not np.all(seq2 == seq3):
        raise ValueError("Sample alignment mismatch between charge_from and charge_to runs.")

    sas = _rowwise_sas(pred2, pred3)
    sa = 1.0 - sas
    pcc = _rowwise_pcc(pred2, pred3)
    detail = pd.DataFrame(
        {
            "run_label": ctx.run_tag,
            "experiment": "B_CHARGE_MODE_COLLAPSE",
            "model": model,
            "sample_idx": np.arange(len(sas), dtype=np.int64),
            "modified_sequence": seq2.astype(str),
            "charge_from": int(charge_from),
            "charge_to": int(charge_to),
            "space_tag": "shared174",
            "sas_pred_z2_vs_pred_z3": sas,
            "sa_pred_z2_vs_pred_z3": sa,
            "pcc_pred_z2_vs_pred_z3": pcc,
        }
    )
    detail_csv = exp_root / f"fullrun_charge_mode_collapse_{model}.csv"
    detail.to_csv(detail_csv, index=False)

    summary = pd.DataFrame(
        [
            {
                "run_label": ctx.run_tag,
                "experiment": "B_CHARGE_MODE_COLLAPSE",
                "model": model,
                "charge_from": int(charge_from),
                "charge_to": int(charge_to),
                "space_tag": "shared174",
                "n_target_subset_all": int(subset_n),
                "n_eval_used": int(len(sas)),
                "mean_sas": float(np.mean(sas)),
                "median_sas": float(np.median(sas)),
                "q25_sas": float(np.quantile(sas, 0.25)),
                "q75_sas": float(np.quantile(sas, 0.75)),
                "high_sas_threshold": float(high_sas_threshold),
                "high_sas_fraction": float(np.mean(sas > high_sas_threshold)),
            }
        ]
    )
    summary_csv = exp_root / f"fullrun_charge_mode_collapse_summary_{model}.csv"
    summary.to_csv(summary_csv, index=False)
    return detail_csv, summary_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Execute metadata sensitivity inference runs (full-run only).")
    p.add_argument("--source-run-dir", type=Path, required=True)
    p.add_argument("--model", type=str, default="prosit")
    p.add_argument("--experiment", choices=["a1-gradient", "a2-blind", "b-charge"], default="a1-gradient")
    p.add_argument("--target-nce", type=int, default=30)
    p.add_argument("--nce-grid", type=str, default="20,25,30,35,40")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--label", type=str, default="full-run")
    p.add_argument("--full-run", action="store_true", help="Required: run full-set evaluation (no smoke subsampling).")
    p.add_argument("--charge-from", type=int, default=2)
    p.add_argument("--charge-to", type=int, default=3)
    p.add_argument("--high-sas-threshold", type=float, default=0.90)
    p.add_argument(
        "--conda-env",
        type=str,
        default="ms2benchmark",
        help="Conda environment used to run run_benchmark.py (forced).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dir(args.output_root)
    if not args.full_run:
        raise ValueError("Full run is mandatory. Re-run with --full-run.")
    conda_env = args.conda_env.strip() if args.conda_env is not None else ""
    if not conda_env:
        raise ValueError("--conda-env must be non-empty; use ms2benchmark for formal runs.")
    ctx = _build_run_context(args.source_run_dir, args.output_root, args.label, conda_env)
    grid = [int(x.strip()) for x in args.nce_grid.split(",") if x.strip()]
    if args.experiment == "a1-gradient":
        detail_csv, summary_csv = run_a1_gradient_full(
            ctx=ctx,
            model=args.model,
            target_nce=args.target_nce,
            nce_grid=grid,
        )
        print(f"[OK] detail: {detail_csv}")
        print(f"[OK] summary: {summary_csv}")
    elif args.experiment == "a2-blind":
        detail_csv, summary_csv = run_a2_blind_full(
            ctx=ctx,
            model=args.model,
        )
        print(f"[OK] detail: {detail_csv}")
        print(f"[OK] summary: {summary_csv}")
    elif args.experiment == "b-charge":
        detail_csv, summary_csv = run_b_charge_mode_collapse_full(
            ctx=ctx,
            model=args.model,
            charge_from=args.charge_from,
            charge_to=args.charge_to,
            high_sas_threshold=args.high_sas_threshold,
        )
        print(f"[OK] detail: {detail_csv}")
        print(f"[OK] summary: {summary_csv}")


if __name__ == "__main__":
    main()
