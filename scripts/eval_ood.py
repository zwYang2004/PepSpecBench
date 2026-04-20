#!/usr/bin/env python

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_ood")


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class PreprocessStats:
    input_rows: int
    kept_rows: int
    dropped_invalid_ptm: int
    dropped_missing_fields: int
    added_level1_labels: bool


def _resolve_registry_key(model_name: str) -> str:
    from src.models.registry import MODEL_REGISTRY

    if model_name in MODEL_REGISTRY:
        return model_name

    # Try common patterns like "prospect_prosit" -> "prosit"
    parts = [p for p in str(model_name).split("_") if p]
    for p in reversed(parts):
        if p in MODEL_REGISTRY:
            return p

    raise KeyError(
        f"Unknown model_name '{model_name}'. Available: {sorted(MODEL_REGISTRY.keys())}"
    )


def _expected_ckpt_filename(registry_key: str) -> str:
    # Mirrors runner.predict() expectations
    if registry_key == "prosit":
        return "prosit.pt"
    if registry_key == "prosit_transformer":
        return "prosit_transformer.pt"
    if registry_key == "alphapeptdeep":
        return "alphapeptdeep.pt"
    if registry_key == "unispec":
        return "unispec.pt"
    if registry_key in ("predfull_torch", "predfull_torch_234d"):
        return "predfull_torch.pt"
    if registry_key == "fastspel":
        return "X.npz"
    return "model.pt"


def _ckpt_candidates_for_dir(registry_key: str) -> List[str]:
    """Return ordered list of checkpoint filenames to try when ckpt_path is a directory."""
    primary = _expected_ckpt_filename(registry_key)
    if registry_key in ("prosit", "prosit_transformer"):
        # Training saves best.ckpt, predict accepts both
        return ["best.ckpt", "last.ckpt", primary]
    if registry_key == "alphapeptdeep":
        # Early-stopped or killed mid-training saves alphapeptdeep_best.pt
        return ["alphapeptdeep_best.pt", primary, "best.ckpt"]
    if registry_key in ("predfull_torch", "predfull_torch_234d"):
        return [primary, "predfull_torch.pt", "best.ckpt"]
    return [primary]


def _normalize_ood_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, PreprocessStats]:
    from scripts.data_process.sample_smart_splits import _normalize_sequence

    input_rows = int(len(df))
    dropped_missing = 0

    seq_col = "modified_sequence" if "modified_sequence" in df.columns else "sequence"
    if seq_col not in df.columns:
        raise KeyError("Missing sequence column (expected 'sequence' or 'modified_sequence')")

    charge_col = "precursor_charge" if "precursor_charge" in df.columns else "charge"
    if charge_col not in df.columns:
        raise KeyError("Missing charge column (expected 'charge' or 'precursor_charge')")

    ce_col = None
    for cand in ("collision_energy", "nce", "ce"):
        if cand in df.columns:
            ce_col = cand
            break

    if ce_col is None:
        dropped_missing += input_rows
        stats = PreprocessStats(
            input_rows=input_rows,
            kept_rows=0,
            dropped_invalid_ptm=0,
            dropped_missing_fields=dropped_missing,
            added_level1_labels=False,
        )
        return pd.DataFrame(), stats

    norms: List[Optional[str]] = []
    nakeds: List[Optional[str]] = []
    valids: List[bool] = []

    for s in df[seq_col].astype(str).tolist():
        norm, naked, ok, _unimods = _normalize_sequence(s)
        norms.append(norm)
        nakeds.append(naked)
        valids.append(bool(ok))

    df = df.copy()
    df["normalized_sequence"] = norms
    df["naked_sequence"] = nakeds
    valid_mask = pd.Series(valids, index=df.index).astype(bool)

    dropped_invalid = int((~valid_mask).sum())

    df = df[valid_mask].copy()
    df = df[df["normalized_sequence"].notna() & df["naked_sequence"].notna()].copy()

    # Ensure canonical columns expected by datasets
    df["sequence"] = df["normalized_sequence"].astype(str)
    df["charge"] = pd.to_numeric(df[charge_col], errors="coerce").astype("Int64")
    df["collision_energy"] = pd.to_numeric(df[ce_col], errors="coerce")

    # Drop rows with missing essential numeric fields
    missing_numeric = df["charge"].isna() | df["collision_energy"].isna()
    dropped_missing += int(missing_numeric.sum())
    df = df[~missing_numeric].copy()

    df["charge"] = df["charge"].astype(int)
    df["collision_energy"] = df["collision_energy"].astype(float)

    # Apply benchmark scope (L=6-40, z=1-6) for fair 100% match across all models
    from src.constraints import DataConstraints
    default_constraints = DataConstraints()
    len_ok = df["sequence"].str.len().between(default_constraints.min_len, default_constraints.max_len)
    charge_ok = df["charge"].between(1, default_constraints.max_charge)
    scope_mask = len_ok & charge_ok
    dropped_scope = int((~scope_mask).sum())
    df = df[scope_mask].copy()

    stats = PreprocessStats(
        input_rows=input_rows,
        kept_rows=int(len(df)),
        dropped_invalid_ptm=dropped_invalid,
        dropped_missing_fields=int(dropped_missing + dropped_scope),
        added_level1_labels=False,
    )
    return df, stats


def _maybe_add_level1_labels(df: pd.DataFrame, *, label_column: str) -> Tuple[pd.DataFrame, bool]:
    if label_column in df.columns:
        return df, False

    # Need mz/intensity columns to compute 234d labels
    mz_col = None
    int_col = None
    for c in df.columns:
        cl = str(c).lower()
        if mz_col is None and "mz" == cl:
            mz_col = c
        if int_col is None and (cl == "intensity" or cl == "intensities"):
            int_col = c

    if mz_col is None or int_col is None:
        return df, False

    from scripts.data_process.add_level1_labels import compute_level1_for_row

    labels = []
    seq_col = "sequence" if "sequence" in df.columns else "modified_sequence"

    for _idx, row in df.iterrows():
        labels.append(compute_level1_for_row(row, seq_col, "charge", mz_col, int_col))

    out = df.copy()
    out[label_column] = labels
    return out, True


def _prepare_model_dir(ckpt_path: Path, *, registry_key: str, out_root: Path) -> Path:
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    model_dir = out_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    expected_name = _expected_ckpt_filename(registry_key)
    if ckpt_path.is_dir():
        # Allow pointing to an existing run directory or model directory.
        if registry_key == "fastspel":
            # FastSpel stores weights as checkpoints/X.npz
            cand = ckpt_path / "checkpoints" / expected_name
            if not cand.exists():
                cand = ckpt_path / expected_name
            ckpt_file = cand
        else:
            # Try multiple candidates (e.g. best.ckpt for prosit when primary is prosit.pt)
            candidates = _ckpt_candidates_for_dir(registry_key)
            ckpt_file = None
            for name in candidates:
                cand = ckpt_path / name
                if cand.exists():
                    ckpt_file = cand
                    break
            if ckpt_file is None:
                # Also try models/ subdir for run_benchmark layout
                for name in candidates:
                    cand = ckpt_path / "models" / registry_key / name
                    if cand.exists():
                        ckpt_file = cand
                        break
            if ckpt_file is None:
                ckpt_file = ckpt_path / expected_name  # Will raise FileNotFoundError below
    else:
        ckpt_file = ckpt_path

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found for model={registry_key}: {ckpt_file}")

    # Place checkpoint where each runner expects it.
    if registry_key == "fastspel":
        expected = model_dir / "checkpoints" / expected_name
        expected.parent.mkdir(parents=True, exist_ok=True)
    else:
        expected = model_dir / expected_name

    if expected.resolve() != ckpt_file.resolve():
        # Copy to expected name (small overhead, robust across filesystems)
        import shutil

        shutil.copy2(str(ckpt_file), str(expected))

    return model_dir


def _build_predict_config(
    *,
    registry_key: str,
    batch_size: int,
    device: str,
    num_workers: int,
    label_column: str,
    eval_max_samples: Optional[int] = None,
    export_per_sample: bool = False,
    output_dim: Optional[int] = None,
) -> Dict[str, Any]:
    prefer_gpu = str(device).lower() == "cuda"
    model_cfg: Dict[str, Any] = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "label_column": str(label_column),
        "eval_max_samples": eval_max_samples,
    }
    if output_dim is not None and registry_key in ("predfull_torch", "predfull_torch_234d"):
        model_cfg["output_dim"] = int(output_dim)
    return {
        "__model_name": registry_key,
        "protocol": {
            "prefer_gpu": prefer_gpu,
            "analysis": {
                "export_per_sample": export_per_sample,
            },
        },
        "models": {
            registry_key: model_cfg,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate trained models on OOD reference data.")
    p.add_argument("--model_name", required=True, type=str, help="Model name (registry key or e.g. prospect_prosit)")
    p.add_argument("--ckpt_path", required=True, type=str, help="Checkpoint path")
    p.add_argument("--ood_dir", default=os.environ.get("PEPSPECBENCH_OOD_DIR", "data/ood_reference"), type=str)
    p.add_argument("--output-dir", default=None, type=str, help="Output dir for ood_results_*.json (default: output/)")
    p.add_argument("--resume", action="store_true", help="Skip files already in existing ood_results_*.json (requires --output-dir)")
    p.add_argument("--use-234d", action="store_true", help="Use ood_dir/234d/ if exists (pre-computed 234d labels, faster)")
    p.add_argument("--batch_size", default=1024, type=int)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num-workers", default=None, type=int)
    p.add_argument("--max-samples", default=None, type=int, help="Max samples per OOD file (for quick timing estimate)")
    p.add_argument("--export-per-sample", action="store_true", help="Export per-sample SA to CSV for bootstrap CI (slower)")
    p.add_argument("--output-dim", type=int, default=None, help="PredFull output dim (234 for predfull_torch_234d, 20000 for predfull_torch)")

    args = p.parse_args()

    registry_key = _resolve_registry_key(args.model_name)

    ood_dir = Path(args.ood_dir)
    if not ood_dir.exists() or not ood_dir.is_dir():
        raise FileNotFoundError(str(ood_dir))

    if getattr(args, "use_234d", False):
        ood_234d = ood_dir / "234d"
        if ood_234d.exists() and ood_234d.is_dir():
            ood_dir = ood_234d
            logger.info("Using 234d subdir: %s", ood_dir)

    parquet_files = sorted([p for p in ood_dir.iterdir() if p.is_file() and p.suffix == ".parquet"])
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {ood_dir}")

    out_dir = Path(args.output_dir) if args.output_dir else _ROOT / "output"
    out_root = out_dir / f"ood_eval_{args.model_name}"
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = out_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Prepare model_dir expected by runners
    model_dir = _prepare_model_dir(Path(args.ckpt_path), registry_key=registry_key, out_root=out_root)

    # Initialize runner
    from src.models.registry import MODEL_REGISTRY

    runner_cls = MODEL_REGISTRY[registry_key]
    runner = runner_cls()

    label_column = "intensities_raw"

    if args.num_workers is None:
        num_workers = 0 if str(args.device).lower() == "cpu" else 4
    else:
        num_workers = int(args.num_workers)
    output_dim = getattr(args, "output_dim", None)
    if output_dim is None and registry_key == "predfull_torch_234d":
        output_dim = 234
    config = _build_predict_config(
        registry_key=registry_key,
        batch_size=int(args.batch_size),
        device=str(args.device),
        num_workers=int(num_workers),
        label_column=label_column,
        eval_max_samples=getattr(args, "max_samples", None),
        export_per_sample=getattr(args, "export_per_sample", False),
        output_dim=output_dim,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"ood_results_{args.model_name}.json"

    all_results: Dict[str, Any] = {
        "model_name": str(args.model_name),
        "registry_key": str(registry_key),
        "ckpt_path": str(Path(args.ckpt_path).resolve()),
        "ood_dir": str(ood_dir.resolve()),
        "files": [],
    }

    # Resume: load existing results and skip completed files
    completed_paths: set = set()
    if getattr(args, "resume", False) and args.output_dir and out_json.exists():
        try:
            with open(out_json) as f:
                prev = json.load(f)
            for entry in prev.get("files", []):
                fp = entry.get("file")
                if fp and "error" not in entry and "unified" in entry:
                    completed_paths.add(Path(fp).name)
            all_results["files"] = list(prev.get("files", []))
            logger.info("Resume: skipping %d already-completed files: %s", len(completed_paths), sorted(completed_paths))
        except Exception as e:
            logger.warning("Resume load failed: %s, starting fresh", e)

    logger.info("Evaluating model=%s (registry=%s) on %d OOD parquet files", args.model_name, registry_key, len(parquet_files))

    # Print header
    print("\nOOD Evaluation Results")
    print("dataset\trows\tmatched%\tmedian_level1_sa\tmedian_level1_pcc\tnote")
    sys.stdout.flush()

    def _save_results() -> None:
        with open(out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Saved OOD results to %s", out_json)

    for i, fp in enumerate(parquet_files):
        if fp.name in completed_paths:
            print(f"\n[{i+1}/{len(parquet_files)}] Skipping {fp.name} (already done)...", flush=True)
            continue
        note = ""
        n_total = len(parquet_files)
        print(f"\n[{i+1}/{n_total}] Processing {fp.name}...", flush=True)
        try:
            df_raw = pd.read_parquet(fp)
        except Exception as e:
            logger.error("Failed to read %s: %s", fp, e)
            print(f"{fp.name}\t0\tNA\tNA\tNA\tread_error")
            all_results["files"].append({"file": str(fp), "error": str(e)})
            _save_results()
            continue

        # Normalize sequences / metadata
        try:
            df_norm, stats = _normalize_ood_df(df_raw)
        except Exception as e:
            logger.error("Failed to normalize %s: %s", fp, e)
            print(f"{fp.name}\t0\tNA\tNA\tNA\tnormalize_error")
            all_results["files"].append({"file": str(fp), "error": str(e)})
            continue

        if df_norm.empty:
            print(f"{fp.name}\t0\tNA\tNA\tNA\tempty_after_norm")
            all_results["files"].append({"file": str(fp), "preprocess": stats.__dict__, "skipped": True})
            continue

        # Add 234d labels if needed
        try:
            df_ready, added = _maybe_add_level1_labels(df_norm, label_column=label_column)
            stats.added_level1_labels = bool(added)
            if label_column not in df_ready.columns:
                note = "missing_labels"
                print(f"{fp.name}\t{len(df_ready)}\tNA\tNA\tNA\t{note}")
                all_results["files"].append({"file": str(fp), "preprocess": stats.__dict__, "skipped": True, "note": note})
                continue
        except Exception as e:
            logger.error("Failed to add level1 labels for %s: %s", fp, e)
            print(f"{fp.name}\t{len(df_norm)}\tNA\tNA\tNA\tlabel_error")
            all_results["files"].append({"file": str(fp), "preprocess": stats.__dict__, "error": str(e)})
            continue

        # Write cached parquet and run runner.predict()
        cached = cache_dir / f"{fp.stem}_normalized.parquet"
        try:
            df_ready.to_parquet(cached, index=False)
        except Exception as e:
            logger.error("Failed to write cache parquet %s: %s", cached, e)
            print(f"{fp.name}\t{len(df_ready)}\tNA\tNA\tNA\tcache_write_error")
            all_results["files"].append({"file": str(fp), "preprocess": stats.__dict__, "error": str(e)})
            continue

        print(f"  [{i+1}/{n_total}] Running inference on {fp.name} ({len(df_ready)} rows)...", flush=True)
        t0 = time.perf_counter()
        try:
            pred_out = runner.predict(str(cached), str(model_dir), config)
            unified = pred_out.get("unified", {}) if isinstance(pred_out, dict) else {}

            median_sa = unified.get("level1_median_sa")
            median_pcc = unified.get("level1_median_pcc")
            matched_pct = unified.get("matched_percent")
            if matched_pct is None:
                matched_pct = 100.0 if unified.get("n", len(df_ready)) == len(df_ready) else 0.0
            else:
                matched_pct = float(matched_pct)

            # Compute inference_total_time_seconds if not present
            n_eval = unified.get("n", len(df_ready))
            if "inference_total_time_seconds" not in unified and isinstance(pred_out, dict):
                # PredFull/FastSpel return timing at top level (subprocess wall time)
                top_level = pred_out.get("inference_total_time_seconds")
                if top_level is not None:
                    unified["inference_total_time_seconds"] = float(top_level)
            mean_ms = unified.get("inference_mean_ms_per_spectrum")
            if mean_ms is not None and n_eval > 0 and "inference_total_time_seconds" not in unified:
                unified["inference_total_time_seconds"] = float(n_eval) * float(mean_ms) / 1000.0

            elapsed = time.perf_counter() - t0
            print(f"{fp.name}\t{len(df_ready)}\t{matched_pct:.1f}%\t{median_sa}\t{median_pcc}\t{note}")
            print(f"  [{i+1}/{n_total}] Done {fp.name} in {elapsed:.1f}s (SA={median_sa})", flush=True)

            all_results["files"].append(
                {
                    "file": str(fp),
                    "cached_parquet": str(cached),
                    "preprocess": stats.__dict__,
                    "unified": unified,
                }
            )
            _save_results()
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error("Predict failed on %s: %s", fp, e)
            print(f"{fp.name}\t{len(df_ready)}\tNA\tNA\tNA\tpredict_error")
            print(f"  [{i+1}/{n_total}] Failed {fp.name} after {elapsed:.1f}s: {e}", flush=True)
            all_results["files"].append({"file": str(fp), "cached_parquet": str(cached), "preprocess": stats.__dict__, "error": str(e)})
            continue

    _save_results()


if __name__ == "__main__":
    main()
