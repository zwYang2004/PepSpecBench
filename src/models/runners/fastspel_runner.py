from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.config import resolve_path
from src.constraints import DataConstraints
from src.models.base_model import BenchmarkModel
from src.utils.hardware import get_optimal_device, get_optimal_num_workers
from src.utils.mass_calc import strip_modifications

logger = logging.getLogger(__name__)

class FastSpelRunner(BenchmarkModel):
    def __init__(self, constraints: Optional[DataConstraints] = None) -> None:
        super().__init__()
        self._constraints = constraints or DataConstraints(max_len=40)

    def get_data_constraints(self) -> DataConstraints:
        return self._constraints

    def fit(self, train_path: str, val_path: Optional[str], output_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
        paths = config.get("paths", {})
        wrapper_rel = paths.get("fastspel_wrapper", "scripts/train/run_fastspel.py")
        wrapper = resolve_path(wrapper_rel)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            wrapper,
            "--train-parquet",
            str(train_path),
            "--output-dir",
            str(out_dir),
            "--dataset-name",
            "fastspel",
        ]

        args = config.get("models", {}).get("fastspel", {})
        max_samples = args.get("max_samples")
        try:
            max_samples_i = int(max_samples) if max_samples is not None else 0
        except Exception:
            max_samples_i = 0
        if max_samples_i > 0:
            cmd.extend(["--max-samples", str(max_samples_i)])
        start = float(time.perf_counter())
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed_seconds = float(time.perf_counter() - start)
        result = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "model_dir": str(out_dir),
            "train_wall_time_seconds": float(elapsed_seconds),
            "time_to_best_seconds": float(elapsed_seconds),
        }
        if "inference_throughput" in proc.stdout:
            result["inference_throughput"] = float(proc.stdout.split("inference_throughput:")[1].strip())
        summary_path = out_dir / "train_summary.json"
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2)
        if proc.returncode != 0:
            err = proc.stderr or ""
            tail = err[-2000:] if err else ""
            raise RuntimeError(f"FastSpel training failed (returncode={proc.returncode}). Last stderr lines:\n{tail}")
        return result

    def predict(self, parquet_path: str, model_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
        paths = config.get("paths", {})
        wrapper_rel = paths.get("fastspel_wrapper", "scripts/train/run_fastspel.py")
        wrapper = resolve_path(wrapper_rel)
        analysis_cfg = config.get("protocol", {}).get("analysis", {}) if isinstance(config.get("protocol", {}), dict) else {}
        meta_override = analysis_cfg.get("metadata_override", {})
        override_charge = meta_override.get("precursor_charge")
        override_nce = meta_override.get("collision_energy")

        cmd = [
            sys.executable,
            wrapper,
            "--skip-training",
            "--eval-parquet",
            str(parquet_path),
            "--output-dir",
            str(model_dir),
            "--dataset-name",
            "fastspel",
        ]

        if override_charge is not None:
            cmd.extend(["--override-charge", str(override_charge)])
        if override_nce is not None:
            cmd.extend(["--override-nce", str(override_nce)])

        args = config.get("models", {}).get("fastspel", {})
        eval_max_samples = args.get("eval_max_samples")
        try:
            eval_max_samples_i = int(eval_max_samples) if eval_max_samples is not None else 0
        except Exception:
            eval_max_samples_i = 0
        if eval_max_samples_i > 0:
            cmd.extend(["--max-samples", str(eval_max_samples_i)])
        logger.info("[FastSpel] Starting inference (subprocess)...")
        start = float(time.perf_counter())
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed_seconds = float(time.perf_counter() - start)
        if proc.returncode != 0:
            err = proc.stderr or ""
            tail = err[-2000:] if err else ""
            raise RuntimeError(f"FastSpel inference failed (returncode={proc.returncode}). Last stderr lines:\n{tail}")

        result = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "inference_total_time_seconds": float(elapsed_seconds),
        }

        # Parse summary.json for unified metrics (used by eval_ood, run_benchmark)
        summary_path = Path(model_dir) / "predictions" / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
                eval_data = summary.get("evaluation", {})
                unified = eval_data.get("unified", {})
                if unified:
                    matched = int(eval_data.get("matched", 0))
                    total = int(eval_data.get("total_samples", 0))
                    unified["n"] = matched
                    unified["matched_percent"] = 100.0 * matched / total if total > 0 else 0.0
                    result["unified"] = unified
            except Exception:
                pass

        # Convert results_test.csv to per_sample format for run_posthoc_analysis (stratification)
        export_per_sample = bool(analysis_cfg.get("export_per_sample", False))
        parquet_stem = Path(parquet_path).stem
        results_csv = Path(model_dir) / "predictions" / f"results_{parquet_stem}.csv"
        per_sample_csv = Path(model_dir) / f"per_sample_{parquet_stem}.csv"
        if export_per_sample and results_csv.exists():
            try:
                df = pd.read_csv(results_csv)
                if "modified_sequence" in df.columns and "level1_sas" in df.columns:
                    df["naked_sequence"] = df["modified_sequence"].astype(str).apply(strip_modifications)
                    df["length"] = df["naked_sequence"].str.len()
                    df["input_precursor_charge"] = df["precursor_charge"]
                    df["input_collision_energy"] = df["collision_energy"]
                    df["has_ptm"] = df["modified_sequence"].astype(str).str.upper().str.contains("UNIMOD").astype(int)
                    out_cols = [
                        "modified_sequence", "naked_sequence", "length", "precursor_charge", "collision_energy",
                        "input_precursor_charge", "input_collision_energy", "has_ptm",
                        "level1_sa", "level1_sas", "level1_pcc", "level2_sa", "level2_sas", "level2_pcc",
                    ]
                    df[out_cols].to_csv(per_sample_csv, index=False)
                    logger.info("[FastSpel] Exported per_sample CSV: %s", per_sample_csv.name)
            except Exception as e:
                logger.warning("[FastSpel] Failed to convert to per_sample format: %s", e)

        return result
