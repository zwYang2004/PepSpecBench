from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
import os
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import resolve_path

logger = logging.getLogger(__name__)
from src.constraints import DataConstraints
from src.models.base_model import BenchmarkModel


class PredFullRunner(BenchmarkModel):
    def __init__(self, constraints: Optional[DataConstraints] = None) -> None:
        super().__init__()
        self._constraints = constraints or DataConstraints(max_len=40)

    def get_data_constraints(self) -> DataConstraints:
        return self._constraints

    def fit(self, train_path: str, val_path: Optional[str], output_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
        # Get train script path with default fallback
        paths = config.get("paths", {})
        train_script_rel = paths.get("predfull_train_torch", "external/PredFull/train_model_torch.py")
        train_script = resolve_path(train_script_rel)

        out_path = Path(output_dir) / "predfull_torch.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Full training-state checkpoint (for robust resume)
        last_path = Path(output_dir) / "predfull_torch_last.pt"

        model_key = config.get("__model_name", "predfull_torch")
        args = config.get("models", {}).get(model_key, config.get("models", {}).get("predfull_torch", {}))
        batch_size = int(args.get("batch_size", 64))
        lr = float(args.get("base_lr", 3e-4))
        max_epochs = int(args.get("max_epochs", 100))
        max_samples = args.get("max_samples")
        try:
            max_samples_i = int(max_samples) if max_samples is not None else 0
        except Exception:
            max_samples_i = 0
        num_workers = int(args.get("num_workers", 4))
        if num_workers < 1:
            num_workers = 1
        patience = int(config.get("protocol", {}).get("early_stopping", {}).get("patience", 10))

        schema = str(config.get("__dataset_schema", ""))
        dataset_type = "massive" if schema.startswith("massive") else "prospect"

        output_dim = int(args.get("output_dim", 20000))
        cmd = [
            sys.executable,
            train_script,
            "--train_parquet",
            str(train_path),
            "--val_parquet",
            str(val_path) if val_path else str(train_path),
            "--dataset_type",
            dataset_type,
            "--out",
            str(out_path),
            "--last",
            str(last_path),
            "--batch_size",
            str(batch_size),
            "--epochs",
            str(max_epochs),
            "--max_samples",
            str(max_samples_i),
            "--peak_lr",
            str(lr),
            "--patience",
            str(patience),
            "--num_workers",
            str(num_workers),
            "--output-dim",
            str(output_dim),
        ]

        resume_path = args.get("resume_from")
        if not resume_path:
            resume_cfg = args.get("resume", {}) if isinstance(args.get("resume", {}), dict) else {}
            resume_path = resume_cfg.get("path")

        # Auto-resume if caller didn't specify a resume path:
        # 1) last checkpoint (full training state)
        # 2) output checkpoint (best, full training state)
        # 3) unified best.ckpt (state_dict only)
        if not resume_path:
            if last_path.exists():
                resume_path = str(last_path)
            elif out_path.exists():
                resume_path = str(out_path)
            else:
                weights_only_ckpt = Path(output_dir) / "models" / "predfull_torch" / "best.ckpt"
                if weights_only_ckpt.exists():
                    resume_path = str(weights_only_ckpt)

        if resume_path:
            cmd.extend(["--resume", str(resume_path)])

        start = time.perf_counter()
        log_path = Path(output_dir) / "predfull_train.log"
        env = dict(os.environ)
        env.setdefault("TQDM_DISABLE", "1")
        env.setdefault("TQDM_MININTERVAL", "60")
        env.setdefault("TQDM_MINITERS", "1000")
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            # Stream output to both terminal and log file
            full_stdout = []
            if proc.stdout:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_file.write(line)
                    log_file.flush()
                    full_stdout.append(line)
            
            proc.wait()
            stdout_str = "".join(full_stdout)

        elapsed_seconds = time.perf_counter() - start

        # PredFull saves the best checkpoint to --out; we treat wall-clock time as time-to-best
        # because the subprocess does not emit structured timing per epoch.
        result = {
            "returncode": proc.returncode,
            "stdout": stdout_str,
            "stderr": "",  # stderr was merged into stdout
            "model_path": str(out_path),
            "train_total_time_seconds": elapsed_seconds,
            "time_to_best_seconds": elapsed_seconds,
        }
        summary_path = Path(output_dir) / "train_summary.json"
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2)
        if proc.returncode != 0:
            # stderr was merged into stdout; use the tail of stdout for debugging.
            tail = stdout_str[-2000:] if stdout_str else ""
            raise RuntimeError(f"PredFull training failed (returncode={proc.returncode}). Last output:\n{tail}")
        return result

    def predict(self, parquet_path: str, model_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
        paths = config.get("paths", {})
        eval_script_rel = paths.get("predfull_eval_torch", "external/PredFull/eval_predfull_torch.py")
        eval_script = resolve_path(eval_script_rel)
        model_dir_p = Path(model_dir)
        ckpt_candidates = [
            model_dir_p / "predfull_torch.pt",
            model_dir_p / "models" / "predfull_torch" / "best.ckpt",
            model_dir_p / "best.ckpt",
        ]
        ckpt = None
        for c in ckpt_candidates:
            if c.exists():
                ckpt = c
                break
        if ckpt is None:
            raise FileNotFoundError(
                f"PredFull checkpoint not found. Searched: {[str(c) for c in ckpt_candidates]}"
            )

        model_key = config.get("__model_name", "predfull_torch")
        args = config.get("models", {}).get(model_key, config.get("models", {}).get("predfull_torch", {}))
        batch_size = int(args.get("batch_size", 64))
        num_workers = int(args.get("num_workers", 4))
        eval_max_samples = args.get("eval_max_samples")
        try:
            eval_max_samples_i = int(eval_max_samples) if eval_max_samples is not None else 0
        except Exception:
            eval_max_samples_i = 0

        schema = str(config.get("__dataset_schema", ""))
        dataset_type = "massive" if schema.startswith("massive") else "prospect"

        out_json = Path(model_dir) / f"eval_{Path(parquet_path).stem}.json"

        analysis_cfg = config.get("protocol", {}).get("analysis", {}) if isinstance(config.get("protocol", {}), dict) else {}
        meta_override = analysis_cfg.get("metadata_override", {})
        override_charge = meta_override.get("precursor_charge")
        override_nce = meta_override.get("collision_energy")

        export_per_sample = bool(analysis_cfg.get("export_per_sample", False) or bool(args.get("export_per_sample", False)))
        export_max_samples = analysis_cfg.get("export_max_samples")
        try:
            export_max_samples_i = int(export_max_samples) if export_max_samples is not None else 0
        except Exception:
            export_max_samples_i = 0
        export_csv_path = str(Path(model_dir) / f"per_sample_{Path(parquet_path).stem}.csv")

        output_dim = int(args.get("output_dim", 20000))
        cmd = [
            sys.executable,
            eval_script,
            "--checkpoint",
            str(ckpt),
            "--parquet-path",
            str(parquet_path),
            "--dataset-type",
            dataset_type,
            "--subset-name",
            Path(parquet_path).stem,
            "--batch-size",
            str(batch_size),
            "--num-workers",
            str(num_workers),
            "--max-samples",
            str(eval_max_samples_i),
            "--output-json",
            str(out_json),
            "--output-dim",
            str(output_dim),
        ]

        if override_charge is not None:
            cmd.extend(["--override-charge", str(override_charge)])
        if override_nce is not None:
            cmd.extend(["--override-nce", str(override_nce)])

        if export_per_sample:
            cmd.extend(
                [
                    "--export-per-sample",
                    "--export-csv-path",
                    str(export_csv_path),
                    "--export-max-samples",
                    str(int(export_max_samples_i)),
                ]
            )

        logger.info("[PredFull] Starting inference (subprocess)...")
        start = float(time.perf_counter())
        
        # Use Popen to stream progress from the subprocess to the main log
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        
        full_stdout = []
        if proc.stdout:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                full_stdout.append(line)
        
        proc.wait()
        elapsed_seconds = float(time.perf_counter() - start)
        stdout_str = "".join(full_stdout)

        if proc.returncode != 0:
            raise RuntimeError(f"PredFull eval failed (code {proc.returncode}): {stdout_str[-2000:]}")

        n_samples: Optional[int] = None
        unified: Optional[Dict[str, Any]] = None
        try:
            import json as _json

            with open(out_json, "r") as f:
                payload = _json.load(f)
            if isinstance(payload, dict) and "unified" in payload and isinstance(payload["unified"], dict):
                unified = payload["unified"]
                n_samples = unified.get("n") or payload.get("usable_samples") or payload.get("total_rows")
            if n_samples is None and isinstance(payload, dict):
                n_samples = payload.get("usable_samples") or payload.get("total_rows")
        except Exception:
            pass

        result: Dict[str, Any] = {
            "returncode": proc.returncode,
            "stdout": stdout_str,
            "stderr": "",  # merged into stdout
            "metrics_json": str(out_json),
            "inference_total_time_seconds": float(elapsed_seconds),
            "inference_samples_per_second": (float(n_samples) / float(elapsed_seconds)) if n_samples and elapsed_seconds > 0 else None,
        }
        if unified is not None:
            unified = dict(unified)  # copy for safe mutation
            if "n" not in unified and n_samples is not None:
                unified["n"] = int(n_samples)
            if "matched_percent" not in unified:
                mp = payload.get("matched_percent")
                if mp is not None:
                    unified["matched_percent"] = float(mp)
                elif "total_rows" in payload and "usable_samples" in payload:
                    total = int(payload.get("total_rows", 1))
                    usable = int(payload.get("usable_samples", 0))
                    unified["matched_percent"] = 100.0 * usable / total if total > 0 else 0.0
            result["unified"] = unified
        return result