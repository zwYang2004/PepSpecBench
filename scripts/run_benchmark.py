#!/usr/bin/env python
"""Unified benchmark entry point for training and evaluating all models.

Usage:
    python scripts/run_benchmark.py --config configs/benchmark_unimod135.yaml --dataset prospect_unimod135 --models all
    python scripts/run_benchmark.py --config configs/benchmark_unimod135.yaml --dataset prospect_unimod135 --models prosit,unispec
    python scripts/run_benchmark.py --config configs/benchmark_unimod135.yaml --dataset prospect_unimod135 --models prosit --eval-only
    python scripts/run_benchmark.py --config configs/benchmark_unimod135.yaml --dataset prospect_unimod135 --models prosit --smoke-test
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import copy
import os
import random
import traceback
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.constraints import DataConstraints

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_benchmark")


def _configure_quiet_progress_for_nohup() -> None:
    """Reduce tqdm/CLI progress spam in nohup logs.

    When stdout is not a TTY (e.g. nohup), tqdm-like progress bars can flood
    the log file. We set conservative environment variables that are respected
    by many tqdm users. Interactive runs (TTY) are left unchanged.
    """
    try:
        is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    except Exception:
        is_tty = False
    if is_tty:
        return
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TQDM_MININTERVAL", "60")
    os.environ.setdefault("TQDM_MINITERS", "1000")


def _set_global_seed(seed: int, *, deterministic: bool) -> None:
    try:
        seed_i = int(seed)
    except Exception:
        seed_i = 42

    os.environ.setdefault("DEEPSPECBENCH_SEED", str(seed_i))
    os.environ.setdefault("PYTHONHASHSEED", str(seed_i))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed_i)
    np.random.seed(seed_i)

    try:
        import torch

        torch.manual_seed(seed_i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_i)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
    except Exception:
        pass


def _append_jsonl(path: Path, rec: Dict[str, Any]) -> None:
    payload = dict(rec)
    payload["ts"] = float(time.time())
    with open(path, "a") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return resolve_config_paths(config, base_dir=_root)


def _resolve_path_str(path: Any, *, base_dir: Path) -> Any:
    if not isinstance(path, str):
        return path
    s = path.strip()
    if not s:
        return path
    p = Path(s)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def resolve_config_paths(config: Dict[str, Any], *, base_dir: Path) -> Dict[str, Any]:
    """Resolve relative paths in config against repo root so CWD doesn't matter."""
    # datasets
    datasets = config.get("datasets", {})
    if isinstance(datasets, dict):
        for _ds_name, ds in datasets.items():
            if not isinstance(ds, dict):
                continue
            for split in ("train", "val", "test"):
                block = ds.get(split)
                if isinstance(block, dict) and "path" in block:
                    block["path"] = _resolve_path_str(block.get("path"), base_dir=base_dir)
                if split == "test" and isinstance(block, dict):
                    subsets = block.get("subsets")
                    if isinstance(subsets, dict):
                        for _k, sub in subsets.items():
                            if isinstance(sub, dict) and "path" in sub:
                                sub["path"] = _resolve_path_str(sub.get("path"), base_dir=base_dir)

    # external script paths
    paths = config.get("paths")
    if isinstance(paths, dict):
        for k, v in list(paths.items()):
            paths[k] = _resolve_path_str(v, base_dir=base_dir)

    # output root
    exp = config.get("experiment")
    if isinstance(exp, dict) and "output_root" in exp:
        exp["output_root"] = _resolve_path_str(exp.get("output_root"), base_dir=base_dir)

    return config


def get_dataset_paths(config: Dict[str, Any], dataset_name: str) -> Dict[str, str]:
    datasets = config.get("datasets", {})
    if dataset_name not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' not found in config. Available: {list(datasets.keys())}")
    
    ds = datasets[dataset_name]
    return {
        "train": ds.get("train", {}).get("path", ""),
        "val": ds.get("val", {}).get("path", ""),
        "test": ds.get("test", {}).get("path", ""),
        "schema": ds.get("schema", ""),
    }


def get_active_models(config: Dict[str, Any], model_filter: str) -> List[str]:
    models_cfg = config.get("models", {})
    
    if model_filter.lower() == "all":
        return [name for name, cfg in models_cfg.items() if cfg.get("active", True)]
    
    requested = [m.strip() for m in model_filter.split(",")]
    available = list(models_cfg.keys())
    
    for m in requested:
        if m not in available:
            raise ValueError(f"Model '{m}' not found in config. Available: {available}")
    
    return requested


def _first_parquet_file(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(str(path))
    for p in sorted(path.iterdir()):
        if p.is_file() and p.suffix == ".parquet":
            return p
    raise FileNotFoundError(f"No parquet files found in directory: {path}")


def _read_parquet_schema_names(path: str) -> List[str]:
    p = Path(path)
    fp = _first_parquet_file(p)
    try:
        import pyarrow.parquet as pq  # type: ignore

        schema = pq.read_schema(str(fp))
        return list(schema.names)
    except Exception:
        # Fallback: pandas (may be slow for large files)
        import pandas as pd

        df0 = pd.read_parquet(str(fp), engine="pyarrow")
        return list(df0.columns)


def _require_any(names: List[str], candidates: List[str], desc: str) -> str:
    for c in candidates:
        if c in names:
            return c
    raise KeyError(f"Missing required column for {desc}. Tried: {candidates}. Found: {names}")


def check_dataset_for_model(model_name: str, dataset_paths: Dict[str, str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight schema/path check without loading full datasets."""
    report: Dict[str, Any] = {
        "model": model_name,
        "status": "pending",
        "train": {},
        "val": {},
        "test": {},
        "error": None,
    }

    try:
        schema_name = str(dataset_paths.get("schema", ""))
        for split in ("train", "val", "test"):
            split_path = dataset_paths.get(split, "")
            if not split_path:
                continue
            names = _read_parquet_schema_names(split_path)
            report[split] = {
                "path": split_path,
                "columns": names,
            }

        # Decide which split schema to validate on (train preferred)
        probe_path = dataset_paths.get("train") or dataset_paths.get("test")
        if not probe_path:
            raise ValueError("Dataset is missing both train and test paths")
        names0 = _read_parquet_schema_names(probe_path)

        # Common requirements
        _require_any(names0, ["collision_energy", "nce", "orig_collision_energy"], "collision energy")
        _require_any(names0, ["precursor_charge", "charge"], "precursor charge")
        _require_any(names0, ["modified_sequence", "sequence"], "sequence")

        # Model-specific requirements
        if model_name in ("prosit", "prosit_transformer", "alphapeptdeep", "fastspel"):
            # 234-d ion-aligned labels are required for these runners
            if "intensities_raw" not in names0:
                raise KeyError("Missing required column: intensities_raw")

        if model_name == "unispec":
            # UniSpec: 234d 用 intensities_raw (canonical); 135 用 intensities_raw/intensity_array/intensities
            _require_any(names0, ["intensities_raw", "intensity_array", "intensities"], "UniSpec label")

        if model_name in ("predfull_torch", "predfull_torch_234d"):
            model_cfg = config.get("models", {}).get(model_name, {})
            output_dim = model_cfg.get("output_dim", 20000) if isinstance(model_cfg, dict) else 20000
            if output_dim in (174, 234):
                # PredFull-234d: requires 234d parquet with intensities_raw
                _require_any(names0, ["intensities_raw"], "intensities_raw (234-d)")
            else:
                if schema_name == "massive_kb":
                    _require_any(names0, ["mz_array"], "m/z array")
                    _require_any(names0, ["intensity_array"], "intensity array")
                    _require_any(names0, ["sequence"], "sequence")
                    _require_any(names0, ["charge"], "charge")
                elif schema_name == "prospect":
                    _require_any(names0, ["mz"], "m/z")
                    _require_any(names0, ["intensities"], "intensities")
                    _require_any(names0, ["modified_sequence"], "modified_sequence")
                    _require_any(names0, ["precursor_charge"], "precursor_charge")
                else:
                    raise ValueError(f"Unknown dataset schema for PredFull: {schema_name}")

        report["status"] = "success"
        return report
    except Exception as e:
        report["status"] = "failed"
        report["error"] = str(e)
        return report


def _verify_dataset_version(model_name: str, dataset_name: str, model_cfg: Dict[str, Any]) -> None:
    """Verify that the model is using the correct dataset version (234d vs 135)."""
    required_version = model_cfg.get("dataset_version", "234d")
    output_dim = model_cfg.get("output_dim", 20000)
    
    # PredFull-234d/174d variant: requires 234d dataset
    if model_name in ("predfull_torch", "predfull_torch_234d") and output_dim in (174, 234):
        if not str(dataset_name).endswith("_234d"):
            raise ValueError(
                f"[{model_name}] with output_dim={output_dim} (PredFull-{output_dim}d) requires 234d dataset, "
                f"but dataset '{dataset_name}' does not end with '_234d'."
            )
        return
    
    # UniSpec 方案 A: 支持 234d，与 Prosit/AlphaPeptDeep 对齐
    if model_name == "unispec" and "234d" in required_version:
        if not str(dataset_name).endswith("_234d"):
            raise ValueError(
                f"[{model_name}] with dataset_version '{required_version}' requires 234d dataset, "
                f"but dataset '{dataset_name}' does not end with '_234d'."
            )
        return

    # Models that require 135-dim full spectrum labels (original PredFull only)
    full_spectrum_models = {"predfull_torch"}
    if model_name in full_spectrum_models:
        if "135" not in required_version:
            raise ValueError(
                f"[{model_name}] requires 135-dim full spectrum labels, "
                f"but dataset_version is set to '{required_version}'. "
                f"Dataset must be '*_135' version, not '*_234d'."
            )
        if str(dataset_name).endswith("_234d"):
            raise ValueError(
                f"[{model_name}] requires 135-dim labels, "
                f"but dataset '{dataset_name}' appears to be 234d version. "
                f"Please use the non-'_234d' 135-dataset variant."
            )
    else:
        # Other models should use 234d
        if "_234d" not in dataset_name:
            logger.warning(
                f"[{model_name}] typically uses 234d dataset, "
                f"but dataset '{dataset_name}' does not contain '_234d'. "
                f"Proceeding anyway, but verify this is intentional."
            )


def _verify_dataset_readability(dataset_paths: Dict[str, str], model_name: str) -> None:
    """Verify that train/val/test datasets are readable and contain data."""
    # Keep this lightweight: avoid pandas full reads on large parquet files.
    # Use pyarrow metadata / small batch reads when available.

    for split_name in ["train", "val", "test"]:
        path = dataset_paths.get(split_name)
        if not path:
            logger.warning(f"[{model_name}] No {split_name} path specified")
            continue
        
        try:
            p = Path(str(path))
            if not p.exists():
                raise FileNotFoundError(f"{split_name} path does not exist: {path}")

            fp = p
            if p.is_dir():
                parquet_files = sorted(list(p.glob("*.parquet")))
                if not parquet_files:
                    raise ValueError(f"No parquet files found in {split_name} directory: {path}")
                fp = parquet_files[0]

            try:
                import pyarrow.parquet as pq  # type: ignore

                pf = pq.ParquetFile(str(fp))
                names = list(pf.schema.names)
                nrows = int(pf.metadata.num_rows) if pf.metadata is not None else None

                # Read a tiny batch to ensure payload is actually readable.
                seen = 0
                for batch in pf.iter_batches(batch_size=16):
                    seen = int(len(batch))
                    break

                if (nrows is not None and int(nrows) <= 0) or seen <= 0:
                    raise ValueError(f"{split_name} dataset is empty: {path}")

                logger.info(
                    f"[{model_name}] {split_name}: ~{nrows if nrows is not None else 'unknown'} rows, columns={names[:3]}..."
                )
            except Exception:
                # Fallback: pandas but limit read to a small head
                import pandas as pd

                df = pd.read_parquet(str(fp), engine="pyarrow")
                if len(df) == 0:
                    raise ValueError(f"{split_name} dataset is empty: {path}")
                logger.info(f"[{model_name}] {split_name}: {len(df)} samples, columns={list(df.columns[:3])}...")

        except Exception as e:
            raise RuntimeError(f"[{model_name}] Failed to read {split_name} dataset at '{path}': {e}")


def run_single_model(
    model_name: str,
    config: Dict[str, Any],
    dataset_paths: Dict[str, str],
    output_dir: Path,
    *,
    override_charge: Optional[int] = None,
    override_nce: Optional[float] = None,
    train: bool = True,
    evaluate: bool = True,
) -> Dict[str, Any]:
    """Run training and/or evaluation for a single model."""
    logger.info(f"{'='*60}")
    logger.info(f"Starting model: {model_name}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    result = {
        "model": model_name,
        "status": "pending",
        "train_result": None,
        "eval_result": None,
        "error": None,
        "duration_seconds": 0,
    }
    
    try:
        # Import registry lazily so that --check-only can run without pulling
        # in heavyweight runtime deps (torch/psutil/etc.).
        from src.models.registry import MODEL_REGISTRY

        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' not registered in MODEL_REGISTRY")

        runner_cls = MODEL_REGISTRY[model_name]
        
        model_cfg = config.get("models", {}).get(model_name, {})
        constraints_cfg = model_cfg.get("constraints", {})
        global_filters = config.get("filters", {}) if isinstance(config.get("filters", {}), dict) else {}
        default_min_len = int(global_filters.get("min_len", 6))
        constraints = DataConstraints(
            min_len=int(constraints_cfg.get("min_len", default_min_len)),
            max_len=int(constraints_cfg.get("max_len", global_filters.get("max_len", 40))),
            max_charge=int(constraints_cfg.get("max_charge", global_filters.get("max_charge", 6))),
            allowed_unimod_ids=tuple(constraints_cfg.get("allowed_unimod_ids", global_filters.get("allowed_unimod_ids", [1, 4, 35]))),
        )
        
        runner = runner_cls(constraints=constraints)
        
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        config_with_schema = dict(config)
        config_with_schema["__dataset_schema"] = dataset_paths.get("schema", "")
        config_with_schema["__model_name"] = model_name
        if override_charge is not None or override_nce is not None:
            cfg_protocol = config_with_schema.setdefault("protocol", {})
            cfg_analysis = cfg_protocol.setdefault("analysis", {})
            meta_override = dict(cfg_analysis.get("metadata_override", {}))
            if override_charge is not None:
                meta_override["precursor_charge"] = int(override_charge)
            if override_nce is not None:
                meta_override["collision_energy"] = float(override_nce)
            cfg_analysis["metadata_override"] = meta_override
        
        # Verify dataset version before training
        dataset_name = dataset_paths.get("name", "")
        logger.info(f"[{model_name}] Verifying dataset version...")
        _verify_dataset_version(model_name, dataset_name, model_cfg)
        
        # Verify dataset readability before training
        logger.info(f"[{model_name}] Verifying dataset readability...")
        _verify_dataset_readability(dataset_paths, model_name)
        
        if train and model_cfg.get("train", {}).get("enabled", True):
            logger.info(f"[{model_name}] Starting training...")
            train_path = dataset_paths["train"]
            val_path = dataset_paths["val"]
            
            if not train_path:
                raise ValueError(f"No training path specified for dataset")
            
            train_result = runner.fit(
                train_path=train_path,
                val_path=val_path if val_path else None,
                output_dir=str(model_output_dir),
                config=config_with_schema,
            )
            result["train_result"] = train_result
            logger.info(f"[{model_name}] Training completed")
        
        if evaluate and model_cfg.get("eval", {}).get("enabled", True):
            logger.info(f"[{model_name}] Starting evaluation...")
            test_path = dataset_paths["test"]
            
            if not test_path:
                raise ValueError(f"No test path specified for dataset")
            
            eval_result = runner.predict(
                parquet_path=test_path,
                model_dir=str(model_output_dir),
                config=config_with_schema,
            )
            result["eval_result"] = eval_result
            logger.info(f"[{model_name}] Evaluation completed")
        
        result["status"] = "success"
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[{model_name}] Failed: {e}\n{tb}")
        result["status"] = "failed"
        result["error"] = str(e)
        result["traceback"] = tb
    
    result["duration_seconds"] = time.time() - start_time
    logger.info(f"[{model_name}] Finished in {result['duration_seconds']:.1f}s with status: {result['status']}")
    
    return result


from src.utils.hardware import get_optimal_device, get_optimal_num_workers

def run_benchmark(
    config_path: str,
    dataset_name: str,
    model_filter: str,
    output_dir: Optional[str] = None,
    *,
    resume_path: Optional[str] = None,
    run_dir: Optional[str] = None,
    overwrite_run_dir: bool = False,
    continue_run: bool = False,
    override_charge: Optional[int] = None,
    override_nce: Optional[float] = None,
    override_tag: Optional[str] = None,
    train: bool = True,
    evaluate: bool = True,
    check_only: bool = False,
    smoke_test: bool = False,
    smoke_max_samples: int = 512,
    smoke_max_epochs: int = 1,
    seed_override: Optional[int] = None,
    deterministic_override: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run benchmark for specified models on a dataset."""
    config = load_config(config_path)

    if seed_override is not None:
        config.setdefault("experiment", {})["seed"] = int(seed_override)
    if deterministic_override is not None:
        config.setdefault("experiment", {})["deterministic"] = bool(deterministic_override)

    exp_cfg = config.get("experiment", {}) if isinstance(config.get("experiment", {}), dict) else {}
    seed_i = int(exp_cfg.get("seed", 42))
    deterministic = bool(exp_cfg.get("deterministic", False))
    _set_global_seed(seed_i, deterministic=deterministic)

    requested_device = exp_cfg.get("device", "auto")
    device = get_optimal_device(requested_device)

    output_root = output_dir or exp_cfg.get("output_root") or exp_cfg.get("output_dir") or "output"
    output_root_p = Path(str(output_root))
    if not output_root_p.is_absolute():
        output_root_p = (_root / output_root_p).resolve()
    output_root_p.mkdir(parents=True, exist_ok=True)

    run_dir_p: Path
    if run_dir is not None:
        run_dir_p = Path(str(run_dir))
        if not run_dir_p.is_absolute():
            # Resolve relative to project root, NOT output_root, to avoid
            # double-nesting like output/output/benchmark_...
            run_dir_p = (_root / run_dir_p).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"_{override_tag}" if override_tag else ""
        run_name = f"benchmark_{dataset_name}_{timestamp}{tag}"
        run_dir_p = (output_root_p / run_name).resolve()

    if overwrite_run_dir and run_dir_p.exists():
        # Safety guard: only allow overwriting if run_dir is under output_root
        # and looks like a benchmark run directory.
        try:
            run_dir_resolved = run_dir_p.resolve()
            output_root_resolved = output_root_p.resolve()
        except Exception:
            run_dir_resolved = run_dir_p
            output_root_resolved = output_root_p

        if output_root_resolved not in run_dir_resolved.parents:
            raise ValueError(f"Refusing to overwrite run_dir outside output_root: {run_dir_resolved}")

        name = str(run_dir_resolved.name)
        if not (name.startswith("benchmark_") or name.startswith("latest_") or "latest_run" in name):
            raise ValueError(
                f"Refusing to overwrite suspicious run_dir name '{name}'. "
                "Please choose a run_dir like output/latest_run_* or output/benchmark_*"
            )

        shutil.rmtree(run_dir_resolved)

    run_dir_p.mkdir(parents=True, exist_ok=True)
    
    # Resolve which models to run based on the filter string and config
    models = get_active_models(config, model_filter)

    # ... (rest of the orchestrator logic using run_dir_p and config)
    
    run_log = run_dir_p / "run.log"
    fh = logging.FileHandler(str(run_log))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    root_logger = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(run_log) for h in root_logger.handlers):
        root_logger.addHandler(fh)

    # Create a stable, discoverable symlink for the run log.
    try:
        unified_dir = (output_root_p / "logs" / "unified" / str(dataset_name)).resolve()
        unified_dir.mkdir(parents=True, exist_ok=True)
        link_path = unified_dir / "run.log"
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        os.symlink(str(run_log), str(link_path))
    except Exception:
        pass

    progress_path = run_dir_p / "progress.jsonl"
    _append_jsonl(
        progress_path,
        {
            "event": "run_start",
            "dataset": str(dataset_name),
            "models": list(models),
            "train": bool(train),
            "evaluate": bool(evaluate),
            "continue_run": bool(continue_run),
            "pid": int(os.getpid()) if hasattr(os, "getpid") else None,
        },
    )

    logger.info(f"Benchmark run directory: {run_dir_p}")
    logger.info(f"Dataset (base): {dataset_name}")
    logger.info(f"Models to run: {models}")
    logger.info(f"Train: {train}, Evaluate: {evaluate}")
    if resume_path:
        logger.info(f"Resume from: {resume_path}")
    
    with open(run_dir_p / "config_snapshot.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Expose run-level context to runners (for progress logging / auto-resume).
    config["__run_dir"] = str(run_dir_p)
    config["__continue_run"] = bool(continue_run)
    
    results = []
    total_start = time.time()

    datasets_cfg = config.get("datasets", {})

    # Treat the CLI --dataset as a base name; models can map it to either
    # the 234d (ion) or 135 (full-spectrum) variant as needed.
    if str(dataset_name).endswith("_234d"):
        base_dataset_name = str(dataset_name)[:-5]
    else:
        base_dataset_name = str(dataset_name)
    canonical_candidate = f"{base_dataset_name}_234d"
    full_spectrum_candidate = base_dataset_name
    
    # predfull_torch 固定用 135-d; unispec 按 dataset_version 决定 (135 或 234d)
    full_spectrum_models = {"predfull_torch"}
    
    done_status: Dict[str, str] = {}
    if continue_run:
        for p in run_dir_p.glob("*_result.json"):
            try:
                with open(p, "r") as f:
                    obj = json.load(f)
                if isinstance(obj, dict) and isinstance(obj.get("model"), str):
                    done_status[str(obj["model"])] = str(obj.get("status", ""))
            except Exception:
                continue

    for model_name in models:
        if continue_run and done_status.get(model_name) == "success":
            logger.info(f"[{model_name}] Skipping (already success in existing run_dir)")
            _append_jsonl(progress_path, {"event": "model_skip", "model": str(model_name), "status": "success"})
            continue

        _append_jsonl(progress_path, {"event": "model_start", "model": str(model_name)})
        model_cfg = config.setdefault("models", {}).setdefault(model_name, {})

        # If the user provided a CLI --resume, wire it into the model config for runners
        # that support resuming. We currently scope this to PredFull, because different
        # runners expect different checkpoint formats.
        if resume_path and model_name == "predfull_torch":
            try:
                if not model_cfg.get("resume_from"):
                    model_cfg["resume_from"] = str(resume_path)
                resume_cfg = model_cfg.get("resume") if isinstance(model_cfg.get("resume"), dict) else {}
                if not resume_cfg.get("path"):
                    resume_cfg["path"] = str(resume_path)
                model_cfg["resume"] = resume_cfg
            except Exception:
                pass

        # Auto-resume: when continuing an existing run, inject last checkpoint path for
        # models that support it (prosit, prosit_transformer, unispec, alphapeptdeep).
        if continue_run and run_dir_p.exists():
            model_dir = run_dir_p / model_name
            if model_name in ("prosit", "prosit_transformer", "unispec"):
                ckpt = model_dir / "last.ckpt"
                if ckpt.exists():
                    resume_cfg = model_cfg.get("resume") if isinstance(model_cfg.get("resume"), dict) else {}
                    if not resume_cfg.get("path"):
                        resume_cfg["path"] = str(ckpt)
                        model_cfg["resume"] = resume_cfg
                        logger.info(f"[{model_name}] Auto-resume from {ckpt}")
                    if model_name == "prosit_transformer" and not model_cfg.get("resume_from"):
                        model_cfg["resume_from"] = str(ckpt)
            elif model_name == "alphapeptdeep":
                for ckpt_name in ("alphapeptdeep.pt", "alphapeptdeep_best.pt"):
                    ckpt = model_dir / ckpt_name
                    if ckpt.exists():
                        if not model_cfg.get("resume_from"):
                            model_cfg["resume_from"] = str(ckpt)
                            logger.info(f"[{model_name}] Auto-resume from {ckpt}")
                        break

        ds_version = str(model_cfg.get("dataset_version", "234d"))

        # Default: use the CLI dataset as-is
        model_dataset = dataset_name

        if model_name in full_spectrum_models or "135" in ds_version:
            # Full-spectrum models should use the 135-dim dataset (no _234d suffix)
            if full_spectrum_candidate in datasets_cfg:
                model_dataset = full_spectrum_candidate
            elif base_dataset_name in datasets_cfg:
                model_dataset = base_dataset_name
            else:
                logger.warning(
                    f"[{model_name}] Full-spectrum model expected 135-dataset, "
                    f"but neither '{full_spectrum_candidate}' nor '{base_dataset_name}' "
                    f"found in config.datasets; using CLI dataset '{dataset_name}'"
                )
        else:
            # Ion models should preferentially use the 234d dataset variant
            if canonical_candidate in datasets_cfg:
                model_dataset = canonical_candidate
            elif str(dataset_name) in datasets_cfg:
                model_dataset = dataset_name
            else:
                logger.warning(
                    f"[{model_name}] Expected 234d dataset, but neither '{canonical_candidate}' "
                    f"nor '{dataset_name}' found in config.datasets; falling back to CLI name"
                )

        dataset_paths = get_dataset_paths(config, model_dataset)
        dataset_paths["name"] = model_dataset  # Add dataset name for version verification

        if smoke_test:
            model_cfg = config.setdefault("models", {}).setdefault(model_name, {})
            model_cfg["max_epochs"] = int(smoke_max_epochs)
            if model_name == "predfull_torch":
                model_cfg["max_samples"] = int(min(int(smoke_max_samples), 512))
            else:
                model_cfg["max_samples"] = int(smoke_max_samples)
            model_cfg["val_max_samples"] = int(max(1, int(smoke_max_samples) // 4))
            model_cfg["eval_max_samples"] = int(max(1, int(smoke_max_samples) // 4))

        if check_only:
            model_result = check_dataset_for_model(model_name, dataset_paths, config)
        else:
            model_result = run_single_model(
                model_name=model_name,
                config=config,
                dataset_paths=dataset_paths,
                output_dir=run_dir_p,
                override_charge=override_charge,
                override_nce=override_nce,
                train=train,
                evaluate=evaluate,
            )
        results.append(model_result)
        
        with open(run_dir_p / f"{model_name}_result.json", "w") as f:
            json.dump(model_result, f, indent=2, default=str)

        _append_jsonl(
            progress_path,
            {
                "event": "model_end",
                "model": str(model_name),
                "status": str(model_result.get("status")),
                "duration_seconds": model_result.get("duration_seconds"),
            },
        )
    
    total_duration = time.time() - total_start
    
    summary = {
        "dataset": dataset_name,
        "models": models,
        "train": train,
        "evaluate": evaluate,
        "total_duration_seconds": total_duration,
        "results": results,
        "success_count": sum(1 for r in results if r["status"] == "success"),
        "failed_count": sum(1 for r in results if r["status"] == "failed"),
    }
    
    with open(run_dir_p / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"{'='*60}")
    logger.info(f"Benchmark completed in {total_duration:.1f}s")
    logger.info(f"Success: {summary['success_count']}/{len(models)}, Failed: {summary['failed_count']}/{len(models)}")
    logger.info(f"Results saved to: {run_dir_p}")
    logger.info(f"{'='*60}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Unified benchmark entry point")
    parser.add_argument("--config", required=True, help="Path to benchmark config YAML")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., prospect_unimod135)")
    parser.add_argument("--models", default="all", help="Comma-separated model names or 'all'")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--train-only", action="store_true", help="Only run training")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--check-only", action="store_true", help="Only check dataset/model compatibility (no training/eval)")
    parser.add_argument("--smoke-test", action="store_true", help="Run a small smoke test (limits epochs/samples)")
    parser.add_argument("--smoke-max-samples", type=int, default=512, help="Max samples per split for smoke test")
    parser.add_argument("--smoke-max-epochs", type=int, default=1, help="Max epochs for smoke test")
    parser.add_argument("--resume", help="Path to checkpoint for resuming training")
    parser.add_argument("--run-dir", type=str, default=None, help="Use an existing output run directory instead of creating a new timestamped one")
    parser.add_argument("--overwrite-run-dir", action="store_true", help="Delete --run-dir if it exists before starting (safe-guarded under output_root)")
    parser.add_argument("--continue-run", action="store_true", help="Continue an existing run directory by skipping models that already succeeded")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment.seed (for reproducibility)")
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic execution (overrides experiment.deterministic)")
    parser.add_argument("--override-charge", type=int, default=None, help="Eval-time override for precursor charge (metadata ablation)")
    parser.add_argument("--override-nce", type=float, default=None, help="Eval-time override for collision energy / NCE in [0,1] (metadata ablation)")
    parser.add_argument("--override-tag", type=str, default=None, help="Optional tag appended to output run directory")
    
    args = parser.parse_args()

    _configure_quiet_progress_for_nohup()
    
    train = not args.eval_only
    evaluate = not args.train_only

    if (args.override_charge is not None or args.override_nce is not None) and train:
        parser.error("--override-charge/--override-nce are only supported for post-training evaluation. Please use --eval-only.")

    if args.check_only:
        train = False
        evaluate = False
    
    if args.train_only and args.eval_only:
        parser.error("Cannot specify both --train-only and --eval-only")
    
    summary = run_benchmark(
        config_path=args.config,
        dataset_name=args.dataset,
        model_filter=args.models,
        output_dir=args.output_dir,
        resume_path=args.resume,
        run_dir=args.run_dir,
        overwrite_run_dir=bool(args.overwrite_run_dir),
        continue_run=bool(args.continue_run),
        seed_override=args.seed,
        deterministic_override=True if bool(args.deterministic) else None,
        override_charge=args.override_charge,
        override_nce=args.override_nce,
        override_tag=args.override_tag,
        train=train,
        evaluate=evaluate,
        check_only=bool(args.check_only),
        smoke_test=bool(args.smoke_test),
        smoke_max_samples=int(args.smoke_max_samples),
        smoke_max_epochs=int(args.smoke_max_epochs),
    )
    
    if summary["failed_count"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
