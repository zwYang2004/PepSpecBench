from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging
import numpy as np
import pandas as pd
import torch

from src.config import project_root
from src.constraints import DataConstraints
from src.evaluator import UnifiedEvaluator
from src.metrics import summarize
from src.models.base_model import BenchmarkModel
from src.schedulers import create_scheduler, EarlyStopping
from src.utils.mass_calc import canonical_by_dim, canonical_by_mask, ion_index, strip_modifications
from src.utils.profiling import InferenceProfiler, get_cuda_peak_vram_gb, get_process_rss_gb, reset_cuda_peak_memory
from src.datasets.peptdeep_adapter import (
    build_precursor_df,
    build_fragment_df_from_level1,
    level1_from_fragment_df,
    normalize_nce,
    parse_mods,
)


logger = logging.getLogger("alphapeptdeep_runner")


def _first_parquet_file(parquet_path: str | Path) -> Optional[Path]:
    p = Path(str(parquet_path))
    if p.is_file():
        return p
    if p.is_dir():
        for fp in sorted(p.iterdir()):
            if fp.is_file() and fp.suffix == ".parquet":
                return fp
    return None


def _read_schema_names(parquet_path: str) -> List[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        fp = _first_parquet_file(parquet_path)
        if fp is None:
            return []
        return list(pq.read_schema(str(fp)).names)
    except Exception:
        return []


def _read_parquet_head(parquet_path: str, *, max_rows: int, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if max_rows <= 0:
        return pd.read_parquet(parquet_path, columns=columns)

    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        df = pd.read_parquet(parquet_path, columns=columns)
        if len(df) > int(max_rows):
            df = df.iloc[: int(max_rows)].copy()
        return df

    p = Path(str(parquet_path))
    files: List[Path]
    if p.is_dir():
        files = sorted([x for x in p.iterdir() if x.is_file() and x.suffix == ".parquet"])
    else:
        files = [p]

    batches = []
    remaining = int(max_rows)
    for fp in files:
        if remaining <= 0:
            break
        pf = pq.ParquetFile(str(fp))
        for batch in pf.iter_batches(batch_size=min(8192, remaining), columns=columns):
            batches.append(batch)
            remaining -= int(len(batch))
            if remaining <= 0:
                break

    if not batches:
        return pd.DataFrame()
    table = pa.Table.from_batches(batches)
    return table.to_pandas()


def _ensure_alphapeptdeep_imports() -> None:
    root = project_root() / "external" / "alphapeptdeep"
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


# Adapter functions are now imported from src.datasets.peptdeep_adapter
# - normalize_nce, parse_mods, build_precursor_df, build_fragment_df_from_level1, level1_from_fragment_df


def _predict_alphapeptdeep_manual(
    model,
    prec: pd.DataFrame,
    df: pd.DataFrame,
    seq_col: str,
    charge_col: str,
    charged_frag_types: List[str],
    level1_dim: int,
    max_len: int,
    batch_size: int,
) -> np.ndarray:
    """Run AlphaPeptDeep inference and build pred_level1 directly.
    Bypasses model.predict() which has a bug in update_sliced_fragment_dataframe.
    """
    import torch
    pred_level1 = np.zeros((len(prec), int(level1_dim)), dtype=np.float32)
    model.model.eval()

    _grouped = prec.groupby("nAA")
    with torch.no_grad():
        for _nAA, df_group in _grouped:
            for i in range(0, len(df_group), batch_size):
                batch_end = min(i + batch_size, len(df_group))
                batch_df = df_group.iloc[i:batch_end]

                features = model._get_features_from_batch_df(batch_df)
                predicts = model._predict_one_batch(*features)

                # Same post-processing as _set_batch_predict_data
                apex_intens = predicts.reshape((len(batch_df), -1)).max(axis=1)
                apex_intens[apex_intens <= 0] = 1
                predicts = predicts / apex_intens.reshape((-1, 1, 1))
                predicts[predicts < model.min_inten] = 0.0
                columns_mask = np.isin(
                    model.model.supported_charged_frag_types, model.charged_frag_types
                )
                predicts = predicts[:, :, columns_mask]

                # Map to pred_level1; batch_df.index gives original prec row indices
                for k, prec_idx in enumerate(batch_df.index):
                    seq = str(df.iloc[prec_idx][seq_col])
                    charge = int(df.iloc[prec_idx][charge_col])
                    naked = strip_modifications(seq)
                    L = len(naked)
                    mask = canonical_by_mask(L, charge, max_len=max_len, max_frag_charge=3)
                    block = predicts[k]
                    positions = block.shape[0]
                    for p in range(1, positions + 1):
                        for j, ft in enumerate(charged_frag_types):
                            parts = ft.split("_")
                            ion_type = parts[0]
                            z = int(parts[1][1:])
                            if z > 3:
                                continue
                            idx = ion_index(p, ion_type, z, max_frag_charge=3)
                            pred_level1[prec_idx, idx] = float(block[p - 1, j])
                    pred_level1[prec_idx, ~mask] = 0.0

    return pred_level1


@torch.no_grad()
def _evaluate_level1(pred_level1: np.ndarray, true_level1: np.ndarray, meta_seq: List[str], meta_charge: List[int], *, max_len: int) -> Dict[str, Any]:
    evaluator = UnifiedEvaluator(max_len=int(max_len), max_frag_charge=3)

    l1_sa: List[float] = []
    l1_sas: List[float] = []
    l1_pcc: List[float] = []
    l2_sa: List[float] = []
    l2_sas: List[float] = []
    l2_pcc: List[float] = []

    for i in range(int(pred_level1.shape[0])):
        meta = {"modified_sequence": str(meta_seq[i]), "precursor_charge": int(meta_charge[i])}
        try:
            pred_std = evaluator.standardize(pred_level1[i], meta)
            true_std = evaluator.standardize(true_level1[i], meta)
            m = evaluator.compute_metrics(
                pred_std.level1,
                true_std.level1,
                true_std.level1_mask,
                pred_std.level2,
                true_std.level2,
            )
            l1_sa.append(float(m["level1_sa"]))
            l1_sas.append(float(m["level1_sas"]))
            l1_pcc.append(float(m["level1_pcc"]))
            l2_sa.append(float(m["level2_sa"]))
            l2_sas.append(float(m["level2_sas"]))
            l2_pcc.append(float(m["level2_pcc"]))
        except Exception as e:
            if i == 0:
                logger.warning(
                    "AlphaPeptDeep _evaluate_level1: sample 0 failed (repr=%s): %s",
                    type(e).__name__,
                    str(e)[:200],
                )
            continue

    l1_mean_sa, l1_median_sa = summarize(l1_sa)
    l1_mean_sas, l1_median_sas = summarize(l1_sas)
    l1_mean_pcc, l1_median_pcc = summarize(l1_pcc)
    l2_mean_sa, l2_median_sa = summarize(l2_sa)
    l2_mean_sas, l2_median_sas = summarize(l2_sas)
    l2_mean_pcc, l2_median_pcc = summarize(l2_pcc)

    return {
        "level1_mean_sa": l1_mean_sa,
        "level1_median_sa": l1_median_sa,
        "level1_mean_sas": l1_mean_sas,
        "level1_median_sas": l1_median_sas,
        "level1_mean_pcc": l1_mean_pcc,
        "level1_median_pcc": l1_median_pcc,
        "level2_mean_sa": l2_mean_sa,
        "level2_median_sa": l2_median_sa,
        "level2_mean_sas": l2_mean_sas,
        "level2_median_sas": l2_median_sas,
        "level2_mean_pcc": l2_mean_pcc,
        "level2_median_pcc": l2_median_pcc,
        "n": int(len(l1_sa)),
    }


from src.utils.experiment import setup_experiment, save_best_model
from src.utils.hardware import get_optimal_device, get_optimal_num_workers

class AlphaPeptDeepRunner(BenchmarkModel):
    def __init__(self, constraints: Optional[DataConstraints] = None) -> None:
        super().__init__()
        self._constraints = constraints or DataConstraints(max_len=40)

    def get_data_constraints(self) -> DataConstraints:
        return self._constraints

    def fit(self, train_path: str, val_path: Optional[str], output_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_alphapeptdeep_imports()

        from peptdeep.model.ms2 import ModelMS2Bert, pDeepModel
        from peptdeep.model.model_interface import CallbackHandler
        from alphabase.peptide.fragment import get_charged_frag_types

        # Unified Setup
        device = get_optimal_device(config.get("experiment", {}).get("device", "auto"))
        # For AlphaPeptDeep we treat the configured num_workers as a simple
        # "parallelism" knob: values <= 1 mean "run single-GPU/single-worker"
        # and will also disable DataParallel below.
        model_workers_cfg = int(config.get("models", {}).get("alphapeptdeep", {}).get("num_workers", 8))

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        args = config.get("models", {}).get("alphapeptdeep", {})
        batch_size = int(args.get("batch_size", 512))
        lr = float(args.get("base_lr", 1e-5))
        max_epochs = int(args.get("max_epochs", 200))
        patience = int(args.get("patience", 5))
        resume_from = args.get("resume_from")
        
        # Load-to-RAM via pandas read if small enough (already implemented in part, but unifying)
        # Note: AlphaPeptDeep handles its own data structures, so we load full DF here.
        # For AlphaPeptDeep, we use the raw parquet read but can add a check.
        from src.utils.hardware import should_load_to_ram
        
        file_size_gb = Path(train_path).stat().st_size / (1024**3) if Path(train_path).is_file() else 1.0
        use_ram = should_load_to_ram(file_size_gb)
        
        if use_ram:
            logger.info(f"Loading {train_path} into RAM for AlphaPeptDeep")
            train_df = pd.read_parquet(train_path)
        else:
            # Fallback to current head reading logic for very large files
            train_n = int(args.get("max_samples", -1))
            if train_n > 0:
                train_df = _read_parquet_head(train_path, max_rows=train_n)
            else:
                train_df = pd.read_parquet(train_path)

        if "normalized_sequence" in train_df.columns:
            seq_col0 = "normalized_sequence"
        else:
            seq_col0 = "modified_sequence" if "modified_sequence" in train_df.columns else "sequence"
        charge_col0 = "precursor_charge" if "precursor_charge" in train_df.columns else "charge"
        if "collision_energy" in train_df.columns:
            ce_col0 = "collision_energy"
        else:
            ce_col0 = "orig_collision_energy"
        label_col = str(args.get("label_column", "intensities_raw"))
        read_cols = [seq_col0, charge_col0, ce_col0, label_col]
        
        # Model setup
        frag_types = list(args.get("frag_types", ["b", "y"]))
        max_frag_charge = int(args.get("max_frag_charge", 2))
        charged_frag_types = get_charged_frag_types(frag_types, max_frag_charge)
        model = pDeepModel(charged_frag_types=charged_frag_types, mask_modloss=True)

        # Resume from checkpoint if specified (loads weights; training runs full max_epochs from scratch)
        if resume_from and Path(resume_from).exists():
            logger.info(f"Resuming AlphaPeptDeep from checkpoint: {resume_from}")
            model.load(str(resume_from))

        # Early stopping callback: stop when train loss does not improve for `patience` epochs
        ckpt_path = out_dir / "alphapeptdeep.pt"
        best_ckpt_path = out_dir / "alphapeptdeep_best.pt"

        class EarlyStoppingCallback(CallbackHandler):
            def __init__(self, patience: int, model_ref, ckpt_path: Path, best_ckpt_path: Path):
                super().__init__()
                self.patience = patience
                self.best_loss = float("inf")
                self.patience_counter = 0
                self.model_ref = model_ref
                self.ckpt_path = ckpt_path
                self.best_ckpt_path = best_ckpt_path

            def epoch_callback(self, epoch: int, epoch_loss: float) -> bool:
                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.patience_counter = 0
                    # Save best model (strip DataParallel prefix for portable checkpoint)
                    raw_sd = self.model_ref.model.state_dict()
                    clean_sd = {}
                    for k, v in raw_sd.items():
                        name = k
                        while name.startswith("module."):
                            name = name[7:]
                        clean_sd[name] = v
                    # Temporarily use inner module for save (model.save uses model.state_dict())
                    old_model = self.model_ref.model
                    if hasattr(old_model, "module"):
                        self.model_ref.model = old_model.module
                    self.model_ref.model.load_state_dict(clean_sd)
                    self.model_ref.save(str(self.best_ckpt_path))
                    self.model_ref.model = old_model
                    return True
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping: no improvement for {self.patience} epochs")
                    return False
                return True

        model.set_callback_handler(EarlyStoppingCallback(patience, model, ckpt_path, best_ckpt_path))
        logger.info(f"Early stopping patience: {patience}")

        # Only enable multi-GPU DataParallel when explicitly requested via
        # num_workers > 1. This lets the user set num_workers: 0 or 1 in the
        # config to force a simpler, single-GPU execution path.
        if model_workers_cfg > 1 and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            logger.info(
                f"Using DataParallel on {torch.cuda.device_count()} GPUs for AlphaPeptDeep (fit)"
            )
            model.model = torch.nn.DataParallel(model.model)

        warmup_epochs = int(args.get("warmup_epochs", 5))

        # Calculate level1_dim dynamically to match data
        from src.utils.mass_calc import canonical_by_dim
        max_len = int(self._constraints.max_len)
        min_len = 6
        level1_dim = canonical_by_dim(max_len=max_len, max_frag_charge=3)
        logger.info(f"Using dynamic level1_dim: {level1_dim}")

        if val_path:
            val_max_samples = args.get("val_max_samples")
            try:
                val_n = int(val_max_samples) if val_max_samples is not None else -1
            except Exception:
                val_n = -1
            if val_n > 0:
                val_df = _read_parquet_head(val_path, max_rows=val_n, columns=read_cols)
            else:
                val_df = pd.read_parquet(val_path, columns=read_cols)
        else:
            val_df = train_df

        if "normalized_sequence" in train_df.columns:
            seq_col = "normalized_sequence"
        else:
            seq_col = "modified_sequence" if "modified_sequence" in train_df.columns else "sequence"
        charge_col = "precursor_charge" if "precursor_charge" in train_df.columns else "charge"

        train_prec = build_precursor_df(train_df)
        y_train = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in train_df[label_col].values], axis=0)
        if y_train.shape[1] != int(level1_dim):
            raise ValueError(f"Label dim mismatch: expected {level1_dim}, got {y_train.shape[1]}")

        train_prec, train_frag = build_fragment_df_from_level1(
            train_prec,
            y_train,
            charged_frag_types,
            max_len=max_len,
            max_frag_charge=3,
        )

        reset_cuda_peak_memory()
        train_start = float(time.perf_counter())
        model.train(train_prec, train_frag, batch_size=batch_size, epoch=max_epochs, lr=lr, warmup_epoch=warmup_epochs, verbose=True)
        train_total_time_seconds = float(time.perf_counter() - train_start)
        peak_vram_gb = float(get_cuda_peak_vram_gb())
        rss_gb = float(get_process_rss_gb())

        # Use best checkpoint if early stopping saved one; otherwise save current model
        if best_ckpt_path.exists():
            import shutil
            shutil.copy(best_ckpt_path, ckpt_path)
            logger.info(f"Using best model from early stopping: {best_ckpt_path}")
        else:
            # Strip DataParallel "module." prefix before saving if present
            raw_sd = model.model.state_dict()
            clean_sd = {}
            for k, v in raw_sd.items():
                name = k
                while name.startswith("module."):
                    name = name[7:]
                clean_sd[name] = v
            model.model.load_state_dict(clean_sd)
            model.save(str(ckpt_path))

        val_prec = build_precursor_df(val_df)
        y_val = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in val_df[label_col].values], axis=0)
        val_prec, val_frag = build_fragment_df_from_level1(
            val_prec,
            y_val,
            charged_frag_types,
            max_len=max_len,
            max_frag_charge=3,
        )

        pred_frag = model.predict(val_prec, batch_size=batch_size, verbose=False)

        pred_level1 = np.zeros((len(val_prec), int(level1_dim)), dtype=np.float32)
        for i in range(len(val_prec)):
            seq = str(val_df.iloc[i][seq_col])
            charge = int(val_df.iloc[i][charge_col])
            naked = strip_modifications(seq)
            L = len(naked)
            mask = canonical_by_mask(L, charge, max_len=max_len, max_frag_charge=3)
            start = int(val_prec.iloc[i]["frag_start_idx"])
            stop = int(val_prec.iloc[i]["frag_stop_idx"])
            block = pred_frag.iloc[start:stop].to_numpy(dtype=np.float32)

            positions = stop - start
            for p in range(1, positions + 1):
                for j, ft in enumerate(charged_frag_types):
                    parts = ft.split("_")
                    ion_type = parts[0]
                    z = int(parts[1][1:])
                    if z > 3:
                        continue
                    idx = ion_index(p, ion_type, z, max_frag_charge=3)
                    pred_level1[i, idx] = float(block[p - 1, j])
            pred_level1[i, ~mask] = 0.0

        meta_seq = [str(val_df.iloc[i][seq_col]) for i in range(len(val_df))]
        meta_charge = [int(val_df.iloc[i][charge_col]) for i in range(len(val_df))]
        unified = _evaluate_level1(pred_level1, y_val, meta_seq, meta_charge, max_len=max_len)

        best_value = float(unified.get("level1_median_sas", float("nan")))
        best_epoch = int(max_epochs) - 1 if int(max_epochs) > 0 else 0
        time_to_best_seconds = float(train_total_time_seconds)

        result = {
            "model_path": str(ckpt_path),
            "unified": unified,
            "best_value": float(best_value) if best_value is not None else None,
            "best_epoch": int(best_epoch),
            "time_to_best_seconds": float(time_to_best_seconds),
            "train_total_time_seconds": float(train_total_time_seconds),
            "peak_vram_gb": float(peak_vram_gb),
            "rss_gb": float(rss_gb),
            "wall_clock_training_time": float(train_total_time_seconds),
            "peak_memory": float(peak_vram_gb),
        }
        # Save best model logic
        is_best = True  # AlphaPeptDeep currently trains for fixed epochs in its .train() call
        save_best_model(model.model, str(out_dir.parent.parent), "alphapeptdeep", result, is_best)

        with open(out_dir / "train_summary.json", "w") as f:
            json.dump(result, f, indent=2)

        return result

    def predict(self, parquet_path: str, model_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_alphapeptdeep_imports()

        from peptdeep.model.ms2 import pDeepModel
        from alphabase.peptide.fragment import get_charged_frag_types

        model_dir_p = Path(model_dir)
        ckpt_candidates = [
            model_dir_p / "alphapeptdeep_best.pt",  # Early-stopped or killed mid-training
            model_dir_p / "alphapeptdeep.pt",
            model_dir_p / "best.ckpt",
            model_dir_p / "models" / "alphapeptdeep" / "best.ckpt",
        ]
        ckpt_path = None
        for c in ckpt_candidates:
            if c.exists():
                ckpt_path = c
                break
        if ckpt_path is None:
            raise FileNotFoundError(
                f"AlphaPeptDeep checkpoint not found. Searched: {[str(c) for c in ckpt_candidates]}"
            )

        args = config.get("models", {}).get("alphapeptdeep", {})
        batch_size = int(args.get("batch_size", 512))
        max_len = int(self._constraints.max_len)
        level1_dim = canonical_by_dim(max_len=max_len, max_frag_charge=3)

        frag_types = list(args.get("frag_types", ["b", "y"]))
        max_frag_charge = int(args.get("max_frag_charge", 2))
        charged_frag_types = get_charged_frag_types(frag_types, max_frag_charge)

        model = pDeepModel(charged_frag_types=charged_frag_types, mask_modloss=True)
        logger.info("[AlphaPeptDeep] Loading checkpoint from %s...", ckpt_path)
        # Load checkpoint. Do NOT wrap with DataParallel for predict: pDeepModel
        # internally accesses model.model.supported_charged_frag_types, which
        # DataParallel does not forward (it lives on the inner .module).
        model.load(str(ckpt_path))
        logger.info("[AlphaPeptDeep] Checkpoint loaded.")

        label_col = str(args.get("label_column", "intensities_raw"))
        names = set(_read_schema_names(parquet_path))
        seq_col0 = "modified_sequence" if "modified_sequence" in names else "sequence"
        charge_col0 = "precursor_charge" if "precursor_charge" in names else "charge"
        if "collision_energy" in names:
            ce_col0 = "collision_energy"
        elif "nce" in names:
            ce_col0 = "nce"
        else:
            ce_col0 = "orig_collision_energy"

        read_cols = [seq_col0, charge_col0, ce_col0, label_col]
        for inst_cand in ("instrument", "instrument_name", "instrument_type", "ms_instrument", "instrument_model"):
            if inst_cand in names and inst_cand not in read_cols:
                read_cols.append(inst_cand)
                break

        eval_max_samples = args.get("eval_max_samples")
        try:
            n = int(eval_max_samples) if eval_max_samples is not None else -1
        except Exception:
            n = -1
        if n > 0:
            df = _read_parquet_head(parquet_path, max_rows=n, columns=read_cols)
        else:
            df = pd.read_parquet(parquet_path, columns=read_cols)
        seq_col = "modified_sequence" if "modified_sequence" in df.columns else "sequence"
        charge_col = "precursor_charge" if "precursor_charge" in df.columns else "charge"
        label_col = str(args.get("label_column", "intensities_raw"))

        prec = build_precursor_df(df)

        # Check for metadata overrides
        analysis_cfg = config.get("protocol", {}).get("analysis", {}) if isinstance(config.get("protocol", {}), dict) else {}
        meta_override = analysis_cfg.get("metadata_override", {})
        override_charge = meta_override.get("precursor_charge")
        override_nce = meta_override.get("collision_energy")

        if override_charge is not None:
            prec["charge"] = int(override_charge)

        if override_nce is not None:
            # AlphaPeptDeep uses 0-100 scale for NCE, while input override is 0-1
            prec["nce"] = float(override_nce) * 100.0

        y_true = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in df[label_col].values], axis=0)
        if y_true.shape[1] != int(level1_dim):
            raise ValueError(f"Label dim mismatch: expected {level1_dim}, got {y_true.shape[1]}")

        prec, _dummy = build_fragment_df_from_level1(
            prec,
            np.zeros_like(y_true),
            charged_frag_types,
            max_len=max_len,
            max_frag_charge=3,
        )

        logger.info("[AlphaPeptDeep] Starting inference on %d rows...", len(prec))
        profiler = InferenceProfiler()
        start = float(time.perf_counter())
        # Bypass model.predict() which has a bug in update_sliced_fragment_dataframe
        # (pandas iloc assignment doesn't update in-place). Run inference manually.
        pred_level1 = _predict_alphapeptdeep_manual(
            model, prec, df, seq_col, charge_col, charged_frag_types,
            level1_dim, max_len, batch_size,
        )
        elapsed_ms = float(time.perf_counter() - start) * 1000.0
        logger.info("[AlphaPeptDeep] Inference done in %.1fs", elapsed_ms / 1000.0)
        n = int(len(prec))
        profiler.add_measurement(elapsed_ms / float(max(1, n)), 0.0)

        meta_seq = [str(df.iloc[i][seq_col]) for i in range(len(df))]
        meta_charge = [int(df.iloc[i][charge_col]) for i in range(len(df))]
        unified = _evaluate_level1(pred_level1, y_true, meta_seq, meta_charge, max_len=max_len)

        analysis_cfg = config.get("protocol", {}).get("analysis", {}) if isinstance(config.get("protocol", {}), dict) else {}
        export_per_sample = bool(analysis_cfg.get("export_per_sample", False) or bool(args.get("export_per_sample", False)))
        export_max_samples = analysis_cfg.get("export_max_samples")
        export_vectors_path = analysis_cfg.get("export_vectors_path")
        try:
            export_max_samples_i = int(export_max_samples) if export_max_samples is not None else None
        except Exception:
            export_max_samples_i = None

        if export_per_sample:
            import csv

            out_csv = model_dir_p / f"per_sample_{Path(parquet_path).stem}.csv"
            evaluator = UnifiedEvaluator(max_len=int(max_len), max_frag_charge=3)
            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "modified_sequence",
                        "naked_sequence",
                        "length",
                        "precursor_charge",
                        "collision_energy",
                        "input_precursor_charge",
                        "input_collision_energy",
                        "has_ptm",
                        "level1_sa",
                        "level1_sas",
                        "level1_pcc",
                        "level2_sa",
                        "level2_sas",
                        "level2_pcc",
                    ]
                )
                written = 0
                ce_vals = df[ce_col0].to_numpy() if ce_col0 in df.columns else np.full((len(df),), np.nan, dtype=np.float32)
                for i in range(int(pred_level1.shape[0])):
                    if export_max_samples_i is not None and int(written) >= int(export_max_samples_i):
                        break
                    meta = {"modified_sequence": str(meta_seq[i]), "precursor_charge": int(meta_charge[i])}
                    try:
                        pred_std = evaluator.standardize(pred_level1[i], meta)
                        true_std = evaluator.standardize(y_true[i], meta)
                        m = evaluator.compute_metrics(
                            pred_std.level1,
                            true_std.level1,
                            true_std.level1_mask,
                            pred_std.level2,
                            true_std.level2,
                        )
                    except Exception:
                        continue
                    mod_seq = str(meta_seq[i])
                    naked = strip_modifications(mod_seq)
                    ce_i = float(ce_vals[i]) if i < int(len(ce_vals)) else float("nan")

                    input_charge = int(override_charge) if override_charge is not None else int(meta_charge[i])
                    input_ce = float(override_nce) if override_nce is not None else (float(normalize_nce(ce_i)) / 100.0)

                    w.writerow(
                        [
                            mod_seq,
                            naked,
                            int(len(naked)),
                            int(meta_charge[i]),
                            float(normalize_nce(ce_i)),
                            input_charge,
                            input_ce,
                            1 if ("UNIMOD" in mod_seq.upper()) else 0,
                            float(m["level1_sa"]),
                            float(m["level1_sas"]),
                            float(m["level1_pcc"]),
                            float(m["level2_sa"]),
                            float(m["level2_sas"]),
                            float(m["level2_pcc"]),
                        ]
                    )
                    written += 1
        if export_vectors_path:
            np.savez(
                str(export_vectors_path),
                pred_level1=np.asarray(pred_level1, dtype=np.float32),
                true_level1=np.asarray(y_true, dtype=np.float32),
                modified_sequence=np.array(meta_seq, dtype=object),
                precursor_charge=np.array(meta_charge, dtype=np.int32),
            )

        prof_summary = profiler.get_summary()
        unified.update(
            {
                "inference_median_ms_per_spectrum": float(prof_summary["median_time_ms"]),
                "inference_mean_ms_per_spectrum": float(prof_summary["mean_time_ms"]),
                "inference_samples_per_second": float(n) / float(elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0,
                "inference_mean_vram_gb": float(prof_summary["mean_vram_gb"]),
                "inference_max_vram_gb": float(prof_summary["max_vram_gb"]),
                "rss_gb": float(get_process_rss_gb()),
            }
        )

        out_json = model_dir_p / f"eval_{Path(parquet_path).stem}.json"
        payload = {
            "dataset": str(parquet_path),
            "checkpoint": str(ckpt_path),
            "unified": unified,
        }
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)

        return {"metrics_json": str(out_json), "unified": unified}
