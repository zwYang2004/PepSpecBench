from __future__ import annotations

import json
import os
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import project_root
from src.constraints import DataConstraints
from src.evaluator import UnifiedEvaluator
from src.metrics import summarize
from src.models.base_model import BenchmarkModel
from src.schedulers import create_scheduler, EarlyStopping
from src.utils.mass_calc import canonical_by_dim, canonical_by_mask, strip_modifications
from src.utils.profiling import InferenceProfiler, cuda_profile, get_cuda_peak_vram_gb, get_process_rss_gb, reset_cuda_peak_memory

logger = logging.getLogger(__name__)
_UNISPEC_DEBUG = bool(int(os.environ.get("MS2BENCHMARK_DEBUG_UNISPEC", "0")))


def _ensure_unispec_imports() -> None:
    unispec_root = project_root() / "external" / "UniSpec"
    unispec_root_str = str(unispec_root)
    if unispec_root_str not in sys.path:
        sys.path.insert(0, unispec_root_str)


def _device_from_config(config: Dict[str, Any]) -> torch.device:
    prefer_gpu = bool(config.get("protocol", {}).get("prefer_gpu", True))
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _normalize_nce(value: Any) -> float:
    try:
        ce = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(ce):
        return 0.0
    if ce > 1.5:
        ce = ce / 100.0
    return float(np.clip(ce, 0.0, 1.0))


def _unispec_mod_string(mod_seq: str) -> str:
    if not isinstance(mod_seq, str):
        return "0"

    s = str(mod_seq).strip()
    mods: List[str] = []

    if s.lower().startswith("[unimod:1]"):
        mods.append("(0,X,Acetyl)")

    pos = -1
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if not ch.isalpha():
            i += 1
            continue

        aa = ch.upper()
        pos += 1
        i += 1

        unimod_id: Optional[int] = None
        if i < n and s[i] == "[":
            j = s.find("]", i + 1)
            if j != -1:
                token = s[i + 1 : j]
                import re

                m = re.search(r"unimod:(\d+)", token, flags=re.IGNORECASE)
                if m:
                    try:
                        unimod_id = int(m.group(1))
                    except Exception:
                        unimod_id = None
                i = j + 1

        if i + 4 <= n and s[i : i + 4].lower() == "(ox)":
            unimod_id = 35
            i += 4

        if aa == "C" and unimod_id == 4:
            mods.append(f"({pos},C,Carbamidomethyl)")
        elif aa == "M" and unimod_id == 35:
            mods.append(f"({pos},M,Oxidation)")

    if not mods:
        return "0"

    return f"{len(mods)}" + "".join(mods)


def _unispec_label(seq: str, charge: int, mod_seq: str, nce: float) -> str:
    mods = _unispec_mod_string(mod_seq)
    return f"{seq}/{int(charge)}_{mods}_0.0eV_NCE{float(nce):.4f}"


def _read_parquet_head(parquet_path: str, *, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0:
        return pd.read_parquet(parquet_path)

    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        df = pd.read_parquet(parquet_path)
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
        for batch in pf.iter_batches(batch_size=min(8192, remaining)):
            batches.append(batch)
            remaining -= int(len(batch))
            if remaining <= 0:
                break

    if not batches:
        return pd.DataFrame()
    table = pa.Table.from_batches(batches)
    return table.to_pandas()


from src.datasets.base_dataset import BaseParquetDataset
from src.utils.mass_calc import strip_modifications

class UniSpecParquetDataset(BaseParquetDataset):
    def __init__(
        self,
        parquet_path: str,
        *,
        max_len: int = 40,
        max_charge: int = 6,
        label_column: str = "intensities_raw",
        max_samples: Optional[int] = None,
        load_to_ram: Optional[bool] = None,
    ) -> None:
        self.max_len = int(max_len)
        self.min_len = 6
        self.max_charge = int(max_charge)
        self.label_column = str(label_column)

        # Base class handles Load-to-RAM and directory vs file logic
        super().__init__(parquet_path, load_to_ram=load_to_ram, max_samples=max_samples)

        cols = set(self._df.columns)
        if "normalized_sequence" in cols:
            self._seq_col = "normalized_sequence"
        else:
            self._seq_col = "modified_sequence" if "modified_sequence" in cols else "sequence"
        self._charge_col = "precursor_charge" if "precursor_charge" in cols else "charge"
        if "collision_energy" in cols:
            self._ce_col = "collision_energy"
        elif "nce" in cols:
            self._ce_col = "nce"
        else:
            self._ce_col = "orig_collision_energy"

        # Resolve label column
        candidate_labels = [self.label_column, "intensities_raw", "intensity_array", "intensities"]
        resolved = next((c for c in candidate_labels if c in cols), self.label_column)
        self.label_column = str(resolved)

        self._df["_naked"] = self._df[self._seq_col].map(lambda x: strip_modifications(x) if isinstance(x, str) else "")
        self._df["_len"] = self._df["_naked"].map(lambda x: len(x) if isinstance(x, str) else 0)

        # Determine label dimensionality
        sample_dim = 0
        for v in self._df[self.label_column]:
            try:
                arr = np.asarray(v, dtype=np.float32).reshape(-1)
                if arr.size > 0:
                    sample_dim = int(arr.size)
                    break
            except: continue
        
        self.level1_dim = sample_dim
        logger.info(f"UniSpecParquetDataset: total={len(self._df)}, dim={self.level1_dim}, load_to_ram={self.load_to_ram}")

    def __len__(self) -> int:
        return int(len(self._df))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if _UNISPEC_DEBUG and (int(idx) % 1000 == 0):
            print(f"DEBUG: __getitem__ {idx}")
        row = self._df.iloc[int(idx)]
        mod_seq = str(row[self._seq_col])
        seq = strip_modifications(mod_seq)
        seq = seq[: int(self.max_len)]
        charge_raw = int(row[self._charge_col])
        charge = int(np.clip(charge_raw, 1, int(self.max_charge)))
        nce = _normalize_nce(row[self._ce_col])

        label = _unispec_label(seq, charge, mod_seq, nce)

        y = np.asarray(row[self.label_column], dtype=np.float32).reshape(-1)
        # Ensure per-sample label vectors match the inferred dimensionality by
        # padding or truncating as needed instead of failing hard when there
        # is mild variation in stored length.
        if y.size < int(self.level1_dim):
            y_padded = np.zeros(int(self.level1_dim), dtype=np.float32)
            if y.size > 0:
                y_padded[: y.size] = y
            y = y_padded
        elif y.size > int(self.level1_dim):
            y = y[: int(self.level1_dim)].copy()

        # Full-spectrum mode only: do not apply canonical_by_mask. The model is
        # trained directly on the full spectral vector of length `level1_dim`,
        # with negative intensities clipped to zero.
        y_train = np.asarray(y, dtype=np.float32).copy()
        y_train[y_train < 0] = 0.0

        return {
            "label": label,
            "y_train": y_train,
            "y_eval": y,
            "modified_sequence": mod_seq,
            "precursor_charge": int(charge),
            "collision_energy": float(nce),
        }


def unispec_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = [b["label"] for b in batch]
    y_train = torch.from_numpy(np.stack([b["y_train"] for b in batch], axis=0)).float()
    y_eval = torch.from_numpy(np.stack([b["y_eval"] for b in batch], axis=0)).float()
    modified_sequence = [str(b["modified_sequence"]) for b in batch]
    precursor_charge = torch.tensor([int(b["precursor_charge"]) for b in batch], dtype=torch.long)
    collision_energy = torch.tensor([float(b.get("collision_energy", 0.0)) for b in batch], dtype=torch.float32)

    return {
        "labels": labels,
        "y_train": y_train,
        "y_eval": y_eval,
        "modified_sequence": modified_sequence,
        "precursor_charge": precursor_charge,
        "collision_energy": collision_energy,
    }


@torch.no_grad()
def _evaluate_unified(
    pred_level1: np.ndarray,
    true_level1: np.ndarray,
    meta_seq: List[str],
    meta_charge: List[int],
    meta_ce: Optional[List[float]] = None,
    meta_charge_in: Optional[List[int]] = None,
    meta_ce_in: Optional[List[float]] = None,
    *,
    max_len: int,
    export_csv_path: Optional[str] = None,
    export_max_samples: Optional[int] = None,
    export_vectors_path: Optional[str] = None,
) -> Dict[str, Any]:
    evaluator = UnifiedEvaluator(max_len=int(max_len), max_frag_charge=3)

    l1_sa: List[float] = []
    l1_sas: List[float] = []
    l1_pcc: List[float] = []
    l2_sa: List[float] = []
    l2_sas: List[float] = []
    l2_pcc: List[float] = []

    export_fh = None
    export_writer = None
    export_written = 0
    if export_csv_path:
        import csv

        export_fh = open(str(export_csv_path), "w", newline="")
        export_writer = csv.writer(export_fh)
        export_writer.writerow(
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
                "input_modified_sequence",
                "input_collision_energy",
                "input_precursor_charge",
            ]
        )

    try:
        for i in range(int(pred_level1.shape[0])):
            meta = {
                "modified_sequence": str(meta_seq[i]),
                "precursor_charge": int(meta_charge[i]),
            }
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
            except Exception:
                continue

            l1_sa.append(float(m["level1_sa"]))
            l1_sas.append(float(m["level1_sas"]))
            l1_pcc.append(float(m["level1_pcc"]))
            l2_sa.append(float(m["level2_sa"]))
            l2_sas.append(float(m["level2_sas"]))
            l2_pcc.append(float(m["level2_pcc"]))

            if export_writer is not None:
                if export_max_samples is not None and int(export_written) >= int(export_max_samples):
                    continue
                mod_seq = str(meta_seq[i])
                naked = strip_modifications(mod_seq)
                ce_i = float(meta_ce[i]) if meta_ce is not None and i < int(len(meta_ce)) else float("nan")
                charge_in = int(meta_charge_in[i]) if meta_charge_in is not None and i < int(len(meta_charge_in)) else int(meta_charge[i])
                ce_in = float(meta_ce_in[i]) if meta_ce_in is not None and i < int(len(meta_ce_in)) else float(ce_i)
                export_writer.writerow(
                    [
                        mod_seq,
                        naked,
                        int(len(naked)),
                        int(meta_charge[i]),
                        float(ce_i),
                        int(charge_in),
                        float(ce_in),
                        1 if ("UNIMOD" in mod_seq.upper()) else 0,
                        float(m["level1_sa"]),
                        float(m["level1_sas"]),
                        float(m["level1_pcc"]),
                        float(m["level2_sa"]),
                        float(m["level2_sas"]),
                        float(m["level2_pcc"]),
                        mod_seq,
                        float(ce_in),
                        int(charge_in),
                    ]
                )
                export_written += 1
    finally:
        if export_fh is not None:
            try:
                export_fh.close()
            except Exception:
                pass

    l1_mean_sa, l1_median_sa = summarize(l1_sa)
    l1_mean_sas, l1_median_sas = summarize(l1_sas)
    l1_mean_pcc, l1_median_pcc = summarize(l1_pcc)
    l2_mean_sa, l2_median_sa = summarize(l2_sa)
    l2_mean_sas, l2_median_sas = summarize(l2_sas)
    l2_mean_pcc, l2_median_pcc = summarize(l2_pcc)
    if export_vectors_path and int(pred_level1.shape[0]) > 0:
        np.savez(
            export_vectors_path,
            pred_level1=np.asarray(pred_level1, dtype=np.float32),
            true_level1=np.asarray(true_level1, dtype=np.float32),
            modified_sequence=np.array(meta_seq, dtype=object),
            precursor_charge=np.array(meta_charge, dtype=np.int32),
            input_precursor_charge=np.array(meta_charge_in if meta_charge_in is not None else meta_charge, dtype=np.int32),
        )

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

class UniSpecRunner(BenchmarkModel):
    def __init__(self, constraints: Optional[DataConstraints] = None) -> None:
        super().__init__()
        self._constraints = constraints or DataConstraints(max_len=40)

    def get_data_constraints(self) -> DataConstraints:
        return self._constraints

    def fit(self, train_path: str, val_path: Optional[str], output_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_unispec_imports()

        # Unified Setup
        device = get_optimal_device(config.get("experiment", {}).get("device", "auto"))
        num_workers = get_optimal_num_workers(config.get("models", {}).get("unispec", {}).get("num_workers", 8))

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / "best.ckpt"
        ckpt_last_path = out_dir / "last.ckpt"
        
        args = config.get("models", {}).get("unispec", {})
        batch_size = int(args.get("batch_size", 128))
        lr = float(args.get("base_lr", 1e-4))
        max_epochs = int(args.get("max_epochs", 200))

        # ... (rest of the loading and training logic using num_workers and device)
        logger.info(f"Using device: {device}, num_workers: {num_workers}")

        max_len = int(self._constraints.max_len)

        dict_dir = args.get("dict_dir")
        if dict_dir is None:
            dict_dir_p = project_root() / "external" / "UniSpec" / "saved_models" / "unispec23"
        else:
            dict_dir_p = Path(str(dict_dir))
            if not dict_dir_p.is_absolute():
                dict_dir_p = (project_root() / dict_dir_p).resolve()

        # Check if UNIMOD135 dictionary exists, rebuild if not, and always use it once available.
        unimod135_dir = dict_dir_p / "unimod135"
        if not unimod135_dir.exists():
            print("UNIMOD135 dictionary not found, rebuilding...")
            import subprocess
            rebuild_script = project_root() / "scripts" / "unispec" / "rebuild_dictionary.py"
            cmd = [
                sys.executable,
                str(rebuild_script),
                "--output_dir",
                str(unimod135_dir),
                "--max_len",
                str(max_len),
                "--max_frag_charge",
                "3",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root()))
            if result.returncode != 0:
                print(f"Dictionary rebuild failed: {result.stderr}")
                raise RuntimeError("Failed to rebuild UniSpec dictionary")
            print("Dictionary rebuild complete.")
        # Always point to the UNIMOD135-specific dictionary directory.
        dict_dir_p = unimod135_dir

        from utils import DicObj, LoadObj
        from models import FlipyFlopy

        dic = DicObj(
            seq_len=max_len,
            chlim=[1, 8],
            criteria_path=str(dict_dir_p / "criteria.txt"),
            stats_path=str(dict_dir_p / "ion_stats_train.txt"),
            mod_path=str(dict_dir_p / "modifications.txt"),
            mass_path=str(project_root() / "external" / "UniSpec" / "input_data" / "masses.txt"),
        )

        # Ensure the UniSpec ion dictionary is non-empty for LoadObj and
        # internal mass/label utilities, but decouple it from the output
        # dimensionality used for training.
        dicsz = getattr(dic, "dicsz", None)
        if dicsz is None:
            dicsz = int(len(getattr(dic, "dictionary", {}) or {}))
            try:
                setattr(dic, "dicsz", int(dicsz))
            except Exception:
                pass
        if int(dicsz) <= 0:
            raise RuntimeError(f"UniSpec dictionary at {dict_dir_p} is empty (dicsz={dicsz}).")

        train_ds = UniSpecParquetDataset(
            train_path,
            max_len=max_len,
            max_charge=int(self._constraints.max_charge),
            label_column=str(args.get("label_column", "intensities_raw")),
            max_samples=args.get("max_samples"),
        )
        val_ds = UniSpecParquetDataset(
            val_path if val_path else train_path,
            max_len=max_len,
            max_charge=int(self._constraints.max_charge),
            label_column=str(args.get("label_column", "intensities_raw")),
            max_samples=args.get("val_max_samples"),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=unispec_collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=unispec_collate_fn,
        )

        # Output dimensionality follows the label vector dimensionality so that
        # UniSpec can operate either in canonical 234-d mode (with fragment
        # masking in the dataset) or in full-spectrum mode without masking.
        out_dim = int(getattr(train_ds, "level1_dim", 0))
        if out_dim <= 0:
            raise RuntimeError("UniSpecParquetDataset reported non-positive label dimension")

        model_cfg = {
            "in_ch": int(dic.seq_channels),
            "seq_len": int(dic.seq_len),
            "out_dim": int(out_dim),
            "CEembed": True,
            "blocks": int(args.get("blocks", 6)),
            "embedsz": int(args.get("embedsz", 256)),
            "filtlast": int(args.get("filtlast", 512)),
            "head": args.get("head", (16, 16, 64)),
            "units": "None" if args.get("units") is None else str(args.get("units")),
            "drop": float(args.get("drop", 0.0)),
            "CEembed_units": int(args.get("CEembed_units", 256)),
        }

        device = _device_from_config(config)
        model = FlipyFlopy(**model_cfg, device=device)
        model.to(device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)

        load = LoadObj(dic, embed=True)

        # Use AdamW with weight decay for unified protocol
        weight_decay = float(config.get("protocol", {}).get("optimizer", {}).get("weight_decay", 0.01))
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Scheduler setup
        sched_cfg = config.get("protocol", {}).get("scheduler", {})
        scheduler_type = str(sched_cfg.get("type", "cosine"))
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 5))
        min_lr = float(sched_cfg.get("min_lr", 1e-6))
        scheduler = create_scheduler(
            opt,
            scheduler_type=scheduler_type,
            warmup_epochs=warmup_epochs,
            total_epochs=max_epochs,
            min_lr=min_lr,
        )
        
        # Early stopping setup
        es_cfg = config.get("protocol", {}).get("early_stopping", {})
        patience = int(es_cfg.get("patience", 10))
        early_stopping = EarlyStopping(patience=patience, mode="max")

        # Load checkpoint if requested (for resume)
        start_epoch = 0
        resume_path = args.get("resume", {}).get("path", "") if isinstance(args.get("resume", {}), dict) else ""
        if resume_path and Path(resume_path).exists():
            try:
                ckpt = torch.load(resume_path, map_location="cpu")
                state_dict = ckpt.get("model_state_dict")
                if state_dict:
                    if any(k.startswith("module.") for k in state_dict.keys()):
                        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
                    model.load_state_dict(state_dict, strict=True)
                if "optimizer_state_dict" in ckpt:
                    try:
                        opt.load_state_dict(ckpt["optimizer_state_dict"])
                    except Exception as e:
                        logger.warning("Failed to load optimizer state_dict: %s", e)
                if "scheduler_state_dict" in ckpt:
                    try:
                        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    except Exception as e:
                        logger.warning("Failed to load scheduler state_dict: %s", e)
                if "early_stopping" in ckpt:
                    try:
                        early_stopping.load_state_dict(ckpt["early_stopping"])
                    except Exception as e:
                        logger.warning("Failed to load early_stopping state: %s", e)
                start_epoch = int(ckpt.get("epoch", 0)) + 1
                logger.info("[UniSpec] Resumed from epoch %d", start_epoch - 1)
            except Exception as e:
                logger.warning("Failed to load checkpoint from %s: %s. Starting from scratch.", resume_path, e)
                start_epoch = 0
        
        cs = torch.nn.CosineSimilarity(dim=-1)

        def loss_fn(targ: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
            targ = targ.clone()
            pred = pred.clone()
            targ[targ > 0] = torch.sqrt(targ[targ > 0])
            pred[pred > 0] = torch.sqrt(pred[pred > 0])
            return -cs(targ, pred)

        best_value: Optional[float] = None
        best_epoch: Optional[int] = None
        time_to_best_seconds: Optional[float] = None
        best_state: Optional[Dict[str, Any]] = None
        history: List[Dict[str, Any]] = []
        train_start = float(time.perf_counter())

        for epoch in range(start_epoch, int(max_epochs)):
            reset_cuda_peak_memory()
            epoch_start = float(time.perf_counter())
            model.train()
            tr_losses: List[float] = []
            n_train_samples = 0
            n_train_steps = 0
            if _UNISPEC_DEBUG:
                print("DEBUG: Entering training loop")
            for batch in train_loader:
                if _UNISPEC_DEBUG and n_train_steps == 0:
                    print("DEBUG: Got first batch")
                n_train_steps += 1
                labels = batch["labels"]
                y = batch["y_train"].to(device, non_blocking=True)
                n_train_samples += int(y.shape[0])

                xs, _info = load.input_from_str(labels)
                xs_gpu = [t.to(device, non_blocking=True) for t in xs]

                pred, _, _ = model(xs_gpu, test=False)
                loss = loss_fn(y, pred).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                tr_losses.append(float(loss.detach().cpu().item()))

            epoch_time_seconds = float(time.perf_counter() - epoch_start)
            train_samples_per_second = float(n_train_samples) / float(epoch_time_seconds) if epoch_time_seconds > 0 else 0.0
            peak_vram_gb = float(get_cuda_peak_vram_gb())
            rss_gb = float(get_process_rss_gb())

            model.eval()
            pred_l1: List[np.ndarray] = []
            true_l1: List[np.ndarray] = []
            meta_seq: List[str] = []
            meta_charge: List[int] = []
            meta_ce: List[float] = []
            meta_charge_in: List[int] = []
            meta_ce_in: List[float] = []

            # No metadata override during training validation
            override_charge_i = None
            override_ce_f = None

            profiler = InferenceProfiler()
            with torch.no_grad():
                for batch in val_loader:
                    labels = batch["labels"]
                    y_eval = batch["y_eval"].cpu().numpy()

                    xs, _info = load.input_from_str(labels)
                    xs_gpu = [t.to(device, non_blocking=True) for t in xs]

                    with cuda_profile() as prof:
                        pred, _, _ = model(xs_gpu, test=False)
                    pred_np = pred.detach().cpu().numpy()

                    bs = int(pred_np.shape[0]) if hasattr(pred_np, "shape") else 1
                    profiler.add_measurement(float(prof["elapsed_ms"]) / float(max(1, bs)), float(prof["peak_vram_gb"]))

                    pred_l1.append(pred_np)
                    true_l1.append(y_eval)
                    meta_seq.extend(batch["modified_sequence"])
                    ch_true_list = [int(x) for x in batch["precursor_charge"].cpu().numpy().tolist()]
                    ce_true_list = [float(x) for x in batch.get("collision_energy").cpu().numpy().reshape(-1).tolist()]
                    meta_charge.extend(ch_true_list)
                    meta_ce.extend(ce_true_list)
                    if override_charge_i is not None:
                        meta_charge_in.extend([int(override_charge_i)] * int(len(ch_true_list)))
                    else:
                        meta_charge_in.extend([int(x) for x in ch_true_list])
                    if override_ce_f is not None:
                        meta_ce_in.extend([float(override_ce_f)] * int(len(ce_true_list)))
                    else:
                        meta_ce_in.extend([float(x) for x in ce_true_list])

            pred_l1_np = np.concatenate(pred_l1, axis=0) if pred_l1 else np.zeros((0, 1), dtype=np.float32)
            true_l1_np = np.concatenate(true_l1, axis=0) if true_l1 else np.zeros((0, 1), dtype=np.float32)

            unified = _evaluate_unified(
                pred_l1_np,
                true_l1_np,
                meta_seq,
                meta_charge,
                meta_ce,
                meta_charge_in,
                meta_ce_in,
                max_len=max_len,
            )

            prof_summary = profiler.get_summary()
            unified.update(
                {
                    "inference_median_ms_per_spectrum": float(prof_summary["median_time_ms"]),
                    "inference_mean_ms_per_spectrum": float(prof_summary["mean_time_ms"]),
                    "inference_mean_vram_gb": float(prof_summary["mean_vram_gb"]),
                    "inference_max_vram_gb": float(prof_summary["max_vram_gb"]),
                }
            )
            val_median_sas = float(unified.get("level1_median_sas", float("nan")))

            rec: Dict[str, Any] = {
                "epoch": int(epoch),
                "train_loss": float(np.mean(tr_losses)) if tr_losses else float("nan"),
                "val_median_sas": float(val_median_sas),
                "lr": float(scheduler.get_last_lr()[0]),
                "epoch_time_seconds": float(epoch_time_seconds),
                "train_samples": int(n_train_samples),
                "train_steps": int(n_train_steps),
                "train_samples_per_second": float(train_samples_per_second),
                "peak_vram_gb": float(peak_vram_gb),
                "rss_gb": float(rss_gb),
                "elapsed_seconds": float(time.perf_counter() - train_start),
            }
            history.append(rec)
            
            # Add explicit logging for epoch progress
            logger.info(
                f"[UniSpec] Epoch {epoch} complete: loss={rec['train_loss']:.4f}, "
                f"val_sas={rec['val_median_sas']:.4f}, time={rec['epoch_time_seconds']:.1f}s, "
                f"vram={rec['peak_vram_gb']:.1f}GB"
            )

            # Check if this is the best model
            if np.isfinite(val_median_sas) and early_stopping.is_best(val_median_sas):
                best_value = float(val_median_sas)
                best_epoch = int(epoch)
                time_to_best_seconds = float(time.perf_counter() - train_start)
                # Strip DataParallel "module." prefix so predict() can load without DP
                raw_sd = model.state_dict()
                clean_sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in raw_sd.items()}
                best_state = {
                    "model_state_dict": clean_sd,
                    "model_cfg": model_cfg,
                    "dict_dir": str(dict_dir_p),
                }
            
            # Always save last checkpoint (for resume)
            raw_sd = model.state_dict()
            clean_sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in raw_sd.items()}
            torch.save(
                {
                    "model_state_dict": clean_sd,
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "early_stopping": early_stopping.state_dict(),
                    "epoch": int(epoch),
                },
                ckpt_last_path,
            )

            # Step scheduler and check early stopping
            scheduler.step()
            if np.isfinite(val_median_sas) and early_stopping(val_median_sas, epoch):
                break

        ckpt_path = out_dir / "unispec.pt"
        if best_state is not None:
            best_state["best_epoch"] = int(best_epoch) if best_epoch is not None else None
            best_state["best_value"] = float(best_value) if best_value is not None else None
            best_state["time_to_best_seconds"] = float(time_to_best_seconds) if time_to_best_seconds is not None else None
            torch.save(best_state, ckpt_path)
        else:
            raw_sd = model.state_dict()
            clean_sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in raw_sd.items()}
            torch.save({"model_state_dict": clean_sd, "model_cfg": model_cfg, "dict_dir": str(dict_dir_p)}, ckpt_path)

        total_train_time_seconds = float(time.perf_counter() - train_start)

        result = {
            "model_path": str(ckpt_path),
            "best_value": float(best_value) if best_value is not None else None,
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "time_to_best_seconds": float(time_to_best_seconds) if time_to_best_seconds is not None else None,
            "train_total_time_seconds": float(total_train_time_seconds),
            "history": history,
        }

        with open(out_dir / "train_summary.json", "w") as f:
            json.dump(result, f, indent=2)

        return result

    def predict(self, parquet_path: str, model_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_unispec_imports()

        from utils import DicObj, LoadObj
        from models import FlipyFlopy

        model_dir_p = Path(model_dir)
        ckpt_candidates = [
            model_dir_p / "unispec.pt",
            model_dir_p / "best.ckpt",
            model_dir_p / "models" / "unispec" / "best.ckpt",
        ]
        ckpt_path = None
        for c in ckpt_candidates:
            if c.exists():
                ckpt_path = c
                break
        if ckpt_path is None:
            raise FileNotFoundError(
                f"UniSpec checkpoint not found. Searched: {[str(c) for c in ckpt_candidates]}"
            )

        args = config.get("models", {}).get("unispec", {})
        batch_size = int(args.get("batch_size", 128))
        num_workers = int(args.get("num_workers", 4))
        if num_workers < 0:
            num_workers = 0

        max_len = int(self._constraints.max_len)

        logger.info("[UniSpec] Loading checkpoint from %s...", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        logger.info("[UniSpec] Checkpoint loaded, building model...")
        dict_dir_raw = ckpt.get("dict_dir")
        dict_dir_p = Path(str(dict_dir_raw)) if dict_dir_raw is not None else Path("external/UniSpec/saved_models/unispec23/unimod135")
        if not dict_dir_p.is_absolute():
            dict_dir_p = (project_root() / dict_dir_p).resolve()
        if not dict_dir_p.exists():
            # Backward-compatible fallback when older checkpoints store
            # relative paths from a different CWD.
            dict_dir_p = (project_root() / "external" / "UniSpec" / "saved_models" / "unispec23" / "unimod135").resolve()
        model_cfg = dict(ckpt.get("model_cfg"))

        dic = DicObj(
            seq_len=max_len,
            chlim=[1, 8],
            criteria_path=str(dict_dir_p / "criteria.txt"),
            stats_path=str(dict_dir_p / "ion_stats_train.txt"),
            mod_path=str(dict_dir_p / "modifications.txt"),
            mass_path=str(project_root() / "external" / "UniSpec" / "input_data" / "masses.txt"),
        )

        device = _device_from_config(config)
        model = FlipyFlopy(**model_cfg, device=device)
        # Auto-strip DataParallel "module." prefix if present
        state = ckpt["model_state_dict"]
        if any(k.startswith("module.") for k in state):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()

        load = LoadObj(dic, embed=True)

        ds = UniSpecParquetDataset(
            parquet_path,
            max_len=max_len,
            max_charge=int(self._constraints.max_charge),
            label_column=str(args.get("label_column", "intensities_raw")),
            max_samples=args.get("eval_max_samples"),
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=unispec_collate_fn,
        )

        pred_l1: List[np.ndarray] = []
        true_l1: List[np.ndarray] = []
        meta_seq: List[str] = []
        meta_charge: List[int] = []
        meta_ce: List[float] = []
        meta_charge_in: List[int] = []
        meta_ce_in: List[float] = []

        analysis_cfg = config.get("protocol", {}).get("analysis", {}) if isinstance(config.get("protocol", {}), dict) else {}
        meta_override = analysis_cfg.get("metadata_override", {}) if isinstance(analysis_cfg, dict) else {}
        override_charge = meta_override.get("precursor_charge")
        override_ce = meta_override.get("collision_energy")
        try:
            override_charge_i = int(override_charge) if override_charge is not None else None
        except Exception:
            override_charge_i = None
        try:
            override_ce_f = float(override_ce) if override_ce is not None else None
        except Exception:
            override_ce_f = None

        profiler = InferenceProfiler()
        total_samples = 0
        total_elapsed_ms = 0.0
        n_batches = len(loader)
        logger.info("[UniSpec] Starting inference on %d batches...", n_batches)
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                labels = batch["labels"]
                y_eval = batch["y_eval"].cpu().numpy()

                if override_charge_i is not None or override_ce_f is not None:
                    labels_in: List[str] = []
                    ch_true = batch["precursor_charge"].cpu().numpy().reshape(-1).tolist()
                    ce_true = batch.get("collision_energy").cpu().numpy().reshape(-1).tolist()
                    for j, mod_seq in enumerate(batch["modified_sequence"]):
                        seq = strip_modifications(str(mod_seq))[: int(max_len)]
                        ch_in = int(override_charge_i) if override_charge_i is not None else int(ch_true[j])
                        ce_in = float(override_ce_f) if override_ce_f is not None else float(ce_true[j])
                        labels_in.append(_unispec_label(seq, ch_in, str(mod_seq), float(ce_in)))
                    labels = labels_in

                xs, _info = load.input_from_str(labels)
                xs_gpu = [t.to(device, non_blocking=True) for t in xs]

                with cuda_profile() as prof:
                    pred, _, _ = model(xs_gpu, test=False)
                pred_np = pred.detach().cpu().numpy()

                bs = int(pred_np.shape[0]) if hasattr(pred_np, "shape") else 1
                total_samples += int(bs)
                total_elapsed_ms += float(prof["elapsed_ms"])
                profiler.add_measurement(float(prof["elapsed_ms"]) / float(max(1, bs)), float(prof["peak_vram_gb"]))

                pred_l1.append(pred_np)
                true_l1.append(y_eval)
                meta_seq.extend(batch["modified_sequence"])
                ch_true_list = [int(x) for x in batch["precursor_charge"].cpu().numpy().tolist()]
                ce_true_list = [float(x) for x in batch.get("collision_energy").cpu().numpy().reshape(-1).tolist()]
                meta_charge.extend(ch_true_list)
                meta_ce.extend(ce_true_list)
                if override_charge_i is not None:
                    meta_charge_in.extend([int(override_charge_i)] * int(len(ch_true_list)))
                else:
                    meta_charge_in.extend([int(x) for x in ch_true_list])
                if override_ce_f is not None:
                    meta_ce_in.extend([float(override_ce_f)] * int(len(ce_true_list)))
                else:
                    meta_ce_in.extend([float(x) for x in ce_true_list])

                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
                    logger.info("[UniSpec] Inference batch %d/%d", batch_idx + 1, n_batches)

        pred_l1_np = np.concatenate(pred_l1, axis=0) if pred_l1 else np.zeros((0, 1), dtype=np.float32)
        true_l1_np = np.concatenate(true_l1, axis=0) if true_l1 else np.zeros((0, 1), dtype=np.float32)

        export_per_sample = bool(analysis_cfg.get("export_per_sample", False) or bool(args.get("export_per_sample", False)))
        export_max_samples = analysis_cfg.get("export_max_samples")
        export_vectors_path = analysis_cfg.get("export_vectors_path")
        try:
            export_max_samples_i = int(export_max_samples) if export_max_samples is not None else None
        except Exception:
            export_max_samples_i = None
        export_csv_path = str(model_dir_p / f"per_sample_{Path(parquet_path).stem}.csv") if export_per_sample else None

        # Optional: save pred/true for diagnose_unispec_output.py
        diagnose_save = analysis_cfg.get("diagnose_save_path") or config.get("_diagnose_save_path")
        if diagnose_save:
            np.savez(
                str(diagnose_save),
                pred=pred_l1_np,
                true=true_l1_np,
                meta_seq=np.array(meta_seq, dtype=object),
                meta_charge=np.array(meta_charge, dtype=np.int32),
            )

        unified = _evaluate_unified(
            pred_l1_np,
            true_l1_np,
            meta_seq,
            meta_charge,
            meta_ce,
            meta_charge_in,
            meta_ce_in,
            max_len=max_len,
            export_csv_path=export_csv_path,
            export_max_samples=export_max_samples_i,
            export_vectors_path=str(export_vectors_path) if export_vectors_path else None,
        )

        prof_summary = profiler.get_summary()
        unified.update(
            {
                "inference_median_ms_per_spectrum": float(prof_summary["median_time_ms"]),
                "inference_mean_ms_per_spectrum": float(prof_summary["mean_time_ms"]),
                "inference_samples_per_second": float(total_samples) / float(total_elapsed_ms / 1000.0) if total_elapsed_ms > 0 else 0.0,
                "inference_mean_vram_gb": float(prof_summary["mean_vram_gb"]),
                "inference_max_vram_gb": float(prof_summary["max_vram_gb"]),
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
