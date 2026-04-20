from __future__ import annotations

import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import project_root
from src.constraints import DataConstraints
from src.datasets.prosit_dataset import PrositParquetDataset, PrositParquetIterableDataset, prosit_collate_fn
from src.evaluator import UnifiedEvaluator
from src.metrics import summarize
from src.models.base_model import BenchmarkModel
from src.schedulers import create_scheduler, EarlyStopping
from src.utils.experiment import save_best_model
from src.utils.hardware import get_optimal_device, get_optimal_num_workers
from src.utils.mass_calc import strip_modifications
from src.utils.profiling import InferenceProfiler, cuda_profile, get_cuda_peak_vram_gb, get_process_rss_gb, reset_cuda_peak_memory


logger = logging.getLogger(__name__)


def _append_jsonl(path: Path, rec: Dict[str, Any]) -> None:
    payload = dict(rec)
    payload["ts"] = float(time.time())
    with open(path, "a") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def _ensure_dlomix_torch_backend() -> None:
    os.environ.setdefault("DLOMIX_BACKEND", "torch")

    dlomix_src = project_root() / "external" / "dlomix" / "src"
    dlomix_src_str = str(dlomix_src)
    if dlomix_src_str not in sys.path:
        sys.path.insert(0, dlomix_src_str)


def _device_from_config(config: Dict[str, Any]) -> torch.device:
    prefer_gpu = bool(config.get("protocol", {}).get("prefer_gpu", True))
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def _evaluate_unified(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    max_len: int,
    export_csv_path: Optional[str] = None,
    export_max_samples: Optional[int] = None,
    export_vectors_path: Optional[str] = None,
    override_charge: Optional[int] = None,
    override_nce: Optional[float] = None,
) -> Dict[str, Any]:
    model.eval()

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
            ]
        )
    vec_pred_l1: List[np.ndarray] = []
    vec_true_l1: List[np.ndarray] = []
    vec_seq: List[str] = []
    vec_charge: List[int] = []

    try:
        total_batches = len(dataloader)
        print(f"  [Evaluation] Starting evaluation on {total_batches} batches...", flush=True)
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                print(f"  [Evaluation] Batch {batch_idx+1}/{total_batches} - CPU RSS: {get_process_rss_gb():.2f} GB", flush=True)
            
            inp = {
                "sequence": batch["sequence"].to(device, non_blocking=True),
                "collision_energy": batch["collision_energy"].to(device, non_blocking=True),
                "precursor_charge": batch["precursor_charge"].to(device, non_blocking=True),
            }
            labels = batch["labels"].to(device, non_blocking=True)

            # Apply metadata overrides if requested
            if override_charge is not None:
                # Prosit uses one-hot encoding for charge (dim=6)
                B = inp["precursor_charge"].shape[0]
                dtype = inp["precursor_charge"].dtype
                new_charge = torch.zeros((B, 6), device=device, dtype=dtype)
                c_idx = int(override_charge) - 1
                if 0 <= c_idx < 6:
                    new_charge[:, c_idx] = 1.0
                inp["precursor_charge"] = new_charge

            if override_nce is not None:
                # Prosit uses normalized collision energy (0-1)
                B = inp["collision_energy"].shape[0]
                dtype = inp["collision_energy"].dtype
                # Clamp NCE to reasonable range if needed, but here we trust the override value
                inp["collision_energy"] = torch.full((B, 1), float(override_nce), device=device, dtype=dtype)

            pred = model(inp)

            pred_np = pred.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            seq_raw = batch.get("modified_sequence", [""] * int(pred_np.shape[0]))
            charges = batch.get("precursor_charge_int", [0] * int(pred_np.shape[0]))

            ce_vals = batch.get("collision_energy")
            if isinstance(ce_vals, torch.Tensor):
                ce_vals_np = ce_vals.detach().cpu().numpy().reshape(-1)
            else:
                ce_vals_np = None

            for i in range(int(pred_np.shape[0])):
                meta = {
                    "modified_sequence": str(seq_raw[i]),
                    "precursor_charge": int(charges[i]),
                }
                try:
                    pred_std = evaluator.standardize(pred_np[i], meta)
                    true_std = evaluator.standardize(labels_np[i], meta)
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
                vec_pred_l1.append(np.asarray(pred_std.level1, dtype=np.float32))
                vec_true_l1.append(np.asarray(true_std.level1, dtype=np.float32))
                vec_seq.append(str(seq_raw[i]))
                vec_charge.append(int(charges[i]))

                if export_writer is not None:
                    if export_max_samples is not None and int(export_written) >= int(export_max_samples):
                        continue
                    mod_seq = str(seq_raw[i])
                    naked = strip_modifications(mod_seq)
                    ce_i = float(ce_vals_np[i]) if ce_vals_np is not None and i < int(ce_vals_np.size) else float("nan")
                    
                    input_charge = int(override_charge) if override_charge is not None else int(charges[i])
                    input_ce = float(override_nce) if override_nce is not None else ce_i

                    export_writer.writerow(
                        [
                            mod_seq,
                            naked,
                            int(len(naked)),
                            int(charges[i]),
                            float(ce_i),
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
    if export_vectors_path and vec_pred_l1:
        np.savez(
            export_vectors_path,
            pred_level1=np.stack(vec_pred_l1, axis=0),
            true_level1=np.stack(vec_true_l1, axis=0),
            modified_sequence=np.array(vec_seq, dtype=object),
            precursor_charge=np.array(vec_charge, dtype=np.int32),
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


class PrositRunner(BenchmarkModel):
    def __init__(self, constraints: Optional[DataConstraints] = None) -> None:
        super().__init__()
        self._constraints = constraints or DataConstraints(max_len=40)

    def get_data_constraints(self) -> DataConstraints:
        return self._constraints

    def fit(self, train_path: str, val_path: Optional[str], output_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_dlomix_torch_backend()

        from dlomix.losses.intensity_torch import masked_spectral_distance
        from dlomix.models import PrositIntensityPredictor

        # Unified Setup (now handled by orchestrator, but runner uses derived params)
        device = get_optimal_device(config.get("experiment", {}).get("device", "auto"))
        num_workers = get_optimal_num_workers(config.get("models", {}).get("prosit", {}).get("num_workers", 8))
        
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / "best.ckpt"
        ckpt_last_path = out_dir / "last.ckpt"
        progress_path = out_dir / "train_progress.jsonl"

        args = config.get("models", {}).get("prosit", {})
        batch_size = int(args.get("batch_size", 128))
        lr = float(args.get("base_lr", 1e-4))
        max_epochs = int(args.get("max_epochs", 200))
        
        seed = int(config.get("experiment", {}).get("seed", 42))
        generator = torch.Generator()
        generator.manual_seed(seed)

        max_len = int(self._constraints.max_len)
        
        # Load-to-RAM handled via PrositParquetDataset inheritance from BaseParquetDataset
        train_ds = PrositParquetDataset(
            train_path,
            max_length=max_len,
            min_peptide_len=int(self._constraints.min_len),
            max_peptide_len=int(self._constraints.max_len),
            max_charge=int(self._constraints.max_charge),
            label_column=str(args.get("label_column", "intensities_raw")),
            max_samples=args.get("max_samples"),
        )
        val_ds = PrositParquetDataset(
            val_path if val_path else train_path,
            max_length=max_len,
            min_peptide_len=int(self._constraints.min_len),
            max_peptide_len=int(self._constraints.max_len),
            max_charge=int(self._constraints.max_charge),
            label_column=str(args.get("label_column", "intensities_raw")),
            max_samples=args.get("val_max_samples"),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=prosit_collate_fn,
            generator=generator,
        )
        val_num_workers = 0
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=prosit_collate_fn,
        )

        model = PrositIntensityPredictor(
            seq_length=max_len,
            len_fion=6,
            use_prosit_ptm_features=False,
            input_keys={"SEQUENCE_KEY": "sequence"},
            meta_data_keys={
                "COLLISION_ENERGY_KEY": "collision_energy",
                "PRECURSOR_CHARGE_KEY": "precursor_charge",
            },
            with_termini=False,
        )
        model.to(device)
        
        # Initialize LazyModule parameters with a dummy forward pass before DataParallel
        # This is required because DataParallel needs all parameters to be initialized
        model.eval()
        with torch.no_grad():
            dummy_seq = torch.zeros((1, max_len), dtype=torch.long, device=device)
            dummy_ce = torch.zeros((1, 1), dtype=torch.float32, device=device)
            dummy_charge = torch.zeros((1, 6), dtype=torch.float32, device=device)
            dummy_inp = {
                "sequence": dummy_seq,
                "collision_energy": dummy_ce,
                "precursor_charge": dummy_charge,
            }
            _ = model(dummy_inp)
        model.train()
        
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            logging.info("[PrositRunner] Using DataParallel on %d GPUs", torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
            logging.info("[PrositRunner] DataParallel wrapper applied successfully")

        # Optimizer with weight decay for unified protocol
        logging.info("[PrositRunner] Creating optimizer...")
        weight_decay = float(config.get("protocol", {}).get("optimizer", {}).get("weight_decay", 0.01))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        logging.info("[PrositRunner] Optimizer created successfully")

        # Scheduler setup
        sched_cfg = config.get("protocol", {}).get("scheduler", {})
        scheduler = create_scheduler(
            optimizer,
            scheduler_type=str(sched_cfg.get("type", "cosine")),
            warmup_epochs=int(sched_cfg.get("warmup_epochs", 5)),
            total_epochs=max_epochs,
            min_lr=float(sched_cfg.get("min_lr", 1e-6)),
        )

        # Early stopping setup
        es_cfg = config.get("protocol", {}).get("early_stopping", {})
        early_stopping = EarlyStopping(
            patience=int(es_cfg.get("patience", 10)), 
            mode=str(es_cfg.get("mode", "max")).lower()
        )
        metric_name = str(es_cfg.get("metric", "val_median_sas"))

        # Load checkpoint if requested
        resume_path = args.get("resume", {}).get("path", "") if isinstance(args.get("resume", {}), dict) else ""
        if resume_path and Path(resume_path).exists():
            try:
                ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
                state_dict = ckpt["model_state_dict"]
                # Match model's expected keys: DataParallel expects "module.xxx", checkpoint may have "xxx"
                model_keys = list(model.state_dict().keys())
                ckpt_has_module = any(k.startswith("module.") for k in state_dict.keys())
                model_has_module = any(k.startswith("module.") for k in model_keys)
                if model_has_module and not ckpt_has_module:
                    state_dict = {"module." + k: v for k, v in state_dict.items()}
                elif not model_has_module and ckpt_has_module:
                    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                if "optimizer_state_dict" in ckpt:
                    try:
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    except Exception as e:
                        logging.warning(f"Failed to load optimizer state_dict: {e}")
                if "scheduler_state_dict" in ckpt:
                    try:
                        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    except Exception as e:
                        logging.warning(f"Failed to load scheduler state_dict: {e}")
                if "early_stopping" in ckpt:
                    try:
                        early_stopping.load_state_dict(ckpt["early_stopping"])
                    except Exception as e:
                        logging.warning(f"Failed to load early_stopping state: {e}")
                start_epoch = ckpt.get("epoch", 0)
            except Exception as e:
                logging.warning(f"Failed to load checkpoint from {resume_path}: {e}. Starting from scratch.")
                start_epoch = 0
        else:
            start_epoch = 0

        progress_path = out_dir / "train_progress.jsonl"
        best_value: Optional[float] = None
        best_epoch: Optional[int] = None
        time_to_best_seconds: Optional[float] = None
        history: List[Dict[str, Any]] = []

        train_start = float(time.perf_counter())
        val_results: Dict[str, Any] = {}
        logging.info("[PrositRunner] Starting training loop, max_epochs=%d, start_epoch=%d", max_epochs, start_epoch)
        for epoch in range(start_epoch, max_epochs):
            generator.manual_seed(seed + epoch)
            reset_cuda_peak_memory()
            epoch_start = float(time.perf_counter())
            model.train()
            logging.info("[PrositRunner] Epoch %d/%d: Starting training", epoch + 1, max_epochs)

            train_losses: List[float] = []
            n_train_samples = 0

            logging.info("[PrositRunner] Epoch %d: Starting to iterate over train_loader", epoch + 1)
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx % 100 == 0:
                    logging.info("[PrositRunner] Epoch %d: Batch %d/%d", epoch + 1, batch_idx, len(train_loader))
                if batch_idx == 0:
                    logging.info("[PrositRunner] Epoch %d: Got first batch, batch_size=%d", epoch + 1, batch["sequence"].shape[0] if "sequence" in batch else 0)
                optimizer.zero_grad(set_to_none=True)

                inp = {
                    "sequence": batch["sequence"].to(device, non_blocking=True),
                    "collision_energy": batch["collision_energy"].to(device, non_blocking=True),
                    "precursor_charge": batch["precursor_charge"].to(device, non_blocking=True),
                }
                labels = batch["labels"].to(device, non_blocking=True)
                n_train_samples += int(labels.shape[0])

                if batch_idx == 0:
                    logging.info("[PrositRunner] Epoch %d: Starting forward pass on first batch", epoch + 1)
                pred = model(inp)
                loss = masked_spectral_distance(labels, pred).mean()
                if batch_idx == 0:
                    logging.info("[PrositRunner] Epoch %d: Completed forward pass, loss=%.4f", epoch + 1, float(loss.item()))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(float(loss.detach().cpu().item()))

            epoch_time = float(time.perf_counter() - epoch_start)
            train_mean_loss = float(np.mean(train_losses)) if train_losses else float("nan")

            # Validation and Best Model Check
            val_results = _evaluate_unified(model, val_loader, device, max_len=max_len)
            current_val_metric = val_results.get("level1_median_sas", 0.0)

            rec = {
                "epoch": int(epoch),
                "train_loss_mean": float(train_mean_loss),
                "val_median_sas": float(current_val_metric),
                "lr": float(scheduler.get_last_lr()[0]),
                "epoch_time_seconds": float(epoch_time),
                "train_samples": int(n_train_samples),
                "elapsed_seconds": float(time.perf_counter() - train_start),
            }
            history.append(rec)
            _append_jsonl(progress_path, {"event": "epoch_end", **rec})

            is_best = early_stopping.is_best(float(current_val_metric))
            should_stop = bool(early_stopping(float(current_val_metric), epoch))

            if is_best:
                best_value = float(current_val_metric)
                best_epoch = int(epoch)
                time_to_best_seconds = float(time.perf_counter() - train_start)
                # Strip DataParallel "module." prefix so predict() can load without DP
                raw_sd = model.state_dict()
                clean_sd = {}
                for k, v in raw_sd.items():
                    name = k
                    while name.startswith("module."):
                        name = name[7:]
                    clean_sd[name] = v
                best_state = {
                    "epoch": epoch,
                    "model_state_dict": clean_sd,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_median_sas": current_val_metric,
                }
                torch.save(best_state, ckpt_path)
            # Always save last checkpoint
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "early_stopping": early_stopping.state_dict(),
                    "epoch": int(epoch),
                },
                ckpt_last_path,
            )

            save_best_model(model, output_dir, "prosit", val_results, is_best)

            if should_stop:
                break
            scheduler.step()

        result = {
            "model_path": str(ckpt_path),
            "best_value": float(best_value) if best_value is not None else None,
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "time_to_best_seconds": float(time_to_best_seconds) if time_to_best_seconds is not None else None,
            "train_total_time_seconds": float(time.perf_counter() - train_start),
            "unified": val_results,
            "history": history,
        }
        with open(out_dir / "train_summary.json", "w") as f:
            json.dump(result, f, indent=2)

        return result

    def predict(self, parquet_path: str, model_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_dlomix_torch_backend()

        from dlomix.models import PrositIntensityPredictor

        model_dir_p = Path(model_dir)
        # Check multiple possible checkpoint locations
        ckpt_candidates = [
            model_dir_p / "prosit.pt",
            model_dir_p / "best.ckpt",
            model_dir_p / "last.ckpt",
            model_dir_p / "models" / "prosit" / "best.ckpt",
        ]
        ckpt_path = None
        for c in ckpt_candidates:
            if c.exists():
                ckpt_path = c
                break
        if ckpt_path is None:
            raise FileNotFoundError(
                f"Prosit checkpoint not found. Searched: {[str(c) for c in ckpt_candidates]}"
            )

        args = config.get("models", {}).get("prosit", {})
        batch_size = int(args.get("batch_size", 128))
        num_workers = int(args.get("num_workers", 4))
        if num_workers < 0:
            num_workers = 0

        max_len = int(self._constraints.max_len)

        ds = PrositParquetDataset(
            parquet_path,
            max_length=max_len,
            min_peptide_len=int(self._constraints.min_len),
            max_peptide_len=int(self._constraints.max_len),
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
            collate_fn=prosit_collate_fn,
        )

        device = _device_from_config(config)

        model = PrositIntensityPredictor(
            seq_length=max_len,
            len_fion=6,
            use_prosit_ptm_features=False,
            input_keys={"SEQUENCE_KEY": "sequence"},
            meta_data_keys={
                "COLLISION_ENERGY_KEY": "collision_energy",
                "PRECURSOR_CHARGE_KEY": "precursor_charge",
            },
            with_termini=False,
        )

        ckpt_obj = torch.load(ckpt_path, map_location="cpu")
        # Support both formats:
        # 1) runner checkpoints: {"model_state_dict": ...}
        # 2) save_best_model checkpoints: raw state_dict
        if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
            state = ckpt_obj["model_state_dict"]
        else:
            state = ckpt_obj
        # Strip DataParallel "module." prefix if present
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.to(device)

        profiler = InferenceProfiler()
        with torch.no_grad():
            for batch in loader:
                inp = {
                    "sequence": batch["sequence"].to(device, non_blocking=True),
                    "collision_energy": batch["collision_energy"].to(device, non_blocking=True),
                    "precursor_charge": batch["precursor_charge"].to(device, non_blocking=True),
                }
                with cuda_profile() as prof:
                    pred = model(inp)
                bs = int(pred.shape[0]) if hasattr(pred, "shape") else 1
                profiler.add_measurement(float(prof["elapsed_ms"]) / float(max(1, bs)), float(prof["peak_vram_gb"]))

        analysis_cfg = config.get("protocol", {}).get("analysis", {}) if isinstance(config.get("protocol", {}), dict) else {}
        
        # Check for metadata overrides
        meta_override = analysis_cfg.get("metadata_override", {})
        override_charge = meta_override.get("precursor_charge")
        override_nce = meta_override.get("collision_energy")
        
        export_per_sample = bool(analysis_cfg.get("export_per_sample", False) or bool(args.get("export_per_sample", False)))
        export_max_samples = analysis_cfg.get("export_max_samples")
        export_vectors_path = analysis_cfg.get("export_vectors_path")
        try:
            export_max_samples_i = int(export_max_samples) if export_max_samples is not None else None
        except Exception:
            export_max_samples_i = None
        export_csv_path = str(model_dir_p / f"per_sample_{Path(parquet_path).stem}.csv") if export_per_sample else None

        unified = _evaluate_unified(
            model,
            loader,
            device,
            max_len=max_len,
            export_csv_path=export_csv_path,
            export_max_samples=export_max_samples_i,
            export_vectors_path=str(export_vectors_path) if export_vectors_path else None,
            override_charge=override_charge,
            override_nce=override_nce,
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

        out_json = model_dir_p / f"eval_{Path(parquet_path).stem}.json"
        payload = {
            "dataset": str(parquet_path),
            "checkpoint": str(ckpt_path),
            "unified": unified,
        }
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)

        return {"metrics_json": str(out_json), "unified": unified}
