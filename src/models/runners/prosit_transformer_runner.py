from __future__ import annotations
import faulthandler
import signal
faulthandler.enable()
if hasattr(signal, 'SIGUSR1'):
    faulthandler.register(signal.SIGUSR1)

import os
import sys
import time
import logging

_tape_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../..", "external", "tape")
)
if os.path.isdir(_tape_path) and _tape_path not in sys.path:
    sys.path.insert(0, _tape_path)

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.constraints import DataConstraints
from src.datasets.prosit_transformer_dataset import PrositTransformerParquetDataset, prosit_transformer_collate_fn
from src.evaluator import UnifiedEvaluator
from src.metrics import summarize
from src.models.base_model import BenchmarkModel
from src.schedulers import EarlyStopping, create_scheduler
from src.utils.mass_calc import strip_modifications
from src.utils.profiling import InferenceProfiler, cuda_profile, get_cuda_peak_vram_gb, get_process_rss_gb, reset_cuda_peak_memory


logger = logging.getLogger(__name__)


def _append_jsonl(path: Path, rec: Dict[str, Any]) -> None:
    payload = dict(rec)
    payload["ts"] = float(time.time())
    with open(path, "a") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def _device_from_config(config: Dict[str, Any]) -> torch.device:
    prefer_gpu = bool(config.get("protocol", {}).get("prefer_gpu", True))
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _masked_spectral_angle_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(y_pred.shape)}, true={tuple(y_true.shape)}")

    mask = (y_true >= 0).to(dtype=y_pred.dtype)
    yp = torch.clamp(y_pred, min=0.0)
    yt = torch.clamp(y_true, min=0.0)

    dot = (yp * yt * mask).sum(dim=-1)
    yp_n = torch.sqrt((yp * yp * mask).sum(dim=-1) + eps)
    yt_n = torch.sqrt((yt * yt * mask).sum(dim=-1) + eps)
    cos = dot / (yp_n * yt_n + eps)
    cos = torch.clamp(cos, -1.0 + 1e-7, 1.0 - 1e-7)

    sa = torch.acos(cos) / float(np.pi)
    sa = torch.nan_to_num(sa, nan=0.0, posinf=1.0, neginf=1.0)
    return sa.mean()


@torch.no_grad()
def _evaluate_unified(
    model: nn.Module,
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
        for batch in dataloader:
            inp = {
                "input_ids": batch["input_ids"].to(device, non_blocking=True),
                "input_mask": batch["input_mask"].to(device, non_blocking=True),
                "collision_energy": batch["collision_energy"].to(device, non_blocking=True),
                "precursor_charge": batch["precursor_charge"].to(device, non_blocking=True),
            }
            labels = batch["labels"].to(device, non_blocking=True)

            # Apply metadata overrides if requested
            if override_charge is not None:
                # Prosit/Prosit-Transformer uses one-hot encoding for charge (dim=6)
                B = inp["precursor_charge"].shape[0]
                dtype = inp["precursor_charge"].dtype
                new_charge = torch.zeros((B, 6), device=device, dtype=dtype)
                c_idx = int(override_charge) - 1
                if 0 <= c_idx < 6:
                    new_charge[:, c_idx] = 1.0
                inp["precursor_charge"] = new_charge

            if override_nce is not None:
                # NCE is a scalar (tensor of shape (B, 1))
                B = inp["collision_energy"].shape[0]
                dtype = inp["collision_energy"].dtype
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


class PrositTransformerV2(nn.Module):
    def __init__(
        self,
        *,
        max_len: int = 40,
        hidden_size: int = 768,
        nhead: int = 12,
        num_layers: int = 4,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()

        self.max_len = int(max_len)
        self.num_ions = int((int(max_len) - 1) * 6)

        try:
            from tape.models.modeling_bert import ProteinBertModel

            print("DEBUG: Starting TAPE ProteinBertModel.from_pretrained('bert-base')...", flush=True)
            self.encoder = ProteinBertModel.from_pretrained("bert-base")
            print("DEBUG: TAPE model loaded successfully.", flush=True)
        except Exception as e:
            raise RuntimeError(
                "Missing dependency 'tape'. Install tape-proteins to use PrositTransformerV2 with pretrained encoder. "
                f"Original error: {e}"
            )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.meta_proj = nn.Sequential(
            nn.Linear(1 + 6, hidden_size),
            nn.GELU(),
        )

        self.pos_emb = nn.Embedding(int(max_len), int(hidden_size))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_size),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))

        self.head = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), 6),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"]
        input_mask = batch["input_mask"]
        ce = batch["collision_energy"]
        charge = batch["precursor_charge"]

        enc = self.encoder(input_ids=input_ids, input_mask=input_mask)
        last_hidden = enc[0] if isinstance(enc, (tuple, list)) else enc.last_hidden_state

        residue = last_hidden[:, 1 : 1 + int(self.max_len), :]
        cleavage = residue[:, :-1, :] + residue[:, 1:, :]

        pos = torch.arange(int(self.max_len) - 1, device=cleavage.device).unsqueeze(0)
        x = cleavage + self.pos_emb(pos)

        meta = torch.cat([ce, charge], dim=-1)
        meta_e = self.meta_proj(meta).unsqueeze(1)
        x = x + meta_e

        x = self.decoder(x)
        y = self.head(x)
        y = y.reshape(int(y.shape[0]), -1)
        if int(y.shape[-1]) != int(self.num_ions):
            raise RuntimeError(f"Output dim mismatch: got {int(y.shape[-1])}, expected {int(self.num_ions)}")
        return y


from src.utils.experiment import setup_experiment, save_best_model
from src.utils.hardware import get_optimal_device, get_optimal_num_workers

class PrositTransformerRunner(BenchmarkModel):
    def __init__(self, constraints: Optional[DataConstraints] = None) -> None:
        super().__init__()
        self._constraints = constraints or DataConstraints(max_len=40)

    def get_data_constraints(self) -> DataConstraints:
        return self._constraints

    def fit(self, train_path: str, val_path: Optional[str], output_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
        # Unified Setup
        device = get_optimal_device(config.get("experiment", {}).get("device", "auto"))
        num_workers = get_optimal_num_workers(config.get("models", {}).get("prosit_transformer", {}).get("num_workers", 8))

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / "best.ckpt"
        ckpt_last_path = out_dir / "last.ckpt"
        progress_path = out_dir / "train_progress.jsonl"

        args = config.get("models", {}).get("prosit_transformer", {})
        batch_size = int(args.get("batch_size", 64))
        lr = float(args.get("base_lr", 1e-4))
        max_epochs = int(args.get("max_epochs", 200))

        try:
            progress_every_steps_i = int(args.get("progress_every_steps", 0) or 0)
        except Exception:
            progress_every_steps_i = 0
        try:
            save_every_steps_i = int(args.get("save_every_steps", 0) or 0)
        except Exception:
            save_every_steps_i = 0

        # Optional resume configuration
        resume_cfg = args.get("resume", {}) if isinstance(args.get("resume", {}), dict) else {}
        resume_path = resume_cfg.get("path") or args.get("resume_from")
        # Initialize start_epoch and resume_step, will be updated from checkpoint if available
        try:
            start_epoch = int(resume_cfg.get("start_epoch", 0))
        except Exception:
            start_epoch = 0
        try:
            resume_step = int(resume_cfg.get("resume_step", 0))
        except Exception:
            resume_step = 0
        
        seed = int(config.get("experiment", {}).get("seed", 42))
        generator = torch.Generator()
        generator.manual_seed(seed)

        max_len = int(self._constraints.max_len)

        train_ds = PrositTransformerParquetDataset(
            train_path, 
            max_len=max_len, 
            label_column=str(args.get("label_column", "intensities_raw"))
        )
        val_ds = PrositTransformerParquetDataset(
            val_path if val_path else train_path,
            max_len=max_len,
            label_column=str(args.get("label_column", "intensities_raw")),
        )

        max_samples = args.get("max_samples")
        if max_samples is not None:
            try:
                n = int(max_samples)
            except Exception:
                n = -1
            if n > 0:
                from torch.utils.data import Subset

                train_ds = Subset(train_ds, list(range(min(n, len(train_ds)))))

        val_max_samples = args.get("val_max_samples")
        if val_max_samples is not None:
            try:
                n = int(val_max_samples)
            except Exception:
                n = -1
            if n > 0:
                from torch.utils.data import Subset

                val_ds = Subset(val_ds, list(range(min(n, len(val_ds)))))

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=prosit_transformer_collate_fn,
            generator=generator,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=prosit_transformer_collate_fn,
        )

        device = _device_from_config(config)

        model = PrositTransformerV2(max_len=max_len, num_layers=int(args.get("decoder_layers", 4)), freeze_encoder=bool(args.get("freeze_encoder", False)))
        model.to(device)
        
        # Check pos_emb before DataParallel wrapping
        if hasattr(model, "pos_emb"):
            print(f"prosit_transformer: max_len={max_len}, head_out={int((max_len-1)*6)}, pos_emb={int(model.pos_emb.num_embeddings)}")
        
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            logging.info("[PrositTransformerRunner] Using DataParallel on %d GPUs", torch.cuda.device_count())
            model = torch.nn.DataParallel(model)

        # Optimizer with weight decay
        weight_decay = float(config.get("protocol", {}).get("optimizer", {}).get("weight_decay", 0.01))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

        # Scheduler setup (Warmup + Cosine) - critical for Transformer stability
        sched_cfg = config.get("protocol", {}).get("scheduler", {})
        scheduler_type = str(sched_cfg.get("type", "cosine"))
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 5))
        min_lr = float(sched_cfg.get("min_lr", 1e-6))
        scheduler = create_scheduler(
            optimizer,
            scheduler_type=scheduler_type,
            warmup_epochs=warmup_epochs,
            total_epochs=max_epochs,
            min_lr=min_lr,
        )

        # Early stopping setup
        es_cfg = config.get("protocol", {}).get("early_stopping", {})
        patience = int(es_cfg.get("patience", 10))
        metric_name = str(es_cfg.get("metric", "val_median_sas"))
        mode = str(es_cfg.get("mode", "max")).lower()
        early_stopping = EarlyStopping(patience=patience, mode=mode)

        # Load checkpoint if requested
        ckpt = None
        if resume_path:
            ckpt_p = Path(resume_path)
            if not ckpt_p.is_absolute():
                ckpt_p = out_dir / ckpt_p
            if ckpt_p.exists():
                try:
                    ckpt = torch.load(str(ckpt_p), map_location="cpu")
                    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                        sd = ckpt["model_state_dict"]
                        clean_sd = {}
                        for k, v in sd.items():
                            name = k
                            while name.startswith("module."):
                                name = name[7:]
                            clean_sd[name] = v
                        model.load_state_dict(clean_sd, strict=True)
                    if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
                        try:
                            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                        except Exception as e:
                            logging.warning(f"Failed to load optimizer state_dict: {e}")
                    if isinstance(ckpt, dict) and "scheduler_state_dict" in ckpt:
                        try:
                            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                        except Exception as e:
                            logging.warning(f"Failed to load scheduler state_dict: {e}")
                    if isinstance(ckpt, dict) and "early_stopping" in ckpt and isinstance(ckpt["early_stopping"], dict):
                        try:
                            early_stopping.load_state_dict(ckpt["early_stopping"])
                        except Exception as e:
                            logging.warning(f"Failed to load early_stopping state: {e}")
                    if isinstance(ckpt, dict) and "rng_state" in ckpt:
                        try:
                            torch.set_rng_state(ckpt["rng_state"])
                        except Exception:
                            pass
                    # If CLI/config didn't override, use checkpoint epoch/step
                    if int(resume_cfg.get("start_epoch", -1)) < 0 and isinstance(ckpt, dict) and "epoch" in ckpt:
                        start_epoch = int(ckpt["epoch"])
                    if int(resume_cfg.get("resume_step", -1)) < 0 and isinstance(ckpt, dict) and "epoch_step" in ckpt:
                        resume_step = int(ckpt["epoch_step"]) + 1
                except Exception as e:
                    logging.warning(f"Failed to load checkpoint from {ckpt_p}: {e}. Starting from scratch.")
                    ckpt = None

        best_value: Optional[float] = None
        best_epoch: Optional[int] = None
        time_to_best_seconds: Optional[float] = None
        history: List[Dict[str, Any]] = []
        train_start = float(time.perf_counter())

        for epoch in range(start_epoch, int(max_epochs)):
            # Make shuffling deterministic per epoch so step-level resume can
            # skip batches reproducibly after a restart.
            generator.manual_seed(int(seed) + int(epoch))
            reset_cuda_peak_memory()
            epoch_start = float(time.perf_counter())
            current_resume_step = resume_step if epoch == start_epoch else 0
            
            model.train()
            train_losses: List[float] = []
            n_train_samples = 0
            n_train_steps = 0

            try:
                total_batches = int(len(train_loader))
            except Exception:
                total_batches = 0
            _append_jsonl(
                progress_path,
                {
                    "event": "epoch_start",
                    "epoch": int(epoch),
                    "max_epochs": int(max_epochs),
                    "resume_step": int(current_resume_step),
                    "total_batches": int(total_batches),
                },
            )

            for batch_idx, batch in enumerate(train_loader):
                # Skip batches for resume
                if batch_idx < current_resume_step:
                    continue
                
                # Add progress logging every 100 batches
                if batch_idx % 100 == 0:
                    logging.info("[PrositTransformerRunner] Epoch %d: Batch %d/%d", epoch + 1, batch_idx, total_batches)
                if batch_idx == 0:
                    logging.info("[PrositTransformerRunner] Epoch %d: Got first batch, batch_size=%d", epoch + 1, batch["input_ids"].shape[0] if "input_ids" in batch else 0)

                n_train_steps += 1

                optimizer.zero_grad(set_to_none=True)

                inp = {
                    "input_ids": batch["input_ids"].to(device, non_blocking=True),
                    "input_mask": batch["input_mask"].to(device, non_blocking=True),
                    "collision_energy": batch["collision_energy"].to(device, non_blocking=True),
                    "precursor_charge": batch["precursor_charge"].to(device, non_blocking=True),
                }
                labels = batch["labels"].to(device, non_blocking=True)
                n_train_samples += int(labels.shape[0])

                pred = model(inp)
                loss = _masked_spectral_angle_loss(pred, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(float(loss.detach().cpu().item()))

                if int(progress_every_steps_i) > 0 and (int(batch_idx) % int(progress_every_steps_i) == 0):
                    pct = (float(batch_idx + 1) / float(total_batches)) if total_batches > 0 else None
                    _append_jsonl(
                        progress_path,
                        {
                            "event": "train_step",
                            "epoch": int(epoch),
                            "batch_idx": int(batch_idx),
                            "total_batches": int(total_batches),
                            "percent": float(pct) if pct is not None else None,
                            "loss": float(train_losses[-1]) if train_losses else None,
                        },
                    )

                if int(save_every_steps_i) > 0 and (int(batch_idx) % int(save_every_steps_i) == 0):
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "early_stopping": early_stopping.state_dict(),
                            "epoch": int(epoch),
                            "epoch_step": int(batch_idx),
                            "rng_state": torch.get_rng_state(),
                            "best_value": float(best_value) if best_value is not None else None,
                            "time_to_best_seconds": float(time_to_best_seconds) if time_to_best_seconds is not None else None,
                            "constraints": {
                                "min_len": int(self._constraints.min_len),
                                "max_len": int(self._constraints.max_len),
                                "max_charge": int(self._constraints.max_charge),
                                "allowed_unimod_ids": list(self._constraints.allowed_unimod_ids),
                            },
                            "config": {"models": {"prosit_transformer": args}},
                        },
                        ckpt_last_path,
                    )

            epoch_time_seconds = float(time.perf_counter() - epoch_start)
            train_samples_per_second = float(n_train_samples) / float(epoch_time_seconds) if epoch_time_seconds > 0 else 0.0
            peak_vram_gb = float(get_cuda_peak_vram_gb())
            rss_gb = float(get_process_rss_gb())

            model.eval()
            val_losses: List[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    inp = {
                        "input_ids": batch["input_ids"].to(device, non_blocking=True),
                        "input_mask": batch["input_mask"].to(device, non_blocking=True),
                        "collision_energy": batch["collision_energy"].to(device, non_blocking=True),
                        "precursor_charge": batch["precursor_charge"].to(device, non_blocking=True),
                    }
                    labels = batch["labels"].to(device, non_blocking=True)
                    pred = model(inp)
                    vloss = _masked_spectral_angle_loss(pred, labels)
                    val_losses.append(float(vloss.detach().cpu().item()))

            unified = _evaluate_unified(model, val_loader, device, max_len=max_len)

            train_mean, train_median = summarize(train_losses)
            val_mean, val_median = summarize(val_losses)

            current_lr = scheduler.get_last_lr()[0]
            rec: Dict[str, Any] = {
                "epoch": int(epoch),
                "train_loss_mean": float(train_mean),
                "train_loss_median": float(train_median),
                "val_loss_mean": float(val_mean),
                "val_loss_median": float(val_median),
                "lr": float(current_lr),
                "unified": unified,
                "epoch_time_seconds": float(epoch_time_seconds),
                "train_samples": int(n_train_samples),
                "train_steps": int(n_train_steps),
                "train_samples_per_second": float(train_samples_per_second),
                "peak_vram_gb": float(peak_vram_gb),
                "rss_gb": float(rss_gb),
                "elapsed_seconds": float(time.perf_counter() - train_start),
            }
            history.append(rec)

            _append_jsonl(
                progress_path,
                {
                    "event": "epoch_end",
                    "epoch": int(epoch),
                    "train_loss_mean": float(train_mean),
                    "val_loss_mean": float(val_mean),
                    "level1_median_sas": float(unified.get("level1_median_sas", float("nan"))),
                },
            )

            # Get current metric value
            if metric_name == "val_median_sas":
                current = float(unified.get("level1_median_sas", float("nan")))
            elif metric_name == "val_mean_sas":
                current = float(unified.get("level1_mean_sas", float("nan")))
            elif metric_name == "val_loss_mean":
                current = float(val_mean)
            else:
                current = float(unified.get("level1_median_sas", float("nan")))

            prev_best = early_stopping.best_value
            improved = False
            should_stop = False
            if np.isfinite(current):
                should_stop = bool(early_stopping(float(current), int(epoch)))
                if prev_best is None:
                    improved = True
                elif str(early_stopping.mode).lower() == "max":
                    improved = float(current) > float(prev_best) + float(early_stopping.min_delta)
                else:
                    improved = float(current) < float(prev_best) - float(early_stopping.min_delta)

            # Save best checkpoint using the UPDATED early-stopping state.
            if improved:
                best_value = float(current)
                best_epoch = int(epoch)
                time_to_best_seconds = float(time.perf_counter() - train_start)
                last_batch_idx = int(batch_idx) if "batch_idx" in locals() else (int(max(0, int(total_batches) - 1)) if total_batches > 0 else int(0))
                # Strip DataParallel "module." prefix
                raw_sd = model.state_dict()
                clean_sd = {}
                for k, v in raw_sd.items():
                    name = k
                    while name.startswith("module."):
                        name = name[7:]
                    clean_sd[name] = v
                
                torch.save(
                    {
                        "model_state_dict": clean_sd,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "early_stopping": early_stopping.state_dict(),
                        "epoch": int(epoch),
                        "epoch_step": int(last_batch_idx),
                        "rng_state": torch.get_rng_state(),
                        "best_value": float(best_value) if best_value is not None else None,
                        "time_to_best_seconds": float(time_to_best_seconds),
                        "constraints": {
                            "min_len": int(self._constraints.min_len),
                            "max_len": int(self._constraints.max_len),
                            "max_charge": int(self._constraints.max_charge),
                            "allowed_unimod_ids": list(self._constraints.allowed_unimod_ids),
                        },
                        "config": {"models": {"prosit_transformer": args}},
                    },
                    ckpt_path,
                )

            # Always save latest checkpoint (also with UPDATED early-stopping state).
            last_batch_idx = int(batch_idx) if "batch_idx" in locals() else (int(max(0, int(total_batches) - 1)) if total_batches > 0 else int(0))
            
            raw_sd = model.state_dict()
            clean_sd = {}
            for k, v in raw_sd.items():
                name = k
                while name.startswith("module."):
                    name = name[7:]
                clean_sd[name] = v
            
            torch.save(
                {
                    "model_state_dict": clean_sd,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "early_stopping": early_stopping.state_dict(),
                    "epoch": int(epoch),
                    "epoch_step": int(last_batch_idx),
                    "rng_state": torch.get_rng_state(),
                    "best_value": float(best_value) if best_value is not None else None,
                    "time_to_best_seconds": float(time_to_best_seconds) if time_to_best_seconds is not None else None,
                    "constraints": {
                        "min_len": int(self._constraints.min_len),
                        "max_len": int(self._constraints.max_len),
                        "max_charge": int(self._constraints.max_charge),
                        "allowed_unimod_ids": list(self._constraints.allowed_unimod_ids),
                    },
                    "config": {"models": {"prosit_transformer": args}},
                },
                ckpt_last_path,
            )

            # Step scheduler and check early stopping
            scheduler.step()
            if bool(should_stop):
                break

        total_train_time_seconds = float(time.perf_counter() - train_start)

        result = {
            "model_path": str(ckpt_path),
            "best_value": float(best_value) if best_value is not None else None,
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "time_to_best_seconds": float(time_to_best_seconds) if time_to_best_seconds is not None else None,
            "train_total_time_seconds": float(total_train_time_seconds),
            "history": history,
        }

        with open(out_dir / "train_summary_prosit_transformer.json", "w") as f:
            json.dump(result, f, indent=2)

        with open(out_dir / "train_summary.json", "w") as f:
            json.dump(result, f, indent=2)

        return result

    def predict(self, parquet_path: str, model_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
        args = config.get("models", {}).get("prosit_transformer", {})

        model_dir_p = Path(model_dir)
        ckpt_candidates = [
            model_dir_p / "prosit_transformer.pt",
            model_dir_p / "best.ckpt",
            model_dir_p / "last.ckpt",
            model_dir_p / "models" / "prosit_transformer" / "best.ckpt",
        ]
        ckpt_path = None
        for c in ckpt_candidates:
            if c.exists():
                ckpt_path = c
                break
        if ckpt_path is None:
            raise FileNotFoundError(
                f"Prosit Transformer checkpoint not found. Searched: {[str(c) for c in ckpt_candidates]}"
            )

        max_len = int(self._constraints.max_len)
        batch_size = int(args.get("batch_size", 64))
        num_workers = int(args.get("num_workers", 4))
        if num_workers < 0:
            num_workers = 0

        ds = PrositTransformerParquetDataset(parquet_path, max_len=max_len, label_column=str(args.get("label_column", "intensities_raw")))

        eval_max_samples = args.get("eval_max_samples")
        if eval_max_samples is not None:
            try:
                n = int(eval_max_samples)
            except Exception:
                n = -1
            if n > 0:
                from torch.utils.data import Subset

                ds = Subset(ds, list(range(min(n, len(ds)))))
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=prosit_transformer_collate_fn,
        )

        device = _device_from_config(config)

        model = PrositTransformerV2(
            max_len=max_len,
            num_layers=int(args.get("decoder_layers", 4)),
            freeze_encoder=bool(args.get("freeze_encoder", False)),
        )

        logger.info("[PrositTransformer] Loading checkpoint from %s...", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        logger.info("[PrositTransformer] Checkpoint loaded, loading into model...")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            # Strip DataParallel "module." prefix if present
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        logger.info("[PrositTransformer] Model on device, starting inference on %d batches...", len(loader))

        profiler = InferenceProfiler()
        total_samples = 0
        total_elapsed_ms = 0.0
        n_batches = len(loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                inp = {
                    "input_ids": batch["input_ids"].to(device, non_blocking=True),
                    "input_mask": batch["input_mask"].to(device, non_blocking=True),
                    "collision_energy": batch["collision_energy"].to(device, non_blocking=True),
                    "precursor_charge": batch["precursor_charge"].to(device, non_blocking=True),
                }
                with cuda_profile() as prof:
                    pred = model(inp)
                bs = int(pred.shape[0]) if hasattr(pred, "shape") else 1
                total_samples += int(bs)
                total_elapsed_ms += float(prof["elapsed_ms"])
                profiler.add_measurement(float(prof["elapsed_ms"]) / float(max(1, bs)), float(prof["peak_vram_gb"]))
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
                    logger.info("[PrositTransformer] Inference batch %d/%d", batch_idx + 1, n_batches)

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

        # Add profiling results
        profile_summary = profiler.get_summary()
        unified.update({
            "inference_median_ms_per_spectrum": profile_summary["median_time_ms"],
            "inference_mean_ms_per_spectrum": profile_summary["mean_time_ms"],
            "inference_samples_per_second": float(total_samples) / float(total_elapsed_ms / 1000.0) if total_elapsed_ms > 0 else 0.0,
            "inference_mean_vram_gb": profile_summary["mean_vram_gb"],
            "inference_max_vram_gb": profile_summary["max_vram_gb"],
        })

        out_json = model_dir_p / f"eval_prosit_transformer_{Path(parquet_path).stem}.json"
        payload = {
            "dataset": str(parquet_path),
            "checkpoint": str(ckpt_path),
            "unified": unified,
        }
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)

        return {"metrics_json": str(out_json), "unified": unified}
