from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.metrics import (
    BinningConfig,
    bin_spectrum,
    masked_spectral_angle,
    spectral_angle,
)
from src.utils.mass_calc import (
    canonical_by_dim,
    canonical_by_mask,
    canonical_by_mz,
    extract_by_from_binned_spectrum,
    mz_to_bin_index,
    strip_modifications,
)


def _as_1d_f32(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _safe_pearson(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    xv = _as_1d_f32(x)
    yv = _as_1d_f32(y)
    if xv.shape != yv.shape:
        raise ValueError(f"Shape mismatch: x={xv.shape}, y={yv.shape}")
    if xv.size < 2:
        return 0.0
    xs = float(np.std(xv))
    ys = float(np.std(yv))
    if not np.isfinite(xs) or not np.isfinite(ys) or xs < eps or ys < eps:
        return 0.0
    c = float(np.corrcoef(xv, yv)[0, 1])
    if not np.isfinite(c):
        return 0.0
    return c


def _seq_and_charge(meta: Dict[str, Any]) -> Tuple[str, int]:
    seq = (
        meta.get("modified_sequence")
        or meta.get("sequence")
        or meta.get("seq")
        or ""
    )
    z = meta.get("precursor_charge")
    if z is None:
        z = meta.get("charge")
    try:
        charge = int(z) if z is not None else 0
    except Exception:
        charge = 0
    return str(seq), int(charge)


def _seq_len(seq: str) -> int:
    return len(strip_modifications(seq)) if isinstance(seq, str) else 0


@dataclass
class StandardizedSpectrum:
    level1: np.ndarray
    level1_mask: np.ndarray
    level2: np.ndarray


class UnifiedEvaluator:
    """Unified benchmark evaluator for PepSpecBench.

    All official metrics (SA, SAS, PCC) are computed in the **shared canonical
    evaluation space** ("level1"), defined as:

        dim(C) = (max_len - 1) x 2 x max_frag_charge
               = (40 - 1) x 2 x 3 = 234

    This space enumerates all b/y fragment ions up to max_len=40 residues and
    fragment charge up to 3.  A per-sample valid positional mask excludes
    physically impossible ions (cleavage positions beyond peptide length, or
    fragment charge exceeding precursor charge).

    The "level2" (20 000-bin full-spectrum) space is retained for diagnostic
    purposes but is NOT used for official benchmark reporting.
    """

    def __init__(
        self,
        *,
        max_len: int = 40,
        max_frag_charge: int = 3,
        bin_cfg: BinningConfig = BinningConfig(),
    ) -> None:
        self.max_len = int(max_len)
        self.max_frag_charge = int(max_frag_charge)
        self.bin_cfg = bin_cfg
        self.level1_dim = canonical_by_dim(max_len=self.max_len, max_frag_charge=self.max_frag_charge)
        self.level2_dim = int(self.bin_cfg.num_bins)

    def to_level1(self, pred_raw: Any, meta: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        seq, charge = _seq_and_charge(meta)
        seq_len = _seq_len(seq)
        mask = canonical_by_mask(
            seq_len,
            charge,
            max_len=self.max_len,
            max_frag_charge=self.max_frag_charge,
        )

        if isinstance(pred_raw, dict):
            if "level1" in pred_raw:
                pred_raw = pred_raw["level1"]
            elif "intensities" in pred_raw:
                pred_raw = pred_raw["intensities"]
            elif "spectrum" in pred_raw:
                pred_raw = pred_raw["spectrum"]

        if isinstance(pred_raw, tuple) and len(pred_raw) == 2:
            it, mz = pred_raw
            try:
                if mz is not None and it is not None and hasattr(mz, "__len__") and hasattr(it, "__len__"):
                    if len(mz) == len(it) and len(mz) > 0:
                        binned = bin_spectrum(mz, it, cfg=self.bin_cfg)
                        by, _ = extract_by_from_binned_spectrum(
                            binned,
                            seq,
                            charge,
                            max_len=self.max_len,
                            max_frag_charge=self.max_frag_charge,
                            cfg=self.bin_cfg,
                        )
                        by = np.clip(by, 0.0, None)
                        return by, mask
            except Exception:
                pass
            pred_raw = it

        y = _as_1d_f32(pred_raw)

        if y.size == self.level2_dim:
            by, _mask = extract_by_from_binned_spectrum(
                y,
                seq,
                charge,
                max_len=self.max_len,
                max_frag_charge=self.max_frag_charge,
                cfg=self.bin_cfg,
            )
            by = np.clip(by, 0.0, None)
            return by, mask

        if y.size != self.level1_dim:
            idx = np.where(mask)[0]
            if y.size == int(idx.size):
                out = np.zeros((self.level1_dim,), dtype=np.float32)
                out[idx] = y
                out = np.clip(out, 0.0, None)
                return out, mask

        out = np.zeros((self.level1_dim,), dtype=np.float32)
        n = int(min(out.size, y.size))
        out[:n] = y[:n]
        out = np.clip(out, 0.0, None)
        return out, mask

    def to_level2(self, pred_raw: Any, meta: Dict[str, Any]) -> np.ndarray:
        seq, charge = _seq_and_charge(meta)

        mz = None
        it = None

        if isinstance(pred_raw, dict):
            if "level2" in pred_raw:
                pred_raw = pred_raw["level2"]
            elif "spectrum" in pred_raw:
                pred_raw = pred_raw["spectrum"]
            elif "mz" in pred_raw and "intensity" in pred_raw:
                mz = pred_raw.get("mz")
                it = pred_raw.get("intensity")
            elif "mz" in pred_raw and "intensities" in pred_raw:
                mz = pred_raw.get("mz")
                it = pred_raw.get("intensities")

        if isinstance(pred_raw, tuple) and len(pred_raw) == 2:
            it, mz = pred_raw

        if mz is not None and it is not None:
            return bin_spectrum(mz, it, cfg=self.bin_cfg)

        y = _as_1d_f32(pred_raw)
        if y.size == self.level2_dim:
            return np.clip(y, 0.0, None)

        if y.size == self.level1_dim:
            mz_vec, _ = canonical_by_mz(
                seq,
                max_len=self.max_len,
                max_frag_charge=self.max_frag_charge,
            )
            binned = np.zeros((self.level2_dim,), dtype=np.float32)
            y = np.clip(y, 0.0, None)
            for i in range(int(min(y.size, mz_vec.size))):
                if y[i] <= 0:
                    continue
                idx = mz_to_bin_index(float(mz_vec[i]), cfg=self.bin_cfg)
                if idx is None:
                    continue
                binned[idx] += float(y[i])
            return np.sqrt(np.clip(binned, 0.0, None))

        out = np.zeros((self.level2_dim,), dtype=np.float32)
        n = int(min(out.size, y.size))
        out[:n] = y[:n]
        return np.clip(out, 0.0, None)

    def standardize(self, pred_raw: Any, meta: Dict[str, Any]) -> StandardizedSpectrum:
        l1, m1 = self.to_level1(pred_raw, meta)
        l2 = self.to_level2(pred_raw, meta)
        return StandardizedSpectrum(level1=l1, level1_mask=m1, level2=l2)

    def compute_metrics(
        self,
        pred_level1: np.ndarray,
        true_level1: np.ndarray,
        level1_mask: np.ndarray,
        pred_level2: np.ndarray,
        true_level2: np.ndarray,
    ) -> Dict[str, float]:
        m = np.asarray(level1_mask, dtype=bool).reshape(-1)
        p1 = _as_1d_f32(pred_level1)
        t1 = _as_1d_f32(true_level1)
        p2 = _as_1d_f32(pred_level2)
        t2 = _as_1d_f32(true_level2)

        if p1.shape != t1.shape:
            raise ValueError(f"Level1 shape mismatch: pred={p1.shape}, true={t1.shape}")
        if p2.shape != t2.shape:
            raise ValueError(f"Level2 shape mismatch: pred={p2.shape}, true={t2.shape}")
        if m.shape != p1.shape:
            raise ValueError(f"Level1 mask mismatch: mask={m.shape}, vec={p1.shape}")

        p1 = np.clip(p1, 0.0, None)
        t1 = np.clip(t1, 0.0, None)
        p2 = np.clip(p2, 0.0, None)
        t2 = np.clip(t2, 0.0, None)

        l1_sa = masked_spectral_angle(p1, t1, mask=m)
        l2_sa = spectral_angle(p2, t2)

        p1m = p1[m]
        t1m = t1[m]

        l1_pcc = _safe_pearson(p1m, t1m)
        l2_pcc = _safe_pearson(p2, t2)

        # Official benchmark metrics: shared canonical space (level1, dim=234).
        # level2 (20k-bin full-spectrum) keys are retained for backward
        # compatibility with existing runners but are NOT official metrics.
        return {
            "level1_sa": float(l1_sa),
            "level1_sas": float(1.0 - l1_sa) if np.isfinite(l1_sa) else float("nan"),
            "level1_pcc": float(l1_pcc),
            "level2_sa": float(l2_sa),
            "level2_sas": float(1.0 - l2_sa) if np.isfinite(l2_sa) else float("nan"),
            "level2_pcc": float(l2_pcc),
        }

    def compute_level2_metrics(
        self,
        pred_peaks: List[Tuple[float, float]],
        true_peaks: List[Tuple[float, float]],
    ) -> Dict[str, float]:
        pred_mz = np.array([p[0] for p in pred_peaks])
        pred_it = np.array([p[1] for p in pred_peaks])
        true_mz = np.array([p[0] for p in true_peaks])
        true_it = np.array([p[1] for p in true_peaks])

        pred_binned = bin_spectrum(pred_mz, pred_it, cfg=self.bin_cfg)
        true_binned = bin_spectrum(true_mz, true_it, cfg=self.bin_cfg)

        l2_sa = spectral_angle(pred_binned, true_binned)
        return {
            "level2_sa": float(l2_sa),
            "level2_sas": float(1.0 - l2_sa) if np.isfinite(l2_sa) else float("nan"),
            "level2_pcc": float(_safe_pearson(pred_binned, true_binned)),
        }
