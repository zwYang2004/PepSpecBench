
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class BinningConfig:
    bin_size: float = 0.1
    max_mz: float = 2000.0

    @property
    def num_bins(self) -> int:
        return int(round(float(self.max_mz) / float(self.bin_size)))


def _as_1d_f32(x: np.ndarray | Iterable[float]) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _safe_normalize_l2(x: np.ndarray, eps: float) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if not np.isfinite(n) or n < eps:
        return np.zeros_like(x)
    return x / n


def spectral_angle(y_pred: np.ndarray | Iterable[float], y_true: np.ndarray | Iterable[float], eps: float = 1e-8) -> float:
    yp = _as_1d_f32(y_pred)
    yt = _as_1d_f32(y_true)
    if yp.shape != yt.shape:
        raise ValueError(f"Shape mismatch: pred={yp.shape}, true={yt.shape}")

    yp = np.clip(yp, 0.0, None)
    yt = np.clip(yt, 0.0, None)
    yp = _safe_normalize_l2(yp, eps)
    yt = _safe_normalize_l2(yt, eps)
    cos_sim = float(np.clip(np.dot(yp, yt), -1.0, 1.0))
    return float(np.arccos(cos_sim) / np.pi)


def spectral_angle_similarity(y_pred: np.ndarray | Iterable[float], y_true: np.ndarray | Iterable[float], eps: float = 1e-8) -> float:
    sa = spectral_angle(y_pred, y_true, eps=eps)
    if not np.isfinite(sa):
        return float("nan")
    return float(1.0 - sa)


def masked_spectral_angle(
    y_pred: np.ndarray | Iterable[float],
    y_true: np.ndarray | Iterable[float],
    mask: Optional[np.ndarray | Iterable[bool]] = None,
    eps: float = 1e-8,
) -> float:
    yp = _as_1d_f32(y_pred)
    yt = _as_1d_f32(y_true)
    if yp.shape != yt.shape:
        raise ValueError(f"Shape mismatch: pred={yp.shape}, true={yt.shape}")

    if mask is None:
        m = yt >= 0
    else:
        m = np.asarray(mask, dtype=bool).reshape(-1)
        if m.shape != yp.shape:
            raise ValueError(f"Mask shape mismatch: mask={m.shape}, vec={yp.shape}")

    if int(m.sum()) == 0:
        return float("nan")

    yp_m = np.clip(yp[m], 0.0, None)
    yt_m = np.clip(yt[m], 0.0, None)
    yp_m = _safe_normalize_l2(yp_m, eps)
    yt_m = _safe_normalize_l2(yt_m, eps)
    cos_sim = float(np.clip(np.dot(yp_m, yt_m), -1.0, 1.0))
    return float(np.arccos(cos_sim) / np.pi)


def masked_spectral_angle_similarity(
    y_pred: np.ndarray | Iterable[float],
    y_true: np.ndarray | Iterable[float],
    mask: Optional[np.ndarray | Iterable[bool]] = None,
    eps: float = 1e-8,
) -> float:
    sa = masked_spectral_angle(y_pred, y_true, mask=mask, eps=eps)
    if not np.isfinite(sa):
        return float("nan")
    return float(1.0 - sa)


def summarize(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(np.median(arr))


def bin_spectrum(
    mz: Optional[Iterable[float]],
    intensity: Optional[Iterable[float]],
    cfg: BinningConfig = BinningConfig(),
    eps: float = 1e-8,
) -> np.ndarray:
    num_bins = int(cfg.num_bins)
    out = np.zeros((num_bins,), dtype=np.float32)
    if mz is None or intensity is None:
        return out

    mz_arr = np.asarray(list(mz), dtype=np.float32).reshape(-1)
    it_arr = np.asarray(list(intensity), dtype=np.float32).reshape(-1)
    if mz_arr.size == 0 or it_arr.size == 0:
        return out

    n = int(min(mz_arr.size, it_arr.size))
    mz_arr = mz_arr[:n]
    it_arr = it_arr[:n]

    it_max = float(np.max(it_arr)) if it_arr.size else 0.0
    if it_max > eps:
        it_arr = it_arr / it_max
    else:
        it_arr = np.zeros_like(it_arr)

    idx = np.around(mz_arr / float(cfg.bin_size)).astype(np.int64)
    valid = (idx >= 0) & (idx < num_bins)
    if int(valid.sum()) == 0:
        return out

    np.add.at(out, idx[valid], it_arr[valid])
    out = np.sqrt(np.clip(out, 0.0, None))
    return out
