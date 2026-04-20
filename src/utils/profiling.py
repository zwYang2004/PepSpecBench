"""
Profiling Utilities for Inference Efficiency Measurement
=======================================================

Provides timing and memory profiling tools for benchmarking model inference
performance in terms of latency and VRAM usage.
"""

from contextlib import contextmanager
import os
import time
from typing import Dict, Generator, List, Tuple

import torch

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


@contextmanager
def cuda_profile() -> Generator[Dict[str, float], None, None]:
    """
    Context manager for profiling CUDA inference time and peak memory usage.

    Yields:
        Dict with keys: elapsed_ms, peak_vram_gb

    Usage:
        with cuda_profile() as prof:
            pred = model(batch)
        print(prof["elapsed_ms"], prof["peak_vram_gb"])
    """
    prof: Dict[str, float] = {"elapsed_ms": 0.0, "peak_vram_gb": 0.0}

    if not torch.cuda.is_available():
        start = time.perf_counter()
        try:
            yield prof
        finally:
            prof["elapsed_ms"] = (time.perf_counter() - start) * 1000
            prof["peak_vram_gb"] = 0.0
        return

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()  # Ensure all operations are complete

    start = time.perf_counter()

    try:
        yield prof
    finally:
        torch.cuda.synchronize()
        prof["elapsed_ms"] = (time.perf_counter() - start) * 1000
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        prof["peak_vram_gb"] = peak_mem_bytes / (1024**3)


def profile_inference_time(func, *args, **kwargs) -> Tuple[float, float]:
    """
    Profile a single function call for time and memory.

    Args:
        func: Function to profile
        *args, **kwargs: Arguments to pass to func

    Returns:
        Tuple of (elapsed_ms, peak_vram_gb)
    """
    with cuda_profile() as prof:
        func(*args, **kwargs)
    return float(prof["elapsed_ms"]), float(prof["peak_vram_gb"])


class InferenceProfiler:
    """
    Accumulates profiling statistics across multiple inference calls.
    """

    def __init__(self):
        self.times: List[float] = []
        self.memories: List[float] = []

    def add_measurement(self, elapsed_ms: float, peak_vram_gb: float):
        """Add a single measurement."""
        self.times.append(elapsed_ms)
        self.memories.append(peak_vram_gb)

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics.

        Returns:
            Dict with keys: mean_time_ms, median_time_ms, std_time_ms,
                           mean_vram_gb, max_vram_gb, std_vram_gb, count
        """
        if not self.times:
            return {
                "mean_time_ms": 0.0,
                "median_time_ms": 0.0,
                "std_time_ms": 0.0,
                "mean_vram_gb": 0.0,
                "max_vram_gb": 0.0,
                "std_vram_gb": 0.0,
                "count": 0,
            }

        import numpy as np
        return {
            "mean_time_ms": float(np.mean(self.times)),
            "median_time_ms": float(np.median(self.times)),
            "std_time_ms": float(np.std(self.times)),
            "mean_vram_gb": float(np.mean(self.memories)),
            "max_vram_gb": float(np.max(self.memories)),
            "std_vram_gb": float(np.std(self.memories)),
            "count": len(self.times),
        }


def _now() -> float:
    return float(time.perf_counter())


def get_process_rss_gb() -> float:
    try:
        if psutil is None:
            return 0.0
        p = psutil.Process(os.getpid())
        return float(p.memory_info().rss) / float(1024**3)
    except Exception:
        return 0.0


def reset_cuda_peak_memory() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            return


def get_cuda_peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        return float(torch.cuda.max_memory_allocated()) / float(1024**3)
    except Exception:
        return 0.0
