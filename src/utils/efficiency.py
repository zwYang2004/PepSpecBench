from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PerformanceSnapshot:
    wall_time_sec: float
    samples: int
    max_gpu_mem_bytes: Optional[int]

    @property
    def samples_per_sec(self) -> float:
        if self.wall_time_sec <= 0:
            return 0.0
        return float(self.samples) / float(self.wall_time_sec)


class PerformanceMonitor:
    def __init__(self) -> None:
        self._t0: Optional[float] = None
        self._samples: int = 0

    def start(self) -> None:
        self._t0 = time.time()
        self._samples = 0

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def add_samples(self, n: int) -> None:
        self._samples += int(n)

    def stop(self) -> PerformanceSnapshot:
        if self._t0 is None:
            raise RuntimeError("PerformanceMonitor.stop() called before start()")

        wall = time.time() - self._t0
        max_mem = None
        try:
            import torch

            if torch.cuda.is_available():
                max_mem = int(torch.cuda.max_memory_allocated())
        except Exception:
            max_mem = None

        return PerformanceSnapshot(wall_time_sec=float(wall), samples=int(self._samples), max_gpu_mem_bytes=max_mem)

    @staticmethod
    def as_dict(snapshot: PerformanceSnapshot) -> Dict[str, Any]:
        return {
            "wall_time_sec": snapshot.wall_time_sec,
            "samples": snapshot.samples,
            "samples_per_sec": snapshot.samples_per_sec,
            "max_gpu_mem_bytes": snapshot.max_gpu_mem_bytes,
        }
