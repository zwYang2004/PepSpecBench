from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.constraints import DataConstraints


class BenchmarkModel(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_data_constraints(self) -> DataConstraints:
        raise NotImplementedError

    @abstractmethod
    def fit(self, train_path: str, val_path: Optional[str], output_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, parquet_path: str, model_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
