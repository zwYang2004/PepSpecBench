from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.utils.hardware import should_load_to_ram, get_optimal_num_workers

class BaseParquetDataset(Dataset):
    """
    Base class for Parquet-based datasets with smart loading:
    - Load-to-RAM: If memory allows, keep data in RAM to eliminate Disk IO.
    - Uniform Metadata: Ensures consistent columns across datasets.
    """
    def __init__(
        self,
        parquet_path: str,
        *,
        load_to_ram: Optional[bool] = None,
        max_samples: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> None:
        self.parquet_path = Path(parquet_path)
        self.max_samples = max_samples
        
        # 1. Estimate file size and decide Load-to-RAM
        if self.parquet_path.is_file():
            file_size_gb = self.parquet_path.stat().st_size / (1024**3)
        else:
            # For directories, estimate from all parquets
            file_size_gb = sum(f.stat().st_size for f in self.parquet_path.glob("*.parquet")) / (1024**3)
            
        if load_to_ram is None:
            self.load_to_ram = should_load_to_ram(file_size_gb)
        else:
            self.load_to_ram = load_to_ram
            
        # 2. Load the data
        if self.load_to_ram:
            self._df = self._load_full_df(columns)
            if self.max_samples and len(self._df) > self.max_samples:
                self._df = self._df.sample(n=self.max_samples, random_state=42).reset_index(drop=True)
            self._use_ram = True
        else:
            # Fallback to standard disk-based index or row-groups if too big
            # For now, most mini datasets will fit in RAM. 
            # If not in RAM, we still load the index/metadata.
            self._df = pd.read_parquet(self.parquet_path, columns=columns)
            if self.max_samples and len(self._df) > self.max_samples:
                self._df = self._df.sample(n=self.max_samples, random_state=42).reset_index(drop=True)
            self._use_ram = False

    def _load_full_df(self, columns: Optional[List[str]]) -> pd.DataFrame:
        """Load Parquet files into a single DataFrame using memory mapping and fast engine."""
        read_kwargs = {
            "columns": columns,
            "engine": "pyarrow",
            "memory_map": True
        }
        if self.parquet_path.is_file():
            return pd.read_parquet(self.parquet_path, **read_kwargs)
        else:
            files = sorted(list(self.parquet_path.glob("*.parquet")))
            dfs = [pd.read_parquet(f, **read_kwargs) for f in files]
            return pd.concat(dfs, ignore_index=True)

    def __len__(self) -> int:
        return len(self._df)

    def _get_row(self, idx: int) -> pd.Series:
        return self._df.iloc[idx]
