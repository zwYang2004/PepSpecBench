from __future__ import annotations

import os
import sys

_tape_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..", "external", "tape")
)
if os.path.isdir(_tape_path) and _tape_path not in sys.path:
    sys.path.insert(0, _tape_path)

from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from src.datasets.prosit_parquet import apply_prosit_mask
from src.utils.mass_calc import strip_modifications


_TAPE_IUPAC_VOCAB: Dict[str, int] = {
    "<pad>": 0,
    "<mask>": 1,
    "<cls>": 2,
    "<sep>": 3,
    "<unk>": 4,
    "A": 5,
    "B": 6,
    "C": 7,
    "D": 8,
    "E": 9,
    "F": 10,
    "G": 11,
    "H": 12,
    "I": 13,
    "K": 14,
    "L": 15,
    "M": 16,
    "N": 17,
    "O": 18,
    "P": 19,
    "Q": 20,
    "R": 21,
    "S": 22,
    "T": 23,
    "U": 24,
    "V": 25,
    "W": 26,
    "X": 27,
    "Y": 28,
    "Z": 29,
}


class _FallbackTapeIUPACTokenizer:
    def __init__(self) -> None:
        self.vocab = dict(_TAPE_IUPAC_VOCAB)

    def encode(self, sequence: str) -> List[int]:
        s = str(sequence) if sequence is not None else ""
        ids = [int(self.vocab["<cls>"])]
        for ch in s:
            aa = ch.upper()
            ids.append(int(self.vocab.get(aa, int(self.vocab["<unk>"]))))
        ids.append(int(self.vocab["<sep>"]))
        return ids


def _get_tape_tokenizer() -> Any:
    try:
        from tape import TAPETokenizer

        return TAPETokenizer(vocab="iupac")
    except Exception:
        return _FallbackTapeIUPACTokenizer()


def _list_parquet_files(parquet_path: str | Path) -> List[Path]:
    p = Path(parquet_path)
    if p.is_dir():
        return sorted([x for x in p.iterdir() if x.suffix == ".parquet"])
    return [p]


def _detect_columns(names: List[str]) -> Tuple[str, str, str]:
    cols = set(names)
    if "normalized_sequence" in cols:
        seq_col = "normalized_sequence"
    else:
        seq_col = "modified_sequence" if "modified_sequence" in cols else "sequence"
    charge_col = "precursor_charge" if "precursor_charge" in cols else "charge"
    if "collision_energy" in cols:
        ce_col = "collision_energy"
    elif "nce" in cols:
        ce_col = "nce"
    else:
        ce_col = "orig_collision_energy"
    return seq_col, charge_col, ce_col


def _charge_onehot(charge: int, *, max_charge: int = 6) -> np.ndarray:
    out = np.zeros((int(max_charge),), dtype=np.float32)
    if 1 <= int(charge) <= int(max_charge):
        out[int(charge) - 1] = 1.0
    return out


def _normalize_collision_energy(value: Any) -> float:
    try:
        ce = float(value)
    except Exception:
        return 0.0

    if not np.isfinite(ce):
        return 0.0

    if ce > 1.5:
        ce = ce / 100.0

    return float(np.clip(ce, 0.0, 1.0))


class PrositTransformerParquetDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        *,
        max_len: int = 40,
        label_column: str = "intensities_raw",
    ) -> None:
        self.parquet_path = str(parquet_path)
        self.max_len = int(max_len)
        self.label_column = str(label_column)

        self.num_ions = int((int(max_len) - 1) * 6)

        self._tokenizer = _get_tape_tokenizer()

        vocab = getattr(self._tokenizer, "vocab", None)
        if isinstance(vocab, dict):
            self._pad_id = int(vocab.get("<pad>", 0))
            self._cls_id = int(vocab.get("<cls>", 2))
            self._sep_id = int(vocab.get("<sep>", 3))
        else:
            self._pad_id = 0
            self._cls_id = 2
            self._sep_id = 3

        self._files = _list_parquet_files(self.parquet_path)
        if not self._files:
            raise FileNotFoundError(f"No parquet files found: {self.parquet_path}")

        self._row_groups: List[Tuple[int, int, int]] = []
        self._starts: List[int] = []
        total = 0

        self._seq_col = "modified_sequence"
        self._charge_col = "precursor_charge"
        self._ce_col = "collision_energy"

        from tqdm import tqdm
        for fi, fp in enumerate(tqdm(self._files, desc='Scanning Parquet Files')):
            pf = pq.ParquetFile(str(fp))
            if fi == 0:
                self._seq_col, self._charge_col, self._ce_col = _detect_columns(pf.schema.names)

            nrg = int(pf.num_row_groups)
            for rg in range(nrg):
                rg_meta = pf.metadata.row_group(rg)
                nrows = int(rg_meta.num_rows)
                self._starts.append(int(total))
                self._row_groups.append((fi, rg, nrows))
                total += nrows

        self._length = int(total)
        self._cache_key: Optional[Tuple[int, int]] = None
        self._cache_df: Optional[pd.DataFrame] = None

    def __len__(self) -> int:
        return int(self._length)

    def _encode(self, seq: str) -> Tuple[np.ndarray, np.ndarray, int]:
        clean = strip_modifications(str(seq))
        clean = clean[: int(self.max_len)]

        ids = list(self._tokenizer.encode(clean))
        if not ids:
            ids = [int(self._cls_id), int(self._sep_id)]
        if int(ids[0]) != int(self._cls_id):
            ids = [int(self._cls_id)] + ids
        if int(ids[-1]) != int(self._sep_id):
            ids = ids + [int(self._sep_id)]

        max_tokens = int(self.max_len) + 2
        if len(ids) > max_tokens:
            ids = ids[: max_tokens - 1] + [int(self._sep_id)]

        input_ids = np.full((max_tokens,), int(self._pad_id), dtype=np.int64)
        n = int(len(ids))
        input_ids[:n] = np.asarray(ids, dtype=np.int64)
        input_mask = np.zeros((max_tokens,), dtype=np.int64)
        input_mask[:n] = 1

        return input_ids, input_mask, int(len(clean))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ix = int(idx)
        if ix < 0 or ix >= int(self._length):
            raise IndexError(ix)

        rg_pos = int(bisect_right(self._starts, ix) - 1)
        if rg_pos < 0:
            rg_pos = 0

        start = int(self._starts[rg_pos])
        fi, rg, nrows = self._row_groups[rg_pos]
        if os.environ.get("MS2BENCHMARK_DATASET_DEBUG", "") == "1" and idx % 1000 == 0:
            print(f"DEBUG: Dataset access idx={idx}, file_idx={fi}, rg={rg}", flush=True)
        local = int(ix - start)
        if local < 0 or local >= int(nrows):
            raise IndexError(ix)

        cache_key = (int(fi), int(rg))
        if self._cache_key != cache_key or self._cache_df is None:
            fp = self._files[int(fi)]
            pf = pq.ParquetFile(str(fp))
            cols = [self._seq_col, self._charge_col, self._ce_col]
            if self.label_column:
                cols.append(self.label_column)
            table = pf.read_row_group(int(rg), columns=cols)
            self._cache_df = table.to_pandas()
            self._cache_key = cache_key

        assert self._cache_df is not None
        row = self._cache_df.iloc[int(local)]

        seq = row[self._seq_col]
        charge = int(row[self._charge_col])
        ce = _normalize_collision_energy(row[self._ce_col])

        input_ids, input_mask, seq_len = self._encode(str(seq))
        charge_onehot = _charge_onehot(charge, max_charge=6)

        y = np.asarray(row.get(self.label_column), dtype=np.float32).reshape(-1)
        if y.shape[0] != int(self.num_ions):
            raise ValueError(f"Label column '{self.label_column}' is not {self.num_ions}-d")

        y = apply_prosit_mask(y, seq_len=seq_len, precursor_charge=charge)

        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "collision_energy": np.asarray([ce], dtype=np.float32),
            "precursor_charge": charge_onehot,
            "labels": y,
            "modified_sequence": str(seq),
            "precursor_charge_int": int(charge),
            "seq_len": int(seq_len),
        }


def prosit_transformer_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = torch.from_numpy(np.stack([b["input_ids"] for b in batch], axis=0)).long()
    input_mask = torch.from_numpy(np.stack([b["input_mask"] for b in batch], axis=0)).long()
    collision_energy = torch.from_numpy(np.stack([b["collision_energy"] for b in batch], axis=0)).float()
    precursor_charge = torch.from_numpy(np.stack([b["precursor_charge"] for b in batch], axis=0)).float()
    labels = torch.from_numpy(np.stack([b["labels"] for b in batch], axis=0)).float()
    precursor_charge_int = torch.tensor([int(b["precursor_charge_int"]) for b in batch], dtype=torch.long)
    seq_len = torch.tensor([int(b["seq_len"]) for b in batch], dtype=torch.long)
    modified_sequence = [str(b["modified_sequence"]) for b in batch]

    return {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "collision_energy": collision_energy,
        "precursor_charge": precursor_charge,
        "labels": labels,
        "precursor_charge_int": precursor_charge_int,
        "seq_len": seq_len,
        "modified_sequence": modified_sequence,
    }
