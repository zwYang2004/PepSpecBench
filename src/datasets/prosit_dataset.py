from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset

from src.datasets.base_dataset import BaseParquetDataset
from src.datasets.prosit_parquet import apply_prosit_mask
from src.utils.mass_calc import strip_modifications


_PROSIT_CLASSIC_ALPHABET_ORDERED = "ACDEFGHIKLMNPQRSTVWY"
_PROSIT_CLASSIC_AA_TO_INT: Dict[str, int] = {
    aa: i + 1 for i, aa in enumerate(_PROSIT_CLASSIC_ALPHABET_ORDERED)
}

_PROSIT_CLASSIC_AA_TO_INT["C[UNIMOD:4]"] = len(_PROSIT_CLASSIC_AA_TO_INT) + 1
_PROSIT_CLASSIC_AA_TO_INT["M[UNIMOD:35]"] = len(_PROSIT_CLASSIC_AA_TO_INT) + 1


_NTERM_RE = re.compile(r"^(\[.*?\]-|\[\]-)")
_CTERM_RE = re.compile(r"(-\[\]|-\[.*?\])$")


def _normalize_modified_sequence(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    s = str(seq).strip()
    s = _NTERM_RE.sub("", s)
    s = _CTERM_RE.sub("", s)
    return s


def _encode_prosit_classic(seq: str, *, max_length: int) -> np.ndarray:
    s = _normalize_modified_sequence(seq)
    if not s:
        return np.zeros((int(max_length),), dtype=np.int64)

    aa_ids: List[int] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if not ch.isalpha():
            i += 1
            continue

        aa = ch.upper()
        i += 1

        unimod_id: Optional[int] = None
        if i < n and s[i] == "[":
            j = s.find("]", i + 1)
            if j != -1:
                token = s[i + 1 : j]
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

        token = aa
        if aa == "C" and unimod_id == 4:
            token = "C[UNIMOD:4]"
        if aa == "M" and unimod_id == 35:
            token = "M[UNIMOD:35]"

        aa_id = _PROSIT_CLASSIC_AA_TO_INT.get(token)
        if aa_id is None:
            aa_id = _PROSIT_CLASSIC_AA_TO_INT.get(aa, 0)

        aa_ids.append(int(aa_id))
        if len(aa_ids) >= int(max_length):
            break

    out = np.zeros((int(max_length),), dtype=np.int64)
    if aa_ids:
        out[: len(aa_ids)] = np.asarray(aa_ids, dtype=np.int64)
    return out


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


def _read_parquet_head(parquet_path: str, *, max_rows: int, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if max_rows <= 0:
        return pd.read_parquet(parquet_path, columns=columns)

    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        # Fallback to pandas full read if pyarrow is unavailable
        df = pd.read_parquet(parquet_path, columns=columns)
        if len(df) > int(max_rows):
            df = df.iloc[: int(max_rows)].copy()
        return df

    p = Path(str(parquet_path))
    files: List[Path] = []
    if p.is_dir():
        files = sorted([x for x in p.iterdir() if x.is_file() and x.suffix == ".parquet"])
    else:
        files = [p]

    tables = []
    remaining = int(max_rows)
    for fp in files:
        if remaining <= 0:
            break
        pf = pq.ParquetFile(str(fp))
        for batch in pf.iter_batches(batch_size=min(8192, remaining), columns=columns):
            tables.append(batch)
            remaining -= int(len(batch))
            if remaining <= 0:
                break

    if not tables:
        return pd.DataFrame()

    import pyarrow as pa  # type: ignore

    table = pa.Table.from_batches(tables)
    return table.to_pandas()


class PrositParquetDataset(BaseParquetDataset):
    def __init__(
        self,
        parquet_path: str,
        *,
        max_length: int = 40,
        min_peptide_len: int = 6,
        max_peptide_len: int = 40,
        max_charge: int = 6,
        label_column: str = "intensities_raw",
        max_samples: Optional[int] = None,
        load_to_ram: Optional[bool] = None,
    ) -> None:
        self.max_length = int(max_length)
        self.label_column = str(label_column)
        self.max_charge = int(max_charge)
        self.num_ions = int((int(max_length) - 1) * 6)

        # Base class handles Load-to-RAM logic and self._df initialization
        super().__init__(parquet_path, load_to_ram=load_to_ram, max_samples=max_samples)

        # Detect columns on loaded DataFrame
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
        candidate_label_cols = [self.label_column, "intensities_raw", "intensities", "intensity_array"]
        resolved_label_col = next((c for c in candidate_label_cols if c in cols), self.label_column)
        self.label_column = str(resolved_label_col)

        # Pre-calculate sequence lengths for speed
        self._df["_seq_len"] = self._df[self._seq_col].map(
            lambda s: len(strip_modifications(s)) if isinstance(s, str) else 0
        )

    def __len__(self) -> int:
        return int(len(self._df))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._df.iloc[int(idx)]
        seq = row[self._seq_col]
        charge_raw = int(row[self._charge_col])
        charge = int(np.clip(charge_raw, 1, int(self.max_charge)))
        ce = _normalize_collision_energy(row[self._ce_col])

        input_ids = _encode_prosit_classic(str(seq), max_length=self.max_length)
        attention_mask = (input_ids != 0).astype(np.int64)
        charge_onehot = _charge_onehot(charge, max_charge=self.max_charge)

        seq_len_raw = int(row.get("_seq_len", 0))
        seq_len = int(min(seq_len_raw, int(self.max_length)))

        if self.label_column in row.index and row[self.label_column] is not None:
            y = np.asarray(row[self.label_column], dtype=np.float32).reshape(-1)
            if y.shape[0] != self.num_ions:
                raise ValueError(f"Label column '{self.label_column}' is not {self.num_ions}-d")
        else:
            y = np.full(self.num_ions, -1.0, dtype=np.float32)

        y = apply_prosit_mask(y, seq_len=seq_len, precursor_charge=charge)

        return {
            "sequence": input_ids,
            "attention_mask": attention_mask,
            "collision_energy": np.asarray([ce], dtype=np.float32),
            "precursor_charge": charge_onehot,
            "labels": y,
            "modified_sequence": str(seq),
            "precursor_charge_int": int(charge),
            "seq_len": int(seq_len),
        }


class PrositParquetIterableDataset(IterableDataset):
    def __init__(
        self,
        parquet_path: str,
        *,
        max_length: int = 40,
        max_charge: int = 6,
        label_column: str = "intensities_raw",
        batch_rows: int = 8192,
    ) -> None:
        super().__init__()
        self.parquet_path = str(parquet_path)
        self.max_length = int(max_length)
        self.max_charge = int(max_charge)
        self.label_column = str(label_column)
        self.batch_rows = int(batch_rows)
        self.num_ions = int((int(max_length) - 1) * 6)

    def __iter__(self):
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as e:
            raise RuntimeError(f"PrositParquetIterableDataset requires pyarrow: {e}")

        fp0 = Path(self.parquet_path)
        files: List[Path]
        if fp0.is_dir():
            files = sorted([x for x in fp0.iterdir() if x.is_file() and x.suffix == ".parquet"])
        else:
            files = [fp0]

        schema_cols: List[str] = []
        for f in files:
            if f.exists():
                try:
                    schema_cols = [str(x) for x in pq.read_schema(str(f)).names]
                except Exception:
                    schema_cols = []
                break

        cols = set(schema_cols)
        seq_col = "modified_sequence" if "modified_sequence" in cols else "sequence"
        charge_col = "precursor_charge" if "precursor_charge" in cols else "charge"
        if "collision_energy" in cols:
            ce_col = "collision_energy"
        elif "nce" in cols:
            ce_col = "nce"
        else:
            ce_col = "orig_collision_energy"

        candidate_label_cols = [
            str(self.label_column) if self.label_column else "",
            "intensities_raw",
            "intensities",
            "intensity_array",
        ]
        candidate_label_cols = [c for c in candidate_label_cols if c]
        label_col = None
        for cand in candidate_label_cols:
            if cand in cols:
                label_col = cand
                break
        if label_col is None:
            raise KeyError(f"PrositParquetIterableDataset: none of candidate label columns {candidate_label_cols} found")

        read_cols = [seq_col, charge_col, ce_col, label_col]

        for f in files:
            pf = pq.ParquetFile(str(f))
            for batch in pf.iter_batches(batch_size=int(self.batch_rows), columns=read_cols):
                df = batch.to_pandas()
                for _idx, row in df.iterrows():
                    seq = row.get(seq_col, "")
                    charge_raw = int(row.get(charge_col, 0) or 0)
                    charge = int(np.clip(charge_raw, 1, int(self.max_charge)))
                    ce = _normalize_collision_energy(row.get(ce_col, 0.0))

                    input_ids = _encode_prosit_classic(str(seq), max_length=self.max_length)
                    attention_mask = (input_ids != 0).astype(np.int64)
                    charge_onehot = _charge_onehot(charge, max_charge=self.max_charge)

                    seq_len_raw = len(strip_modifications(str(seq))) if isinstance(seq, str) else 0
                    seq_len = int(min(int(seq_len_raw), int(self.max_length)))

                    y = np.asarray(row.get(label_col, None), dtype=np.float32).reshape(-1)
                    if y.shape[0] != self.num_ions:
                        continue

                    y = apply_prosit_mask(y, seq_len=seq_len, precursor_charge=charge)

                    yield {
                        "sequence": input_ids,
                        "attention_mask": attention_mask,
                        "collision_energy": np.asarray([ce], dtype=np.float32),
                        "precursor_charge": charge_onehot,
                        "labels": y,
                        "modified_sequence": str(seq),
                        "precursor_charge_int": int(charge),
                        "seq_len": int(seq_len),
                    }


def prosit_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    sequence = torch.from_numpy(np.stack([b["sequence"] for b in batch], axis=0)).long()
    attention_mask = torch.from_numpy(np.stack([b["attention_mask"] for b in batch], axis=0)).long()
    collision_energy = torch.from_numpy(np.stack([b["collision_energy"] for b in batch], axis=0)).float()
    precursor_charge = torch.from_numpy(np.stack([b["precursor_charge"] for b in batch], axis=0)).float()
    labels = torch.from_numpy(np.stack([b["labels"] for b in batch], axis=0)).float()
    precursor_charge_int = torch.tensor([int(b["precursor_charge_int"]) for b in batch], dtype=torch.long)
    seq_len = torch.tensor([int(b["seq_len"]) for b in batch], dtype=torch.long)
    modified_sequence = [str(b["modified_sequence"]) for b in batch]

    return {
        "sequence": sequence,
        "attention_mask": attention_mask,
        "collision_energy": collision_energy,
        "precursor_charge": precursor_charge,
        "labels": labels,
        "precursor_charge_int": precursor_charge_int,
        "seq_len": seq_len,
        "modified_sequence": modified_sequence,
    }


def prosit_classic_alphabet() -> Dict[str, int]:
    return dict(_PROSIT_CLASSIC_AA_TO_INT)
