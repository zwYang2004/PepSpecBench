import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


_AMINO_ACID_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_INT: Dict[str, int] = {aa: i + 1 for i, aa in enumerate(_AMINO_ACID_ALPHABET)}
_AA_TO_INT["M(ox)"] = 21
_AA_TO_INT["C"] = 2

_MOD_SITE_RE = re.compile(r"([A-Z])\[([^\]]+)\]")
_UNIMOD_RE = re.compile(r"unimod:(\d+)", re.IGNORECASE)


def _strip_modifications(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    return re.sub(r"\[.*?\]", "", seq)


def _parse_unimod_id(token: str) -> Optional[int]:
    if not isinstance(token, str) or not token:
        return None

    m = _UNIMOD_RE.search(token)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None

    t = token.lower()
    if "oxidation" in t:
        return 35
    if "carbamidomethyl" in t:
        return 4

    return None


@dataclass(frozen=True)
class PrositTokenization:
    input_ids: np.ndarray
    ptm_ids: np.ndarray


class PrositTokenizer:
    def __init__(self, max_length: int = 40, ptm_max_unimod_id: int = 512):
        self.max_length = int(max_length)
        self.ptm_max_unimod_id = int(ptm_max_unimod_id)

    def tokenize(self, seq: str) -> PrositTokenization:
        if not isinstance(seq, str) or not seq:
            aa_ids = np.zeros(self.max_length, dtype=np.int64)
            ptm_ids = np.zeros(self.max_length, dtype=np.int64)
            return PrositTokenization(input_ids=aa_ids, ptm_ids=ptm_ids)

        seq = str(seq).strip()

        aa_list: List[int] = []
        ptm_list: List[int] = []

        i = 0
        n = len(seq)
        while i < n:
            ch = seq[i]
            if not ch.isalpha():
                i += 1
                continue

            aa = ch.upper()
            i += 1

            unimod_id: Optional[int] = None
            if i < n and seq[i] == "[":
                j = seq.find("]", i + 1)
                if j != -1:
                    token = seq[i + 1 : j]
                    unimod_id = _parse_unimod_id(token)
                    i = j + 1

            if aa == "M" and unimod_id == 35:
                aa_id = _AA_TO_INT["M(ox)"]
                ptm_id = 0
            else:
                aa_id = _AA_TO_INT.get(aa, 0)
                if unimod_id is None:
                    ptm_id = 0
                else:
                    ptm_id = unimod_id if unimod_id <= self.ptm_max_unimod_id else 0

            aa_list.append(aa_id)
            ptm_list.append(ptm_id)

            if len(aa_list) >= self.max_length:
                break

        aa_ids = np.zeros(self.max_length, dtype=np.int64)
        ptm_ids = np.zeros(self.max_length, dtype=np.int64)

        length = min(len(aa_list), self.max_length)
        if length > 0:
            aa_ids[:length] = np.asarray(aa_list[:length], dtype=np.int64)
            ptm_ids[:length] = np.asarray(ptm_list[:length], dtype=np.int64)

        return PrositTokenization(input_ids=aa_ids, ptm_ids=ptm_ids)


def _charge_onehot(charge: int, max_charge: int = 6) -> np.ndarray:
    out = np.zeros(max_charge, dtype=np.float32)
    if 1 <= int(charge) <= max_charge:
        out[int(charge) - 1] = 1.0
    return out


def _ion_index(pos_1idx: int, ion_type: str, frag_charge_1idx: int) -> int:
    ion_offset = 0 if ion_type == "y" else 3
    return (pos_1idx - 1) * 6 + ion_offset + (frag_charge_1idx - 1)


def _infer_max_fragment_positions(num_ions: int) -> int:
    if int(num_ions) <= 0 or int(num_ions) % 6 != 0:
        raise ValueError(f"Invalid Prosit intensity vector length: {num_ions}")
    return int(num_ions) // 6


def apply_prosit_mask(
    intensities: np.ndarray,
    seq_len: int,
    precursor_charge: int,
) -> np.ndarray:
    y = np.asarray(intensities, dtype=np.float32).reshape(-1)

    max_positions = _infer_max_fragment_positions(y.shape[0])

    valid_positions = int(min(max(seq_len - 1, 0), max_positions))
    max_frag_charge = int(min(3, max(int(precursor_charge), 0)))

    valid = np.zeros(y.shape[0], dtype=bool)
    for pos in range(1, valid_positions + 1):
        for ion_type in ("y", "b"):
            for z in range(1, max_frag_charge + 1):
                valid[_ion_index(pos, ion_type, z)] = True

    y_out = y.copy()
    y_out[~valid] = -1.0
    y_out[valid & (y_out < 0)] = 0.0

    return y_out


class PrositParquetDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        max_length: int = 40,
        ptm_max_unimod_id: int = 512,
        min_peptide_len: int = 6,
        max_peptide_len: int = 40,
        max_charge: int = 6,
        label_column: str = "intensities_raw",
        max_samples: Optional[int] = None,
    ):
        self.parquet_path = str(parquet_path)
        self.tokenizer = PrositTokenizer(max_length=max_length, ptm_max_unimod_id=ptm_max_unimod_id)
        self.label_column = str(label_column)
        self.max_charge = int(max_charge)
        self.num_ions = int((int(max_length) - 1) * 6)

        df = pd.read_parquet(self.parquet_path)

        if "normalized_sequence" in df.columns:
            seq_col = "normalized_sequence"
        else:
            seq_col = "modified_sequence" if "modified_sequence" in df.columns else "sequence"
        charge_col = "precursor_charge" if "precursor_charge" in df.columns else "charge"
        ce_col = "collision_energy" if "collision_energy" in df.columns else "orig_collision_energy"

        df = df.copy()
        df["_naked_seq"] = df[seq_col].map(_strip_modifications)
        df["_seq_len"] = df["_naked_seq"].map(lambda s: len(s) if isinstance(s, str) else 0)

        if max_samples is not None and len(df) > int(max_samples):
            df = df.sample(n=int(max_samples), random_state=42)

        df = df.reset_index(drop=True)

        self._df = df
        self._seq_col = seq_col
        self._charge_col = charge_col
        self._ce_col = ce_col

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        row = self._df.iloc[int(idx)]
        seq = row[self._seq_col]
        charge_raw = int(row[self._charge_col])
        charge = int(np.clip(charge_raw, 1, int(self.max_charge)))
        ce = float(row[self._ce_col])
        if ce > 1.5:
            ce = ce / 100.0
        ce = float(np.clip(ce, 0.0, 1.0))

        tok = self.tokenizer.tokenize(seq)
        input_ids = tok.input_ids
        ptm_ids = tok.ptm_ids

        seq_len_raw = int(row["_seq_len"])
        seq_len = int(min(seq_len_raw, int(self.tokenizer.max_length)))

        if self.label_column in row.index and row[self.label_column] is not None:
            y = np.asarray(row[self.label_column], dtype=np.float32).reshape(-1)
            if y.shape[0] != self.num_ions:
                raise ValueError(f"Label column '{self.label_column}' is not {self.num_ions}-d")
        else:
            y = np.full(self.num_ions, -1.0, dtype=np.float32)

        y = apply_prosit_mask(y, seq_len=seq_len, precursor_charge=charge)

        attention_mask = (input_ids != 0).astype(np.int64)
        charge_onehot = _charge_onehot(charge, max_charge=self.max_charge)

        return {
            "input_ids": input_ids,
            "ptm_ids": ptm_ids,
            "attention_mask": attention_mask,
            "collision_energy": np.asarray([ce], dtype=np.float32),
            "charge": charge_onehot,
            "labels": y,
        }


def prosit_collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.from_numpy(np.stack([b["input_ids"] for b in batch], axis=0)).long()
    ptm_ids = torch.from_numpy(np.stack([b["ptm_ids"] for b in batch], axis=0)).long()
    attention_mask = torch.from_numpy(np.stack([b["attention_mask"] for b in batch], axis=0)).long()
    collision_energy = torch.from_numpy(np.stack([b["collision_energy"] for b in batch], axis=0)).float()
    charge = torch.from_numpy(np.stack([b["charge"] for b in batch], axis=0)).float()
    labels = torch.from_numpy(np.stack([b["labels"] for b in batch], axis=0)).float()

    return {
        "input_ids": input_ids,
        "ptm_ids": ptm_ids,
        "attention_mask": attention_mask,
        "collision_energy": collision_energy,
        "charge": charge,
        "labels": labels,
    }
