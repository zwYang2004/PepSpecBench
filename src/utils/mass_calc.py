from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.metrics import BinningConfig


PROTON_MASS = 1.00727646688
WATER_MASS = 18.010564684

UNIMOD_1_ACETYL = 42.010565
UNIMOD_4_CARBAMIDOMETHYL = 57.021464
UNIMOD_35_OXIDATION = 15.994915


_AA_MASS = {
    "A": 71.037113805,
    "R": 156.101111050,
    "N": 114.042927470,
    "D": 115.026943065,
    "C": 103.009184505,
    "E": 129.042593135,
    "Q": 128.058577540,
    "G": 57.021463735,
    "H": 137.058911875,
    "I": 113.084064015,
    "L": 113.084064015,
    "K": 128.094963050,
    "M": 131.040484645,
    "F": 147.068413945,
    "P": 97.052763875,
    "S": 87.032028435,
    "T": 101.047678505,
    "W": 186.079312980,
    "Y": 163.063328575,
    "V": 99.068413945,
}


_UNIMOD_RE = re.compile(r"unimod:(\d+)", re.IGNORECASE)


def strip_modifications(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    s = re.sub(r"\[[^\]]+\]", "", seq)
    s = re.sub(r"\(ox\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\(o\)", "", s, flags=re.IGNORECASE)
    return s


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
    if "acetyl" in t:
        return 1
    return None


@dataclass(frozen=True)
class ParsedSequence:
    aa: str
    residue_masses: Tuple[float, ...]
    nterm_mod_mass: float

    @property
    def length(self) -> int:
        return len(self.residue_masses)


def parse_modified_sequence(
    seq: str,
    *,
    assume_carbamidomethyl_c: bool = True,
) -> ParsedSequence:
    if not isinstance(seq, str):
        return ParsedSequence(aa="", residue_masses=tuple(), nterm_mod_mass=0.0)

    s = str(seq).strip()

    nterm_mod_mass = 0.0
    s_l = s.lstrip()
    if s_l.startswith("[UNIMOD:1]") or s_l.lower().startswith("[unimod:1]") or s_l.lower().startswith("[acetyl]"):
        nterm_mod_mass += UNIMOD_1_ACETYL
        s = re.sub(r"^\s*\[UNIMOD:1\]", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^\s*\[acetyl\]", "", s, flags=re.IGNORECASE)

    aa_list: List[str] = []
    masses: List[float] = []

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
                unimod_id = _parse_unimod_id(token)
                i = j + 1

        if i + 4 <= n and s[i : i + 4].lower() == "(ox)":
            unimod_id = 35
            i += 4

        base = float(_AA_MASS.get(aa, 0.0))
        if base <= 0.0:
            continue

        if aa == "C" and (assume_carbamidomethyl_c or unimod_id == 4):
            base += UNIMOD_4_CARBAMIDOMETHYL
        if aa == "M" and unimod_id == 35:
            base += UNIMOD_35_OXIDATION

        aa_list.append(aa)
        masses.append(base)

    return ParsedSequence(
        aa="".join(aa_list),
        residue_masses=tuple(masses),
        nterm_mod_mass=float(nterm_mod_mass),
    )


def ion_index(pos_1idx: int, ion_type: str, frag_charge_1idx: int, *, max_frag_charge: int = 3) -> int:
    ion_offset = 0 if ion_type == "y" else int(max_frag_charge)
    return (int(pos_1idx) - 1) * (2 * int(max_frag_charge)) + ion_offset + (int(frag_charge_1idx) - 1)


def canonical_by_dim(max_len: int = 40, max_frag_charge: int = 3) -> int:
    return int(max_len - 1) * 2 * int(max_frag_charge)


def canonical_by_mask(seq_len: int, precursor_charge: int, *, max_len: int = 40, max_frag_charge: int = 3) -> np.ndarray:
    dim = canonical_by_dim(max_len=max_len, max_frag_charge=max_frag_charge)
    valid = np.zeros((dim,), dtype=bool)

    valid_positions = int(min(max(int(seq_len) - 1, 0), int(max_len) - 1))
    valid_charges = int(min(int(max_frag_charge), max(int(precursor_charge), 0)))

    for pos in range(1, valid_positions + 1):
        for ion_type in ("y", "b"):
            for z in range(1, valid_charges + 1):
                valid[ion_index(pos, ion_type, z, max_frag_charge=max_frag_charge)] = True

    return valid


def canonical_by_mz(
    seq: str,
    *,
    max_len: int = 40,
    max_frag_charge: int = 3,
    assume_carbamidomethyl_c: bool = True,
) -> Tuple[np.ndarray, int]:
    parsed = parse_modified_sequence(seq, assume_carbamidomethyl_c=assume_carbamidomethyl_c)
    L = int(parsed.length)
    dim = canonical_by_dim(max_len=max_len, max_frag_charge=max_frag_charge)
    out = np.zeros((dim,), dtype=np.float32)

    if L <= 1:
        return out, L

    masses = np.asarray(parsed.residue_masses, dtype=np.float64)
    prefix = np.concatenate([[0.0], np.cumsum(masses)])

    for pos in range(1, min(L, int(max_len))):
        b_neutral = float(parsed.nterm_mod_mass) + float(prefix[pos])
        for z in range(1, int(max_frag_charge) + 1):
            mz = (b_neutral + float(z) * PROTON_MASS) / float(z)
            out[ion_index(pos, "b", z, max_frag_charge=max_frag_charge)] = float(mz)

        y_neutral = float(masses[L - pos :].sum()) + WATER_MASS
        for z in range(1, int(max_frag_charge) + 1):
            mz = (y_neutral + float(z) * PROTON_MASS) / float(z)
            out[ion_index(pos, "y", z, max_frag_charge=max_frag_charge)] = float(mz)

    return out, L


def mz_to_bin_index(mz: float, cfg: BinningConfig = BinningConfig()) -> Optional[int]:
    try:
        m = float(mz)
    except Exception:
        return None
    if not np.isfinite(m) or m <= 0:
        return None
    idx = int(np.around(m / float(cfg.bin_size)))
    if idx < 0 or idx >= int(cfg.num_bins):
        return None
    return idx


def extract_by_from_binned_spectrum(
    binned: np.ndarray,
    seq: str,
    precursor_charge: int,
    *,
    max_len: int = 40,
    max_frag_charge: int = 3,
    cfg: BinningConfig = BinningConfig(),
    assume_carbamidomethyl_c: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    mz_vec, seq_len = canonical_by_mz(
        seq,
        max_len=max_len,
        max_frag_charge=max_frag_charge,
        assume_carbamidomethyl_c=assume_carbamidomethyl_c,
    )
    mask = canonical_by_mask(seq_len, precursor_charge, max_len=max_len, max_frag_charge=max_frag_charge)

    b = np.asarray(binned, dtype=np.float32).reshape(-1)
    out = np.zeros_like(mz_vec, dtype=np.float32)

    for i in np.where(mask)[0].tolist():
        idx = mz_to_bin_index(float(mz_vec[i]), cfg=cfg)
        if idx is None:
            continue
        if idx < b.shape[0]:
            out[i] = float(b[idx])

    return out, mask
