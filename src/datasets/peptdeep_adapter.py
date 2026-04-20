"""AlphaPeptDeep parquet adapter.

This module provides utilities to convert benchmark parquet data to AlphaPeptDeep's
expected input format (precursor_df + fragment_df).

Key functions:
- parse_mods: Parse modified sequence to extract naked sequence, mods, and sites
- build_precursor_df: Build precursor DataFrame from benchmark parquet
- build_fragment_df_from_level1: Build fragment DataFrame from 234-d level1 labels
- level1_from_fragment_df: Convert fragment predictions back to 234-d level1 format
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.mass_calc import (
    canonical_by_dim,
    canonical_by_mask,
    ion_index,
    strip_modifications,
)


def normalize_nce(value: Any) -> float:
    """Normalize collision energy to 0-100 scale.
    
    Args:
        value: Raw collision energy value (can be float, int, or string)
        
    Returns:
        Normalized NCE in range [0, 100]
    """
    try:
        ce = float(value)
    except Exception:
        return 30.0
    if not np.isfinite(ce):
        return 30.0
    if ce <= 1.5:
        ce = ce * 100.0
    return float(np.clip(ce, 0.0, 100.0))


def parse_mods(mod_seq: str) -> Tuple[str, str, str]:
    """Parse modified sequence to extract naked sequence, mods, and sites.
    
    Supports UNIMOD notation and (ox) notation for oxidation.
    
    Args:
        mod_seq: Modified sequence string (e.g., "[UNIMOD:1]PEPTC[UNIMOD:4]M[UNIMOD:35]IDE")
        
    Returns:
        Tuple of (naked_sequence, mods_string, sites_string)
        - mods_string: semicolon-separated mod names (e.g., "Acetyl@Any_N-term;Carbamidomethyl@C")
        - sites_string: semicolon-separated positions (e.g., "0;4")
    """
    if not isinstance(mod_seq, str) or not mod_seq:
        return "", "", ""

    s = str(mod_seq).strip()

    mods: List[str] = []
    sites: List[str] = []

    if s.lower().startswith("[unimod:1]"):
        mods.append("Acetyl@Any_N-term")
        sites.append("0")
        # Remove the tag so it doesn't get parsed as amino acids
        s = s[len("[UNIMOD:1]"):]

    pos = -1
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if not ch.isalpha():
            i += 1
            continue

        aa = ch.upper()
        pos += 1
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

        if aa == "C" and unimod_id == 4:
            mods.append("Carbamidomethyl@C")
            sites.append(str(pos))
        elif aa == "M" and unimod_id == 35:
            mods.append("Oxidation@M")
            sites.append(str(pos))

    seq = strip_modifications(s)
    return seq, ";".join(mods), ";".join(sites)


def build_precursor_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build AlphaPeptDeep precursor DataFrame from benchmark parquet.
    
    Args:
        df: Benchmark parquet DataFrame with columns:
            - modified_sequence or sequence
            - precursor_charge or charge
            - collision_energy or nce or orig_collision_energy
            
    Returns:
        Precursor DataFrame with columns:
            - sequence: naked sequence
            - mods: semicolon-separated mod names
            - mod_sites: semicolon-separated positions
            - charge: precursor charge
            - nAA: sequence length
            - nce: normalized collision energy
            - instrument: instrument name (default "Lumos")
    """
    if "modified_sequence" in df.columns:
        seq_col = "modified_sequence"
    elif "normalized_sequence" in df.columns:
        seq_col = "normalized_sequence"
    else:
        seq_col = "sequence"
    charge_col = "precursor_charge" if "precursor_charge" in df.columns else "charge"
    if "collision_energy" in df.columns:
        ce_col = "collision_energy"
    elif "nce" in df.columns:
        ce_col = "nce"
    else:
        ce_col = "orig_collision_energy"

    parsed = df[seq_col].map(parse_mods)

    out = pd.DataFrame({
        "sequence": parsed.map(lambda x: x[0]),
        "mods": parsed.map(lambda x: x[1]),
        "mod_sites": parsed.map(lambda x: x[2]),
        "charge": df[charge_col].astype(int),
    })
    out["nAA"] = out["sequence"].map(lambda x: len(x) if isinstance(x, str) else 0)
    out["nce"] = df[ce_col].map(normalize_nce).astype(float)

    instrument_col: Optional[str] = None
    for cand in ["instrument", "instrument_name", "instrument_type", "ms_instrument", "instrument_model"]:
        if cand in df.columns:
            instrument_col = str(cand)
            break
    # Map instrument to AlphaPeptDeep's known set (QE, Lumos, timsTOF, SciexTOF, ThermoTOF).
    # "Orbitrap" and variants map to Lumos per peptdeep instrument_group.
    _INSTRUMENT_MAP = {
        "orbitrap": "Lumos",
        "orbitrap tribrid": "Lumos",
        "orbitrap tribridlumos": "Lumos",
        "thermotribrid": "Lumos",
        "fusion": "Lumos",
        "eclipse": "Lumos",
        "velos": "Lumos",
        "elite": "Lumos",
        "exploris": "QE",
        "exploris480": "QE",
    }

    if instrument_col is not None:
        out["instrument"] = df[instrument_col].astype(str)
        s = out["instrument"].str.strip()
        lower = s.str.lower()
        missing = (s.str.len() == 0) | lower.isin(["nan", "none", "<na>", "null"])
        out.loc[missing, "instrument"] = "Lumos"
        # Normalize unknown instruments to Lumos (fixes zero predictions for Orbitrap-labeled data)
        _KNOWN = {"qe": "QE", "lumos": "Lumos", "timstof": "timsTOF", "sciextof": "SciexTOF", "thermotof": "ThermoTOF"}

        def _norm_inst(v: str) -> str:
            k = v.strip().lower()
            if k in _INSTRUMENT_MAP:
                return _INSTRUMENT_MAP[k]
            if k in _KNOWN:
                return _KNOWN[k]
            return "Lumos" if k else "Lumos"

        out["instrument"] = out["instrument"].apply(_norm_inst)
    else:
        out["instrument"] = "Lumos"
    return out


def build_fragment_df_from_level1(
    precursor_df: pd.DataFrame,
    level1: np.ndarray,
    charged_frag_types: List[str],
    *,
    max_len: int = 40,
    max_frag_charge: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build AlphaPeptDeep fragment DataFrame from 234-d level1 labels.
    
    Args:
        precursor_df: Precursor DataFrame with 'sequence' and 'charge' columns
        level1: Level1 labels array of shape (N, 234)
        charged_frag_types: List of charged fragment types (e.g., ["b_z1", "b_z2", "y_z1", "y_z2"])
        max_len: Maximum sequence length
        max_frag_charge: Maximum fragment charge for level1 indexing
        
    Returns:
        Tuple of (updated_precursor_df, fragment_df)
        - updated_precursor_df: precursor_df with frag_start_idx and frag_stop_idx columns
        - fragment_df: Fragment intensities DataFrame with charged_frag_types as columns
    """
    level1 = np.asarray(level1, dtype=np.float32)
    dim = canonical_by_dim(max_len=int(max_len), max_frag_charge=int(max_frag_charge))
    if level1.ndim != 2 or level1.shape[1] != int(dim):
        raise ValueError(f"Expected level1 shape (N,{dim}), got {level1.shape}")

    n = int(level1.shape[0])

    frag_start_idx: List[int] = []
    frag_stop_idx: List[int] = []

    rows: List[np.ndarray] = []
    start = 0

    for i in range(n):
        seq = str(precursor_df.iloc[i]["sequence"])
        charge = int(precursor_df.iloc[i]["charge"])
        L = len(seq)
        positions = int(max(min(L - 1, int(max_len) - 1), 0))

        frag_start_idx.append(int(start))
        stop = start + positions
        frag_stop_idx.append(int(stop))

        mat = np.zeros((positions, len(charged_frag_types)), dtype=np.float32)

        for p in range(1, positions + 1):
            for j, ft in enumerate(charged_frag_types):
                parts = ft.split("_")
                if len(parts) != 2:
                    continue
                ion_type = parts[0]
                z = int(parts[1][1:])
                if z > int(max_frag_charge):
                    continue
                if z > int(charge):
                    continue
                idx = ion_index(p, ion_type, z, max_frag_charge=int(max_frag_charge))
                v = float(level1[i, idx])
                if v < 0:
                    v = 0.0
                mat[p - 1, j] = v

        rows.append(mat)
        start = stop

    precursor_df = precursor_df.copy()
    precursor_df["frag_start_idx"] = frag_start_idx
    precursor_df["frag_stop_idx"] = frag_stop_idx

    if rows:
        frag = np.concatenate(rows, axis=0)
    else:
        frag = np.zeros((0, len(charged_frag_types)), dtype=np.float32)

    return precursor_df, pd.DataFrame(frag, columns=charged_frag_types)


def level1_from_fragment_df(
    precursor_df: pd.DataFrame,
    fragment_df: pd.DataFrame,
    charged_frag_types: List[str],
    original_df: pd.DataFrame,
    *,
    max_len: int = 40,
    max_frag_charge: int = 3,
) -> np.ndarray:
    """Convert AlphaPeptDeep fragment predictions back to 234-d level1 format.
    
    Args:
        precursor_df: Precursor DataFrame with frag_start_idx and frag_stop_idx
        fragment_df: Fragment predictions DataFrame
        charged_frag_types: List of charged fragment types
        original_df: Original benchmark DataFrame with modified_sequence and charge
        max_len: Maximum sequence length
        max_frag_charge: Maximum fragment charge for level1 indexing
        
    Returns:
        Level1 predictions array of shape (N, 234)
    """
    seq_col = "modified_sequence" if "modified_sequence" in original_df.columns else "sequence"
    charge_col = "precursor_charge" if "precursor_charge" in original_df.columns else "charge"
    
    level1_dim = canonical_by_dim(max_len=max_len, max_frag_charge=max_frag_charge)
    pred_level1 = np.zeros((len(precursor_df), int(level1_dim)), dtype=np.float32)
    
    for i in range(len(precursor_df)):
        seq = str(original_df.iloc[i][seq_col])
        charge = int(original_df.iloc[i][charge_col])
        naked = strip_modifications(seq)
        L = len(naked)
        mask = canonical_by_mask(L, charge, max_len=max_len, max_frag_charge=max_frag_charge)
        
        start = int(precursor_df.iloc[i]["frag_start_idx"])
        stop = int(precursor_df.iloc[i]["frag_stop_idx"])
        block = fragment_df.iloc[start:stop].to_numpy(dtype=np.float32)

        positions = stop - start
        for p in range(1, positions + 1):
            for j, ft in enumerate(charged_frag_types):
                parts = ft.split("_")
                ion_type = parts[0]
                z = int(parts[1][1:])
                if z > max_frag_charge:
                    continue
                idx = ion_index(p, ion_type, z, max_frag_charge=max_frag_charge)
                pred_level1[i, idx] = float(block[p - 1, j])
        pred_level1[i, ~mask] = 0.0
    
    return pred_level1
