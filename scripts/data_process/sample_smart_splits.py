#!/usr/bin/env python
"""Smart stratified sampling for PROSPECT and MassIVE-KB.

This script creates *mini* versions of the benchmark datasets that are:
- Deterministic (MD5-based ranking)
- KDD-aligned in filtering (charge<=6, len<=40, UNIMOD 1/4/35 only)
- Balanced for PROSPECT (Unmod vs PTM, and PTM Ox/CAM/Acetyl)
- Natural for MassIVE-KB (respecting official train/val/test boundaries)

Outputs:
  data/reconstructed/prospect/all/{train,val,test}.parquet
  data/reconstructed/massive_kb/all/{train,val,test}.parquet

Defaults: 500k/50k/50k (matches paper). Use --convert-234d to produce
234d versions via add_level1_labels.py.
"""

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm

# Project root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PROSPECT_FILTERED_DIR = _ROOT / "data" / "Prospect_parquet" / "Prospect_merged_unimod135"
MASSIVE_FILTERED_DIR = _ROOT / "data" / "MassIVE-KB" / "processed_charge_le6_unimod135"

PROSPECT_ALL_DIR = _ROOT / "data" / "reconstructed" / "prospect" / "all"
PROSPECT_234D_DIR = _ROOT / "data" / "reconstructed" / "prospect" / "234d"
MASSIVE_ALL_DIR = _ROOT / "data" / "reconstructed" / "massive_kb" / "all"
MASSIVE_234D_DIR = _ROOT / "data" / "reconstructed" / "massive_kb" / "234d"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sample_smart_splits")


_MOD_BRACKET_RE = re.compile(r"\[([^\]]+)\]")


def _detect_cols(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    """Detect (seq_col, charge_col, ce_col)."""
    if "modified_sequence" in df.columns:
        seq_col = "modified_sequence"
    elif "sequence" in df.columns:
        seq_col = "sequence"
    else:
        raise KeyError("Sequence column not found (expected 'modified_sequence' or 'sequence').")

    if "precursor_charge" in df.columns:
        charge_col = "precursor_charge"
    elif "charge" in df.columns:
        charge_col = "charge"
    else:
        raise KeyError("Charge column not found (expected 'precursor_charge' or 'charge').")

    ce_col: Optional[str] = None
    for cand in ("collision_energy", "nce", "ce"):
        if cand in df.columns:
            ce_col = cand
            break

    return seq_col, charge_col, ce_col


def _classify_mod_token(tok: str) -> Optional[int]:
    """Classify a single modification token inside [...].

    Returns UNIMOD id (1,4,35) or None if not allowed.
    """
    t = str(tok).strip().lower()
    if not t:
        return None

    # Explicit UNIMOD first
    if "unimod:35" in t:
        return 35
    if "unimod:4" in t:
        return 4
    if "unimod:1" in t:
        return 1

    # Name / delta-based fallbacks
    if "oxidation" in t or "+15.99" in t or "+15.994" in t:
        return 35
    if "carbamidomethyl" in t or "+57.02" in t or "+57.021" in t:
        return 4
    if "acetyl" in t or "+42.01" in t or "+42.011" in t:
        return 1

    return None


def _normalize_sequence(seq: str) -> Tuple[Optional[str], Optional[str], bool, List[int]]:
    """Normalize a modified sequence.

    - Only allow UNIMOD {1,4,35} (Acetyl/CAM/Oxidation).
    - Replace any allowed bracket token with canonical [UNIMOD:x].
    - Drop sequences containing any other PTM (return is_valid=False).

    Returns (normalized_seq, naked_seq, is_valid, unimods_list).
    """
    s = str(seq)
    if "[" not in s:
        # Unmodified sequence
        return s, s, True, []

    parts: List[str] = []
    last = 0
    unimods: List[int] = []
    for m in _MOD_BRACKET_RE.finditer(s):
        parts.append(s[last:m.start()])
        inner = m.group(1)
        u = _classify_mod_token(inner)
        if u is None:
            return None, None, False, []
        parts.append(f"[UNIMOD:{u}]")
        unimods.append(u)
        last = m.end()
    parts.append(s[last:])
    norm = "".join(parts)
    naked = _MOD_BRACKET_RE.sub("", norm)
    return norm, naked, True, unimods


def _compute_sample_key(naked_seq: str, charge: int, ce: Optional[float], seed: int) -> int:
    ce_val = 0.0 if ce is None else float(ce)
    # Use :.1f to ensure byte-level reproducibility across different python/float versions
    s = f"{naked_seq}|{charge}|{ce_val:.1f}|{seed}"
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    # Use only the first 15 hex digits so the value fits safely into a signed int64
    # (15 hex digits < 2**60). This is still fully deterministic and sufficient
    # for providing a stable ordering key while avoiding OverflowError in
    # pyarrow/pandas parquet conversion.
    return int(h[:15], 16)


def _get_split_by_hash(naked_seq: str) -> str:
    """Deterministic split based on naked sequence hash.
    Buckets: 0-79 -> train (80%), 80-89 -> val (10%), 90-99 -> test (10%).
    Matches paper Section 2.2 and Appendix A (8:1:1)."""
    h = int(hashlib.md5(str(naked_seq).encode()).hexdigest()[:8], 16) % 100
    if h < 80: return "train"
    if h < 90: return "val"
    return "test"


def _get_split_by_sequence(mod_seq: str) -> str:
    """Sequence-level split: partition by modified_sequence (allows PTM-variant leakage)."""
    h = int(hashlib.md5(str(mod_seq).encode()).hexdigest()[:8], 16) % 100
    if h < 80: return "train"
    if h < 90: return "val"
    return "test"


def _get_split_random(sample_key: int, rng) -> str:
    """Random split: partition by random assignment (maximizes leakage)."""
    u = rng.random()
    if u < 0.8: return "train"
    if u < 0.9: return "val"
    return "test"


def _apply_global_filters(df: pd.DataFrame, max_charge: int = 6, max_len: int = 40) -> pd.DataFrame:
    """Enforce global constraints: charge<=max_charge, length<=max_len."""
    if df.empty:
        return df

    seq_col, charge_col, _ = _detect_cols(df)

    # Charge filter
    df = df[df[charge_col] <= max_charge].copy()

    # Length filter (prefer existing peptide_length, else compute from naked)
    if "peptide_length" in df.columns:
        df = df[df["peptide_length"] <= max_len].copy()
    else:
        # Compute length from raw sequence (before normalization). This is a fallback;
        # downstream normalization also uses naked lengths implicitly.
        lengths = df[seq_col].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.len()
        df = df[lengths <= max_len].copy()

    return df


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply PTM normalization and drop disallowed PTMs.

    Adds columns:
      - normalized_sequence
      - naked_sequence
      - has_ptm (bool)
      - ptm_bucket (for PROSPECT; one of {"ox","cam","ace"} or None)
    """
    if df.empty:
        return df

    seq_col, charge_col, ce_col = _detect_cols(df)

    norms: List[Optional[str]] = []
    nakeds: List[Optional[str]] = []
    valids: List[bool] = []
    buckets: List[Optional[str]] = []

    for _, row in df.iterrows():
        norm, naked, ok, unimods = _normalize_sequence(row[seq_col])
        if not ok or norm is None or naked is None:
            norms.append(None)
            nakeds.append(None)
            valids.append(False)
            buckets.append(None)
            continue
        norms.append(norm)
        nakeds.append(naked)
        valids.append(True)
        if not unimods:
            buckets.append(None)
        else:
            # Assign bucket by priority 35 > 4 > 1
            u_set = set(unimods)
            if 35 in u_set:
                buckets.append("ox")
            elif 4 in u_set:
                buckets.append("cam")
            elif 1 in u_set:
                buckets.append("ace")
            else:
                buckets.append(None)

    df = df.copy()
    df["normalized_sequence"] = norms
    df["naked_sequence"] = nakeds
    df["_valid_norm"] = valids
    df["ptm_bucket"] = buckets

    df = df[df["_valid_norm"]].copy()
    df.drop(columns=["_valid_norm"], inplace=True)

    df["has_ptm"] = df["ptm_bucket"].notna()

    # Compute sample_key for determinism
    ce_series = df[ce_col] if ce_col is not None else None
    df["sample_key"] = [
        _compute_sample_key(naked, int(ch), float(ce_series.iat[i]) if ce_series is not None else None, seed=42)
        for i, (naked, ch) in enumerate(zip(df["naked_sequence"], df[charge_col]))
    ]

    return df


# ---------------------------------------------------------------------------
# PROSPECT sampling (balanced)
# ---------------------------------------------------------------------------


def _sample_prospect_split(df: pd.DataFrame, split: str, total_target: int) -> pd.DataFrame:
    """Sample a single PROSPECT split with balanced Unmod/PTM strategy."""
    if df.empty or total_target <= 0:
        return df.iloc[0:0].copy()

    # Unmod vs PTM
    unmod_df = df[~df["has_ptm"]].copy()
    ptm_df = df[df["has_ptm"]].copy()

    unmod_target = min(total_target // 2, len(unmod_df))
    ptm_target = total_target - unmod_target

    # Unmod: sort by sample_key and take top
    unmod_sel = (
        unmod_df.sort_values("sample_key", kind="mergesort").head(unmod_target) if unmod_target > 0 else unmod_df.iloc[0:0]
    )

    # PTM: three buckets
    bucket_targets: Dict[str, int] = {}
    if ptm_target > 0:
        base = ptm_target // 3
        rem = ptm_target - base * 3
        for b in ("ox", "cam", "ace"):
            bucket_targets[b] = base
        # Distribute remainder deterministically by fixed order
        order = ["ox", "cam", "ace"]
        for i in range(rem):
            bucket_targets[order[i]] += 1

    selected_parts: List[pd.DataFrame] = []
    leftovers: List[pd.DataFrame] = []

    for bucket in ("ox", "cam", "ace"):
        sub = ptm_df[ptm_df["ptm_bucket"] == bucket].copy()
        if sub.empty or ptm_target <= 0:
            continue
        target_b = bucket_targets.get(bucket, 0)
        if target_b <= 0:
            # All rows are leftover
            leftovers.append(sub)
            continue
        sub_sorted = sub.sort_values("sample_key", kind="mergesort")
        use = min(target_b, len(sub_sorted))
        selected_parts.append(sub_sorted.head(use))
        if len(sub_sorted) > use:
            leftovers.append(sub_sorted.iloc[use:])

    selected_ptm = pd.concat(selected_parts, ignore_index=True) if selected_parts else ptm_df.iloc[0:0]
    needed_extra = ptm_target - len(selected_ptm)

def _process_batch_for_prospect(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Apply global filters and normalization to a batch."""
    df = _apply_global_filters(batch_df)
    if df.empty:
        return df
    return _normalize_df(df)


def _report_stats(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        logger.info("%s: empty dataset", label)
        return

    seq_col, charge_col, _ = _detect_cols(df)

    total = len(df)
    unmod = int((~df["has_ptm"]).sum()) if "has_ptm" in df.columns else 0
    ptm = int(df["has_ptm"].sum()) if "has_ptm" in df.columns else 0

    logger.info("[%s] total=%d, unmod=%d (%.2f%%), ptm=%d (%.2f%%)", label, total, unmod, 100 * unmod / total, ptm, 100 * ptm / total)

    if "ptm_bucket" in df.columns:
        bucket_counts = df["ptm_bucket"].value_counts(dropna=True).to_dict()
        logger.info("[%s] PTM buckets: %s", label, bucket_counts)

    charge_dist = df[charge_col].value_counts().sort_index().to_dict()
    logger.info("[%s] charge distribution: %s", label, charge_dist)

    # Simple sanity: length distribution if available
    if "peptide_length" in df.columns:
        len_dist = df["peptide_length"].value_counts().sort_index().head(10).to_dict()
        logger.info("[%s] peptide_length (first 10): %s", label, len_dist)


def run_prospect_sampling_streaming(
    train_limit: int,
    val_limit: int,
    test_limit: int,
    split_strategy: str = "backbone",
    output_subdir: Optional[str] = None,
) -> None:
    """Run PROSPECT sampling. split_strategy: backbone | random | sequence."""
    if output_subdir:
        out_dir = _ROOT / "data" / "reconstructed" / output_subdir
    else:
        out_dir = PROSPECT_ALL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42) if split_strategy == "random" else None

    logger.info("Starting PROSPECT sampling with split_strategy=%s into %s", split_strategy, out_dir)

    storage = {
        "train": {"unmod": [], "ox": [], "cam": [], "ace": []},
        "val":   {"unmod": [], "ox": [], "cam": [], "ace": []},
        "test":  {"unmod": [], "ox": [], "cam": [], "ace": []}
    }

    in_files = [
        PROSPECT_FILTERED_DIR / "train.charge_le6.unimod135.parquet",
        PROSPECT_FILTERED_DIR / "val.charge_le6.unimod135.parquet",
        PROSPECT_FILTERED_DIR / "test.charge_le6.unimod135.parquet",
    ]
    for in_path in in_files:
        if not in_path.exists():
            continue
        logger.info("Streaming from %s", in_path)
        pf = pq.ParquetFile(str(in_path))
        n_batches = max(1, pf.metadata.num_rows // 100000)
        for batch in tqdm(pf.iter_batches(batch_size=100000), total=n_batches):
            df = _process_batch_for_prospect(batch.to_pandas())
            if df.empty:
                continue
            for _, row in df.iterrows():
                if split_strategy == "backbone":
                    split = _get_split_by_hash(row["naked_sequence"])
                elif split_strategy == "sequence":
                    split = _get_split_by_sequence(row.get("normalized_sequence", row.get("modified_sequence", row["naked_sequence"])))
                elif split_strategy == "random":
                    split = _get_split_random(int(row["sample_key"]), rng)
                else:
                    raise ValueError(f"Unknown split_strategy: {split_strategy}")
                bucket = "unmod" if not row["has_ptm"] else row["ptm_bucket"]
                if bucket in storage[split]:
                    storage[split][bucket].append(row)

    # Final assembly for each split
    for split, target in [("train", train_limit), ("val", val_limit), ("test", test_limit)]:
        unmod_target = target // 2
        ptm_target = target - unmod_target
        
        # Unmod
        unmod_df = pd.DataFrame(storage[split]["unmod"])
        if len(unmod_df) > unmod_target:
            unmod_df = unmod_df.sample(n=unmod_target, random_state=42)
        
        # PTM
        ptm_parts = []
        b_target = ptm_target // 3
        for b in ["ox", "cam", "ace"]:
            b_df = pd.DataFrame(storage[split][b])
            if len(b_df) > b_target:
                b_df = b_df.sample(n=b_target, random_state=42)
            ptm_parts.append(b_df)
        
        ptm_df = pd.concat(ptm_parts)
        # If we still need more PTMs to reach target due to small buckets
        if len(ptm_df) < ptm_target:
            # Add remaining from any PTM bucket
            remaining_ptm = []
            for b in ["ox", "cam", "ace"]:
                b_all = pd.DataFrame(storage[split][b])
                remaining_ptm.append(b_all[~b_all.index.isin(ptm_df.index) if not ptm_df.empty else slice(None)])
            if remaining_ptm:
                extra_pool = pd.concat(remaining_ptm)
                if not extra_pool.empty:
                    take = extra_pool.sample(n=min(len(extra_pool), ptm_target - len(ptm_df)), random_state=42)
                    ptm_df = pd.concat([ptm_df, take])
        
        final_df = pd.concat([unmod_df, ptm_df]).sample(frac=1, random_state=42)
        out_path = out_dir / f"{split}.parquet"
        final_df.to_parquet(out_path, index=False)
        logger.info(f"Saved PROSPECT {split}: {len(final_df)} rows to {out_path}")
        _report_stats(final_df, f"PROSPECT_{split}")


def run_massive_sampling_streaming(train_limit: int, val_limit: int, test_limit: int) -> None:
    MASSIVE_ALL_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting MassIVE-KB sampling into data/reconstructed/massive_kb/all (Official Split + Head N)...")
    
    for split, target in [("train", train_limit), ("val", val_limit), ("test", test_limit)]:
        split_src_dir = MASSIVE_FILTERED_DIR / split
        if not split_src_dir.is_dir():
            logger.warning(f"Official split dir {split_src_dir} not found. Skipping.")
            continue
            
        files = sorted(split_src_dir.glob("*.parquet"))
        logger.info(f"Processing MassIVE-KB {split} from {len(files)} files...")
        
        collected = []
        count = 0
        
        for fp in files:
            pf = pq.ParquetFile(str(fp))
            for batch in pf.iter_batches(batch_size=100000):
                df = _process_batch_for_prospect(batch.to_pandas())
                if df.empty: continue
                
                needed = target - count
                if needed <= 0: break
                
                take = df.head(needed)
                collected.append(take)
                count += len(take)
                if count >= target: break
            if count >= target: break
            
        if collected:
            final_df = pd.concat(collected)
            out_path = MASSIVE_ALL_DIR / f"{split}.parquet"
            final_df.to_parquet(out_path, index=False)
            logger.info(f"Saved MassIVE-KB {split}: {len(final_df)} rows to {out_path}")
            _report_stats(final_df, f"MassIVE_{split}")


def run_convert_234d(datasets: Sequence[str], *, extra_splits: Optional[Sequence[str]] = None) -> None:
    try:
        from scripts.data_process.add_level1_labels import process_directory
    except ImportError as e:
        logger.error("Failed to import add_level1_labels: %s", e)
        return

    if "prospect" in datasets or "both" in datasets:
        in_dir = PROSPECT_ALL_DIR
        out_dir = PROSPECT_234D_DIR
        if in_dir.is_dir():
            logger.info("Converting PROSPECT all (135d) to 234d: %s -> %s", in_dir, out_dir)
            process_directory(in_dir, out_dir, recursive=False, overwrite=True, label_column="intensities_raw")

        # Convert ablation splits (prospect_random, prospect_sequence)
        for sub in extra_splits or []:
            in_sub = _ROOT / "data" / "reconstructed" / sub / "all"
            out_sub = _ROOT / "data" / "reconstructed" / sub / "234d"
            if in_sub.is_dir():
                logger.info("Converting %s all to 234d: %s -> %s", sub, in_sub, out_sub)
                process_directory(in_sub, out_sub, recursive=False, overwrite=True, label_column="intensities_raw")

    if "massivekb" in datasets or "both" in datasets:
        in_dir = MASSIVE_ALL_DIR
        out_dir = MASSIVE_234D_DIR
        if in_dir.is_dir():
            logger.info("Converting MassIVE-KB all (135d) to 234d: %s -> %s", in_dir, out_dir)
            process_directory(in_dir, out_dir, recursive=False, overwrite=True, label_column="intensities_raw")


def main() -> None:
    p = argparse.ArgumentParser(description="Smart stratified sampling for PROSPECT and MassIVE-KB (Streaming).")
    p.add_argument("--dataset", choices=["prospect", "massivekb", "both"], default="both")
    p.add_argument("--train-limit", type=int, default=500_000, help="Target train spectra (paper: 500k)")
    p.add_argument("--val-limit", type=int, default=50_000, help="Target val spectra (paper: 50k)")
    p.add_argument("--test-limit", type=int, default=50_000, help="Target test spectra (paper: 50k)")
    p.add_argument("--convert-234d", action="store_true")
    p.add_argument(
        "--split-strategy",
        choices=["backbone", "random", "sequence"],
        default="backbone",
        help="PROSPECT only: backbone (default, zero leakage) | random (max leakage) | sequence (PTM-variant leakage)",
    )
    p.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Output subdir suffix, e.g. 'prospect_random' -> data/reconstructed/prospect_random/all/",
    )

    args = p.parse_args()

    if args.dataset in ("prospect", "both"):
        output_subdir = None
        if args.split_strategy != "backbone":
            output_subdir = args.output_suffix or f"prospect_{args.split_strategy}"
            if "/" not in output_subdir:
                output_subdir = f"{output_subdir}/all"
        run_prospect_sampling_streaming(
            args.train_limit,
            args.val_limit,
            args.test_limit,
            split_strategy=args.split_strategy,
            output_subdir=output_subdir,
        )

    if args.dataset in ("massivekb", "both"):
        run_massive_sampling_streaming(args.train_limit, args.val_limit, args.test_limit)

    if args.convert_234d:
        extra = []
        if args.split_strategy != "backbone" and args.dataset in ("prospect", "both"):
            extra.append(f"prospect_{args.split_strategy}")
        run_convert_234d([args.dataset], extra_splits=extra if extra else None)


if __name__ == "__main__":
    main()
