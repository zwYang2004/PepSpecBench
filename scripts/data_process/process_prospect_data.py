#!/usr/bin/env python3
"""
PROSPECT Data Processing Pipeline - v10 (All Hash-based Splitting)
===================================================================

Features:
- Multi-core Parallel Processing
- Processes BOTH PTM and Unmodified datasets
- Quality Control:
  - Score >= 70 (lowered from 100 for better generalization)
  - Peptide length >= 5 (minimum viable length)
  - Empty Spectra Filtering
  - 20ppm Mass Error Filtering (using Pyteomics)
- NO Deduplication (keeping all PSMs for data diversity)
- MD5 Hash-based Deterministic Train/Val/Test Splitting (80/10/10)
- Checkpoint Recovery System (skips already processed data)
- Final merge of PTM + Unmodified into unified train/val/test

Changes in v10:
- PTM data now uses MD5 hash-based splitting (same as Unmodified)
- No mapping table needed - fully reproducible with hash function:
    bucket = int(hashlib.md5(naked_sequence.encode())[:8], 16) % 10
    Train: bucket 0-7 (80%), Val: bucket 8 (10%), Test: bucket 9 (10%)
- Removed separate PTM-Test processing (now included in hash split)

Usage:
    python3 process_prospect_data.py
"""

import os
import re
import json
import hashlib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pyteomics import mass
import pyarrow as pa
import pyarrow.parquet as pq

# ============================================================================
# Logging Configuration
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(os.environ.get("MS2B_DATA_ROOT", ROOT / "data"))
PROSPECT_ROOT = os.environ.get("MS2B_PROSPECT_ROOT", str(DEFAULT_DATA_ROOT / "raw" / "Prospect"))

# PTM Training Sources
PTM_SOURCES = [
    os.path.join(PROSPECT_ROOT, "PTM/Multi-PTM dataset"),
    os.path.join(PROSPECT_ROOT, "PTM/Tmt dataset"),
    os.path.join(PROSPECT_ROOT, "PTM/TMT-PTM dataset"),
]

# Unmodified Training Source
UNMOD_SOURCES = [
    os.path.join(PROSPECT_ROOT, "unmodified"),
]

# Official Test-PTM dataset (will be merged and re-split by hash)
PTM_TEST_SOURCE = os.path.join(PROSPECT_ROOT, "Test/Test-PTM dataset")

# Add PTM_TEST_SOURCE to PTM_SOURCES for unified processing
PTM_SOURCES.append(PTM_TEST_SOURCE)

OUTPUT_DIR = os.environ.get("MS2B_PROSPECT_OUTPUT_DIR", str(DEFAULT_DATA_ROOT / "Prospect_parquet"))
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Quality Control Thresholds
SCORE_THRESHOLD = 70              # Lowered from 100 for better generalization
MIN_PEPTIDE_LENGTH = 5            # Minimum viable peptide length
MASS_ERROR_THRESHOLD_PPM = 20.0   # 20ppm mass error filter

# Data Splitting
VAL_RATIO = 0.2
RANDOM_SEED = 42

# Multicore settings (1TB RAM, 4 cores safe)
N_WORKERS = 4

# Proton mass for MZ calculation
PROTON = 1.00727646687

# ============================================================================
# Mass Calculation Helpers
# ============================================================================

def parse_prospect_sequence(seq):
    """
    Parse PROSPECT modified sequence and calculate theoretical mass.
    Returns theoretical mass or None if parsing fails.
    """
    # Extract mods: [ModName]
    mods = re.findall(r'\[(.*?)\]', str(seq))
    naked = re.sub(r'\[.*?\]', '', str(seq))
    
    try:
        base_mass = mass.calculate_mass(sequence=naked)
        
        mod_mass = 0.0
        for mod in mods:
            mod_lower = mod.lower()
            if 'oxidation' in mod_lower:
                mod_mass += 15.994915
            elif 'carbamidomethyl' in mod_lower:
                mod_mass += 57.021464
            elif 'acetyl' in mod_lower:
                mod_mass += 42.010565
            elif 'phospho' in mod_lower:
                mod_mass += 79.966331
            elif 'deamidated' in mod_lower:
                mod_mass += 0.984016
            elif 'tmtpro' in mod_lower:
                mod_mass += 304.207146
            elif 'tmt6plex' in mod_lower or 'tmt' in mod_lower:
                mod_mass += 229.162932
            else:
                # Unknown mod - return None to skip this PSM for mass filter
                return None
                
        return base_mass + mod_mass
    except:
        return None


def calculate_ppm_error(row):
    """Calculate ppm error for a single row."""
    try:
        seq = row['modified_sequence']
        exp_mz = row['precursor_mz']
        charge = row['precursor_charge']
        
        theoretical_mass = parse_prospect_sequence(seq)
        
        if theoretical_mass is None:
            return np.nan  # Cannot calculate, will be kept
            
        # Theoretical MZ = (M + z*H) / z
        theoretical_mz = (theoretical_mass + (charge * PROTON)) / charge
        
        ppm_error = abs(exp_mz - theoretical_mz) / theoretical_mz * 1e6
        return ppm_error
    except:
        return np.nan


def apply_mass_error_filter(df, threshold_ppm=MASS_ERROR_THRESHOLD_PPM):
    """Apply 20ppm mass error filter."""
    print("  Calculating mass errors...")
    
    # Calculate ppm errors
    tqdm.pandas(desc="  Mass error calc")
    df['ppm_error'] = df.progress_apply(calculate_ppm_error, axis=1)
    
    # Count how many will be filtered
    valid_errors = df['ppm_error'].notna()
    high_error = (df['ppm_error'] > threshold_ppm) & valid_errors
    
    print(f"  Valid mass calculations: {valid_errors.sum():,}")
    print(f"  High error (>{threshold_ppm}ppm): {high_error.sum():,}")
    
    # Keep: NaN (couldn't calculate, conservative) OR <= threshold
    keep_mask = df['ppm_error'].isna() | (df['ppm_error'] <= threshold_ppm)
    df_filtered = df[keep_mask].copy()
    
    # Drop the helper column
    df_filtered.drop(columns=['ppm_error'], inplace=True)
    
    return df_filtered


def assign_hash_splits(naked_sequences: pd.Series) -> np.ndarray:
    """Deterministically assign each naked_sequence to train/val/test via hashing.

    Strategy (per user spec):
      bucket = md5(naked_sequence) % 10
        - 0-7 -> train (80%)
        - 8   -> val   (10%)
        - 9   -> test  (10%)

    Using naked_sequence ensures backbone exclusivity across splits.
    """
    # Ensure strings
    seq_series = naked_sequences.astype(str)
    unique_seqs = seq_series.unique()
    
    def _bucket(seq: str) -> int:
        h = hashlib.md5(seq.encode("utf-8")).hexdigest()
        # Use first 8 hex digits for a stable int
        return int(h[:8], 16) % 10

    mapping = {s: _bucket(s) for s in unique_seqs}
    buckets = seq_series.map(mapping).to_numpy()
    
    labels = np.full(len(buckets), "train", dtype=object)
    labels[buckets == 8] = "val"
    labels[buckets == 9] = "test"
    return labels

# ============================================================================
# Checkpoint Functions
# ============================================================================

def save_checkpoint(data: pd.DataFrame, name: str):
    """Save checkpoint to disk."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{name}.parquet")
    print(f"\n💾 Saving checkpoint: {checkpoint_path}...")
    data.to_parquet(checkpoint_path, index=False)
    print(f"   Saved: {len(data):,} rows, {os.path.getsize(checkpoint_path) / 1e9:.2f} GB")


def load_checkpoint(name: str) -> Optional[pd.DataFrame]:
    """Load checkpoint from disk if exists."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{name}.parquet")
    if os.path.exists(checkpoint_path):
        print(f"\n📂 Loading checkpoint: {checkpoint_path}")
        df = pd.read_parquet(checkpoint_path)
        print(f"   Loaded: {len(df):,} rows")
        return df
    return None


def checkpoint_exists(name: str) -> bool:
    """Check if checkpoint exists."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{name}.parquet")
    return os.path.exists(checkpoint_path)


# ============================================================================
# Multicore Processing Functions
# ============================================================================

def load_and_aggregate_single_file(file_path: str) -> pd.DataFrame:
    """Load a single annotation file and aggregate by PSM."""
    try:
        df = pd.read_parquet(file_path)
        
        # Aggregate immediately to save memory
        aggregated = df.groupby(['raw_file', 'scan_number']).agg({
            'experimental_mass': list,
            'intensity': list
        }).reset_index()
        
        aggregated.rename(columns={
            'experimental_mass': 'mz',
            'intensity': 'intensities'
        }, inplace=True)
        
        return aggregated
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()


def load_and_aggregate_batch_parallel(batch_files: List[str], batch_num: int, total_batches: int) -> pd.DataFrame:
    """Load and aggregate a batch of files in parallel."""
    print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch_files)} files) [Parallel]...")
    
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(load_and_aggregate_single_file, f): f for f in batch_files}
        
        with tqdm(total=len(batch_files), desc=f"  Loading batch {batch_num}") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if not result.empty:
                        results.append(result)
                    pbar.update(1)
                except Exception as e:
                    print(f"  ⚠️  Batch Error: {e}")
    
    if not results:
        return pd.DataFrame()
        
    df_combined = pd.concat(results, ignore_index=True)
    print(f"  Aggregated into {len(df_combined):,} PSMs")
    return df_combined


# ============================================================================
# Helper Functions
# ============================================================================

def find_parquet_files(root_dir: str) -> Tuple[List[str], List[str]]:
    """Find annotation and metadata Parquet files."""
    root_path = Path(root_dir)
    annotation_files = list(root_path.rglob("*_annotation.parquet"))
    metadata_files = list(root_path.rglob("*_meta_data.parquet"))
    return [str(f) for f in annotation_files], [str(f) for f in metadata_files]


def load_and_aggregate_annotations(annotation_files: List[str], batch_size: int = 100) -> pd.DataFrame:
    """Load annotation files in batches with parallel processing."""
    print(f"\n{'='*60}")
    print(f"STAGE 1.1: Loading Annotation Files (Multi-core: {N_WORKERS} workers)")
    print(f"{'='*60}")
    
    all_aggregated = []
    total_psms = 0
    
    for i in range(0, len(annotation_files), batch_size):
        batch_files = annotation_files[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(annotation_files) + batch_size - 1) // batch_size
        
        batch_result = load_and_aggregate_batch_parallel(batch_files, batch_num, total_batches)
        
        if not batch_result.empty:
            batch_psms = len(batch_result)
            total_psms += batch_psms
            print(f"  Cumulative PSMs: {total_psms:,}")
            all_aggregated.append(batch_result)
    
    print(f"\nCombining all batches...")
    if not all_aggregated:
        return pd.DataFrame()
        
    df_combined = pd.concat(all_aggregated, ignore_index=True)
    print(f"\n✓ Combined into {len(df_combined):,} PSMs")
    return df_combined


def load_metadata(metadata_files: List[str]) -> pd.DataFrame:
    """Load metadata files."""
    print(f"\n{'='*60}")
    print("STAGE 1.2: Loading Metadata Files")
    print(f"{'='*60}")
    
    all_metadata = []
    for file_path in tqdm(metadata_files, desc="Loading metadata"):
        try:
            df = pd.read_parquet(file_path)
            all_metadata.append(df)
        except Exception as e:
            print(f"Error loading metadata {file_path}: {e}")
    
    df_meta = pd.concat(all_metadata, ignore_index=True)
    print(f"✓ Loaded {len(df_meta):,} PSM records from {len(metadata_files)} files")
    
    # Check available columns
    print(f"Metadata columns: {df_meta.columns.tolist()}")
    
    # Correct mapping based on actual file content
    column_mapping = {
        'Fragmentation': 'fragmentation',
        'Mass_analyzer': 'instrument',
        'mass_analyzer': 'instrument',  # Handle lowercase variation
        'modified_sequence': 'modified_sequence',
        'precursor_charge': 'precursor_charge',
        'collision_energy': 'collision_energy',
        'orig_collision_energy': 'collision_energy', # Prioritize this if exists
        'Score': 'score',
        'andromeda_score': 'score',     # Correct mapping for PROSPECT
        'retention_time': 'retention_time',
        'raw_file': 'raw_file',
        'scan_number': 'scan_number',
        'precursor_mz': 'precursor_mz'
    }
    
    df_meta = df_meta.rename(columns=column_mapping)
    
    # Ensure required columns exist
    required_cols = ['raw_file', 'scan_number', 'modified_sequence', 'precursor_charge',
                     'collision_energy', 'fragmentation', 'instrument', 'score', 'retention_time', 'precursor_mz']
    
    # Filter only existing columns to avoid KeyError
    existing_cols = [c for c in required_cols if c in df_meta.columns]
    df_meta = df_meta[existing_cols]
    
    return df_meta


def integrate_data(df_anno: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """Merge annotations with metadata."""
    print(f"\n{'='*60}")
    print("STAGE 1.3: Integrating Data")
    print(f"{'='*60}")
    
    # Inner merge to keep only PSMs with both annotation and metadata
    df_merged = pd.merge(df_anno, df_meta, on=['raw_file', 'scan_number'], how='inner')
    print(f"✓ Merged PSMs: {len(df_merged):,}")
    return df_merged


def extract_naked_sequence(modified_seq: str) -> str:
    """Remove PTM annotations (e.g., A[Oxidation]BC -> ABC)."""
    return re.sub(r'\[.*?\]', '', str(modified_seq))


def clean_data(df: pd.DataFrame, score_threshold: float = SCORE_THRESHOLD) -> pd.DataFrame:
    """Apply quality filters."""
    print(f"\n{'='*60}")
    print("STAGE 2: Cleaning Data")
    print(f"{'='*60}")
    
    initial_count = len(df)
    print(f"Initial count: {initial_count:,}")
    
    # Handle column naming inconsistencies (if not already handled in metadata loading)
    if 'score' not in df.columns:
        for col in ['Score', 'andromeda_score', 'Andromeda Score']:
            if col in df.columns:
                print(f"Renaming '{col}' to 'score'")
                df.rename(columns={col: 'score'}, inplace=True)
                break
    
    # 1. Filter by Score
    if 'score' in df.columns:
        df = df[df['score'] >= score_threshold].copy()
        print(f"✓ After score filter (>={score_threshold}): {len(df):,}")
    else:
        print("⚠️ 'score' column not found, skipping score filter.")
        print(f"Available columns: {df.columns.tolist()}")

    
    # 2. Filter Empty Spectra
    if 'intensities' in df.columns:
        df = df[df['intensities'].apply(len) > 0].copy()
        print(f"✓ After removing empty spectra: {len(df):,}")
    
    # 3. Filter Mass Error (20 ppm)
    if 'precursor_mz' in df.columns:
        before_mass = len(df)
        df = apply_mass_error_filter(df, MASS_ERROR_THRESHOLD_PPM)
        print(f"✓ After {MASS_ERROR_THRESHOLD_PPM}ppm Mass Error filter: {len(df):,} (-{before_mass - len(df):,})")
    else:
        print("⚠️ 'precursor_mz' column not found, skipping mass error filter.")
    
    # 4. Add Naked Sequence
    df['naked_sequence'] = df['modified_sequence'].apply(extract_naked_sequence)
    df['peptide_length'] = df['naked_sequence'].str.len()
    
    # 5. Filter small peptides (minimum viable length)
    df = df[df['peptide_length'] >= MIN_PEPTIDE_LENGTH].copy()
    print(f"✓ After length filter (>={MIN_PEPTIDE_LENGTH}): {len(df):,}")
    
    print(f"✓ Final cleaned PSMs: {len(df):,}")
    return df


def split_train_val_test_hash(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test based on MD5 hash of naked sequence.
    
    Uses the same deterministic hash-based splitting as Unmodified data:
        bucket = int(md5(naked_sequence)[:8], 16) % 10
        0-7 -> train (80%), 8 -> val (10%), 9 -> test (10%)
    
    This ensures reproducibility without needing a mapping table.
    """
    print(f"\n{'='*60}")
    print("STAGE 3: Hash-based Train/Val/Test Split (Deterministic)")
    print(f"{'='*60}")
    
    unique_seqs = df['naked_sequence'].unique()
    print(f"Unique naked_sequence: {len(unique_seqs):,}")
    
    # Use the same hash function as Unmodified data
    split_labels = assign_hash_splits(df['naked_sequence'])
    
    train_df = df[split_labels == 'train'].copy()
    val_df = df[split_labels == 'val'].copy()
    test_df = df[split_labels == 'test'].copy()
    
    print(f"✓ Train: {len(train_df):,} PSMs ({train_df['naked_sequence'].nunique():,} backbones)")
    print(f"✓ Val:   {len(val_df):,} PSMs ({val_df['naked_sequence'].nunique():,} backbones)")
    print(f"✓ Test:  {len(test_df):,} PSMs ({test_df['naked_sequence'].nunique():,} backbones)")
    
    return train_df, val_df, test_df


def flag_test_leakage(test_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'is_in_train' column to test_df.
    True if naked_sequence appears in train_df or val_df.
    """
    print(f"\n{'='*60}")
    print("STAGE 4: Flagging Test Leakage (is_in_train)")
    print(f"{'='*60}")
    
    # Get set of all training sequences
    train_seqs = set(train_df['naked_sequence'].unique())
    val_seqs = set(val_df['naked_sequence'].unique())
    all_train_seqs = train_seqs.union(val_seqs)
    
    print(f"Train/Val unique sequences: {len(all_train_seqs):,}")
    
    # Flag leakage
    test_df['is_in_train'] = test_df['naked_sequence'].isin(all_train_seqs)
    
    leak_count = test_df['is_in_train'].sum()
    print(f"✓ Flagged {leak_count:,} ({leak_count/len(test_df)*100:.1f}%) test PSMs as seen in train.")
    
    return test_df


def save_final_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None):
    """Save final splits.

    If test_df is None, only train/val will be written.
    """
    print(f"\n{'='*60}")
    print("Saving Final Splits")
    print(f"{'='*60}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_path = os.path.join(OUTPUT_DIR, "train.parquet")
    val_path = os.path.join(OUTPUT_DIR, "val.parquet")
    test_path = os.path.join(OUTPUT_DIR, "test.parquet")
    
    print("Saving train...")
    train_df.to_parquet(train_path, index=False)
    print(f"✓ Saved: {train_path}")
    
    print("Saving val...")
    val_df.to_parquet(val_path, index=False)
    print(f"✓ Saved: {val_path}")
    
    if test_df is not None:
        print("Saving test...")
        test_df.to_parquet(test_path, index=False)
        print(f"✓ Saved: {test_path}")


"""Source-level processing helpers"""

# ============================================================================
# Generic source processing (used for PTM)
# ============================================================================

def process_source(source_name: str, sources: List[str], checkpoint_prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process a data source (e.g., PTM) through the full pipeline.

    This path is suitable for sources with moderate size (current PTM sources).
    It keeps the full integrated DataFrame in memory.
    
    Now uses MD5 hash-based deterministic splitting (same as Unmodified data)
    for full reproducibility without needing a mapping table.
    """
    print(f"\n" + "#"*60)
    print(f"# Processing {source_name} Sources")
    print("#"*60)
    
    # Check if final checkpoint exists
    train_ckpt = f"{checkpoint_prefix}_train_final"
    val_ckpt = f"{checkpoint_prefix}_val_final"
    test_ckpt = f"{checkpoint_prefix}_test_final"
    
    if checkpoint_exists(train_ckpt) and checkpoint_exists(val_ckpt) and checkpoint_exists(test_ckpt):
        print(f"\n✅ Found checkpoints: {train_ckpt}, {val_ckpt}, {test_ckpt}")
        train_df = load_checkpoint(train_ckpt)
        val_df = load_checkpoint(val_ckpt)
        test_df = load_checkpoint(test_ckpt)
        return train_df, val_df, test_df
    
    # Check integrated checkpoint
    integrated_ckpt = f"{checkpoint_prefix}_integrated"
    if checkpoint_exists(integrated_ckpt):
        print(f"\n✅ Found checkpoint: {integrated_ckpt}")
        df_raw = load_checkpoint(integrated_ckpt)
    else:
        all_anno = []
        all_meta = []
        
        for source in sources:
            print(f"\nScanning: {source}")
            if not os.path.exists(source):
                print(f"  ⚠️ Source not found, skipping: {source}")
                continue
            anno_files, meta_files = find_parquet_files(source)
            print(f"  Found {len(anno_files)} annotation, {len(meta_files)} metadata")
            all_anno.extend(anno_files)
            all_meta.extend(meta_files)
        
        if not all_anno:
            print(f"\n⚠️ No annotation files found for {source_name}. Returning empty DataFrames.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        print(f"\nTotal: {len(all_anno)} annotation, {len(all_meta)} metadata")
        
        df_anno = load_and_aggregate_annotations(all_anno)
        df_meta = load_metadata(all_meta)
        df_raw = integrate_data(df_anno, df_meta)
        
        save_checkpoint(df_raw, integrated_ckpt)
        del df_anno, df_meta
    
    # Check cleaned checkpoint
    cleaned_ckpt = f"{checkpoint_prefix}_cleaned"
    if checkpoint_exists(cleaned_ckpt):
        print(f"\n✅ Found checkpoint: {cleaned_ckpt}")
        df_clean = load_checkpoint(cleaned_ckpt)
    else:
        df_clean = clean_data(df_raw)
        save_checkpoint(df_clean, cleaned_ckpt)
        del df_raw
    
    # NOTE: Deduplication step REMOVED per user request
    # Keeping all PSMs to preserve data diversity for better model generalization
    
    # Split train/val/test using deterministic MD5 hash (same as Unmodified)
    train_df, val_df, test_df = split_train_val_test_hash(df_clean)
    
    # Add source tag
    train_df['source_type'] = source_name
    val_df['source_type'] = source_name
    test_df['source_type'] = source_name
    
    save_checkpoint(train_df, train_ckpt)
    save_checkpoint(val_df, val_ckpt)
    save_checkpoint(test_df, test_ckpt)
    del df_clean
    
    return train_df, val_df, test_df


# ============================================================================
# Streaming processing for Unmodified (OOM-safe)
# ============================================================================

def process_unmodified_streaming(sources: List[str], checkpoint_prefix: str = "unmod") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hash-based, streaming split for Unmodified sources.

    This implementation follows the user's spec:
      - Load all metadata once (metadata is relatively small).
      - Stream annotation files in batches.
      - For each batch: integrate -> clean -> deduplicate.
      - Compute naked_sequence-based hash bucket to assign each PSM to
        train/val/test *deterministically* (backbone exclusivity).
      - Append rows to on-disk Parquet files:
            unmod_train_hash.parquet
            unmod_val_hash.parquet
            unmod_test_hash.parquet

    To remain OOM-safe, we do not keep full Unmodified data in memory, and we
    do not return large DataFrames. The function returns empty DataFrames and
    is used for its side effects (writing Parquet files).
    """
    source_name = "Unmodified"
    print(f"\n" + "#"*60)
    print(f"# Processing {source_name} Sources (Streaming)")
    print("#"*60)

    # Final output paths for Unmodified hash-split data
    unmod_train_path = Path(OUTPUT_DIR) / "unmod_train.parquet"
    unmod_val_path = Path(OUTPUT_DIR) / "unmod_val.parquet"
    unmod_test_path = Path(OUTPUT_DIR) / "unmod_test.parquet"

    # If all three exist, assume processing already completed
    if unmod_train_path.exists() and unmod_val_path.exists() and unmod_test_path.exists():
        print("\n✅ Found existing Unmodified hash-split outputs. Skipping re-processing.")
        return pd.DataFrame(), pd.DataFrame()

    all_anno: List[str] = []
    all_meta: List[str] = []

    for source in sources:
        print(f"\nScanning: {source}")
        if not os.path.exists(source):
            print(f"  ⚠️ Source not found, skipping: {source}")
            continue
        anno_files, meta_files = find_parquet_files(source)
        print(f"  Found {len(anno_files)} annotation, {len(meta_files)} metadata")
        all_anno.extend(anno_files)
        all_meta.extend(meta_files)

    if not all_anno:
        print(f"\n⚠️ No annotation files found for {source_name}. Returning empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame()

    print(f"\nTotal: {len(all_anno)} annotation, {len(all_meta)} metadata")

    # Load all metadata once (size is manageable)
    df_meta = load_metadata(all_meta)

    batch_size = 100
    total_batches = (len(all_anno) + batch_size - 1) // batch_size

    # Prepare Parquet writers for streaming appends
    train_writer = None
    val_writer = None
    test_writer = None

    def _write_split(df_split: pd.DataFrame, kind: str):
        nonlocal train_writer, val_writer, test_writer
        if df_split.empty:
            return
        table = pa.Table.from_pandas(df_split, preserve_index=False)
        if kind == "train":
            if train_writer is None:
                train_writer = pq.ParquetWriter(str(unmod_train_path), table.schema)
            train_writer.write_table(table)
        elif kind == "val":
            if val_writer is None:
                val_writer = pq.ParquetWriter(str(unmod_val_path), table.schema)
            val_writer.write_table(table)
        elif kind == "test":
            if test_writer is None:
                test_writer = pq.ParquetWriter(str(unmod_test_path), table.schema)
            test_writer.write_table(table)

    for i in range(0, len(all_anno), batch_size):
        batch_files = all_anno[i:i + batch_size]
        batch_num = i // batch_size + 1

        # 1) Aggregate annotations for this batch (parallel)
        df_batch_anno = load_and_aggregate_batch_parallel(batch_files, batch_num, total_batches)
        if df_batch_anno.empty:
            continue

        # 2) Integrate with metadata
        df_batch_raw = integrate_data(df_batch_anno, df_meta)
        del df_batch_anno

        # 3) Clean only (NO deduplication per user request)
        df_batch_clean = clean_data(df_batch_raw)
        del df_batch_raw

        # NOTE: Deduplication step REMOVED per user request
        # Keeping all PSMs to preserve data diversity for better model generalization

        # Ensure naked_sequence exists (added in clean_data) and assign hash-based splits
        if 'naked_sequence' not in df_batch_clean.columns:
            df_batch_clean['naked_sequence'] = df_batch_clean['modified_sequence'].apply(extract_naked_sequence)

        split_labels = assign_hash_splits(df_batch_clean['naked_sequence'])
        df_batch_clean['source_type'] = source_name

        # Split into train / val / test
        mask_train = split_labels == 'train'
        mask_val = split_labels == 'val'
        mask_test = split_labels == 'test'

        df_train_split = df_batch_clean[mask_train].copy()
        df_val_split = df_batch_clean[mask_val].copy()
        df_test_split = df_batch_clean[mask_test].copy()

        # We no longer need the helper array
        del split_labels

        print(f"\nBatch {batch_num}/{total_batches} - rows after clean (no dedup): {len(df_batch_clean):,}")
        print(f"  -> Train: {len(df_train_split):,}, Val: {len(df_val_split):,}, Test: {len(df_test_split):,}")

        # 4) Stream to Parquet files
        _write_split(df_train_split, "train")
        _write_split(df_val_split, "val")
        _write_split(df_test_split, "test")

        del df_batch_clean, df_train_split, df_val_split, df_test_split

    # Close writers
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()
    if test_writer is not None:
        test_writer.close()

    print("\n✓ Unmodified hash-split completed.")
    print(f"  Train file: {unmod_train_path}")
    print(f"  Val file:   {unmod_val_path}")
    print(f"  Test file:  {unmod_test_path}")

    # For the main() pipeline, we do not load these large files back into memory.
    # Return empty DataFrames as placeholders.
    return pd.DataFrame(), pd.DataFrame()


# ============================================================================
# Merge PTM + Unmodified outputs and generate stratified test sets
# ============================================================================

def merge_and_stratify_outputs() -> None:
    """Merge PTM and Unmodified splits, analyze leakage, and create final outputs.

    This step operates purely on disk-level Parquet outputs produced by the
    earlier stages:

        OUTPUT_DIR/
          - train.parquet          # PTM train
          - val.parquet            # PTM val
          - test.parquet           # Official PTM-Test (static input)
          - unmod_train.parquet    # Unmodified train (80%)
          - unmod_val.parquet      # Unmodified val (10%)
          - unmod_test.parquet     # Unmodified test (10%)

    It produces merged and stratified artifacts under:

        OUTPUT_DIR/Prospect_merged/
          - train.parquet         # PTM + Unmodified train
          - val.parquet           # PTM + Unmodified val
          - test.parquet          # PTM-Test + Unmodified test
          - test_clean.parquet    # Test minus any backbones seen in train/val
          - test_leaked.parquet   # Test PSMs whose backbones appear in train/val
          - test_ptm.parquet      # PTM-only test subset
          - test_unmod.parquet    # Unmodified-only test subset
          - SPLIT_INFO.md         # Text summary of backbone stats and leakage

    Additionally, a small checkpoint manifest with input row counts is stored
    via save_checkpoint() under CHECKPOINT_DIR as "merge_inputs_manifest".
    """

    base_dir = Path(OUTPUT_DIR)
    merged_dir = base_dir / "Prospect_merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#"*60)
    print("# Merging PTM + Unmodified Data")
    print("#"*60)
    print(f"\nOutput directory: {merged_dir}")

    # Source files (produced by the main PTM + Unmodified pipeline)
    ptm_train_path = base_dir / "train.parquet"
    ptm_val_path = base_dir / "val.parquet"
    ptm_test_path = base_dir / "test.parquet"  # Official PTM-Test (static)
    unmod_train_path = base_dir / "unmod_train.parquet"
    unmod_val_path = base_dir / "unmod_val.parquet"
    unmod_test_path = base_dir / "unmod_test.parquet"

    required_paths = [
        ptm_train_path,
        ptm_val_path,
        ptm_test_path,
        unmod_train_path,
        unmod_val_path,
        unmod_test_path,
    ]

    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        print("\n⚠️  Cannot run merge step because the following inputs are missing:")
        for m in missing:
            print(f"   - {m}")
        print("   Please ensure PTM and Unmodified processing have completed successfully.")
        return

    def _load_and_tag(path: Path, source_tag: str) -> pd.DataFrame:
        print(f"Loading {path.name}...")
        df = pd.read_parquet(path)
        if "source_type" not in df.columns:
            df["source_type"] = source_tag
        print(f"  -> {len(df):,} rows")
        return df

    # Load all inputs once
    ptm_train = _load_and_tag(ptm_train_path, "PTM")
    ptm_val = _load_and_tag(ptm_val_path, "PTM")
    ptm_test = _load_and_tag(ptm_test_path, "PTM-Test")

    unmod_train = _load_and_tag(unmod_train_path, "Unmodified")
    unmod_val = _load_and_tag(unmod_val_path, "Unmodified")
    unmod_test = _load_and_tag(unmod_test_path, "Unmodified")

    # ----------------------------------------------------------------------
    # Pre-merge checkpoint manifest (small, just row counts)
    # ----------------------------------------------------------------------
    manifest_rows = []
    for df, src, split in [
        (ptm_train, "PTM", "train"),
        (ptm_val, "PTM", "val"),
        (ptm_test, "PTM-Test", "test"),
        (unmod_train, "Unmodified", "train"),
        (unmod_val, "Unmodified", "val"),
        (unmod_test, "Unmodified", "test"),
    ]:
        manifest_rows.append({
            "source": src,
            "split": split,
            "rows": len(df),
        })

    manifest_df = pd.DataFrame(manifest_rows)
    save_checkpoint(manifest_df, "merge_inputs_manifest")

    # ----------------------------------------------------------------------
    # Merge train / val / test
    # ----------------------------------------------------------------------
    print("\n" + "="*60)
    print("Merging Datasets")
    print("="*60)

    merged_train = pd.concat([ptm_train, unmod_train], ignore_index=True)
    print(f"Merged Train: {len(ptm_train):,} (PTM) + {len(unmod_train):,} (Unmod) = {len(merged_train):,}")

    merged_val = pd.concat([ptm_val, unmod_val], ignore_index=True)
    print(f"Merged Val:   {len(ptm_val):,} (PTM) + {len(unmod_val):,} (Unmod) = {len(merged_val):,}")

    merged_test = pd.concat([ptm_test, unmod_test], ignore_index=True)
    print(f"Merged Test:  {len(ptm_test):,} (PTM-Test) + {len(unmod_test):,} (Unmod) = {len(merged_test):,}")

    # ----------------------------------------------------------------------
    # Backbone leakage analysis
    # ----------------------------------------------------------------------
    print("\n" + "="*60)
    print("Checking Backbone Leakage")
    print("="*60)

    for df in [merged_train, merged_val, merged_test]:
        if "naked_sequence" not in df.columns:
            df["naked_sequence"] = df["modified_sequence"].apply(extract_naked_sequence)

    train_backbones = set(merged_train["naked_sequence"].unique())
    val_backbones = set(merged_val["naked_sequence"].unique())
    test_backbones = set(merged_test["naked_sequence"].unique())

    print(f"Train unique backbones: {len(train_backbones):,}")
    print(f"Val unique backbones:   {len(val_backbones):,}")
    print(f"Test unique backbones:  {len(test_backbones):,}")

    train_val_overlap = train_backbones & val_backbones
    train_test_overlap = train_backbones & test_backbones
    val_test_overlap = val_backbones & test_backbones

    print(f"\nTrain ∩ Val overlap:  {len(train_val_overlap):,} backbones")
    print(f"Train ∩ Test overlap: {len(train_test_overlap):,} backbones")
    print(f"Val ∩ Test overlap:   {len(val_test_overlap):,} backbones")

    # Count leaked test PSMs
    test_leaked_from_train = merged_test[merged_test["naked_sequence"].isin(train_test_overlap)] if train_test_overlap else pd.DataFrame()
    test_leaked_from_val = merged_test[merged_test["naked_sequence"].isin(val_test_overlap)] if val_test_overlap else pd.DataFrame()

    if not test_leaked_from_train.empty:
        print(f"  ⚠️ Test PSMs with backbones in Train: {len(test_leaked_from_train):,}")
    if not test_leaked_from_val.empty:
        print(f"  ⚠️ Test PSMs with backbones in Val: {len(test_leaked_from_val):,}")

    leakage_info = {
        "train_backbones": len(train_backbones),
        "val_backbones": len(val_backbones),
        "test_backbones": len(test_backbones),
        "train_val_overlap": len(train_val_overlap),
        "train_test_overlap": len(train_test_overlap),
        "val_test_overlap": len(val_test_overlap),
    }

    # ----------------------------------------------------------------------
    # Save merged train/val/test
    # ----------------------------------------------------------------------
    print("\n" + "="*60)
    print("Saving Merged Files")
    print("="*60)

    merged_train_path = merged_dir / "train.parquet"
    merged_val_path = merged_dir / "val.parquet"
    merged_test_path = merged_dir / "test.parquet"

    merged_train.to_parquet(merged_train_path, index=False)
    print(f"✓ Saved: {merged_train_path} ({len(merged_train):,} rows)")

    merged_val.to_parquet(merged_val_path, index=False)
    print(f"✓ Saved: {merged_val_path} ({len(merged_val):,} rows)")

    merged_test.to_parquet(merged_test_path, index=False)
    print(f"✓ Saved: {merged_test_path} ({len(merged_test):,} rows)")

    # ----------------------------------------------------------------------
    # Stratified test subsets: PTM-only, Unmod-only
    # (test_clean and test_leaked removed - hash-based split guarantees no leakage)
    # ----------------------------------------------------------------------
    print("\n" + "="*60)
    print("Creating Stratified Test Subsets")
    print("="*60)

    test_ptm = merged_test[merged_test["source_type"].isin(["PTM", "PTM-Test"])].copy()
    test_unmod = merged_test[merged_test["source_type"] == "Unmodified"].copy()

    test_ptm_path = merged_dir / "test_ptm.parquet"
    test_unmod_path = merged_dir / "test_unmod.parquet"

    test_ptm.to_parquet(test_ptm_path, index=False)
    test_unmod.to_parquet(test_unmod_path, index=False)

    print(f"  test_ptm.parquet:    {len(test_ptm):,} PSMs (PTM-only)")
    print(f"  test_unmod.parquet:  {len(test_unmod):,} PSMs (Unmodified-only)")

    # ----------------------------------------------------------------------
    # Write SPLIT_INFO.md summary
    # ----------------------------------------------------------------------
    report_path = merged_dir / "SPLIT_INFO.md"

    has_leakage = (leakage_info["train_test_overlap"] > 0 or leakage_info["val_test_overlap"] > 0)

    report_content = f"""# PROSPECT Merged Dataset Split Info

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Statistics

| Split | PTM PSMs | Unmodified PSMs | Total PSMs |
|-------|----------|-----------------|------------|
| Train | {len(ptm_train):,} | {len(unmod_train):,} | {len(merged_train):,} |
| Val   | {len(ptm_val):,} | {len(unmod_val):,} | {len(merged_val):,} |
| Test  | {len(ptm_test):,} | {len(unmod_test):,} | {len(merged_test):,} |

## Backbone (Naked Sequence) Statistics

| Set | Unique Backbones |
|-----|------------------|
| Train | {leakage_info['train_backbones']:,} |
| Val   | {leakage_info['val_backbones']:,} |
| Test  | {leakage_info['test_backbones']:,} |

## Leakage Analysis

| Overlap | Count |
|---------|-------|
| Train ∩ Val | {leakage_info['train_val_overlap']:,} |
| Train ∩ Test | {leakage_info['train_test_overlap']:,} |
| Val ∩ Test | {leakage_info['val_test_overlap']:,} |

{"⚠️ **WARNING**: Test set has backbone overlap with Train/Val. This may cause data leakage during evaluation." if has_leakage else "✅ **No leakage detected**: Test backbones are disjoint from Train/Val."}

## Stratified Test Subsets

- test_ptm.parquet:    {len(test_ptm):,} PSMs (PTM-only)
- test_unmod.parquet:  {len(test_unmod):,} PSMs (Unmodified-only)

## Notes

- **Split Method**: All data (PTM + Unmodified) uses MD5 hash-based deterministic splitting.
- **Reproducibility**: `bucket = int(md5(naked_sequence)[:8], 16) % 10`
  - Train: bucket 0-7 (80%)
  - Val: bucket 8 (10%)
  - Test: bucket 9 (10%)
- **No leakage possible**: Hash-based split guarantees backbone exclusivity across splits.

## Source Files

- PTM Train: `{ptm_train_path}`
- PTM Val: `{ptm_val_path}`
- PTM Test: `{ptm_test_path}`
- Unmod Train: `{unmod_train_path}`
- Unmod Val: `{unmod_val_path}`
- Unmod Test: `{unmod_test_path}`
"""

    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"✓ Saved report: {report_path}")

    print("\n" + "#"*60)
    print("# Merge + Stratified Test Generation Complete")
    print("#"*60)




# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("PROSPECT Pipeline v10 - PTM + Unmodified (All Hash-based)")
    logger.info(f"Using {N_WORKERS} CPU cores")
    logger.info(f"Quality: Score>={SCORE_THRESHOLD}, Length>={MIN_PEPTIDE_LENGTH}, Mass Error<={MASS_ERROR_THRESHOLD_PPM}ppm")
    logger.info("Split method: MD5 hash-based (deterministic, no mapping table needed)")
    logger.info("=" * 60)
    
    # ---- Process PTM Sources (now uses hash-based split like Unmodified) ----
    ptm_train, ptm_val, ptm_test = process_source("PTM", PTM_SOURCES, "ptm")

    # ---- SAVE PTM Final Splits (train/val/test) ----
    print("\n" + "#"*60)
    print("# Saving PTM train/val/test splits")
    print("#"*60)
    save_final_splits(ptm_train, ptm_val, ptm_test)

    # ---- Process Unmodified Sources (hash-based streaming, side-effect only) ----
    process_unmodified_streaming(UNMOD_SOURCES, checkpoint_prefix="unmod")

    # ---- Merge PTM + Unmodified outputs and generate final evaluation splits ----
    merge_and_stratify_outputs()

    print("\n" + "#"*60)
    print("# Pipeline Complete!")
    print("#"*60)
    print("\n📋 Reproducibility: All splits use MD5 hash-based deterministic assignment.")
    print("   No mapping table needed - use: bucket = int(md5(naked_sequence)[:8], 16) % 10")
    print("   Train: bucket 0-7 (80%), Val: bucket 8 (10%), Test: bucket 9 (10%)")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
