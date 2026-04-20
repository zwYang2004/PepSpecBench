#!/usr/bin/env python3
"""
MassIVE-KB Data Cleaning and Standardization Script

This script processes MassIVE-KB MGF files, performs:
- Universal PTM mapping (numerical prefixes -> UNIMOD IDs)
- Missing metadata imputation
- Output to Parquet format with chunking
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from pyteomics import mgf

# Configure logging (portable: no machine-specific file path).
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(os.environ.get("MS2B_DATA_ROOT", ROOT / "data"))
DEFAULT_RAW_ROOT = Path(os.environ.get("MS2B_MASSIVE_RAW_ROOT", DEFAULT_DATA_ROOT / "raw" / "MassIVE-KB"))

# Input MGF files (override with env MS2B_MASSIVE_INPUT_FILES=path1,path2,...)
_input_files_env = os.environ.get("MS2B_MASSIVE_INPUT_FILES", "").strip()
if _input_files_env:
    INPUT_FILES = [p.strip() for p in _input_files_env.split(",") if p.strip()]
else:
    INPUT_FILES = [
        str(DEFAULT_RAW_ROOT / "massivekb_82c0124b_train.mgf"),
        str(DEFAULT_RAW_ROOT / "massivekb_82c0124b_val.mgf"),
        str(DEFAULT_RAW_ROOT / "massivekb_82c0124b_test.mgf"),
    ]

# Output directory
OUTPUT_DIR = Path(os.environ.get("MS2B_MASSIVE_OUTPUT_DIR", DEFAULT_DATA_ROOT / "MassIVE-KB" / "processed"))

# Chunk size for memory management
CHUNK_SIZE = 100000

# ============================================================================
# Universal PTM Mapping Table
# ============================================================================

# Mapping from regex pattern to UNIMOD ID
# Format: (regex_pattern, unimod_id)
PTM_MAPPING = [
    (r'\+57\.021', '[UNIMOD:4]'),    # Carbamidomethyl (C)
    (r'\+15\.995', '[UNIMOD:35]'),   # Oxidation (M)
    (r'-17\.027', '[UNIMOD:28]'),    # Gln->pyro-Glu (Q N-term)
    (r'\+42\.011', '[UNIMOD:1]'),    # Acetyl (N-term)
    (r'\+43\.006', '[UNIMOD:5]'),    # Carbamyl (N-term)
    (r'\+0\.984', '[UNIMOD:7]'),     # Deamidation (N/Q)
]

# Hardcoded metadata values based on MassIVE-KB paper
HARDCODED_METADATA = {
    'instrument': 'Orbitrap',
    'collision_energy': 25,
    'fragmentation': 'HCD',
    'score': 101,
}


def standardize_sequence(seq: str) -> str:
    """
    Convert numerical PTM prefixes to UNIMOD ID format.
    
    Examples:
        -17.027QKAEADKNDK -> [UNIMOD:28]QKAEADKNDK
        M+15.995 -> M[UNIMOD:35]
        C+57.021 -> C[UNIMOD:4]
    
    Args:
        seq: Original sequence with numerical PTM notation
        
    Returns:
        Standardized sequence with UNIMOD IDs
    """
    if not seq:
        return seq
    
    result = seq
    
    # Apply each PTM mapping
    for pattern, unimod_id in PTM_MAPPING:
        result = re.sub(pattern, unimod_id, result)
    
    return result


def parse_title(title: str) -> Tuple[str, Optional[int]]:
    """
    Parse TITLE field to extract raw_file and scan_number.
    
    TITLE format: SulfenM_RKO_LCA_A01.mzXML:scan:6867
    
    Args:
        title: TITLE string from MGF
        
    Returns:
        Tuple of (raw_file, scan_number)
    """
    if not title:
        return '', None
    
    # Pattern: filename.mzXML:scan:number
    match = re.match(r'^(.+?)\.mzXML:scan:(\d+)$', title)
    if match:
        raw_file = match.group(1)
        scan_number = int(match.group(2))
        return raw_file, scan_number
    
    # Fallback: try other patterns
    # Pattern: filename:scan:number (without extension)
    match = re.match(r'^(.+?):scan:(\d+)$', title)
    if match:
        raw_file = match.group(1).replace('.mzXML', '').replace('.mzML', '')
        scan_number = int(match.group(2))
        return raw_file, scan_number
    
    logger.warning(f"Could not parse TITLE: {title}")
    return title, None


def process_spectrum(spectrum: Dict) -> Dict:
    """
    Process a single spectrum entry from MGF.
    
    Args:
        spectrum: Dictionary from pyteomics.mgf
        
    Returns:
        Processed spectrum dictionary
    """
    params = spectrum.get('params', {})
    
    # Extract basic fields
    title = params.get('title', '')
    raw_file, scan_number = parse_title(title)
    
    # Get and standardize sequence
    seq = params.get('seq', '')
    standardized_seq = standardize_sequence(seq)
    
    # Get precursor info
    pepmass = params.get('pepmass', (0.0,))
    if isinstance(pepmass, (list, tuple)):
        precursor_mz = float(pepmass[0]) if pepmass else 0.0
        # Handle case where pepmass[1] might be None
        if len(pepmass) > 1 and pepmass[1] is not None:
            precursor_intensity = float(pepmass[1])
        else:
            precursor_intensity = None
    else:
        precursor_mz = float(pepmass) if pepmass is not None else 0.0
        precursor_intensity = None
    
    # Get charge
    charge = params.get('charge', [0])
    if isinstance(charge, (list, tuple)):
        charge = int(charge[0]) if charge else 0
    else:
        charge = int(str(charge).replace('+', '').replace('-', ''))
    
    # Get retention time
    rt = params.get('rtinseconds', 0.0)
    if isinstance(rt, str):
        rt = float(rt)
    
    # Get m/z and intensity arrays
    mz_array = spectrum.get('m/z array', [])
    intensity_array = spectrum.get('intensity array', [])
    
    # Build result dictionary
    result = {
        'sequence': standardized_seq,
        'original_sequence': seq,
        'precursor_mz': precursor_mz,
        'precursor_intensity': precursor_intensity,
        'charge': charge,
        'retention_time': rt,
        'mz_array': list(mz_array),
        'intensity_array': list(intensity_array),
        'raw_file': raw_file,
        'scan_number': scan_number,
        'title': title,
        # Hardcoded metadata
        'instrument': HARDCODED_METADATA['instrument'],
        'collision_energy': HARDCODED_METADATA['collision_energy'],
        'fragmentation': HARDCODED_METADATA['fragmentation'],
        'score': HARDCODED_METADATA['score'],
    }
    
    return result


def process_mgf_file(input_path: str, output_dir: Path, split_name: str) -> int:
    """
    Process a single MGF file and save to Parquet chunks.
    
    Args:
        input_path: Path to input MGF file
        output_dir: Output directory
        split_name: 'train', 'val', or 'test'
        
    Returns:
        Total number of spectra processed
    """
    logger.info(f"Processing {input_path}")
    
    # Create output subdirectory
    split_output_dir = output_dir / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)
    
    spectra_buffer = []
    chunk_idx = 0
    total_count = 0
    
    # Read MGF file using pyteomics
    with mgf.MGF(input_path) as reader:
        for spectrum in reader:
            processed = process_spectrum(spectrum)
            spectra_buffer.append(processed)
            total_count += 1
            
            # Write chunk when buffer is full
            if len(spectra_buffer) >= CHUNK_SIZE:
                df = pd.DataFrame(spectra_buffer)
                output_file = split_output_dir / f'MassIVE-KB_{split_name}_chunk_{chunk_idx:04d}.parquet'
                df.to_parquet(output_file, index=False, engine='pyarrow')
                logger.info(f"Saved chunk {chunk_idx} with {len(spectra_buffer)} spectra to {output_file}")
                
                spectra_buffer = []
                chunk_idx += 1
            
            # Progress logging
            if total_count % 500000 == 0:
                logger.info(f"Processed {total_count:,} spectra...")
    
    # Write remaining spectra
    if spectra_buffer:
        df = pd.DataFrame(spectra_buffer)
        output_file = split_output_dir / f'MassIVE-KB_{split_name}_chunk_{chunk_idx:04d}.parquet'
        df.to_parquet(output_file, index=False, engine='pyarrow')
        logger.info(f"Saved final chunk {chunk_idx} with {len(spectra_buffer)} spectra to {output_file}")
    
    logger.info(f"Completed {split_name}: {total_count:,} total spectra in {chunk_idx + 1} chunks")
    return total_count


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("MassIVE-KB Data Cleaning and Standardization")
    logger.info("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Process each file
    total_spectra = 0
    split_counts = {}
    
    for input_file in INPUT_FILES:
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            continue
        
        # Determine split name from filename
        if 'train' in input_file:
            split_name = 'train'
        elif 'val' in input_file:
            split_name = 'val'
        elif 'test' in input_file:
            split_name = 'test'
        else:
            split_name = 'unknown'
        
        count = process_mgf_file(input_file, OUTPUT_DIR, split_name)
        split_counts[split_name] = count
        total_spectra += count
    
    # Summary
    logger.info("=" * 60)
    logger.info("Processing Complete!")
    logger.info("=" * 60)
    logger.info(f"Total spectra processed: {total_spectra:,}")
    for split, count in split_counts.items():
        logger.info(f"  {split}: {count:,}")
    logger.info(f"Output saved to: {OUTPUT_DIR}")
    
    # Save summary statistics
    summary = {
        'total_spectra': total_spectra,
        'splits': split_counts,
        'ptm_mapping': {p: u for p, u in PTM_MAPPING},
        'hardcoded_metadata': HARDCODED_METADATA,
    }
    
    summary_df = pd.DataFrame([{
        'split': k,
        'count': v
    } for k, v in split_counts.items()])
    summary_df.to_csv(OUTPUT_DIR / 'MassIVE-KB_processing_summary.csv', index=False)
    logger.info(f"Summary saved to {OUTPUT_DIR / 'MassIVE-KB_processing_summary.csv'}")


if __name__ == '__main__':
    main()
