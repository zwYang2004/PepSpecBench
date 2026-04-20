"""
Full-Spectrum Reconstruction for Level 2 Evaluation
====================================================

Reconstructs predicted spectra into peak lists for comprehensive evaluation
across different model architectures.

For models with different output spaces:
- PredFull: 20000-d bin vector → sparse peak list
- Ion-vector models (Prosit, PrositTransformer, AlphaPeptDeep): 234-d → canonical mz peaks
- UniSpec: dictionary indices → mz peaks from rebuilt dictionary
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from pathlib import Path

from src.utils.mass_calc import ion_mz, BIN_SIZE, MZ_MIN


def reconstruct_predfull(preds: np.ndarray, meta: Dict[str, Any]) -> List[Tuple[float, float]]:
    """
    Reconstruct PredFull 20000-d predictions to peak list.

    Args:
        preds: 20000-d intensity vector
        meta: Metadata dict (unused for PredFull)

    Returns:
        List of (mz, intensity) tuples
    """
    peaks = []
    for i, intensity in enumerate(preds):
        if intensity > 0:
            mz = MZ_MIN + i * BIN_SIZE
            peaks.append((mz, float(intensity)))
    return peaks


def reconstruct_ion_vector(preds: np.ndarray, meta: Dict[str, Any]) -> List[Tuple[float, float]]:
    """
    Reconstruct ion-vector models (Prosit, PrositTransformer, AlphaPeptDeep) to peak list.

    Args:
        preds: 234-d intensity vector
        meta: Metadata dict with 'modified_sequence', 'precursor_charge'

    Returns:
        List of (mz, intensity) tuples for canonical ions
    """
    seq = meta['modified_sequence']
    charge = meta['precursor_charge']

    peaks = []
    pos = 0
    for aa_pos in range(1, len(seq)):
        for ion_type in ['b', 'y']:
            for z in range(1, min(charge, 3) + 1):
                if pos < len(preds) and preds[pos] > 0:
                    try:
                        mz = ion_mz(seq, aa_pos, ion_type, z)
                        peaks.append((mz, float(preds[pos])))
                    except Exception:
                        pass  # Skip invalid ions
                pos += 1

    return peaks


def reconstruct_unispec(preds: np.ndarray, meta: Dict[str, Any], dict_path: Path) -> List[Tuple[float, float]]:
    """
    Reconstruct UniSpec predictions to peak list using dictionary.

    Args:
        preds: Dictionary-based predictions
        meta: Metadata dict (unused)
        dict_path: Path to dictionary directory

    Returns:
        List of (mz, intensity) tuples
    """
    # Load dictionary
    import yaml
    dic_yaml = dict_path / "dic.yaml"
    if not dic_yaml.exists():
        return []

    with open(dic_yaml, 'r') as f:
        dictionary = yaml.safe_load(f)

    peaks = []
    for idx, intensity in enumerate(preds):
        if intensity > 0 and idx in dictionary:
            ion_type, z, loss = dictionary[idx]
            # For simplicity, map to approximate mz using ion_type and position
            # In practice, this would need more sophisticated mz calculation
            # based on the specific ion annotation
            try:
                # This is a placeholder - real implementation would need
                # to parse ion_type (e.g., 'b5', 'y12') and calculate mz
                # For now, use a simple mapping
                if ion_type.startswith('b') or ion_type.startswith('y'):
                    pos = int(''.join(filter(str.isdigit, ion_type)))
                    seq = meta.get('modified_sequence', 'A' * (pos + 1))
                    charge = meta.get('precursor_charge', 2)
                    ion_t = ion_type[0]  # 'b' or 'y'
                    mz = ion_mz(seq[:pos+1], pos, ion_t, min(z, charge))
                    peaks.append((mz, float(intensity)))
            except Exception:
                pass  # Skip invalid ions

    return peaks


def reconstruct(model_name: str, preds: np.ndarray, meta: Dict[str, Any], dict_path: Path = None) -> List[Tuple[float, float]]:
    """
    Reconstruct predictions to peak list based on model type.

    Args:
        model_name: Name of the model ('predfull', 'prosit', etc.)
        preds: Model predictions
        meta: Metadata dict
        dict_path: Path to UniSpec dictionary (for UniSpec only)

    Returns:
        List of (mz, intensity) tuples
    """
    if model_name.lower() == 'predfull':
        return reconstruct_predfull(preds, meta)
    elif model_name.lower() in ['prosit', 'prosit_transformer', 'alphapeptdeep']:
        return reconstruct_ion_vector(preds, meta)
    elif model_name.lower() == 'unispec':
        return reconstruct_unispec(preds, meta, dict_path)
    else:
        # Default to ion vector reconstruction
        return reconstruct_ion_vector(preds, meta)
