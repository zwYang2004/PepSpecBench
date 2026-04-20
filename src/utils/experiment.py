import os
import time
import yaml
import torch
import random
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from src.utils.hardware import get_optimal_device

def setup_experiment(
    config_path: str,
    run_tag: Optional[str] = None,
    *,
    dataset_name_override: Optional[str] = None,
    run_dir_override: Optional[str] = None,
) -> Tuple[Dict[Any, Any], torch.device, str]:
    """
    Unified experiment setup:
    1. Load config
    2. Set seeds
    3. Setup device
    4. Create output directories
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Global Seed
    seed = config.get("experiment", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Determinism
    if config.get("experiment", {}).get("deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Device
    requested_device = config.get("experiment", {}).get("device", "auto")
    device = get_optimal_device(requested_device)
    
    # Run directory
    output_root = config.get("experiment", {}).get("output_dir", "output")
    if run_dir_override:
        run_dir = str(run_dir_override)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dataset_name = dataset_name_override or config.get("dataset", "unknown")
        tag = f"_{run_tag}" if run_tag else ""
        run_name = f"benchmark_{dataset_name}_{timestamp}{tag}"
        run_dir = os.path.join(output_root, run_name)

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    
    # Logger setup
    log_path = os.path.join(run_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    return config, device, run_dir

def save_best_model(model: torch.nn.Module, run_dir: str, model_name: str, metrics: Dict[str, float], is_best: bool):
    """Save the best model and a training summary."""
    model_dir = os.path.join(run_dir, "models", model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    if is_best:
        path = os.path.join(model_dir, "best.ckpt")
        # Save standard PyTorch state_dict, stripping DataParallel wrapper if present
        sd = model.state_dict()
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        torch.save(sd, path)
        
    summary_path = os.path.join(model_dir, "training_summary.json")
    import json
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)
