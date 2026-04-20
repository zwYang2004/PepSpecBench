import os
import torch
import psutil
import logging

logger = logging.getLogger(__name__)

def get_optimal_device(requested_device: str = "auto") -> torch.device:
    """
    Find the most suitable device for training/inference.
    - If 'auto', looks for an available GPU.
    - If a specific device is requested (e.g., 'cuda:1'), validates its existence.
    """
    if requested_device == "cpu":
        return torch.device("cpu")
    
    if torch.cuda.is_available():
        if requested_device == "auto" or requested_device.startswith("cuda"):
            # If auto, we could pick the one with most free memory if needed
            # For now, just use cuda:0 or the requested index
            target = requested_device if requested_device != "auto" else "cuda:0"
            try:
                device = torch.device(target)
                # Simple check
                torch.cuda.get_device_name(device)
                return device
            except Exception as e:
                logger.warning(f"Requested device {target} failed: {e}. Falling back to cuda:0 or cpu.")
                return torch.device("cuda:0")
    
    return torch.device("cpu")

def get_optimal_num_workers(max_workers: int = 8) -> int:
    """
    Calculate optimal num_workers for DataLoader based on CPU count.
    Avoids overloading the system.
    """
    try:
        cpu_count = os.cpu_count()
        if cpu_count is None:
            return 0
        
        # Rule of thumb: min(cpu_count, max_workers)
        # Often we use cpu_count // 2 to leave room for other processes
        suggested = max(0, min(cpu_count // 2, max_workers))
        return suggested
    except Exception:
        return 0

def get_ram_info():
    """Returns available RAM in GB."""
    vm = psutil.virtual_memory()
    return vm.available / (1024**3)

def should_load_to_ram(file_size_gb: float, safety_margin_gb: float = 8.0) -> bool:
    """
    Check if a file (or dataset) of given size should be fully loaded to RAM.
    safety_margin_gb: minimum RAM to leave free for OS and other tasks.
    """
    available = get_ram_info()
    return available > (file_size_gb + safety_margin_gb)
