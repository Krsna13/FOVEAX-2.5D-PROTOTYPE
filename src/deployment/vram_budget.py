"""VRAM Budget logic to prevent OOM on memory-constrained GPUs (e.g., RTX 5050 8GB)."""

import os

def get_available_vram_mb() -> float:
    """Detects actual free VRAM using PyTorch or PyNVML."""
    try:
        import torch
        if torch.cuda.is_available():
            # torch.cuda.mem_get_info returns (free, total) in bytes
            free, total = torch.cuda.mem_get_info()
            return free / (1024 * 1024)
    except ImportError:
        pass

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.free / (1024 * 1024)
    except Exception:
        pass
        
    return -1.0

def get_total_vram_mb() -> float:
    """Detects actual total VRAM using PyTorch or PyNVML."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return total / (1024 * 1024)
    except ImportError:
        pass

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.total / (1024 * 1024)
    except Exception:
        pass
        
    return -1.0

def estimate_engine_vram_mb(engine_path: str) -> float:
    """
    Best-effort estimate of VRAM required to load an engine.
    Typically, TensorRT engines require at least their file size + context memory.
    """
    if not os.path.exists(engine_path):
        return 0.0
    file_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    # Heuristic: TensorRT engines often use 1.5x - 2.5x their serialized size in activation/context memory
    return file_size_mb * 2.0

def check_concurrent_fit(engine_paths: list[str], safety_margin_pct: float = 20.0) -> bool:
    """
    Returns True if all requested engines are likely to fit concurrently within
    the *currently available* free VRAM.
    """
    free_mb = get_available_vram_mb()
    
    if free_mb < 0:
        # Cannot measure VRAM (CPU mode or missing libs). Assume it fits to avoid blocking, 
        # but warn the user.
        print("[WARNING] Could not measure VRAM. Skipping concurrent fit check.")
        return True
        
    total_estimated_mb = sum(estimate_engine_vram_mb(p) for p in engine_paths)
    
    safety_margin = (safety_margin_pct / 100.0) * get_total_vram_mb()
    if get_total_vram_mb() < 0:
        safety_margin = (safety_margin_pct / 100.0) * free_mb
        
    required_mb = total_estimated_mb + safety_margin
    
    if required_mb > free_mb:
        print(f"[WARNING] VRAM BUDGET EXCEEDED: Engines require ~{total_estimated_mb:.0f} MB + {safety_margin:.0f} MB safety margin.")
        print(f"          Currently available free VRAM: {free_mb:.0f} MB.")
        return False
        
    return True
