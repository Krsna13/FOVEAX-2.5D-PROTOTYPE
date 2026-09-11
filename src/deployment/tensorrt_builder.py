"""TensorRT Engine Builder for FOVEAX Pipeline."""

import os
import json
import time
from pathlib import Path
from typing import Optional

from src.deployment.vram_budget import get_available_vram_mb

def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp16",
    max_workspace_size_gb: float = 2.0,
    batch_size: int = 1,
    extra_args: Optional[list] = None
):
    """
    Builds a TensorRT engine from an ONNX model.
    Defaults to 2GB workspace. Do NOT use 8GB workspace on an 8GB RTX 5050.
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
        
    out_p = Path(engine_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path("outputs/phase11/tensorrt")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    free_vram_mb = get_available_vram_mb()
    workspace_mb = max_workspace_size_gb * 1024
    
    if free_vram_mb > 0 and workspace_mb > (free_vram_mb * 0.4):
        print(f"[WARNING] Requested TensorRT workspace ({workspace_mb:.0f} MB) exceeds 40% of currently free VRAM ({free_vram_mb:.0f} MB).")
        print("[WARNING] This may starve the OS or cause OOM during build on an 8GB card.")
        
    try:
        import tensorrt as trt
        TRT_AVAILABLE = True
    except ImportError:
        TRT_AVAILABLE = False
        
    build_metadata = {
        "onnx_path": onnx_path,
        "engine_path": engine_path,
        "precision": precision,
        "workspace_size_gb_requested": max_workspace_size_gb,
        "free_vram_mb_at_build": free_vram_mb,
        "batch_size": batch_size,
        "timestamp": time.time(),
        "tensorrt_version": trt.__version__ if TRT_AVAILABLE else "unknown"
    }
    
    with open(log_dir / "engine_metadata.json", "w") as f:
        json.dump(build_metadata, f, indent=4)
        
    if not TRT_AVAILABLE:
        print("[ERROR] TensorRT Python API is not installed (`import tensorrt` failed).")
        print("To build the engine manually, use `trtexec` from the command line.")
        print(f"\nExample command for RTX 5050 (8GB):")
        cmd = f"trtexec --onnx={onnx_path} --saveEngine={engine_path} --workspace={int(workspace_mb)}"
        if precision == "fp16":
            cmd += " --fp16"
        elif precision == "int8":
            cmd += " --int8 --calib=<path_to_calibration_data>"
        print(f"  {cmd}\n")
        
        with open(log_dir / "engine_build_log.txt", "w") as f:
            f.write(f"Build failed: TensorRT Python API missing.\nManual command:\n{cmd}\n")
            
        raise ImportError("tensorrt is required for automated build. Please use trtexec manually.")

    print(f"[*] Building TensorRT engine (Precision: {precision.upper()}) using Python API...")
    # Standard TRT build process
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mb * 1024 * 1024))
    
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("[WARNING] Platform does not have fast FP16. Using FP16 anyway, but expect suboptimal performance.")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        # Note: INT8 requires a calibrator which is beyond this basic snippet, 
        # but would be hooked up here via config.int8_calibrator = MyCalibrator()
        
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(f"[TRT Parser Error] {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX file.")
            
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Failed to build TensorRT engine.")
        
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
        
    print(f"[+] Engine successfully saved to {engine_path}")
    with open(log_dir / "engine_build_log.txt", "w") as f:
        f.write("Build successful via TensorRT Python API.\n")
