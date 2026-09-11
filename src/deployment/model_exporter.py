"""ONNX Export Utility for FOVEAX Pipeline."""

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List

def export_pytorch_to_onnx(
    model: Any,
    dummy_input: Any,
    output_path: str,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]] = None,
    opset_version: int = 14
):
    """
    Exports a PyTorch model to ONNX.
    
    In FOVEAX, models like SalsaNext and PointPillars are loaded via external
    repositories. If the 'model' object is not available (e.g. mock mode or 
    missing dependencies), this function will log a clear error and instruction.
    """
    if model is None:
        raise ValueError(
            "PyTorch model is None. "
            "FOVEAX Phase 11 requires the native PyTorch model object to export to ONNX. "
            "Since SalsaNext and PointPillars are external repositories, you must either:\n"
            "1. Instantiate them properly with their respective checkpoints before calling this.\n"
            "2. Use the official export scripts provided by OpenPCDet or SalsaNext.\n"
            "Example: python -m tools.export_onnx --cfg_file ... (for OpenPCDet)"
        )

    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required to export to ONNX.")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Validating PyTorch model before export...")
    # Validate it runs
    with torch.no_grad():
        try:
            _ = model(dummy_input)
        except Exception as e:
            raise RuntimeError(f"Model forward pass failed with dummy input: {e}")
            
    print(f"[*] Exporting to {output_path} (opset {opset_version})...")
    
    # Deterministic flags if applicable
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
    except (ModuleNotFoundError, ImportError) as e:
        raise RuntimeError(
            f"ONNX export requires additional dependencies ('onnx' and 'onnxscript'): {e}. "
            "Please run 'pip install onnx onnxscript' to enable ONNX model serialization."
        ) from e
    
    print(f"[+] Successfully exported ONNX model to {output_path}")
    
    # Save Metadata
    metadata_path = out_p.with_suffix(".metadata.json")
    metadata = {
        "model_type": str(type(model)),
        "opset_version": opset_version,
        "input_names": input_names,
        "output_names": output_names,
        "timestamp": time.time()
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
