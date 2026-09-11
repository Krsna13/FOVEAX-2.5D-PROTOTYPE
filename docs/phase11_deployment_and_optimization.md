# FOVEAX Phase 11: Deployment and Optimization

## Overview
Phase 11 introduces optimization and deployment configurations targeting hardware-constrained environments, specifically the **NVIDIA RTX 5050 Laptop GPU (Blackwell GB207, 8 GB VRAM)**.

Because 8 GB of VRAM is highly constrained when running multiple deep learning models concurrently alongside an OS and a dashboard, this phase introduces strict **VRAM Budgeting**, **ONNX Exports**, and **TensorRT compilation** to maximize performance while preventing Out-Of-Memory (OOM) crashes.

## Why ONNX and TensorRT?
- **PyTorch overhead**: Native PyTorch relies on eager execution and uses significant VRAM for computational graphs and CUDA contexts.
- **ONNX**: Provides a standardized, framework-agnostic serialization format, allowing models to be decoupled from PyTorch.
- **TensorRT**: NVIDIA's high-performance deep learning inference optimizer. It performs layer fusion, precision calibration, and dynamic memory pooling to drastically reduce VRAM footprint and inference latency.

## Target Hardware Profiles

### **Primary Target: RTX 5050 Laptop**
- **Architecture**: Blackwell GB207
- **CUDA Cores**: 2,560
- **Tensor Cores**: 5th-gen (Excellent FP16 and INT8 acceleration)
- **VRAM**: **8 GB GDDR7 (Hard Constraint)**
- **Budgeting**: We allocate a strict 6 GB maximum concurrent limit for both engines, leaving 2 GB for OS, dashboard, and CUDA context. If engines exceed this, the pipeline refuses to load to prevent OOM.

### Future Targets
- **Jetson Orin Nano**: 8 GB unified memory (highly constrained).
- **Jetson Orin NX**: 16 GB unified memory.
- **Jetson AGX Orin**: 32 GB unified memory.

## Expected Speedups (RTX 5050)
- **FP16 TensorRT** over native PyTorch: 2.5x - 4x throughput increase.
- **VRAM Footprint**: Engine execution context is strictly capped (default `workspace=2GB`), often saving 1-3 GB compared to native PyTorch allocations.

## How the Concurrent-Engine VRAM Fit Check Works
The script `vram_budget.py` actively polls actual available VRAM using `torch.cuda.mem_get_info()` or `pynvml` at startup.
Before loading the models, it estimates the required memory:
`Required VRAM = (Engine 1 Size * 2.0) + (Engine 2 Size * 2.0) + (20% of Total VRAM Margin)`

If `Required VRAM > Measured Free VRAM`, the pipeline throws a `MemoryError`.
**What happens if they don't both fit?**
You must run in single-model mode (`--semantic-model` OR `--detection-model`, not both), or configure sequential unloading/reloading.

## How to Export Models to ONNX
If you have a native PyTorch model instantiated, you can use:
```bash
python src/13_optimize_and_deploy.py --export-onnx --onnx-output outputs/phase11/onnx
```
*Note: Since SalsaNext and PointPillars are external repositories, if they are not natively loaded, the exporter will safely abort and provide instructions on using their official export scripts.*

## How to Build TensorRT Engines
To compile an ONNX file into a TensorRT engine (FP16 recommended for RTX 5050):
```bash
python src/13_optimize_and_deploy.py --build-engine --semantic-model outputs/phase11/onnx/model.onnx --precision fp16
```
If the Python API is missing, the script will output the exact `trtexec` manual command to use, e.g.:
```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16 --workspace=2048
```

## FP16 vs INT8 Tradeoffs
- **FP16**: The recommended default. Blackwell Tensor cores process FP16 rapidly with virtually zero accuracy loss.
- **INT8**: Doubles throughput again but requires calibration. Without a representative calibration dataset, INT8 quantization can silently degrade accuracy by 5-10 mAP points.

## How to Run Benchmarks
```bash
python src/13_optimize_and_deploy.py --benchmark --target rtx_5050_laptop
```
Results are saved to `outputs/phase11/benchmark/`.

## Known Limitations
- The current ONNX exporter stub raises a `ValueError` for the semantic/detection modules because FOVEAX intentionally avoids duplicating the official proprietary architectures (Phase 6/8 design).
- If VRAM is too fragmented by other applications, the `check_concurrent_fit` might still pass, but TRT execution might OOM later during dynamic memory spikes.
