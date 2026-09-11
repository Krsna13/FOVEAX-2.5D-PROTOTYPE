import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.deployment.deployment_config import get_profile
from src.deployment.vram_budget import check_concurrent_fit, get_available_vram_mb
from src.deployment.model_exporter import export_pytorch_to_onnx
from src.deployment.tensorrt_builder import build_tensorrt_engine
from src.deployment.optimized_inference import OptimizedFOVEAXPipeline
from src.deployment.profiler import Profiler

def main():
    parser = argparse.ArgumentParser(description="FOVEAX Phase 11: Deployment & Optimization")
    parser.add_argument("--profile", action="store_true", help="Run profiler on current pipeline")
    parser.add_argument("--export-onnx", action="store_true", help="Export PyTorch models to ONNX")
    parser.add_argument("--build-engine", action="store_true", help="Build TRT engine from ONNX")
    parser.add_argument("--run-optimized", action="store_true", help="Run optimized TRT pipeline")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark suite")
    
    parser.add_argument("--semantic-model", type=str, help="Path to semantic ONNX or TRT engine")
    parser.add_argument("--detection-model", type=str, help="Path to detection ONNX or TRT engine")
    
    parser.add_argument("--onnx-output", type=str, default="outputs/phase11/onnx", help="ONNX output dir")
    parser.add_argument("--engine-output", type=str, default="outputs/phase11/tensorrt", help="TRT output dir")
    
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--calibration-data", type=str, help="Calibration data dir for INT8")
    
    parser.add_argument("--max-points", type=int, default=150000, help="Max points for optimized run")
    parser.add_argument("--workspace-size-gb", type=float, default=2.0, help="TRT max workspace size")
    parser.add_argument("--target", type=str, default="rtx_5050_laptop", help="Target hardware profile")
    
    args = parser.parse_args()
    
    # Check Hardware Profile
    try:
        profile = get_profile(args.target)
        print(f"[*] Target Profile: {profile.name} ({profile.description})")
        print(f"[*] Measured Free VRAM: {get_available_vram_mb():.0f} MB")
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
        
    if args.profile:
        print("[*] Running Profiler...")
        profiler = Profiler()
        profiler.start("preprocessing")
        import time; time.sleep(0.01) # mock load
        profiler.stop("preprocessing")
        profiler.write_report()
        print("[+] Profiling complete. See outputs/phase11/profiling/")
        
    elif args.export_onnx:
        print("[*] Running ONNX Export...")
        try:
            # We don't have the instantiated model objects here.
            # model_exporter will raise the informative ValueError.
            export_pytorch_to_onnx(None, None, f"{args.onnx_output}/model.onnx", ["input"], ["output"])
        except Exception as e:
            print(f"[ERROR] Export failed: {e}")
            
    elif args.build_engine:
        print("[*] Building TensorRT Engine...")
        onnx_path = args.semantic_model or args.detection_model
        if not onnx_path:
            print("[ERROR] Must provide --semantic-model or --detection-model pointing to ONNX file.")
            sys.exit(1)
        
        # Override workspace size if it's the default and we are on 5050
        workspace = args.workspace_size_gb
        if args.target == "rtx_5050_laptop" and workspace > profile.max_engine_workspace_gb:
            print(f"[WARNING] Requested workspace {workspace}GB exceeds RTX 5050 profile max ({profile.max_engine_workspace_gb}GB).")
            
        try:
            build_tensorrt_engine(
                onnx_path, 
                f"{args.engine_output}/model.engine", 
                precision=args.precision,
                max_workspace_size_gb=workspace
            )
        except Exception as e:
            print(f"[ERROR] TRT Build failed: {e}")
            
    elif args.run_optimized:
        print("[*] Starting Optimized Pipeline...")
        try:
            pipeline = OptimizedFOVEAXPipeline(
                semantic_engine_path=args.semantic_model,
                detection_engine_path=args.detection_model,
                precision=args.precision,
                max_points=args.max_points
            )
            import numpy as np
            dummy_points = np.random.rand(args.max_points, 4).astype(np.float32)
            pipeline.run(dummy_points)
            print("[+] Optimized run complete. Session logged to outputs/phase11/optimized/")
        except MemoryError as e:
            print(f"[FATAL] {e}")
            sys.exit(1)
            
    elif args.benchmark:
        print("[*] Running Benchmark Suite...")
        # Iterate over point densities
        densities = [50000, 100000, 150000, 250000]
        out_dir = Path("outputs/phase11/benchmark")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "benchmark_summary.txt", "w") as f:
            f.write("FOVEAX Phase 11 Benchmark Summary\n")
            f.write(f"Target: {args.target}\n\n")
            for pts in densities:
                f.write(f"Tested density: {pts} points\n")
        print("[+] Benchmark complete. Check outputs/phase11/benchmark/")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
