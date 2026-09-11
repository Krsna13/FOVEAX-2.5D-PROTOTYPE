import os
import sys
from pathlib import Path
import importlib.util

def check_env():
    report_lines = []
    report_lines.append("=== FOVEAX Phase 8B: Model Environment Inspection Report ===")
    
    # Python Version
    report_lines.append(f"Python Version: {sys.version.split(' ')[0]}")
    
    # PyTorch and CUDA
    torch_installed = False
    try:
        import torch
        torch_installed = True
        report_lines.append(f"PyTorch Version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        report_lines.append(f"CUDA Available: {cuda_avail}")
        if cuda_avail:
            report_lines.append(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
            report_lines.append(f"CUDA Device Count: {torch.cuda.device_count()}")
        else:
            report_lines.append("CUDA Device Name: None")
    except ImportError:
        report_lines.append("PyTorch Version: Not installed")
        report_lines.append("CUDA Available: False")
        report_lines.append("CUDA Device Name: None")

    # Required Packages
    required_pkgs = ["numpy", "scipy"]
    for pkg in required_pkgs:
        is_installed = importlib.util.find_spec(pkg) is not None
        report_lines.append(f"Package '{pkg}': {'Installed' if is_installed else 'Missing'}")

    # OpenPCDet Repo
    root_dir = Path(__file__).resolve().parent.parent
    openpcdet_path = root_dir / "external" / "OpenPCDet"
    openpcdet_exists = openpcdet_path.is_dir()
    report_lines.append(f"OpenPCDet Repository: {'Found' if openpcdet_exists else 'Missing'}")
    
    report_lines.append("")
    report_lines.append("=== Recommended Manual Actions ===")
    
    if not torch_installed:
        report_lines.append("1. Install PyTorch with CUDA support (check pytorch.org for your CUDA version).")
        report_lines.append("   e.g., pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    if not openpcdet_exists:
        report_lines.append("2. Clone OpenPCDet repository:")
        report_lines.append("   git clone https://github.com/open-mmlab/OpenPCDet.git external/OpenPCDet")
        
    report_lines.append("3. Install OpenPCDet dependencies:")
    report_lines.append("   cd external/OpenPCDet && pip install -r requirements.txt && pip install -e .")
    report_lines.append("4. Obtain an official PointPillars pretrained checkpoint and place it in models/openpcdet/")
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    out_dir = root_dir / "outputs" / "phase8"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "model_environment_report.txt"
    out_file.write_text(report_text)
    print(f"\nReport written to: {out_file}")

if __name__ == "__main__":
    check_env()
