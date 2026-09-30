"""Build the standalone Windows app and a downloadable zip.

    python packaging/build_app.py [--frames 100] [--rellis-root C:/dev/data/rellis3d] [--seq 00001]

Produces dist/FOVEAX/FOVEAX.exe next to a small real RELLIS-3D sample
(data/rellis3d/...) and dist/FOVEAX-windows.zip. Run it from the repository
root with the project's virtual environment. PyInstaller is installed on
demand.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100, help="Number of consecutive frames to bundle.")
    ap.add_argument("--rellis-root", type=Path, default=Path("C:/dev/data/rellis3d"))
    ap.add_argument("--seq", default="00001")
    args = ap.parse_args()

    seq_dir = args.rellis_root / "Rellis-3D" / args.seq
    bins = sorted((seq_dir / "os1_cloud_node_kitti_bin").glob("*.bin"))[: args.frames]
    if not bins:
        sys.exit(f"No RELLIS-3D frames found under {seq_dir}")

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    subprocess.check_call(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
         "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build"),
         str(ROOT / "packaging" / "foveax.spec")],
        cwd=ROOT,
    )

    app_dir = ROOT / "dist" / "FOVEAX"
    dst = app_dir / "data" / "rellis3d" / "Rellis-3D" / args.seq
    for sub, pattern in (("os1_cloud_node_kitti_bin", ".bin"), ("os1_cloud_node_semantickitti_label_id", ".label")):
        (dst / sub).mkdir(parents=True, exist_ok=True)
        for b in bins:
            shutil.copy2(seq_dir / sub / (b.stem + pattern), dst / sub / (b.stem + pattern))
    shutil.copy2(ROOT / "packaging" / "APP_README.txt", app_dir / "README.txt")
    shutil.copy2(ROOT / "LICENSE", app_dir / "LICENSE")

    zip_base = ROOT / "dist" / "FOVEAX-windows"
    shutil.make_archive(str(zip_base), "zip", ROOT / "dist", "FOVEAX")
    print(f"\nBuilt {app_dir / 'FOVEAX.exe'}\nZip:   {zip_base}.zip")


if __name__ == "__main__":
    main()
