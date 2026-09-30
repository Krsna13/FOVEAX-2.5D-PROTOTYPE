# PyInstaller spec for the FOVEAX PyQt5 + Open3D dashboard (Windows).
# Build with:  python packaging/build_app.py     (see docs/packaging.md)
#
# torch is deliberately excluded: the dashboard treats it as optional
# (src/13_realtime_dashboard.py wraps `import torch` in try/except), and the
# packaged demo replays recorded RELLIS-3D frames with ground-truth labels, so
# no neural network runs. Bundling torch cu130 would add several GB.
from PyInstaller.utils.hooks import collect_all

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821  (SPECPATH is provided by PyInstaller)

o3d_datas, o3d_binaries, o3d_hidden = collect_all("open3d")

# Modules that are loaded by name with importlib (numbered file names cannot
# be written as normal imports), so PyInstaller's static analysis misses them.
hidden = [
    "src.07_adaptive_2point5d_map",
    "src.10_ai_semantic_2point5d_map",
    "src.10b_eval_distance_metrics",
    "src.11_object_detection_tracking",
    "src.dashboard.export_web_dashboard_data",
    "src.perception.semantic_predictor",
    "src.perception.semantic_labels",
    "src.perception.rellis3d_loader",
    "win32gui", "win32con", "win32process",
] + o3d_hidden

a = Analysis(  # noqa: F821
    [os.path.join(ROOT, "src", "13_realtime_dashboard.py")],
    pathex=[ROOT],
    binaries=o3d_binaries,
    datas=o3d_datas,
    hiddenimports=hidden,
    excludes=["torch", "torchvision", "tensorrt", "onnx", "pytest", "IPython", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="FOVEAX",
    console=False,          # GUI app: no console window
    disable_windowed_traceback=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="FOVEAX")  # noqa: F821
