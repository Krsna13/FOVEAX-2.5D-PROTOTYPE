# Packaging the PyQt5 dashboard as a Windows app

GitHub Pages cannot run a PyQt5 program (it only serves web pages), so the
dashboard is shipped two ways: as source (clone and run) and as a
downloadable Windows app on the repository's **Releases** page.

## What the packaged app is

`FOVEAX-windows.zip` unpacks to a folder containing `FOVEAX.exe`, the Python
runtime and libraries (PyQt5, Open3D, NumPy, SciPy, Matplotlib, OpenCV), and a
small sample of **real RELLIS-3D frames** (`data/rellis3d/...`, 100 consecutive
frames of sequence `00001`). Double-click `FOVEAX.exe` to open the same
PyQt5 + embedded Open3D dashboard the source install runs.

Differences from the source install:

| | Packaged app | Source install |
|---|---|---|
| Labels | Ground truth from the dataset | Ground truth, mock, or SalsaNext (`--predictor`) |
| PyTorch / GPU model | Not included (`torch` is excluded to keep the download small) | Required for SalsaNext |
| Data | 100 bundled frames of sequence 00001 | Full RELLIS-3D / SemanticKITTI |
| Other sequence buttons | Fall back to a synthetic sample | Load the real sequences |

## Building it

Requirements: Windows, the project's virtual environment (see the README
setup), and the RELLIS-3D data (default `C:\dev\data\rellis3d`).

```bash
python packaging/build_app.py                  # 100 frames of sequence 00001
python packaging/build_app.py --frames 60      # smaller download
```

This installs PyInstaller on first use, builds `dist/FOVEAX/`, copies the
sample frames next to the exe, and writes `dist/FOVEAX-windows.zip`.
`requirements-app.txt` lists the runtime libraries without PyTorch, for a
clean build environment.

Build from a clean checkout of the commit you are releasing, not from a
working tree with uncommitted changes.

## Publishing a release

1. Push the code, then create a tag: `git tag v1.0 && git push origin v1.0`.
2. On GitHub: **Releases -> Draft a new release**, choose the tag, and drag
   `dist/FOVEAX-windows.zip` into the assets box.

GitHub allows release assets up to 2 GB; the zip is well below that.

## How the frozen build differs in code

`src/13_realtime_dashboard.py` has three packaging-related details:

- `multiprocessing.freeze_support()` in `__main__`: the Open3D view runs in a
  child process, and a frozen exe starts it by re-launching itself.
- When frozen, the working directory is set to the exe's folder so the
  sample data and `outputs/` are found next to `FOVEAX.exe`.
- The frame count is clamped to the number of frames actually on disk, so a
  short sample does not wrap around mid-run.

`packaging/foveax.spec` lists the modules that are loaded by name with
`importlib` (the numbered scripts such as `src.10b_eval_distance_metrics`),
which PyInstaller cannot discover on its own.
