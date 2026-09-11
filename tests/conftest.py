# torch must be imported before PyQt5 in this process: on Windows, PyQt5's
# bundled Qt DLLs conflict with torch's bundled CUDA DLLs (c10.dll) when
# PyQt5 loads first, causing an import-time crash (WinError 1114 / access
# violation). Importing torch here, before pytest collects any test module,
# guarantees the safe order for the whole session regardless of which test
# file (dashboard/PyQt5 vs. perception/torch) pytest happens to collect first.
try:
    import torch  # noqa: F401
except ImportError:
    pass
