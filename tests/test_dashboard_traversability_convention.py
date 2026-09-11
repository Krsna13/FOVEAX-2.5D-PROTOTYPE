"""Regression tests for the dashboard traversability sign convention.

Canonical convention (src/06_terrain_traversability.py, src/perception/grid_overlay.py):
    1.0 = SAFE (best), 0.0 = BLOCKED / LETHAL (worst)
    Safe >= 0.70, Caution 0.40-0.70, Blocked < 0.40

data_streamer.py and qt_main_window.py previously inverted this. These tests
pin down the corrected behavior so the inversion can't silently come back.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pytest

pytest.importorskip("PyQt5", reason="PyQt5 not installed; dashboard not runnable in this environment yet")

from src.dashboard.data_streamer import DataStreamerThread
from src.dashboard.qt_main_window import FoveaXDashboardWindow


def test_data_streamer_flat_terrain_is_safe_rough_terrain_is_blocked():
    streamer = DataStreamerThread.__new__(DataStreamerThread)
    streamer.grid_res = 1.0
    streamer.grid_extent = (0.0, 4.0, 0.0, 4.0)
    streamer.x_bins = 4
    streamer.y_bins = 4

    # Cell (0,0): flat terrain -> should be SAFE (traversability near 1.0)
    flat_pts = np.array([
        [0.1, 0.1, 1.0],
        [0.2, 0.2, 1.0],
    ], dtype=np.float32)
    # Cell (3,3): large height range -> should be BLOCKED (traversability near 0.0)
    rough_pts = np.array([
        [3.1, 3.1, 0.0],
        [3.2, 3.2, 5.0],
    ], dtype=np.float32)
    points = np.vstack([flat_pts, rough_pts])
    points = np.hstack([points, np.zeros((points.shape[0], 1), dtype=np.float32)])

    maps = streamer._generate_maps(points, tracks=[])
    trav = np.flipud(maps["traversability"])  # undo the vertical flip applied for image rendering

    assert trav[0, 0] == pytest.approx(1.0)
    assert trav[3, 3] == pytest.approx(0.0)


def test_qt_main_window_uses_correct_colormap_and_lethal_threshold():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    win = FoveaXDashboardWindow.__new__(FoveaXDashboardWindow)
    win.map_label = type("FakeLabel", (), {
        "size": lambda self: __import__("PyQt5.QtCore", fromlist=["QSize"]).QSize(100, 100),
        "setPixmap": lambda self, pm: None,
        "setText": lambda self, t: None,
    })()

    grid = np.full((10, 10), np.nan, dtype=np.float32)
    grid[0, 0] = 1.0   # safe
    grid[9, 9] = 0.0   # blocked/lethal

    win._render_grid_to_label(grid, "traversability")
    # No assertion on rendered pixels needed beyond it not raising: the real
    # regression coverage is the source-level checks below, which confirm the
    # exact convention constants used inside _render_grid_to_label.
    import inspect
    src = inspect.getsource(FoveaXDashboardWindow._render_grid_to_label)
    assert 'mpl.colormaps["RdYlGn"]' in src
    assert '(grid < 0.40)' in src
    assert 'RdYlGn_r' not in src
    assert '(grid > 0.70)' not in src
