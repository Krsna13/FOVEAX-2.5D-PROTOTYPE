import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pytest
from src.dashboard.dashboard_state import FrameState, HardwareMetrics, assert_coordinate_frame_consistency
from src.tracking.multi_object_tracker import TrackState

def test_hardware_metrics_initialization():
    hm = HardwareMetrics(fps=10.0, cpu_percent=50.0)
    assert hm.fps == 10.0
    assert hm.cpu_percent == 50.0
    assert hm.gpu_percent == 0.0
    assert not hm.gpu_available

def test_frame_state_initialization():
    fs = FrameState(frame_idx=5)
    assert fs.frame_idx == 5
    assert isinstance(fs.points, np.ndarray)
    assert fs.points.shape == (0, 4)
    assert isinstance(fs.grid_maps, dict)
    assert len(fs.grid_maps) == 0

def test_frame_state_with_data():
    pts = np.array([[1, 2, 3, 0.5], [4, 5, 6, 0.8]], dtype=np.float32)
    grids = {"elevation": np.zeros((10, 10))}
    hm = HardwareMetrics(fps=30)
    
    fs = FrameState(
        frame_idx=10,
        points=pts,
        grid_maps=grids,
        metrics=hm
    )
    
    assert fs.frame_idx == 10
    assert fs.points.shape == (2, 4)
    assert "elevation" in fs.grid_maps
    assert fs.metrics.fps == 30

def _make_track(track_id: int, x: float, y: float) -> TrackState:
    return TrackState(
        track_id=track_id,
        class_name="vehicle",
        state=np.array([x, y, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        covariance=np.eye(6, dtype=np.float64),
        size_lwh=np.array([4.0, 2.0, 1.5], dtype=np.float32),
        yaw_rad=0.0,
    )

def test_coordinate_frame_consistency_no_warnings_for_consistent_frame():
    pts = np.array([[1.0, 2.0, 0.5, 0.0], [-3.0, -4.0, 0.5, 0.0]], dtype=np.float32)
    fs = FrameState(
        points=pts,
        tracks=[_make_track(1, 2.0, 1.0)],
        grid_extent_m=(-20.0, 20.0, -20.0, 20.0),
    )
    assert assert_coordinate_frame_consistency(fs) == []

def test_coordinate_frame_consistency_warns_on_track_outside_grid_bounds():
    pts = np.array([[1.0, 2.0, 0.5, 0.0], [-3.0, -4.0, 0.5, 0.0]], dtype=np.float32)
    fs = FrameState(
        points=pts,
        tracks=[_make_track(1, 500.0, 500.0)],
        grid_extent_m=(-20.0, 20.0, -20.0, 20.0),
    )
    warnings = assert_coordinate_frame_consistency(fs)
    assert len(warnings) >= 1
    assert any("track 1" in w for w in warnings)
