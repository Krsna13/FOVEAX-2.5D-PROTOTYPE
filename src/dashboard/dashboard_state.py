from dataclasses import dataclass, field
import numpy as np
from typing import Any

from src.tracking.multi_object_tracker import TrackState

@dataclass
class HardwareMetrics:
    fps: float = 0.0
    target_fps: float = 0.0
    achieved_fps: float = 0.0
    latency_ms: float = 0.0
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    # GPU metrics if pynvml is available
    gpu_available: bool = False
    gpu_percent: float = 0.0
    vram_percent: float = 0.0
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0

@dataclass
class FrameState:
    """Holds the unified state for a single dashboard tick."""
    schema_version: str = "1.0"
    frame_idx: int = 0
    timestamp_s: float = 0.0
    
    # Raw Point Cloud (N x 4: [x, y, z, intensity])
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    
    # Tracking Data
    tracks: list[TrackState] = field(default_factory=list)
    track_velocities: dict[int, tuple[float, float]] = field(default_factory=dict)
    
    # 2.5D Maps and Extents
    # Map layers will be 2D float32 arrays. Keys might be 'elevation', 'slope', 'traversability', 'semantic', 'roi'
    grid_maps: dict[str, np.ndarray] = field(default_factory=dict)
    
    # Spatial metadata for the grid maps
    grid_resolution_m: float = 0.0
    grid_extent_m: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0) # x_min, x_max, y_min, y_max
    
    # Performance & System
    metrics: HardwareMetrics = field(default_factory=HardwareMetrics)

    # Cross-panel coordinate-frame sanity warnings (Task O), populated by
    # assert_coordinate_frame_consistency(). Empty when the frame is consistent.
    coordinate_warnings: list[str] = field(default_factory=list)


def assert_coordinate_frame_consistency(frame_state: "FrameState") -> list[str]:
    """
    Coarse sanity check that raw_points, grid_cells, and tracks in a
    FrameState share a consistent ego-centered coordinate convention.

    Returns a list of warning strings (empty if consistent). Does not raise
    or block rendering -- callers should log and surface warnings instead.
    This is not a frame-transform validation (that's tf2's job in Phase 10);
    it's a coarse offline/dashboard-context sanity check.
    """
    warnings_found: list[str] = []
    points = frame_state.points
    x_min, x_max, y_min, y_max = frame_state.grid_extent_m
    has_grid = (x_min, x_max, y_min, y_max) != (0.0, 0.0, 0.0, 0.0)

    if has_grid and not (x_min <= 0 <= x_max and y_min <= 0 <= y_max):
        warnings_found.append(
            f"COORDINATE FRAME WARNING: grid extent {frame_state.grid_extent_m} "
            "does not contain the ego origin (0, 0)."
        )

    p_min = p_max = None
    if len(points) > 0:
        p_min = np.min(points[:, :2], axis=0)
        p_max = np.max(points[:, :2], axis=0)

        if has_grid and (p_max[0] < x_min or p_min[0] > x_max or p_max[1] < y_min or p_min[1] > y_max):
            warnings_found.append(
                f"COORDINATE FRAME WARNING: point cloud bounds "
                f"({p_min[0]:.1f}, {p_min[1]:.1f}) to ({p_max[0]:.1f}, {p_max[1]:.1f}) "
                f"do not overlap grid extent {frame_state.grid_extent_m}."
            )

    for t in frame_state.tracks:
        tx, ty = float(t.state[0]), float(t.state[1])
        if not (-100.0 <= tx <= 100.0 and -100.0 <= ty <= 100.0):
            warnings_found.append(
                f"COORDINATE FRAME WARNING: track {t.track_id} position "
                f"({tx:.1f}, {ty:.1f}) is far outside expected ego bounds."
            )
        elif p_min is not None:
            margin = 5.0
            if not (p_min[0] - margin <= tx <= p_max[0] + margin and p_min[1] - margin <= ty <= p_max[1] + margin):
                warnings_found.append(
                    f"COORDINATE FRAME WARNING: track {t.track_id} at ({tx:.1f}, {ty:.1f}) "
                    f"falls outside the raw point cloud bounds."
                )

    return warnings_found
