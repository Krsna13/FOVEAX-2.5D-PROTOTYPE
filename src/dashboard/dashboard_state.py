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

    # Real-data dashboard features (Task: PyQt5 restyle).
    # Ego-forward drivable corridor width, computed from contiguous
    # traversability>=0.70 (canonical Safe) cells at the ego's row. None
    # when no safe cell exists under the ego column (nothing to measure).
    road_width_m: float | None = None

    # Real elevation profile along the ego-forward centerline (x nearest 0):
    # list of (y_m, z_max) sampled from the real elevation grid.
    elevation_profile: list[tuple[float, float]] = field(default_factory=list)

    # y-positions (m) along that same centerline where traversability<0.40
    # (canonical Blocked) -- real hazard markers, not scripted.
    elevation_hazards_m: list[float] = field(default_factory=list)

    # Connected-component hazard clusters from the real traversability grid
    # (scipy.ndimage.label), same method as
    # src/dashboard/export_web_dashboard_data.py::find_hazard_clusters.
    hazard_clusters: list[dict] = field(default_factory=list)

    # Real Phase 5 (src/08_spatial_importance_roi.py) observation-uncertainty
    # summary, computed with the identical formula on the live local grid:
    # uncertainty = 1 - clip(log1p(point_count)/log(8), 0, 1); unoccupied=1.0.
    mean_uncertainty: float | None = None
    occupied_fraction: float | None = None

    # Real forward-corridor cross-section (src/perception/centerline_profile.py):
    # one dict per distance bin (distance_m, height_above_baseline_m,
    # depth_below_baseline_m, point_count) -- a gap (no real data) has
    # height/depth = None and point_count = 0, never an interpolated guess.
    centerline_profile: list[dict] = field(default_factory=list)

    # Real hazards/tracked objects whose position falls inside the same
    # forward corridor, each carrying the SAME distance_m already shown
    # elsewhere in the dashboard (hazard table / tracked-objects panel)
    # for that object -- reused verbatim, not re-derived.
    centerline_markers: list[dict] = field(default_factory=list)

    # Real distance-bucketed accuracy, computed by the SAME
    # src/10b_eval_distance_metrics.py::compute_metrics() used by the
    # already-validated offline evaluation -- never reimplemented here.
    # None when this frame has no real ground truth to compare against
    # (has_ground_truth=False); otherwise a dict with keys
    # "near"/"mid"/"far"/"overall", each compute_metrics()'s own return
    # value (a dict with n_points/accuracy/confusion, or None if that
    # bucket had zero valid ground-truth points).
    has_ground_truth: bool = False
    predictor_mode: str = "ground_truth"
    accuracy_metrics: dict | None = None


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
