"""FOVEAX Phase 8B — ROI & Traversability Feedback Loop (Task J).

Overlays detected and tracked dynamic-object footprints (center_xyz + size_lwh + yaw)
onto Phase 3 traversability grids and Phase 5 ROI/importance grids.
All outputs are saved as new files in:
    outputs/phase8/<source>/<detector>/dynamic_object_overlay/
Original Phase 3 and Phase 5 outputs are never modified in place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .object_detector import Detection3D
from ..tracking.multi_object_tracker import TrackState


def compute_footprint_mask(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    objects: Sequence[Detection3D | TrackState],
    margin_m: float = 0.0,
) -> np.ndarray:
    """Compute a 2D boolean mask of cells covered by any object footprint.

    Parameters
    ----------
    grid_x : np.ndarray, shape (H, W)
        X coordinate for each grid cell center.
    grid_y : np.ndarray, shape (H, W)
        Y coordinate for each grid cell center.
    objects : Sequence[Detection3D | TrackState]
        Objects with center_xyz, size_lwh, and yaw_rad.
    margin_m : float
        Optional padding margin around box footprint in metres.

    Returns
    -------
    np.ndarray, shape (H, W), dtype bool
        True for cells covered by at least one object footprint.
    """
    mask = np.zeros(grid_x.shape, dtype=bool)
    if not objects:
        return mask

    for obj in objects:
        if isinstance(obj, TrackState):
            cx, cy = float(obj.state[0]), float(obj.state[1])
            l, w = float(obj.size_lwh[0]), float(obj.size_lwh[1])
            yaw = float(obj.yaw_rad)
        else:
            cx, cy = float(obj.center_xyz[0]), float(obj.center_xyz[1])
            l, w = float(obj.size_lwh[0]), float(obj.size_lwh[1])
            yaw = float(obj.yaw_rad)

        half_l = (l / 2.0) + margin_m
        half_w = (w / 2.0) + margin_m

        dx = grid_x - cx
        dy = grid_y - cy

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # Rotate to object local coordinate system
        local_x = dx * cos_yaw + dy * sin_yaw
        local_y = -dx * sin_yaw + dy * cos_yaw

        inside = (np.abs(local_x) <= half_l) & (np.abs(local_y) <= half_w)
        mask |= inside

    return mask


def create_frame_base_grids(
    points_xyzi: np.ndarray,
    resolution_m: float = 0.05,
    margin_m: float = 1.0,
) -> dict[str, np.ndarray]:
    """Create baseline 2.5D, traversability, and ROI grids for a point cloud frame.

    Uses the Phase 3 & Phase 5 definitions (slope, roughness, step, distance,
    point density) to create the reference layers.

    Parameters
    ----------
    points_xyzi : np.ndarray, shape (N, 4)
        LiDAR points [x, y, z, intensity].
    resolution_m : float
        Grid resolution in metres per cell.
    margin_m : float
        Boundary padding in metres.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing coordinate meshes, occupied mask, traversability,
        terrain class, importance score, and ROI class.
    """
    x = points_xyzi[:, 0]
    y = points_xyzi[:, 1]
    z = points_xyzi[:, 2]

    x_min = float(np.min(x)) - margin_m
    x_max = float(np.max(x)) + margin_m
    y_min = float(np.min(y)) - margin_m
    y_max = float(np.max(y)) + margin_m

    width = max(1, int(np.ceil((x_max - x_min) / resolution_m)))
    height = max(1, int(np.ceil((y_max - y_min) / resolution_m)))

    # Compute cell indices
    col = np.clip(np.floor((x - x_min) / resolution_m).astype(np.int32), 0, width - 1)
    row = np.clip(np.floor((y - y_min) / resolution_m).astype(np.int32), 0, height - 1)
    flat_idx = row * width + col
    total_cells = height * width

    point_count = np.bincount(flat_idx, minlength=total_cells).reshape(height, width).astype(np.float32)
    z_sum = np.bincount(flat_idx, weights=z, minlength=total_cells).reshape(height, width).astype(np.float32)

    occupied = point_count > 0
    z_mean = np.full((height, width), np.nan, dtype=np.float32)
    z_mean[occupied] = z_sum[occupied] / point_count[occupied]

    # Coordinate mesh for cell centers
    xs = (np.arange(width) + 0.5) * resolution_m + x_min
    ys = (np.arange(height) + 0.5) * resolution_m + y_min
    grid_x, grid_y = np.meshgrid(xs, ys)

    # Simple elevation gradient for slope
    z_filled = z_mean.copy()
    fill_val = float(np.nanmedian(z_filled)) if np.any(occupied) else 0.0
    z_filled[np.isnan(z_filled)] = fill_val

    gy, gx = np.gradient(z_filled, resolution_m, resolution_m)
    slope_deg = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2))).astype(np.float32)
    slope_deg[~occupied] = np.nan

    # Traversability (0.0 = blocked, 1.0 = safe)
    max_safe_slope = 15.0
    slope_risk = np.clip(slope_deg / max_safe_slope, 0.0, 1.0)
    traversability = (1.0 - slope_risk).astype(np.float32)
    traversability[~occupied] = np.nan

    # Terrain class: 0=Safe, 1=Caution, 2=Blocked, 3=Unknown
    terrain_class = np.full((height, width), 3, dtype=np.uint8)
    terrain_class[occupied & (traversability >= 0.70)] = 0
    terrain_class[occupied & (traversability >= 0.40) & (traversability < 0.70)] = 1
    terrain_class[occupied & (traversability < 0.40)] = 2

    # Distance priority (Phase 5)
    dist = np.sqrt(grid_x**2 + grid_y**2)
    max_d = float(np.max(dist)) if np.max(dist) > 1e-6 else 1.0
    dist_priority = (1.0 - dist / max_d).astype(np.float32)

    # Uncertainty from density
    density_conf = np.clip(np.log1p(point_count) / np.log(8.0), 0.0, 1.0).astype(np.float32)
    uncertainty = 1.0 - density_conf
    uncertainty[~occupied] = 1.0

    # Importance score
    terrain_risk = np.nan_to_num(slope_risk, nan=0.0)
    importance_score = (
        0.25 * dist_priority + 0.50 * terrain_risk + 0.25 * uncertainty
    ).astype(np.float32)
    importance_score[~occupied] = np.nan

    # ROI class: 0=Background, 1=Important, 2=Critical, 3=Unknown
    roi_class = np.full((height, width), 3, dtype=np.uint8)
    roi_class[occupied & (importance_score < 0.40)] = 0
    roi_class[occupied & (importance_score >= 0.40) & (importance_score < 0.70)] = 1
    roi_class[occupied & (importance_score >= 0.70)] = 2

    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "occupied": occupied,
        "z_mean": z_mean,
        "traversability": traversability,
        "terrain_class": terrain_class,
        "importance_score": importance_score,
        "roi_class": roi_class,
        "bounds": np.array([x_min, x_max, y_min, y_max, resolution_m], dtype=np.float32),
    }


def generate_dynamic_object_overlay(
    points_xyzi: np.ndarray,
    dynamic_objects: Sequence[Detection3D | TrackState],
    output_dir: Path,
    frame_idx: int,
    resolution_m: float = 0.05,
) -> Path:
    """Generate and save the dynamic-object overlay for Phase 3 and Phase 5 grids.

    Parameters
    ----------
    points_xyzi : np.ndarray, shape (N, 4)
        LiDAR points of the current frame.
    dynamic_objects : Sequence[Detection3D | TrackState]
        Dynamic objects detected or tracked in this frame.
    output_dir : Path
        Target directory (`outputs/phase8/<source>/<detector>/dynamic_object_overlay`).
    frame_idx : int
        Current frame index.
    resolution_m : float
        Grid resolution.

    Returns
    -------
    Path
        Path to the saved overlay .npz file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate base grids
    base = create_frame_base_grids(points_xyzi, resolution_m=resolution_m)
    grid_x = base["grid_x"]
    grid_y = base["grid_y"]

    # 2. Compute footprint mask
    footprint_mask = compute_footprint_mask(grid_x, grid_y, dynamic_objects)

    # 3. Apply overlay onto traversability: dynamic object cells -> blocked (score=0.0, class=2)
    traversability_overlay = base["traversability"].copy()
    terrain_class_overlay = base["terrain_class"].copy()

    traversability_overlay[footprint_mask] = 0.0
    terrain_class_overlay[footprint_mask] = 2  # Blocked

    # 4. Apply overlay onto ROI: dynamic object cells -> critical (score=1.0, class=2)
    importance_score_overlay = base["importance_score"].copy()
    roi_class_overlay = base["roi_class"].copy()

    importance_score_overlay[footprint_mask] = 1.0
    roi_class_overlay[footprint_mask] = 2  # Critical

    # 5. Save compressed NPZ overlay (DO NOT modify Phase 3/5 files in place)
    npz_path = output_dir / f"frame_{frame_idx:04d}_overlay.npz"
    np.savez_compressed(
        npz_path,
        grid_bounds=base["bounds"],
        dynamic_footprint_mask=footprint_mask,
        traversability_base=base["traversability"],
        traversability_overlay=traversability_overlay,
        terrain_class_base=base["terrain_class"],
        terrain_class_overlay=terrain_class_overlay,
        importance_score_base=base["importance_score"],
        importance_score_overlay=importance_score_overlay,
        roi_class_base=base["roi_class"],
        roi_class_overlay=roi_class_overlay,
        occupied=base["occupied"],
    )

    # 6. Render comparison visualisations
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Traversability overlay plot
    ax1 = axes[0]
    im1 = ax1.imshow(
        np.ma.masked_invalid(traversability_overlay),
        origin="lower",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
    )
    if np.any(footprint_mask):
        ax1.contour(footprint_mask, levels=[0.5], colors=["blue"], linewidths=1.5)
    ax1.set_title(f"Traversability + Dynamic Overlay (Frame {frame_idx:04d})")
    ax1.set_xlabel("Grid X")
    ax1.set_ylabel("Grid Y")
    plt.colorbar(im1, ax=ax1, label="Traversability (0=Blocked, 1=Safe)")

    # ROI importance overlay plot
    ax2 = axes[1]
    im2 = ax2.imshow(
        np.ma.masked_invalid(importance_score_overlay),
        origin="lower",
        cmap="plasma",
        vmin=0.0,
        vmax=1.0,
    )
    if np.any(footprint_mask):
        ax2.contour(footprint_mask, levels=[0.5], colors=["cyan"], linewidths=1.5)
    ax2.set_title(f"ROI Spatial Importance + Dynamic Overlay (Frame {frame_idx:04d})")
    ax2.set_xlabel("Grid X")
    ax2.set_ylabel("Grid Y")
    plt.colorbar(im2, ax=ax2, label="Importance (0=Low, 1=Critical)")

    fig.tight_layout()
    png_path = output_dir / f"frame_{frame_idx:04d}_overlay.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)

    return npz_path
