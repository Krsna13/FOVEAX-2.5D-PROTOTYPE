"""Coarse, informational-only ego displacement estimate between two
consecutive real LiDAR frames, via point-to-point ICP registration.

This pipeline has no odometry (confirmed by inspection of
dashboard_state.py, data_streamer.py, and the tracker -- see
src/perception/centerline_profile.py's module docstring for that
investigation). There is also no absolute-position "world map" view
anywhere in this codebase: every 2D grid and the Open3D 3D view are
built fresh, ego-centered, from each frame's own raw points. This module
does NOT create one -- it only measures how far the real, raw
point-cloud content shifted between two consecutive real frames, as a
real, honestly-labeled diagnostic number (e.g. "Est. relative
displacement: 0.012 m"), never used to reposition any view.

Uses Open3D's existing point-to-point ICP (already a project dependency)
on voxel-downsampled points for speed. Returns None (not zero, not a
guess) whenever registration does not have enough real geometry to
produce a trustworthy estimate -- callers must treat None as "unknown",
not "no motion".
"""
from __future__ import annotations

import numpy as np

# Below this many real downsampled points, ICP has too little real
# geometry to trust -- report unknown rather than guess.
MIN_POINTS_FOR_ICP = 50
DEFAULT_VOXEL_SIZE_M = 0.5
DEFAULT_MAX_CORRESPONDENCE_DISTANCE_M = 1.0


def estimate_relative_displacement_m(
    prev_points_xyz: np.ndarray,
    curr_points_xyz: np.ndarray,
    voxel_size_m: float = DEFAULT_VOXEL_SIZE_M,
    max_correspondence_distance_m: float = DEFAULT_MAX_CORRESPONDENCE_DISTANCE_M,
) -> float | None:
    """Coarse real displacement estimate between two real point clouds.

    Returns the translation magnitude (metres) of the rigid transform
    ICP converges to, or None if either cloud has too few real points
    after downsampling, or Open3D is unavailable.
    """
    try:
        import open3d as o3d
    except ImportError:
        return None

    if len(prev_points_xyz) == 0 or len(curr_points_xyz) == 0:
        return None

    prev_pcd = o3d.geometry.PointCloud()
    prev_pcd.points = o3d.utility.Vector3dVector(prev_points_xyz.astype(np.float64))
    curr_pcd = o3d.geometry.PointCloud()
    curr_pcd.points = o3d.utility.Vector3dVector(curr_points_xyz.astype(np.float64))

    prev_down = prev_pcd.voxel_down_sample(voxel_size_m)
    curr_down = curr_pcd.voxel_down_sample(voxel_size_m)

    if len(prev_down.points) < MIN_POINTS_FOR_ICP or len(curr_down.points) < MIN_POINTS_FOR_ICP:
        return None

    try:
        result = o3d.pipelines.registration.registration_icp(
            curr_down, prev_down, max_correspondence_distance_m,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
    except Exception:
        return None

    if result.fitness <= 0.0:
        # ICP found essentially no real correspondences -- an estimate
        # here would not be trustworthy.
        return None

    translation = result.transformation[:3, 3]
    return float(np.linalg.norm(translation))
