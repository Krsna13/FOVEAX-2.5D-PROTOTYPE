"""3D-box class labels for the Open3D viewport.

The dashboard uses the legacy `open3d.visualization.Visualizer`, which has no
text rendering. Labels are therefore drawn as Qt widgets over the embedded
Open3D window: the Open3D process projects each box's top-center into screen
pixels with its live camera, and the Qt process places a label there.

Everything here derives from a single `TrackState` -- the same object
`FoveaX3DViewer._create_box_lineset` builds the box from -- so a label cannot
show a class, status, or position that its box does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Lift above the box's top face so the text clears the returns on the object.
LABEL_Z_OFFSET_M = 0.3

# Same real Near/Middle/Far boundaries used everywhere else in this project:
# SPEC_ZONES_100M (src/07_adaptive_2point5d_map.py), the RESOLUTION_ZONES
# spec panel (src/dashboard/qt_main_window.py), and the distance buckets in
# src/10b_eval_distance_metrics.py. Kept in one place so a per-zone object
# count can never use different boundaries than the zones it's labelled
# against.
DISTANCE_ZONE_BOUNDS_M = (0.0, 15.0, 35.0, 100.0)
DISTANCE_ZONE_NAMES = ("Near", "Middle", "Far")


def distance_zone_for_track(track) -> str | None:
    """Which real resolution zone a track's real distance falls in.

    None for a track farther than the Far zone's outer bound (100m) --
    outside the spec'd grid entirely, not silently folded into "Far".
    """
    dist = track_distance_m(track)
    lo, near_far, far_outer, max_far = DISTANCE_ZONE_BOUNDS_M
    if lo <= dist < near_far:
        return "Near"
    if near_far <= dist < far_outer:
        return "Middle"
    if far_outer <= dist <= max_far:
        return "Far"
    return None


def zone_object_counts(tracks) -> dict[str, int]:
    """Real per-zone object counts, each zone name initialised to 0 so an
    empty zone is shown as 0, not omitted."""
    counts = {name: 0 for name in DISTANCE_ZONE_NAMES}
    for t in tracks:
        zone = distance_zone_for_track(t)
        if zone is not None:
            counts[zone] += 1
    return counts


def track_status(track) -> str:
    """Kalman-tracker motion status exactly as the dashboard reports it."""
    return "Dynamic" if track.dynamic else "Static"


def track_distance_m(track) -> float:
    """Real 2D ego-relative distance to a track's box centroid.

    2D (x, y only), matching the existing "DYNAMIC OBJECT: ... at X m"
    nearest-threat convention already shown by the dashboard
    (src/dashboard/qt_main_window.py) -- kept identical so the same object
    can never show two different distances in different parts of the UI.
    This view is ego-relative and re-centered every frame (no odometry), so
    a 2D radial distance from the origin is the real, available quantity.
    """
    return float(np.hypot(track.state[0], track.state[1]))


def display_class_for_track(track) -> str:
    """Real-geometry-derived display class for a track.

    Reads only `track.metadata`, populated per-frame from this track's own
    matched detection by DataStreamerThread._attach_terrain_features (see
    src/perception/terrain_features.py) -- nothing here is computed or
    guessed independently, so this can never show a feature the track's own
    geometry didn't produce this frame.

    Priority when more than one applies: an overhang is the most safety-
    relevant (a low structure a vehicle could hit that a ground/obstacle
    label wouldn't warn about), then slope (relevant only for ground-like
    clusters), then the rock heuristic (explicitly marked as such, never
    presented as a trained class), else the real classifier's own class
    name unchanged.
    """
    meta = getattr(track, "metadata", None) or {}

    if meta.get("is_overhang"):
        return f"Overhang ({track.class_name})"
    slope = meta.get("slope_deg")
    if slope is not None:
        return f"Slope ({slope:.0f}°)"
    if meta.get("is_probable_rock"):
        return "Rock (heuristic)"
    return track.class_name


def displayed_speed_mps(track) -> float:
    """Real Kalman speed for Dynamic tracks; 0.0 for Static (Bug 1 fix --
    a Static track's raw speed estimate can still be a real nonzero,
    noisy/transient value; showing it next to "Static" read as
    contradictory, so only the display -- not the classification -- is
    forced to 0.0)."""
    if not track.dynamic:
        return 0.0
    return float(np.linalg.norm(track.state[3:6]))


def tracked_object_row_text(track) -> str:
    """One TRACKED OBJECTS list row, built from the same functions
    (`display_class_for_track`, `track_distance_m`) that build the 3D label
    text for the same track -- so a track's row and its label can never
    show two different classes or distances."""
    return (
        f"ID: {track.track_id} | {display_class_for_track(track)} | "
        f"{track_status(track)} | {displayed_speed_mps(track):.1f} m/s | "
        f"{track_distance_m(track):.1f}m"
    )


def track_label_text(track) -> str:
    return (
        f"{display_class_for_track(track)} ({track_status(track)}) "
        f"- {track_distance_m(track):.1f}m"
    )


def box_top_center(track, z_offset_m: float = LABEL_Z_OFFSET_M) -> np.ndarray:
    """Top-center of the track's 3D box, lifted by `z_offset_m`.

    The box is built centered on `track.state[:3]` spanning +-h/2 in Z, and yaw
    only rotates about Z, so its top face center is (cx, cy, cz + h/2).
    """
    cx, cy, cz = (float(v) for v in track.state[:3])
    h = float(track.size_lwh[2])
    return np.array([cx, cy, cz + h / 2.0 + z_offset_m], dtype=np.float64)


def orthonormalize_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    """Rebuild a world->camera matrix with a truly orthonormal rotation.

    Legacy ViewControl accepts an `up` vector that is not perpendicular to
    `front` (the dashboard's own perspective preset does this). The renderer
    orthogonalizes it, but `convert_to_pinhole_camera_parameters()` returns the
    raw rows, so projecting with them misplaces points vertically. The
    translation was built as -R_raw * eye from those same rows, so the camera
    position is still exactly recoverable.
    """
    ext = np.asarray(extrinsic, dtype=np.float64)
    r_raw, t = ext[:3, :3], ext[:3, 3]
    eye = -np.linalg.solve(r_raw, t)

    forward = r_raw[2] / np.linalg.norm(r_raw[2])
    right = r_raw[0] - np.dot(r_raw[0], forward) * forward
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)

    rot = np.vstack([right, down, forward])
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = -rot @ eye
    return out


def project_to_screen(
    points_world: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project (N,3) world points to pixels with Open3D pinhole parameters.

    `extrinsic` is Open3D's world->camera 4x4; the camera looks down +Z with
    +X right and +Y down, so u = fx*x/z + cx, v = fy*y/z + cy.
    Returns (u, v, visible); points behind the camera or off-screen are not
    visible.
    """
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
    cam = (orthonormalize_extrinsic(extrinsic) @ homo.T).T[:, :3]
    z = cam[:, 2]
    in_front = z > 1e-6
    safe_z = np.where(in_front, z, 1.0)
    k = np.asarray(intrinsic, dtype=np.float64)
    u = k[0, 0] * cam[:, 0] / safe_z + k[0, 2]
    v = k[1, 1] * cam[:, 1] / safe_z + k[1, 2]
    visible = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, visible


@dataclass(frozen=True)
class TrackLabel:
    track_id: int
    text: str
    u: float
    v: float
    visible: bool
    dynamic: bool
    view_width: int
    view_height: int


def build_track_labels(
    tracks,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    width: int,
    height: int,
    z_offset_m: float = LABEL_Z_OFFSET_M,
) -> list[TrackLabel]:
    """One label per track, in track order, projected with the given camera."""
    tracks = list(tracks)
    if not tracks:
        return []
    anchors = np.vstack([box_top_center(t, z_offset_m) for t in tracks])
    u, v, visible = project_to_screen(anchors, intrinsic, extrinsic, width, height)
    return [
        TrackLabel(
            track_id=int(t.track_id),
            text=track_label_text(t),
            u=float(u[i]),
            v=float(v[i]),
            visible=bool(visible[i]),
            dynamic=bool(t.dynamic),
            view_width=int(width),
            view_height=int(height),
        )
        for i, t in enumerate(tracks)
    ]
