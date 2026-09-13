"""Real-geometry derived terrain/object features for display only.

Every function here computes a real number or boolean from real point
coordinates or real grid cells passed in -- nothing here invents a value.
None is returned whenever there isn't enough real data to support a claim
(e.g. fewer than 3 points for a plane fit), and callers must not substitute
a placeholder for None.

Two functions here are explicitly heuristic, not trained-model output, and
are named accordingly so callers can't accidentally present them as such:
`is_probable_rock_heuristic` and the "Rock" tag it feeds. Everything else
(slope angle, ground clearance, pothole/bump depth) is a direct geometric
computation on real coordinates.
"""

from __future__ import annotations

import numpy as np

# A box's bottom face must clear the local ground by more than this to be
# flagged an overhang (e.g. a low branch or overhanging structure a low
# ego vehicle could pass under, as opposed to sitting on the ground).
DEFAULT_OVERHANG_CLEARANCE_M = 0.5

# Classes never eligible for the overhang tag. Confirmed on real RELLIS-3D
# data (frame 0, sequence 00001, track 7): MockObjectDetector's ground-
# rejection threshold clips a VEHICLE/PEDESTRIAN cluster to points above
# roughly ego height, discarding the wheels/feet that actually touch real
# ground -- so the resulting box's "bottom" sits far above true ground even
# though the object is not hanging (real local ground was measured at
# z=-1.33 there, ~1.3m below the sensor origin -- the LiDAR mount height,
# not a ditch). A track whose own semantic class already says VEHICLE or
# PEDESTRIAN is known ground-contact by definition; the clearance
# computation is geometrically real but not a meaningful "is it hanging"
# signal for these two classes specifically, so they are excluded rather
# than mislabeled "Overhang (VEHICLE)".
OVERHANG_EXCLUDED_CLASSES = frozenset({"VEHICLE", "PEDESTRIAN"})

# Minimum angle from vertical (degrees) for a ground-cluster's fitted plane
# to be reported as a slope rather than flat ground.
DEFAULT_SLOPE_THRESHOLD_DEG = 8.0

# Rock heuristic defaults: small, dense, compact clusters that aren't
# already a confidently classified class (vegetation/person/vehicle).
DEFAULT_ROCK_MAX_VOLUME_M3 = 0.5
DEFAULT_ROCK_MIN_DENSITY_PER_M3 = 200.0
DEFAULT_ROCK_EXCLUDED_CLASSES = frozenset(
    {"VEGETATION", "PEDESTRIAN", "VEHICLE", "DRIVABLE_GROUND", "ROUGH_TERRAIN"}
)

# A grid cell must sit at least this far from a local baseline to be called
# a real depression/bump rather than ordinary surface noise -- same default
# magnitude as src/perception/centerline_profile.py's DEFAULT_DEPRESSION_THRESHOLD_M.
DEFAULT_HAZARD_DEPTH_THRESHOLD_M = 0.10
DEFAULT_HAZARD_HEIGHT_THRESHOLD_M = 0.10


def points_in_oriented_box(
    points_xyzi: np.ndarray,
    center_xyz: np.ndarray,
    size_lwh: np.ndarray,
    margin: float = 0.1,
) -> np.ndarray:
    """Boolean mask of points within an axis-aligned padded box.

    Axis-aligned regardless of the box's real yaw, matching the existing
    approximation in `classify_cluster_from_semantics`
    (src/perception/object_detector.py) -- kept consistent rather than
    introducing a second, differently-approximated membership test.
    """
    c = np.asarray(center_xyz, dtype=np.float32)
    s = np.asarray(size_lwh, dtype=np.float32)
    pts = points_xyzi
    return (
        (pts[:, 0] >= c[0] - s[0] / 2.0 - margin)
        & (pts[:, 0] <= c[0] + s[0] / 2.0 + margin)
        & (pts[:, 1] >= c[1] - s[1] / 2.0 - margin)
        & (pts[:, 1] <= c[1] + s[1] / 2.0 + margin)
        & (pts[:, 2] >= c[2] - s[2] / 2.0 - margin)
        & (pts[:, 2] <= c[2] + s[2] / 2.0 + margin)
    )


def plane_normal_pca(points_xyz: np.ndarray) -> np.ndarray | None:
    """Unit surface normal of the best-fit plane through real points.

    PCA plane fit: the normal is the eigenvector of the centered points'
    covariance matrix with the smallest eigenvalue (the direction of least
    variance). Returns None when there are fewer than 3 points, or when the
    points are (numerically) collinear/coincident and no plane is
    determined -- never fabricates a normal.
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        return None
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Degenerate (collinear) point sets: the two smallest eigenvalues are
    # both ~0, so no single plane is well-determined.
    if eigvals[1] <= 1e-12:
        return None
    normal = eigvecs[:, 0]
    if normal[2] < 0:
        normal = -normal
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return None
    return normal / norm


def slope_angle_deg(points_xyz: np.ndarray) -> float | None:
    """Real angle (degrees, 0-90) between a PCA-fit plane's normal and
    vertical (+Z), computed from real ground-cluster points.

    0 degrees = perfectly flat ground; 90 degrees = a vertical wall.
    None when `plane_normal_pca` can't determine a plane.
    """
    normal = plane_normal_pca(points_xyz)
    if normal is None:
        return None
    cos_angle = float(np.clip(abs(normal[2]), 0.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def local_ground_z_percentile(
    points_xyzi: np.ndarray,
    center_xy: np.ndarray,
    radius_m: float,
    percentile: float = 10.0,
) -> float | None:
    """Real local ground height near (center_xy), from real points.

    Takes the given low percentile of Z among all real points within
    `radius_m` of `center_xy` (planar distance) -- a robust ground estimate
    that tolerates a few low-lying outliers without being pulled down by
    them the way a bare minimum would. None when no point falls in radius.
    """
    cx, cy = float(center_xy[0]), float(center_xy[1])
    dx = points_xyzi[:, 0] - cx
    dy = points_xyzi[:, 1] - cy
    within = (dx * dx + dy * dy) <= (radius_m * radius_m)
    if not np.any(within):
        return None
    return float(np.percentile(points_xyzi[within, 2], percentile))


def ground_clearance_m(bbox_bottom_z: float, ground_z: float) -> float:
    """Real vertical gap between a box's bottom face and local ground."""
    return float(bbox_bottom_z - ground_z)


def is_overhang(
    clearance_m: float,
    class_name: str,
    threshold_m: float = DEFAULT_OVERHANG_CLEARANCE_M,
    excluded_classes: frozenset[str] = OVERHANG_EXCLUDED_CLASSES,
) -> bool:
    """True when a box's bottom face clears local ground by more than
    `threshold_m` -- real geometry -- unless `class_name` is already known
    to be ground-contact by definition (see OVERHANG_EXCLUDED_CLASSES),
    in which case the clearance number is real but not a reliable "is it
    hanging" signal for that class (see the module-level comment)."""
    if class_name in excluded_classes:
        return False
    return clearance_m > threshold_m


def is_probable_rock_heuristic(
    volume_m3: float,
    num_points: int,
    class_name: str,
    max_volume_m3: float = DEFAULT_ROCK_MAX_VOLUME_M3,
    min_density_per_m3: float = DEFAULT_ROCK_MIN_DENSITY_PER_M3,
    excluded_classes: frozenset[str] = DEFAULT_ROCK_EXCLUDED_CLASSES,
) -> bool:
    """HEURISTIC fallback flag, NOT a trained-model class.

    Neither RELLIS-3D nor SemanticKITTI's checkpoint has a "rock"/"stone"
    class (see docs/validation_results.md and the ontologies in
    src/perception/semantic_labels.py), so a rock can only be guessed at
    geometrically: small volume, high point density, and not already a
    confidently classified natural/dynamic class. This is real geometry
    computed from a real cluster, but the *label* "rock" itself is an
    unverified guess -- callers must display it as clearly heuristic
    (e.g. "Rock (heuristic)"), never as a plain class name.
    """
    if class_name in excluded_classes:
        return False
    if volume_m3 <= 0.0 or volume_m3 > max_volume_m3:
        return False
    density = num_points / volume_m3
    return density >= min_density_per_m3


def classify_hazard_kind(
    z_max_local: float,
    z_min_local: float,
    baseline_z: float | None,
    depth_threshold_m: float = DEFAULT_HAZARD_DEPTH_THRESHOLD_M,
    height_threshold_m: float = DEFAULT_HAZARD_HEIGHT_THRESHOLD_M,
) -> tuple[str, float | None]:
    """Classify a real hazard grid-cluster as a negative or positive
    obstacle from its real min/max elevation vs. a real local baseline.

    Returns ("Pothole", depth_m) when the cluster's real minimum elevation
    sits more than `depth_threshold_m` below `baseline_z` (depth_m > 0);
    ("Bump", height_m) when its real maximum sits more than
    `height_threshold_m` above baseline (height_m > 0); otherwise
    ("Rough", None) -- a real hazard (blocked traversability) whose
    elevation doesn't clearly read as either, e.g. dense rough texture, or
    when no real baseline could be computed (`baseline_z` is None).

    A cluster that is simultaneously a deep dip and has a high point
    somewhere in it reports "Pothole" if the depth exceeds the height
    (the dip is the dominant, more safety-relevant feature); ties go to
    "Pothole" for the same reason.
    """
    if baseline_z is None or not np.isfinite(baseline_z):
        return "Rough", None

    depth_m = baseline_z - z_min_local
    height_m = z_max_local - baseline_z

    is_pothole = depth_m > depth_threshold_m
    is_bump = height_m > height_threshold_m

    if is_pothole and (not is_bump or depth_m >= height_m):
        return "Pothole", float(depth_m)
    if is_bump:
        return "Bump", float(height_m)
    return "Rough", None
