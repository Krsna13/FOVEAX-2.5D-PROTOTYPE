"""FOVEAX Phase 8A — Model-agnostic 3D object detection interface.

Defines:

    Detection3D        — single 3D detection result.
    ObjectDetector     — abstract detector interface.
    GroundTruthBoxProvider — placeholder (raises NotImplementedError).
    MockObjectDetector      — deterministic geometric clustering baseline.
    OpenPCDetDetector       — Phase 8B placeholder.

All implementations are model-agnostic; real PointPillars / CenterPoint
integration is deferred to Phase 8B and uses OpenPCDet externally.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Detection3D
# ---------------------------------------------------------------------------

@dataclass
class Detection3D:
    """A single 3D bounding-box detection.

    Attributes
    ----------
    center_xyz : np.ndarray, shape (3,), float32
        Box centre in smoothed [x, y, z] (metres).
    size_lwh : np.ndarray, shape (3,), float32
        Box dimensions [length (x), width (y), height (z)] in metres.
    yaw_rad : float
        Heading angle in radians, 0 = +x axis, counter-clockwise positive.
    class_id : int
        Integer class identifier (detector-specific taxonomy).
    class_name : str
        Human-readable class name, e.g. "vehicle", "pedestrian", "cyclist".
    confidence : float
        Detection confidence in [0, 1].
    source : str
        Human-readable label for the detector that produced this result.
    metadata : dict
        Optional extra fields (e.g. velocity, detector internals).
    """

    center_xyz: np.ndarray
    size_lwh: np.ndarray
    yaw_rad: float
    class_id: int
    class_name: str
    confidence: float
    source: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.center_xyz.shape != (3,):
            raise ValueError(
                f"center_xyz must have shape (3,), got {self.center_xyz.shape}."
            )
        if self.center_xyz.dtype != np.float32:
            raise ValueError(
                f"center_xyz must be float32, got {self.center_xyz.dtype}."
            )
        if self.size_lwh.shape != (3,):
            raise ValueError(
                f"size_lwh must have shape (3,), got {self.size_lwh.shape}."
            )
        if self.size_lwh.dtype != np.float32:
            raise ValueError(
                f"size_lwh must be float32, got {self.size_lwh.dtype}."
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}."
            )
        if self.size_lwh[0] <= 0.0 or self.size_lwh[1] <= 0.0 or self.size_lwh[2] <= 0.0:
            raise ValueError(
                f"size_lwh must be positive, got {self.size_lwh}."
            )

    def __repr__(self) -> str:
        c = self.center_xyz
        s = self.size_lwh
        return (
            f"Detection3D(class={self.class_name!r}, conf={self.confidence:.3f}, "
            f"center=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}), "
            f"size=({s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}), "
            f"yaw={np.degrees(self.yaw_rad):.1f}deg, source={self.source!r})"
        )


# ---------------------------------------------------------------------------
# ObjectDetector interface
# ---------------------------------------------------------------------------

def validate_points_xyzi(points: np.ndarray) -> None:
    """Validate that *points* has shape (N, 4) and is finite.

    Raises
    ------
    ValueError
        If shape is wrong or the array contains NaN / Inf.
    """
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(
            f"points_xyzi must have shape (N, 4), got {points.shape}."
        )
    if points.shape[0] == 0:
        raise ValueError("points_xyzi must contain at least one point.")
    if not np.isfinite(points).all():
        raise ValueError("points_xyzi contains NaN or Inf values.")


class ObjectDetector(ABC):
    """Abstract interface for 3D object detectors.

    All concrete detectors must implement :meth:`detect`.
    """

    @abstractmethod
    def detect(
        self,
        points_xyzi: np.ndarray,
        timestamp_s: float | None = None,
        semantic_labels: np.ndarray | None = None,
    ) -> list[Detection3D]:
        """Detect 3D objects in a single LiDAR sweep.

        Parameters
        ----------
        points_xyzi : np.ndarray, shape (N, 4), dtype float32
            Columns are [x, y, z, intensity].  Validated by the interface.
        timestamp_s : float or None
            Optional sensor timestamp (seconds).  Used by trackers to compute
            elapsed time between frames.

        Returns
        -------
        list[Detection3D]
            Detections sorted deterministically (by confidence desc, then
            class_name, then center x, then center y).
        """
        ...


# ---------------------------------------------------------------------------
# GroundTruthBoxProvider — placeholder, never returns detections
# ---------------------------------------------------------------------------

class GroundTruthBoxProvider(ObjectDetector):
    """Placeholder provider for ground-truth 3D boxes.

    ⚠️  SemanticKITTI segmentation labels do NOT provide standard 3D object
    detection boxes (class, x, y, z, l, w, h, yaw).  They provide per-point
    semantic and instance labels only.  Deriving detection boxes from them is
    a separate task and is not implemented here.

    For real 3D ground-truth boxes use one of:
        - nuScenes (3D detection GT available)
        - Waymo Open Dataset (3D detection GT available)
        - KITTI Detection benchmark
        - CARLA with labelled dynamic actors
        - Synthetic detections generated from known actor poses

    This class exists solely to make the interface explicit and to raise a
    clear error if it is accidentally used.
    """

    _SOURCE = "ground_truth_box_provider_not_available"

    def detect(
        self,
        points_xyzi: np.ndarray,
        timestamp_s: float | None = None,
    ) -> list[Detection3D]:
        """Always raises NotImplementedError with a detailed explanation."""
        validate_points_xyzi(points_xyzi)
        raise NotImplementedError(
            "GroundTruthBoxProvider is NOT available.\n\n"
            "SemanticKITTI .label files provide per-point semantic and instance "
            "labels but NOT standard 3D object detection boxes (class, x, y, z, "
            "length, width, height, yaw).  Converting instance segments to 3D "
            "boxes is a non-trivial separate step and is intentionally not "
            "implemented in Phase 8A.\n\n"
            "For real ground-truth 3D boxes use one of:\n"
            "  - nuScenes\n"
            "  - Waymo Open Dataset\n"
            "  - KITTI Detection benchmark\n"
            "  - CARLA with labelled dynamic actors\n"
            "  - Synthetic detections from known actor poses\n\n"
            "MockObjectDetector provides a deterministic geometric baseline for "
            "pipelines that must run without any labelled data."
        )


# ---------------------------------------------------------------------------
# MockObjectDetector — deterministic geometric baseline (NOT AI)
# ---------------------------------------------------------------------------

_VEHICLE_CLASSES = frozenset({"car", "bus", "truck", "van", "trailer"})
_PEDESTRIAN_CLASSES = frozenset({"pedestrian", "person", "walker"})
_CYCLIST_CLASSES = frozenset({"cyclist", "bicycle", "motorcycle", "bicycle"})

# Maximum configuration defaults
_MOCK_DEFAULTS = {
    "ground_threshold_m": 0.25,     # points below this are treated as ground
    "cluster_min_points": 5,        # minimum cluster size to keep
    "cluster_max_gap_m": 0.5,       # max gap between points in BFS clustering
    "cluster_max_dist_m": 1.5,      # max pairwise distance for two points in a cluster
    "voxel_size_m": 0.2,            # voxel size for pre-filtering
}


def _voxel_cluster_key(point: np.ndarray, voxel_size: float) -> tuple[int, int, int]:
    """Voxel index tuple for a single 3-D point.

    Kept for any external caller expecting a single-point key; the hot
    path inside MockObjectDetector.detect() uses
    _voxel_cluster_keys_vectorized() below instead (identical per-point
    formula, computed for every point in one NumPy call rather than a
    Python-level loop).
    """
    return (
        int(np.floor(point[0] / voxel_size)),
        int(np.floor(point[1] / voxel_size)),
        int(np.floor(point[2] / voxel_size)),
    )


def _voxel_cluster_keys_vectorized(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel index triples for every point in *xyz* in one vectorized call.

    Row i of the result is exactly
    _voxel_cluster_key(xyz[i], voxel_size) as a length-3 int64 array --
    same floor-division formula, applied to the whole (N, 3) array at
    once instead of once per point in a Python list comprehension.
    """
    return np.floor(xyz / voxel_size).astype(np.int64)


def _deterministic_cluster_sort_key(d: Detection3D) -> tuple:
    """Sort key: confidence desc, class_name asc, center_x asc, center_y asc."""
    return (-d.confidence, d.class_name, d.center_xyz[0], d.center_xyz[1])


def classify_cluster_from_semantics(
    center_xyz: np.ndarray,
    size_lwh: np.ndarray,
    points_xyzi: np.ndarray,
    semantic_labels: np.ndarray | None,
    margin: float = 0.05,
    fallback_class: str = "unclassified",
) -> tuple[int, str]:
    """Classify a 3D bounding box / cluster against per-point semantic labels.

    Cross-references the cluster's 3D extent against the current frame's
    semantic segmentation output by majority voting over points inside the box.

    Parameters
    ----------
    center_xyz : np.ndarray, shape (3,)
        Cluster centre [x, y, z] in metres.
    size_lwh : np.ndarray, shape (3,)
        Cluster dimensions [length, width, height] in metres.
    points_xyzi : np.ndarray, shape (N, 4)
        Full point cloud for the frame.
    semantic_labels : np.ndarray, shape (N,), optional
        Per-point FOVEAX class IDs (0..7). If None, empty, or only UNKNOWN,
        returns (7, fallback_class).
    margin : float
        Spatial padding around the bounding box (metres).
    fallback_class : str
        Class name to assign if no semantic data or only UNKNOWN points exist.

    Returns
    -------
    tuple[int, str]
        (class_id, class_name) where class_name is a readable FOVEAX class
        (e.g. "VEHICLE", "PEDESTRIAN", "SOLID_OBSTACLE", "BUILDING_WALL",
        "VEGETATION") or fallback_class.
    """
    from collections import Counter
    from src.perception.semantic_labels import FOVEAX_CLASSES

    if semantic_labels is None or len(points_xyzi) == 0:
        return 7, fallback_class

    c = np.asarray(center_xyz, dtype=np.float32)
    s = np.asarray(size_lwh, dtype=np.float32)
    in_box = (
        (points_xyzi[:, 0] >= c[0] - s[0] / 2.0 - margin)
        & (points_xyzi[:, 0] <= c[0] + s[0] / 2.0 + margin)
        & (points_xyzi[:, 1] >= c[1] - s[1] / 2.0 - margin)
        & (points_xyzi[:, 1] <= c[1] + s[1] / 2.0 + margin)
        & (points_xyzi[:, 2] >= c[2] - s[2] / 2.0 - margin)
        & (points_xyzi[:, 2] <= c[2] + s[2] / 2.0 + margin)
    )
    box_labels = semantic_labels[in_box]
    if len(box_labels) == 0:
        return 7, fallback_class

    # Filter out UNKNOWN (7)
    valid_labels = box_labels[box_labels != 7]
    if len(valid_labels) == 0:
        return 7, fallback_class

    # Prioritize obstacle/object classes (2..6: VEGETATION, BUILDING_WALL, SOLID_OBSTACLE, VEHICLE, PEDESTRIAN)
    obstacle_labels = valid_labels[(valid_labels >= 2) & (valid_labels <= 6)]
    if len(obstacle_labels) > 0:
        counts = Counter(obstacle_labels)
        majority_id = int(counts.most_common(1)[0][0])
    else:
        counts = Counter(valid_labels)
        majority_id = int(counts.most_common(1)[0][0])

    class_name = FOVEAX_CLASSES.get(majority_id, fallback_class)
    return majority_id, class_name


class MockObjectDetector(ObjectDetector):
    """Deterministic geometric clustering baseline — NOT an AI detector.

    This detector uses only point cloud geometry:
        1. Reject ground points (z < ground_threshold_m).
        2. Voxel-pre-filter for performance.
        3. BFS cluster within cluster_max_dist_m.
        4. Axis-aligned bounding box from [min, max] extents.
        5. Assign class_name="unknown_obstacle" by default, or cross-reference
           with semantic_labels when provided.

    Confidence is deterministic and bounded in [0, 1].

    Parameters
    ----------
    ground_threshold_m : float
        Discard points with z < ground_threshold_m (metres).
    cluster_min_points : int
        Minimum number of points required to form a cluster.
    cluster_max_gap_m : float
        Not used directly by the BFS, kept for configuration consistency.
    cluster_max_dist_m : float
        Maximum distance between neighbours in the BFS (metres).
    voxel_size_m : float
        Voxel size for pre-filtering (metres).
    default_class_name : str
        Default fallback class name when no semantic labels are provided
        (default "unknown_obstacle").
    """

    _SOURCE: str = "mock_geometric_clusterer"

    def __init__(
        self,
        ground_threshold_m: float = _MOCK_DEFAULTS["ground_threshold_m"],
        cluster_min_points: int = _MOCK_DEFAULTS["cluster_min_points"],
        cluster_max_gap_m: float = _MOCK_DEFAULTS["cluster_max_gap_m"],
        cluster_max_dist_m: float = _MOCK_DEFAULTS["cluster_max_dist_m"],
        voxel_size_m: float = _MOCK_DEFAULTS["voxel_size_m"],
        default_class_name: str = "unknown_obstacle",
    ) -> None:
        """Configure the mock geometric clusterer.

        Parameters
        ----------
        ground_threshold_m : float
            Points with z below this are discarded as ground.
        cluster_min_points : int
            Minimum number of points a cluster must contain to become a detection.
        cluster_max_gap_m : float
            Maximum gap between consecutive BFS neighbours.
        cluster_max_dist_m : float
            Maximum Euclidean distance between two points to be considered
            adjacent in the BFS graph.
        voxel_size_m : float
            Voxel size used for pre-filtering the point cloud before clustering.
        default_class_name : str
            Default class name for detections when no semantic labels are given.
        """
        if ground_threshold_m < 0.0:
            raise ValueError("ground_threshold_m must be >= 0.")
        if cluster_min_points < 1:
            raise ValueError("cluster_min_points must be >= 1.")
        if cluster_max_gap_m <= 0.0:
            raise ValueError("cluster_max_gap_m must be > 0.")
        if cluster_max_dist_m <= 0.0:
            raise ValueError("cluster_max_dist_m must be > 0.")
        if voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be > 0.")

        self.ground_threshold_m = ground_threshold_m
        self.cluster_min_points = cluster_min_points
        self.cluster_max_gap_m = cluster_max_gap_m
        self.cluster_max_dist_m = cluster_max_dist_m
        self.voxel_size_m = voxel_size_m
        self.default_class_name = default_class_name

        warnings.warn(
            "MockObjectDetector is a deterministic GEOMETRIC baseline, "
            "NOT an AI / trained object detector.  Detections are clusters of "
            "points above the ground plane and carry class_name='unknown_obstacle'. "
            "Do not use these results for safety claims.",
            UserWarning,
            stacklevel=2,
        )

    def detect(
        self,
        points_xyzi: np.ndarray,
        timestamp_s: float | None = None,
        semantic_labels: np.ndarray | None = None,
        fallback_class_name: str | None = None,
    ) -> list[Detection3D]:
        """Run geometric clustering on *points_xyzi*.

        Parameters
        ----------
        points_xyzi : np.ndarray, shape (N, 4), dtype float32
            Point cloud with columns [x, y, z, intensity].
        timestamp_s : float or None
            Ignored by the mock detector (no temporal model).
        semantic_labels : np.ndarray, shape (N,), optional
            Per-point FOVEAX class IDs to cross-reference detected clusters.
        fallback_class_name : str, optional
            Override class name when unclassified or no semantic labels given.

        Returns
        -------
        list[Detection3D]
        """
        validate_points_xyzi(points_xyzi)

        warnings.warn(
            "MockObjectDetector is a deterministic GEOMETRIC baseline, "
            "NOT an AI / trained object detector.  Detections are clusters of "
            "points above the ground plane and carry class_name='unknown_obstacle'. "
            "Do not use these results for safety claims.",
            UserWarning,
            stacklevel=2,
        )

        xyz = points_xyzi[:, 0:3].astype(np.float64)
        intensity = points_xyzi[:, 3].astype(np.float64)

        # 1. Remove ground points.
        above_ground = xyz[:, 2] >= self.ground_threshold_m
        if not above_ground.any():
            return []
        xyz = xyz[above_ground]
        intensity = intensity[above_ground]
        n = len(xyz)

        # 2. Voxel pre-filter: keep one representative per occupied voxel.
        #    Only apply when the cloud is large enough; for small clouds the
        #    pre-filter can collapse distinct clusters into a single voxel.
        #    Skip for clouds with fewer than 5000 points after ground removal.
        if self.voxel_size_m > 0.0 and n >= 5000:
            # Vectorized: was a Python list comprehension calling
            # _voxel_cluster_key() once per point (profiled as part of
            # MockObjectDetector.detect()'s dominant cost on real,
            # ~131k-point RELLIS-3D frames). Same per-point formula,
            # computed for the whole array in one NumPy call.
            keys = _voxel_cluster_keys_vectorized(xyz, self.voxel_size_m)
            # Lexicographic sort by voxel key, then by original index for
            # determinism.
            order = np.lexsort((np.arange(n), keys[:, 0], keys[:, 1], keys[:, 2]))
            keys = keys[order]
            xyz = xyz[order]
            intensity = intensity[order]
            # Keep first point per voxel.
            unique_mask = np.concatenate(
                (
                    np.ones(1, dtype=bool),
                    (keys[1:] != keys[:-1]).any(axis=1),
                )
            )
            xyz = xyz[unique_mask]
            intensity = intensity[unique_mask]

        n = len(xyz)
        if n == 0:
            return []

        # 3. Clustering via a vectorized radius-neighbor graph + connected
        # components -- replaces a manual per-point BFS that used
        # cKDTree.query_ball_point plus a per-neighbor-list sorted() call.
        # Profiled at 78% of MockObjectDetector.detect()'s time on a real
        # ~131k-point RELLIS-3D frame (0.168s of 0.214s total), dominated
        # by one query_ball_point() call per point plus 4,774 sorted()
        # calls on their neighbor lists.
        #
        # That sorted() call only ever affected the BFS's internal
        # traversal order *within* a connected component (which of a
        # point's neighbors got visited first) -- it never affected which
        # points ended up in which cluster, how many clusters existed, or
        # any value derived from a cluster (center = mean of its points,
        # size = its AABB extent are both order-invariant over the
        # cluster's point set). Removing it changes nothing about the
        # actual output.
        #
        # cKDTree.query_pairs(r) returns every unordered pair of points
        # within cluster_max_dist_m of each other -- both it and the old
        # query_ball_point(r) use the identical inclusive (<= r)
        # Euclidean-distance adjacency criterion, so the resulting graph's
        # connected components are exactly the same partition the BFS
        # computed, just derived in one vectorized call instead of n
        # sequential Python-level ones.
        clusters: list[list[int]] = []
        if n > 0:
            try:
                from scipy.spatial import cKDTree
                from scipy.sparse import coo_matrix
                from scipy.sparse.csgraph import connected_components

                tree = cKDTree(xyz)
                pairs = tree.query_pairs(self.cluster_max_dist_m, output_type="ndarray")

                # connected_components(..., directed=False) treats the
                # graph as undirected internally (confirmed: edges given
                # in only one direction still connect both endpoints), so
                # query_pairs' single-direction (i < j) pairs are enough
                # -- no need to duplicate each edge in both directions
                # first, which only added sparse-matrix construction
                # overhead (sum_duplicates/sort_indices) without changing
                # the resulting components.
                if len(pairs) > 0:
                    data = np.ones(len(pairs), dtype=bool)
                    adjacency = coo_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
                else:
                    adjacency = coo_matrix((n, n), dtype=bool)

                _, labels = connected_components(adjacency, directed=False)

                # Group point indices by component, then order components
                # the same way the old sequential BFS discovered them: by
                # the smallest point index each component contains (the
                # old loop always started its next unvisited component at
                # the next-smallest untouched index). This is not required
                # for correctness of the final Detection3D list (that is
                # separately, deterministically re-sorted by
                # _deterministic_cluster_sort_key below) but keeps this
                # intermediate ordering identical to the previous
                # implementation's for anyone inspecting it directly.
                order = np.argsort(labels, kind="stable")
                sorted_labels = labels[order]
                boundaries = np.flatnonzero(np.diff(sorted_labels)) + 1
                groups = np.split(order, boundaries)
                groups.sort(key=lambda g: int(g.min()))
                clusters = [g.tolist() for g in groups]
            except ImportError:
                max_dist_sq = self.cluster_max_dist_m ** 2
                visited = np.zeros(n, dtype=bool)
                for start in range(n):
                    if visited[start]:
                        continue
                    stack = [start]
                    visited[start] = True
                    cluster: list[int] = []
                    while stack:
                        idx = stack.pop()
                        cluster.append(idx)
                        px, py, pz = xyz[idx]
                        for j in range(n):
                            if visited[j]:
                                continue
                            qx, qy, qz = xyz[j]
                            dx = px - qx
                            dy = py - qy
                            dz = pz - qz
                            if dx * dx + dy * dy + dz * dz <= max_dist_sq:
                                visited[j] = True
                                stack.append(j)
                    clusters.append(cluster)

        # 4. Build detections for clusters above the minimum size.
        detections: list[Detection3D] = []
        for cluster in clusters:
            if len(cluster) < self.cluster_min_points:
                continue
            pts = xyz[cluster]
            min_xyz = pts.min(axis=0)
            max_xyz = pts.max(axis=0)
            # Real point centroid (mean position), NOT the AABB midpoint.
            # Root cause of a false-positive "moving vegetation" bug: for a
            # diffuse/sparse cluster (e.g. a real vegetation patch), the BFS
            # connectivity graph (cluster_max_dist_m below) can borderline
            # bridge or fail to bridge to a nearby sub-clump depending on a
            # handful of points -- when that happens the AABB midpoint swings
            # by meters between frames even though almost none of the real
            # points moved, since (min+max)/2 depends only on the two most
            # extreme points. The mean of all real points in the cluster is
            # far more stable against that kind of borderline merge/split,
            # while staying numerically identical to the AABB midpoint for
            # compact, roughly-symmetric clusters (vehicles, pedestrians).
            center = pts.mean(axis=0).astype(np.float32)
            size = (max_xyz - min_xyz).astype(np.float32)
            # yaw is 0 for axis-aligned boxes produced by this baseline.
            yaw = 0.0

            # Deterministic confidence from point count and compactness.
            n_pts = float(len(cluster))
            # Compactness: how well does the bounding box fit the points?
            # Compute the actual volume the points occupy vs AABB volume.
            # Use a simple proxy: fraction of points near the AABB centre.
            # Here we use point count saturation + AABB compactness.
            aabb_volume = size[0] * size[1] * size[2]
            if aabb_volume > 0.0:
                # Average distance of points to the centre, normalised.
                dists = np.linalg.norm(pts - center, axis=1)
                avg_dist = float(dists.mean())
                # Smaller avg_dist relative to box size → more compact.
                # Use the maximum box dimension as reference scale.
                max_dim = float(max(size[0], size[1], size[2]))
                compactness = float(np.clip(1.0 - avg_dist / (max_dim + 1e-6), 0.0, 1.0))
            else:
                compactness = 0.0

            # Confidence: saturates with point count, modulated by compactness.
            count_factor = float(np.clip(n_pts / 50.0, 0.0, 1.0))
            confidence = float(np.clip(0.3 + 0.5 * count_factor + 0.2 * compactness, 0.0, 1.0))

            if semantic_labels is not None:
                cid, cname = classify_cluster_from_semantics(
                    center,
                    size,
                    points_xyzi,
                    semantic_labels,
                    fallback_class=fallback_class_name or "unclassified",
                )
            else:
                cid = 0
                cname = fallback_class_name if fallback_class_name is not None else self.default_class_name

            d = Detection3D(
                center_xyz=center,
                size_lwh=size,
                yaw_rad=yaw,
                class_id=cid,
                class_name=cname,
                confidence=confidence,
                source=self._SOURCE,
                metadata={
                    "point_count": len(cluster),
                    "compactness": compactness,
                    "cluster_indices": cluster,
                },
            )
            detections.append(d)

        # 5. Deterministic sort.
        detections.sort(key=_deterministic_cluster_sort_key)
        return detections


# ---------------------------------------------------------------------------
# OpenPCDetDetector — Phase 8B placeholder
# ---------------------------------------------------------------------------

class OpenPCDetDetector(ObjectDetector):
    """Placeholder for OpenPCDet / PointPillars / CenterPoint integration.

    This class is a stub only.  Real integration (Phase 8B) will:
        - Use the OpenPCDet toolbox externally (not vendored here).
        - Load a pretrained PointPillars or CenterPoint checkpoint trained on
          KITTI Detection, nuScenes, or Waymo.
        - Preprocess the LiDAR sweep into the format expected by the model.
        - Post-process model output into list[Detection3D].

    No model weights are included and no repositories are cloned in Phase 8A.
    """

    _SOURCE = "openpcdet_placeholder"

    def __init__(self) -> None:
        """Instantiate the placeholder."""
        pass

    def detect(
        self,
        points_xyzi: np.ndarray,
        timestamp_s: float | None = None,
    ) -> list[Detection3D]:
        """Raises NotImplementedError with integration instructions."""
        validate_points_xyzi(points_xyzi)
        raise NotImplementedError(
            "OpenPCDet / PointPillars / CenterPoint integration is Phase 8B.\n\n"
            "No model weights, OpenPCDet code, or detection checkpoints are "
            "included in Phase 8A.  To integrate a real detector:\n\n"
            "  1. Install the OpenPCDet toolbox externally:\n"
            "       git clone https://github.com/open-mmlab/openPCDet.git\n"
            "       cd openPCDet && pip install -e .\n\n"
            "  2. Download a pretrained checkpoint for your target dataset:\n"
            "       - PointPillars (KITTI Detection, nuScenes, Waymo)\n"
            "       - CenterPoint (nuScenes, Waymo)\n\n"
            "  3. Implement a wrapper that:\n"
            "       a. Converts points_xyzi to the model's input format.\n"
            "       b. Calls the OpenPCDet inference interface.\n"
            "       c. Converts model output to list[Detection3D].\n\n"
            "  4. Replace this placeholder with the real wrapper.\n\n"
            "Do NOT claim placeholder output as AI detection results."
        )
