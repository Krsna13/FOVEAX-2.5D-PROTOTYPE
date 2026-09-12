"""Unit tests for FOVEAX Phase 8A — object_detector module.

Run with:
    python -m pytest tests/test_object_detector.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perception.object_detector import (
    Detection3D,
    GroundTruthBoxProvider,
    MockObjectDetector,
    OpenPCDetDetector,
    ObjectDetector,
    validate_points_xyzi,
    _voxel_cluster_key,
    _voxel_cluster_keys_vectorized,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_points() -> np.ndarray:
    """A small cloud with several distinguishable obstacle clusters.

    Layout (top view, x = forward, y = left):
        Cluster A: around (1, 0, 1.5) — 6 points, small
        Cluster B: around (3, 1, 0.8) — 8 points, medium
        Cluster C: around (5, -1, 2.0) — 12 points, larger
        Ground points: scattered near z=0
    """
    pts = np.array([
        # Cluster A (6 points)
        [1.0, 0.0, 1.5, 0.5],
        [1.1, 0.1, 1.6, 0.5],
        [0.9, -0.1, 1.4, 0.5],
        [1.0, 0.2, 1.7, 0.5],
        [1.2, -0.2, 1.5, 0.5],
        [0.8, 0.0, 1.3, 0.5],
        # Cluster B (8 points)
        [3.0, 1.0, 0.8, 0.6],
        [3.1, 1.1, 0.9, 0.6],
        [2.9, 0.9, 0.7, 0.6],
        [3.2, 1.2, 1.0, 0.6],
        [3.0, 1.3, 0.8, 0.6],
        [2.8, 1.0, 0.6, 0.6],
        [3.3, 0.8, 0.9, 0.6],
        [3.1, 1.4, 0.7, 0.6],
        # Cluster C (12 points)
        [5.0, -1.0, 2.0, 0.7],
        [5.1, -1.1, 2.1, 0.7],
        [4.9, -0.9, 1.9, 0.7],
        [5.2, -1.2, 2.2, 0.7],
        [5.0, -1.3, 2.0, 0.7],
        [4.8, -1.0, 1.8, 0.7],
        [5.3, -0.8, 2.1, 0.7],
        [5.1, -1.4, 1.9, 0.7],
        [4.7, -1.1, 2.0, 0.7],
        [5.4, -0.9, 1.8, 0.7],
        [5.0, -1.5, 2.2, 0.7],
        [4.9, -0.8, 1.7, 0.7],
        # Ground points (z < ground_threshold)
        [0.0, 0.0, 0.0, 0.1],
        [2.0, 0.0, 0.0, 0.1],
        [4.0, 0.0, 0.0, 0.1],
        [6.0, 0.0, 0.0, 0.1],
        [1.0, 2.0, 0.0, 0.1],
        [3.0, -2.0, 0.0, 0.1],
        [5.0, 2.0, 0.0, 0.1],
    ], dtype=np.float32)
    return pts


@pytest.fixture
def tiny_cloud() -> np.ndarray:
    """A cloud with fewer than cluster_min_points points above ground."""
    return np.array([
        [1.0, 0.0, 0.5, 0.5],
        [1.1, 0.1, 0.6, 0.5],
    ], dtype=np.float32)


@pytest.fixture
def empty_above_ground() -> np.ndarray:
    """All points are below the default ground threshold."""
    return np.array([
        [1.0, 0.0, 0.0, 0.5],
        [2.0, 0.0, 0.1, 0.5],
        [3.0, 0.0, 0.0, 0.5],
    ], dtype=np.float32)


@pytest.fixture
def valid_detection() -> Detection3D:
    """A well-formed Detection3D for testing."""
    return Detection3D(
        center_xyz=np.array([1.0, 2.0, 0.5], dtype=np.float32),
        size_lwh=np.array([1.5, 1.8, 2.0], dtype=np.float32),
        yaw_rad=0.0,
        class_id=0,
        class_name="vehicle",
        confidence=0.85,
        source="mock_geometric_clusterer",
        metadata={"point_count": 10},
    )


# ============================================================================
# validate_points_xyzi
# ============================================================================

class TestValidatePointsXYZI:
    """Tests for the input validation utility."""

    def test_valid_input(self, sample_points: np.ndarray) -> None:
        validate_points_xyzi(sample_points)  # Should not raise.

    def test_wrong_ndim(self) -> None:
        with pytest.raises(ValueError, match="shape.*N, 4"):
            validate_points_xyzi(np.array([1.0, 2.0, 3.0, 4.0]))

    def test_wrong_columns(self) -> None:
        with pytest.raises(ValueError, match="shape.*N, 4"):
            validate_points_xyzi(np.ones((5, 3), dtype=np.float32))

    def test_empty_array(self) -> None:
        with pytest.raises(ValueError, match="at least one point"):
            validate_points_xyzi(np.empty((0, 4), dtype=np.float32))

    def test_nan_values(self) -> None:
        pts = np.array([[0.0, 0.0, np.nan, 1.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_points_xyzi(pts)

    def test_inf_values(self) -> None:
        pts = np.array([[0.0, 0.0, np.inf, 1.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_points_xyzi(pts)


# ============================================================================
# Detection3D dataclass
# ============================================================================

class TestDetection3D:
    """Tests for the Detection3D dataclass."""

    def test_valid_creation(self, valid_detection: Detection3D) -> None:
        d = valid_detection
        assert d.center_xyz.shape == (3,)
        assert d.center_xyz.dtype == np.float32
        assert d.size_lwh.shape == (3,)
        assert d.size_lwh.dtype == np.float32
        assert 0.0 <= d.confidence <= 1.0
        assert d.size_lwh[0] > 0 and d.size_lwh[1] > 0 and d.size_lwh[2] > 0
        assert d.source == "mock_geometric_clusterer"
        assert d.metadata == {"point_count": 10}

    def test_center_xyz_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="shape.*3,"):
            Detection3D(
                center_xyz=np.array([1.0, 2.0], dtype=np.float32),
                size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                yaw_rad=0.0,
                class_id=0,
                class_name="test",
                confidence=0.5,
                source="test",
            )

    def test_center_xyz_wrong_dtype(self) -> None:
        with pytest.raises(ValueError, match="float32"):
            Detection3D(
                center_xyz=np.array([1.0, 2.0, 3.0], dtype=np.float64),
                size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                yaw_rad=0.0,
                class_id=0,
                class_name="test",
                confidence=0.5,
                source="test",
            )

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            Detection3D(
                center_xyz=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                yaw_rad=0.0,
                class_id=0,
                class_name="test",
                confidence=1.5,
                source="test",
            )

    def test_size_lwh_non_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            Detection3D(
                center_xyz=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                size_lwh=np.array([0.0, 1.0, 1.0], dtype=np.float32),
                yaw_rad=0.0,
                class_id=0,
                class_name="test",
                confidence=0.5,
                source="test",
            )

    def test_repr(self, valid_detection: Detection3D) -> None:
        r = repr(valid_detection)
        assert "vehicle" in r
        assert "0.850" in r or "0.85" in r
        assert "source" in r


# ============================================================================
# ObjectDetector abstract interface
# ============================================================================

class TestObjectDetectorInterface:
    """Verify the abstract interface is correctly defined."""

    def test_is_abstract(self) -> None:
        from abc import ABC
        assert issubclass(ObjectDetector, ABC)

    def test_has_abstract_detect(self) -> None:
        import inspect
        detect_method = ObjectDetector.detect
        assert inspect.isabstract(detect_method) or getattr(
            detect_method, "__isabstractmethod__", False
        )


# ============================================================================
# GroundTruthBoxProvider
# ============================================================================

class TestGroundTruthBoxProvider:
    """Tests for the ground-truth placeholder provider."""

    def test_raises_not_implemented(self, sample_points: np.ndarray) -> None:
        provider = GroundTruthBoxProvider()
        with pytest.raises(NotImplementedError, match="NOT available"):
            provider.detect(sample_points, timestamp_s=0.0)

    def test_error_mentions_semantickitti(self, sample_points: np.ndarray) -> None:
        provider = GroundTruthBoxProvider()
        with pytest.raises(NotImplementedError) as exc_info:
            provider.detect(sample_points)
        msg = str(exc_info.value)
        assert "SemanticKITTI" in msg
        assert "3D object detection boxes" in msg or "detection boxes" in msg

    def test_error_lists_alternatives(self, sample_points: np.ndarray) -> None:
        provider = GroundTruthBoxProvider()
        with pytest.raises(NotImplementedError) as exc_info:
            provider.detect(sample_points)
        msg = str(exc_info.value)
        for alt in ["nuScenes", "Waymo", "KITTI Detection", "CARLA"]:
            assert alt in msg, f"Expected alternative '{alt}' in error message."

    def test_source_attribute(self) -> None:
        provider = GroundTruthBoxProvider()
        assert provider._SOURCE == "ground_truth_box_provider_not_available"


# ============================================================================
# MockObjectDetector
# ============================================================================

class TestMockObjectDetector:
    """Tests for the deterministic geometric clustering detector."""

    def test_default_configuration(self) -> None:
        det = MockObjectDetector()
        assert det.ground_threshold_m == 0.25
        assert det.cluster_min_points == 5
        assert det.cluster_max_dist_m == 1.5
        assert det.voxel_size_m == 0.2
        assert det._SOURCE == "mock_geometric_clusterer"

    def test_emits_warning_on_init(self) -> None:
        with pytest.warns(UserWarning, match="NOT an AI"):
            MockObjectDetector()

    def test_emits_warning_on_detect(self, sample_points: np.ndarray) -> None:
        det = MockObjectDetector()
        with pytest.warns(UserWarning, match="GEOMETRIC"):
            det.detect(sample_points)

    def test_source_label(self, sample_points: np.ndarray) -> None:
        det = MockObjectDetector()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            detections = det.detect(sample_points)
        for d in detections:
            assert d.source == "mock_geometric_clusterer"

    def test_class_name_unknown_obstacle(self, sample_points: np.ndarray) -> None:
        det = MockObjectDetector()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            detections = det.detect(sample_points)
        for d in detections:
            assert d.class_name == "unknown_obstacle"
            assert d.class_id == 0

    def test_no_ground_points(self, empty_above_ground: np.ndarray) -> None:
        det = MockObjectDetector()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            detections = det.detect(empty_above_ground)
        assert len(detections) == 0

    def test_too_few_points(self, tiny_cloud: np.ndarray) -> None:
        det = MockObjectDetector(cluster_min_points=5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            detections = det.detect(tiny_cloud)
        assert len(detections) == 0

    def test_detections_have_valid_geometry(self, sample_points: np.ndarray) -> None:
        det = MockObjectDetector()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            detections = det.detect(sample_points)
        for d in detections:
            assert d.size_lwh[0] > 0
            assert d.size_lwh[1] > 0
            assert d.size_lwh[2] > 0
            assert np.isfinite(d.center_xyz).all()
            assert 0.0 <= d.confidence <= 1.0

    def test_confidence_determined_by_point_count(
        self, sample_points: np.ndarray
    ) -> None:
        """Larger clusters should tend to have higher confidence."""
        det = MockObjectDetector(cluster_min_points=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            detections = det.detect(sample_points)

        confs = [d.confidence for d in detections]
        sizes = [d.metadata["point_count"] for d in detections]
        # Sort by point count and verify confidence generally increases.
        if len(sizes) >= 2:
            order = np.argsort(sizes)
            sorted_confs = np.array(confs)[order]
            # Not strictly monotonic due to geometry, but the trend should
            # generally hold for our fixture.
            assert sorted_confs[-1] >= sorted_confs[0] - 0.1

    def test_deterministic(self, sample_points: np.ndarray) -> None:
        det = MockObjectDetector()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            d1 = det.detect(sample_points)
            d2 = det.detect(sample_points)
        assert len(d1) == len(d2)
        for a, b in zip(d1, d2):
            np.testing.assert_array_equal(a.center_xyz, b.center_xyz)
            np.testing.assert_array_equal(a.size_lwh, b.size_lwh)
            assert a.confidence == b.confidence
            assert a.class_name == b.class_name
            assert a.yaw_rad == b.yaw_rad

    def test_deterministic_sorting(self, sample_points: np.ndarray) -> None:
        """Detections must be sorted by confidence desc, then class_name,
        then center x, then center y."""
        det = MockObjectDetector(cluster_min_points=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            detections = det.detect(sample_points)

        for i in range(len(detections) - 1):
            a, b = detections[i], detections[i + 1]
            # Primary key: confidence descending.
            if a.confidence != b.confidence:
                assert a.confidence >= b.confidence, (
                    f"Confidence not descending: {a.confidence} vs {b.confidence}"
                )
                continue
            # Secondary key: class_name ascending.
            assert a.class_name <= b.class_name, (
                f"Class name not ascending: {a.class_name!r} vs {b.class_name!r}"
            )
            if a.class_name != b.class_name:
                continue
            # Tertiary key: center x ascending.
            assert a.center_xyz[0] <= b.center_xyz[0] + 1e-6, (
                f"Center x not ascending: {a.center_xyz[0]} vs {b.center_xyz[0]}"
            )
            if abs(a.center_xyz[0] - b.center_xyz[0]) > 1e-6:
                continue
            # Quaternary key: center y ascending.
            assert a.center_xyz[1] <= b.center_xyz[1] + 1e-6, (
                f"Center y not ascending: {a.center_xyz[1]} vs {b.center_xyz[1]}"
            )

    def test_accepts_custom_configuration(self) -> None:
        det = MockObjectDetector(
            ground_threshold_m=0.5,
            cluster_min_points=10,
            cluster_max_dist_m=2.0,
            voxel_size_m=0.3,
        )
        assert det.ground_threshold_m == 0.5
        assert det.cluster_min_points == 10
        assert det.cluster_max_dist_m == 2.0
        assert det.voxel_size_m == 0.3

    def test_rejects_invalid_ground_threshold(self) -> None:
        with pytest.raises(ValueError, match="ground_threshold_m must be >= 0"):
            MockObjectDetector(ground_threshold_m=-0.1)

    def test_rejects_invalid_cluster_min_points(self) -> None:
        with pytest.raises(ValueError, match="cluster_min_points must be >= 1"):
            MockObjectDetector(cluster_min_points=0)

    def test_rejects_invalid_cluster_max_dist(self) -> None:
        with pytest.raises(ValueError, match="cluster_max_dist_m must be > 0"):
            MockObjectDetector(cluster_max_dist_m=0.0)

    def test_rejects_invalid_voxel_size(self) -> None:
        with pytest.raises(ValueError, match="voxel_size_m must be > 0"):
            MockObjectDetector(voxel_size_m=0.0)

    def test_rejects_nan_input(self) -> None:
        det = MockObjectDetector()
        pts = np.array([[np.nan, 0.0, 1.0, 1.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="NaN or Inf"):
            det.detect(pts)

    def test_rejects_wrong_shape(self) -> None:
        det = MockObjectDetector()
        with pytest.raises(ValueError, match="shape.*N, 4"):
            det.detect(np.ones((5, 3), dtype=np.float32))

    def test_rejects_empty_input(self) -> None:
        det = MockObjectDetector()
        with pytest.raises(ValueError, match="at least one point"):
            det.detect(np.empty((0, 4), dtype=np.float32))

    def test_metadata_includes_point_count(self, sample_points: np.ndarray) -> None:
        det = MockObjectDetector(cluster_min_points=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            detections = det.detect(sample_points)
        for d in detections:
            assert "point_count" in d.metadata
            assert d.metadata["point_count"] >= det.cluster_min_points

    def test_yaw_zero_for_axis_aligned(self, sample_points: np.ndarray) -> None:
        det = MockObjectDetector(cluster_min_points=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            detections = det.detect(sample_points)
        for d in detections:
            np.testing.assert_almost_equal(d.yaw_rad, 0.0, decimal=6)

    def test_clusters_separated_by_ground(self) -> None:
        """Points separated by a ground-level gap should form separate clusters."""
        pts = np.array([
            # Cluster 1: 6 points above ground
            [0.0, 0.0, 1.0, 0.5],
            [0.1, 0.1, 1.1, 0.5],
            [-0.1, -0.1, 0.9, 0.5],
            [0.0, 0.2, 1.2, 0.5],
            [0.2, -0.2, 1.0, 0.5],
            [-0.2, 0.0, 0.8, 0.5],
            # Cluster 2: 6 points, separated by a gap
            [5.0, 0.0, 1.0, 0.5],
            [5.1, 0.1, 1.1, 0.5],
            [4.9, -0.1, 0.9, 0.5],
            [5.0, 0.2, 1.2, 0.5],
            [5.2, -0.2, 1.0, 0.5],
            [4.8, 0.0, 0.8, 0.5],
            # Ground points between them (should be filtered out)
            [2.0, 0.0, 0.0, 0.1],
            [3.0, 0.0, 0.0, 0.1],
        ], dtype=np.float32)
        det = MockObjectDetector(cluster_min_points=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            detections = det.detect(pts)
        # Should have at least 2 clusters.
        assert len(detections) >= 2
        centers = np.array([d.center_xyz for d in detections])
        # The two clusters should be far apart (gap of ~5 m).
        if len(centers) >= 2:
            dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            assert dists.min() > 1.0  # far apart


# ============================================================================
# Spec D: Detection3D field validation & geometric detector tests
# ============================================================================

class TestDetection3DFields:
    """validate Detection3D fields (spec section D)."""

    def test_center_xyz_shape_and_dtype(self) -> None:
        d = Detection3D(
            center_xyz=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            size_lwh=np.array([1.5, 1.8, 2.0], dtype=np.float32),
            yaw_rad=0.0,
            class_id=0,
            class_name="test",
            confidence=0.5,
            source="test",
        )
        assert d.center_xyz.shape == (3,)
        assert d.center_xyz.dtype == np.float32

    def test_size_lwh_shape_and_dtype(self) -> None:
        d = Detection3D(
            center_xyz=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            size_lwh=np.array([1.5, 1.8, 2.0], dtype=np.float32),
            yaw_rad=0.0,
            class_id=0,
            class_name="test",
            confidence=0.5,
            source="test",
        )
        assert d.size_lwh.shape == (3,)
        assert d.size_lwh.dtype == np.float32

    def test_confidence_bounds_valid(self) -> None:
        d = Detection3D(
            center_xyz=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            size_lwh=np.array([1.5, 1.8, 2.0], dtype=np.float32),
            yaw_rad=0.0,
            class_id=0,
            class_name="test",
            confidence=0.0,
            source="test",
        )
        assert d.confidence == 0.0
        d2 = Detection3D(
            center_xyz=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            size_lwh=np.array([1.5, 1.8, 2.0], dtype=np.float32),
            yaw_rad=0.0,
            class_id=0,
            class_name="test",
            confidence=1.0,
            source="test",
        )
        assert d2.confidence == 1.0

    def test_box_size_positivity(self) -> None:
        d = Detection3D(
            center_xyz=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            size_lwh=np.array([1.5, 1.8, 2.0], dtype=np.float32),
            yaw_rad=0.0,
            class_id=0,
            class_name="test",
            confidence=0.5,
            source="test",
        )
        assert d.size_lwh[0] > 0
        assert d.size_lwh[1] > 0
        assert d.size_lwh[2] > 0

    def test_invalid_input_rejected(self) -> None:
        """Nx4 input with NaN/Inf or wrong shape is rejected."""
        det = MockObjectDetector()
        with pytest.raises(ValueError, match="shape.*N, 4"):
            det.detect(np.ones((5, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="NaN or Inf"):
            det.detect(np.array([[np.nan, 0.0, 1.0, 1.0]], dtype=np.float32))


class TestDeterministicClusterDetection:
    """Deterministic cluster detection on synthetic point cloud (spec D)."""

    def test_deterministic_on_synthetic_cloud(self) -> None:
        """Same synthetic cloud produces identical detections."""
        rng = np.random.default_rng(42)
        n = 50
        pts = np.column_stack([
            rng.uniform(-5, 5, n),
            rng.uniform(-5, 5, n),
            rng.uniform(0.5, 2.0, n),
            rng.uniform(0.1, 1.0, n),
        ]).astype(np.float32)

        det = MockObjectDetector(cluster_min_points=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            d1 = det.detect(pts)
            d2 = det.detect(pts)

        assert len(d1) == len(d2)
        for a, b in zip(d1, d2):
            np.testing.assert_array_equal(a.center_xyz, b.center_xyz)
            np.testing.assert_array_equal(a.size_lwh, b.size_lwh)
            assert a.confidence == b.confidence

    def test_detections_sorted_deterministic(self) -> None:
        """Detections are sorted deterministically (confidence desc, then class, then x, then y)."""
        rng = np.random.default_rng(123)
        n = 40
        pts = np.column_stack([
            rng.uniform(-5, 5, n),
            rng.uniform(-5, 5, n),
            rng.uniform(0.5, 2.0, n),
            rng.uniform(0.1, 1.0, n),
        ]).astype(np.float32)

        det = MockObjectDetector(cluster_min_points=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            detections = det.detect(pts)

        # Verify sort order.
        for i in range(len(detections) - 1):
            a, b = detections[i], detections[i + 1]
            if a.confidence != b.confidence:
                assert a.confidence >= b.confidence
                continue
            assert a.class_name <= b.class_name
            if a.class_name != b.class_name:
                continue
            if abs(a.center_xyz[0] - b.center_xyz[0]) > 1e-6:
                assert a.center_xyz[0] <= b.center_xyz[0]
                continue
            assert a.center_xyz[1] <= b.center_xyz[1] + 1e-6


class TestClusterCenterStabilityUnderBorderlineMerge:
    """Regression guard for a real false-positive "moving vegetation" bug.

    Root cause (confirmed on real RELLIS-3D data): center_xyz used to be the
    AABB midpoint (min+max)/2, computed only from the two most extreme
    points in a BFS-connected cluster. For a large, diffuse real-world
    cluster (e.g. a vegetation patch), a handful of borderline points can
    determine whether a distant sub-clump bridges into the same connected
    component from one frame to the next -- when that happens the AABB
    midpoint swings by meters even though almost none of the real points
    moved, producing a large false "velocity" once tracked. Fixed by using
    the real point centroid (mean position) instead, which barely moves
    when the dominant sub-clump's own points are unchanged.
    """

    def _build_bridged_cloud(self, include_bridge: bool) -> np.ndarray:
        rng = np.random.default_rng(7)
        # Dominant, well-observed static clump: 100 points near y=0.
        group_a = np.column_stack([
            rng.uniform(-0.15, 0.15, 100),
            rng.uniform(-0.15, 0.15, 100),
            rng.uniform(0.5, 0.8, 100),
        ])
        # Small, sparse static clump 3m away: 10 points near y=3.
        group_b = np.column_stack([
            rng.uniform(-0.15, 0.15, 10),
            3.0 + rng.uniform(-0.15, 0.15, 10),
            rng.uniform(0.5, 0.8, 10),
        ])
        parts = [group_a, group_b]
        if include_bridge:
            # Two intermediate points, <=1.5m apart, that bridge group_a
            # and group_b into a single BFS-connected component (default
            # cluster_max_dist_m=1.5).
            bridge = np.array([[0.0, 1.0, 0.6], [0.0, 2.0, 0.6]])
            parts.append(bridge)
        xyz = np.vstack(parts)
        intensity = rng.uniform(0.1, 1.0, len(xyz)).reshape(-1, 1)
        return np.hstack([xyz, intensity]).astype(np.float32)

    def test_dominant_clump_centroid_barely_shifts_when_bridge_drops(self) -> None:
        """The SAME dominant real clump's reported center must stay close
        whether or not a distant sparse sub-clump happens to bridge into
        the same connected component that frame."""
        det = MockObjectDetector(cluster_min_points=5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            merged = det.detect(self._build_bridged_cloud(include_bridge=True))
            split = det.detect(self._build_bridged_cloud(include_bridge=False))

        # Merged frame: one big cluster (group_a + bridge + group_b).
        assert len(merged) == 1
        merged_center = merged[0].center_xyz

        # Split frame: two separate clusters -- identify the dominant one
        # (near y=0, matching group_a) by point count.
        assert len(split) == 2
        dominant = max(split, key=lambda d: d.metadata["point_count"])
        assert dominant.metadata["point_count"] == 100

        # The real, dominant clump's points did not move between the two
        # calls -- its reported center must stay close to y=0, not drift
        # toward the merged cluster's AABB midpoint (~y=1.5).
        assert abs(dominant.center_xyz[1]) < 0.5, (
            f"dominant clump center y={dominant.center_xyz[1]:.3f} drifted "
            "toward the bridged AABB midpoint -- centroid fix regressed."
        )
        # And the merged-frame center (dragged toward the larger group_a
        # since it now uses the point-count-weighted mean, not the
        # geometric AABB midpoint) should also sit far closer to group_a
        # than to the old AABB-midpoint value of ~1.5.
        assert merged_center[1] < 1.0

    def test_tracked_static_dominant_clump_reports_near_zero_velocity(self) -> None:
        """End-to-end regression: tracking the same real static clump across
        a borderline merge/split must not report it as moving."""
        from src.tracking.multi_object_tracker import MultiObjectTracker

        det = MockObjectDetector(cluster_min_points=5)
        tracker = MultiObjectTracker()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for frame_idx, include_bridge in enumerate([True, False, True, False, False]):
                cloud = self._build_bridged_cloud(include_bridge=include_bridge)
                detections = det.detect(cloud)
                tracks = tracker.update(detections, timestamp_s=frame_idx * 0.2)

        # Find the track that matches the dominant real clump (near y=0).
        dominant_tracks = [t for t in tracks if abs(t.state[1]) < 1.0]
        assert len(dominant_tracks) >= 1, "dominant static clump lost track association"
        track = dominant_tracks[0]
        speed = float(np.linalg.norm(track.state[3:6]))

        assert speed < 0.5, (
            f"static real clump reported speed={speed:.3f} m/s -- "
            "false-positive motion from a borderline cluster merge/split."
        )
        assert not track.dynamic


class TestPlaceholdersRaiseExpectedErrors:
    """Placeholders raise expected errors (spec D)."""

    def test_ground_truth_placeholder_raises(self, sample_points: np.ndarray) -> None:
        provider = GroundTruthBoxProvider()
        with pytest.raises(NotImplementedError, match="NOT available"):
            provider.detect(sample_points)

    def test_openpcdet_placeholder_raises(self, sample_points: np.ndarray) -> None:
        det = OpenPCDetDetector()
        with pytest.raises(NotImplementedError, match="Phase 8B"):
            det.detect(sample_points)


# ============================================================================
# OpenPCDetDetector
# ============================================================================

class TestOpenPCDetDetector:
    """Tests for the Phase 8B placeholder detector."""

    def test_raises_not_implemented(self, sample_points: np.ndarray) -> None:
        det = OpenPCDetDetector()
        with pytest.raises(NotImplementedError, match="Phase 8B"):
            det.detect(sample_points)

    def test_error_mentions_openpcdet(self, sample_points: np.ndarray) -> None:
        det = OpenPCDetDetector()
        with pytest.raises(NotImplementedError) as exc_info:
            det.detect(sample_points)
        assert "OpenPCDet" in str(exc_info.value)

    def test_error_mentions_pointpillars_or_centerpoint(
        self, sample_points: np.ndarray
    ) -> None:
        det = OpenPCDetDetector()
        with pytest.raises(NotImplementedError) as exc_info:
            det.detect(sample_points)
        msg = str(exc_info.value)
        assert "PointPillars" in msg or "CenterPoint" in msg

    def test_source_attribute(self) -> None:
        det = OpenPCDetDetector()
        assert det._SOURCE == "openpcdet_placeholder"

    def test_validates_input_before_raising(self) -> None:
        det = OpenPCDetDetector()
        # Wrong shape should raise ValueError before NotImplementedError.
        with pytest.raises(ValueError, match="shape.*N, 4"):
            det.detect(np.ones((5, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# TestSemanticClusterClassification
# ---------------------------------------------------------------------------

class TestSemanticClusterClassification:
    """Test cross-referencing geometric clusters against per-point semantic labels."""

    def test_majority_voting_assigns_correct_foveax_class(self) -> None:
        from src.perception.object_detector import classify_cluster_from_semantics

        # 10 points centered around [5.0, 5.0, 1.0]
        pts = np.zeros((10, 4), dtype=np.float32)
        pts[:, 0] = np.linspace(4.8, 5.2, 10)
        pts[:, 1] = np.linspace(4.8, 5.2, 10)
        pts[:, 2] = np.linspace(0.8, 1.2, 10)

        # 8 points labeled VEHICLE (5), 2 labeled DRIVABLE_GROUND (0)
        labels = np.array([5, 5, 5, 5, 5, 5, 5, 5, 0, 0], dtype=np.uint8)

        center = np.array([5.0, 5.0, 1.0], dtype=np.float32)
        size = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        cid, cname = classify_cluster_from_semantics(center, size, pts, labels)
        assert cid == 5
        assert cname == "VEHICLE"

    def test_pedestrian_class_assigned(self) -> None:
        from src.perception.object_detector import classify_cluster_from_semantics

        pts = np.zeros((5, 4), dtype=np.float32)
        pts[:, 0] = 2.0
        pts[:, 1] = 3.0
        pts[:, 2] = 1.0
        labels = np.full(5, 6, dtype=np.uint8)  # PEDESTRIAN = 6

        center = np.array([2.0, 3.0, 1.0], dtype=np.float32)
        size = np.array([0.4, 0.4, 1.7], dtype=np.float32)

        cid, cname = classify_cluster_from_semantics(center, size, pts, labels)
        assert cid == 6
        assert cname == "PEDESTRIAN"

    def test_unclassified_fallback_when_no_semantics(self) -> None:
        from src.perception.object_detector import classify_cluster_from_semantics

        pts = np.zeros((5, 4), dtype=np.float32)
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        size = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        cid, cname = classify_cluster_from_semantics(center, size, pts, None)
        assert cid == 7
        assert cname == "unclassified"

    def test_unclassified_fallback_when_only_unknown_points(self) -> None:
        from src.perception.object_detector import classify_cluster_from_semantics

        pts = np.zeros((5, 4), dtype=np.float32)
        labels = np.full(5, 7, dtype=np.uint8)  # UNKNOWN = 7
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        size = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        cid, cname = classify_cluster_from_semantics(center, size, pts, labels)
        assert cid == 7
        assert cname == "unclassified"

    def test_mock_detector_uses_semantic_labels_when_provided(self, sample_points: np.ndarray) -> None:
        det = MockObjectDetector()
        # Create semantic labels where points above ground are VEGETATION (2)
        labels = np.full(len(sample_points), 2, dtype=np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            detections = det.detect(sample_points, semantic_labels=labels)

        assert len(detections) > 0
        for d in detections:
            assert d.class_name == "VEGETATION"
            assert d.class_id == 2


class TestVectorizedClusteringMatchesOldBFS:
    """Regression guard for the clustering performance fix: the vectorized
    cKDTree.query_pairs + scipy.sparse.csgraph.connected_components
    implementation must produce IDENTICAL cluster membership to the
    original per-point BFS (using query_ball_point + a per-neighbor-list
    sorted() call) it replaced. This is a performance fix, not a behavior
    change -- if a future refactor silently changes clustering behavior,
    this test must catch it.
    """

    @staticmethod
    def _old_bfs_clusters(xyz: np.ndarray, cluster_max_dist_m: float) -> list[frozenset]:
        """The exact pre-fix algorithm, reimplemented here only as a
        reference oracle (not reused from production code, since the
        whole point is to catch production code silently drifting from
        this known-correct baseline)."""
        from scipy.spatial import cKDTree

        n = len(xyz)
        clusters = []
        visited = np.zeros(n, dtype=bool)
        tree = cKDTree(xyz)
        for start in range(n):
            if visited[start]:
                continue
            visited[start] = True
            cluster = []
            stack = [start]
            while stack:
                idx = stack.pop()
                cluster.append(idx)
                for nb in sorted(tree.query_ball_point(xyz[idx], cluster_max_dist_m)):
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            clusters.append(cluster)
        return [frozenset(c) for c in clusters]

    def _fixed_synthetic_cloud(self) -> np.ndarray:
        """A fixed synthetic cloud covering several clustering edge cases:
        a dense compact blob, a sparse diffuse blob, two blobs joined by a
        single borderline bridging point, and isolated singleton points."""
        rng = np.random.default_rng(99)

        dense_blob = np.column_stack([
            rng.uniform(-0.2, 0.2, 40),
            rng.uniform(-0.2, 0.2, 40),
            rng.uniform(0.5, 0.8, 40),
        ])
        sparse_blob = np.column_stack([
            10.0 + rng.uniform(-1.0, 1.0, 15),
            rng.uniform(-1.0, 1.0, 15),
            rng.uniform(0.5, 1.5, 15),
        ])
        blob_a = np.column_stack([
            20.0 + rng.uniform(-0.1, 0.1, 20),
            rng.uniform(-0.1, 0.1, 20),
            rng.uniform(0.5, 0.6, 20),
        ])
        blob_b = np.column_stack([
            21.4 + rng.uniform(-0.1, 0.1, 20),  # exactly on the border of
            rng.uniform(-0.1, 0.1, 20),          # cluster_max_dist_m=1.5
            rng.uniform(0.5, 0.6, 20),
        ])
        bridge_point = np.array([[20.7, 0.0, 0.55]])
        # Isolated tiny (non-degenerate-extent) pairs, far from everything
        # else -- not true single-point clusters, since Detection3D
        # requires strictly positive size on every axis (a pre-existing,
        # unrelated constraint) and a single point has zero extent.
        isolated_pairs = np.array([
            [40.0, 0.0, 0.50], [40.05, 0.02, 0.51],
            [45.0, 0.0, 0.50], [45.05, 0.02, 0.51],
            [50.0, 0.0, 0.50], [50.05, 0.02, 0.51],
        ])

        xyz = np.vstack([dense_blob, sparse_blob, blob_a, bridge_point, blob_b, isolated_pairs])
        intensity = rng.uniform(0.1, 1.0, len(xyz)).reshape(-1, 1)
        return np.hstack([xyz, intensity]).astype(np.float32)

    def test_full_partition_matches_old_bfs_exactly(self) -> None:
        det = MockObjectDetector(cluster_min_points=2, ground_threshold_m=0.0, voxel_size_m=0.001)
        cloud = self._fixed_synthetic_cloud()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            detections = det.detect(cloud)

        new_membership = sorted(
            (frozenset(d.metadata["cluster_indices"]) for d in detections),
            key=lambda s: min(s),
        )

        xyz = cloud[:, 0:3].astype(np.float64)
        old_membership = sorted(
            self._old_bfs_clusters(xyz, det.cluster_max_dist_m),
            key=lambda s: min(s),
        )

        assert new_membership == old_membership, (
            "Vectorized clustering produced different cluster membership "
            "than the original BFS -- this must be a behavior change, not "
            "just a performance change."
        )
        # Sanity: the fixed cloud actually exercises >1 cluster (otherwise
        # this test would trivially pass without checking anything real).
        assert len(new_membership) >= 4

    def test_cluster_count_and_sizes_match_old_bfs(self) -> None:
        det = MockObjectDetector(cluster_min_points=2, ground_threshold_m=0.0, voxel_size_m=0.001)
        cloud = self._fixed_synthetic_cloud()
        xyz = cloud[:, 0:3].astype(np.float64)

        old_clusters = self._old_bfs_clusters(xyz, det.cluster_max_dist_m)
        old_sizes = sorted(len(c) for c in old_clusters)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            detections = det.detect(cloud)
        new_sizes = sorted(len(d.metadata["cluster_indices"]) for d in detections)

        assert new_sizes == old_sizes


class TestVectorizedVoxelKeysMatchOldPerPoint:
    """Regression guard for the voxel-key vectorization: the vectorized
    whole-array computation must produce bit-identical keys to the old
    per-point Python-loop implementation for the same input.
    """

    def test_matches_old_per_point_on_fixed_synthetic_cloud(self) -> None:
        rng = np.random.default_rng(2024)
        xyz = rng.uniform(-50.0, 50.0, size=(3000, 3))
        voxel_size = 0.2

        old_keys = np.array([_voxel_cluster_key(p, voxel_size) for p in xyz])
        new_keys = _voxel_cluster_keys_vectorized(xyz, voxel_size)

        assert new_keys.shape == old_keys.shape
        np.testing.assert_array_equal(new_keys, old_keys)

    def test_matches_old_per_point_on_boundary_values(self) -> None:
        """Points sitting exactly on a voxel boundary are the case most
        likely to expose a floor-division rounding discrepancy."""
        voxel_size = 0.5
        xyz = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5],
            [0.4999999, -0.5000001, 1.0],
            [-1.5, 2.5, -3.5],
        ])
        old_keys = np.array([_voxel_cluster_key(p, voxel_size) for p in xyz])
        new_keys = _voxel_cluster_keys_vectorized(xyz, voxel_size)
        np.testing.assert_array_equal(new_keys, old_keys)

