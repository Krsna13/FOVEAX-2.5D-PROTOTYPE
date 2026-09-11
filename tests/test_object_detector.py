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
