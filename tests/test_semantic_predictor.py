"""Unit tests for FOVEAX Phase 7 — semantic_predictor module.

Run with:
    python -m pytest tests/test_semantic_predictor.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

# Ensure src/ is importable when running from project root.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perception.semantic_predictor import (
    FOVEAX_CLASSES,
    NUM_FOVEAX_CLASSES,
    SEMANTICKITTI_TO_FOVEAX,
    GroundTruthSemanticPredictor,
    MockSemanticPredictor,
    PretrainedModelPredictor,
    SemanticPrediction,
    extract_semantickitti_ids,
    remap_semantickitti_to_foveax,
    validate_points_xyzi,
    validate_prediction,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def sample_points() -> np.ndarray:
    """10 points with varied z and intensity values."""
    return np.array(
        [
            [1.0, 0.0, -2.0, 0.1],   # z < -1.5 → UNKNOWN
            [2.0, 0.0, -1.6, 0.2],   # z < -1.5 → UNKNOWN
            [3.0, 0.0, -0.5, 0.3],   # z < 0.25 → DRIVABLE_GROUND
            [4.0, 0.0,  0.0, 0.4],   # z < 0.25 → DRIVABLE_GROUND
            [5.0, 0.0,  0.2, 0.5],   # z < 0.25 → DRIVABLE_GROUND
            [6.0, 0.0,  0.5, 0.6],   # 0.25 ≤ z < 1.8 → SOLID_OBSTACLE
            [7.0, 0.0,  1.0, 0.7],   # 0.25 ≤ z < 1.8 → SOLID_OBSTACLE
            [8.0, 0.0,  1.7, 0.8],   # 0.25 ≤ z < 1.8 → SOLID_OBSTACLE
            [9.0, 0.0,  2.0, 0.9],   # z ≥ 1.8 → BUILDING_WALL
            [10., 0.0,  5.0, 1.0],   # z ≥ 1.8 → BUILDING_WALL
        ],
        dtype=np.float32,
    )


@pytest.fixture
def sample_labels_raw() -> np.ndarray:
    """10 raw SemanticKITTI labels (semantic + instance packed)."""
    # semantic_id in lower 16 bits, instance_id in upper 16 bits.
    # semantic IDs: 40(road), 44(parking), 70(vegetation), 50(building),
    # 10(car), 30(person), 80(pole), 0(unlabeled), 48(sidewalk), 72(terrain)
    semantic = np.array([40, 44, 70, 50, 10, 30, 80, 0, 48, 72], dtype=np.uint32)
    instance = np.array([0, 0, 1, 2, 3, 4, 0, 0, 0, 0], dtype=np.uint32)
    return semantic | (instance << 16)


# =========================================================================
# Input validation tests
# =========================================================================

class TestValidatePointsXYZI:
    """Tests for validate_points_xyzi."""

    def test_valid_input(self, sample_points: np.ndarray) -> None:
        validate_points_xyzi(sample_points)   # Should not raise.

    def test_wrong_ndim(self) -> None:
        with pytest.raises(ValueError, match="shape.*N, 4"):
            validate_points_xyzi(np.array([1.0, 2.0, 3.0, 4.0]))

    def test_wrong_columns(self) -> None:
        with pytest.raises(ValueError, match="shape.*N, 4"):
            validate_points_xyzi(np.ones((5, 3), dtype=np.float32))

    def test_empty_array(self) -> None:
        with pytest.raises(ValueError, match="at least one point"):
            validate_points_xyzi(np.empty((0, 4), dtype=np.float32))


# =========================================================================
# SemanticKITTI label extraction tests
# =========================================================================

class TestExtractSemanticKITTIIds:
    """Tests for extract_semantickitti_ids (lower-16-bit extraction)."""

    def test_lower_16_bits(self, sample_labels_raw: np.ndarray) -> None:
        sem, inst = extract_semantickitti_ids(sample_labels_raw)
        expected_sem = np.array([40, 44, 70, 50, 10, 30, 80, 0, 48, 72],
                                dtype=np.uint32)
        np.testing.assert_array_equal(sem, expected_sem)

    def test_upper_16_bits(self, sample_labels_raw: np.ndarray) -> None:
        sem, inst = extract_semantickitti_ids(sample_labels_raw)
        expected_inst = np.array([0, 0, 1, 2, 3, 4, 0, 0, 0, 0],
                                 dtype=np.uint32)
        np.testing.assert_array_equal(inst, expected_inst)

    def test_max_values(self) -> None:
        """Semantic = 0xFFFF, instance = 0xFFFF packed together."""
        raw = np.array([0xFFFF_FFFF], dtype=np.uint32)
        sem, inst = extract_semantickitti_ids(raw)
        assert sem[0] == 0xFFFF
        assert inst[0] == 0xFFFF


# =========================================================================
# Class mapping tests
# =========================================================================

class TestRemapSemanticKITTI:
    """Tests for remap_semantickitti_to_foveax."""

    def test_road_maps_to_drivable_ground(self) -> None:
        result = remap_semantickitti_to_foveax(np.array([40], dtype=np.uint32))
        assert result[0] == 0  # DRIVABLE_GROUND

    def test_lane_marking_maps_to_drivable_ground(self) -> None:
        result = remap_semantickitti_to_foveax(np.array([60], dtype=np.uint32))
        assert result[0] == 0  # DRIVABLE_GROUND (corrected)

    def test_vegetation_maps_to_vegetation(self) -> None:
        result = remap_semantickitti_to_foveax(np.array([70], dtype=np.uint32))
        assert result[0] == 2  # VEGETATION

    def test_car_maps_to_vehicle(self) -> None:
        result = remap_semantickitti_to_foveax(np.array([10], dtype=np.uint32))
        assert result[0] == 5  # VEHICLE

    def test_person_maps_to_pedestrian(self) -> None:
        result = remap_semantickitti_to_foveax(np.array([30], dtype=np.uint32))
        assert result[0] == 6  # PEDESTRIAN

    def test_unlabeled_maps_to_unknown(self) -> None:
        result = remap_semantickitti_to_foveax(np.array([0], dtype=np.uint32))
        assert result[0] == 7  # UNKNOWN

    def test_unmapped_id_defaults_to_unknown(self) -> None:
        result = remap_semantickitti_to_foveax(np.array([999], dtype=np.uint32))
        assert result[0] == 7  # UNKNOWN

    def test_all_known_mappings(self) -> None:
        """Every key in SEMANTICKITTI_TO_FOVEAX produces a valid FOVEAX id."""
        for raw_id, expected_foveax in SEMANTICKITTI_TO_FOVEAX.items():
            result = remap_semantickitti_to_foveax(
                np.array([raw_id], dtype=np.uint32)
            )
            assert result[0] == expected_foveax, (
                f"raw_id={raw_id}: expected {expected_foveax}, got {result[0]}"
            )

    def test_batch_mapping(self, sample_labels_raw: np.ndarray) -> None:
        sem, _ = extract_semantickitti_ids(sample_labels_raw)
        mapped = remap_semantickitti_to_foveax(sem)
        assert len(mapped) == len(sem)
        assert mapped.dtype == np.uint8
        # All mapped values should be valid FOVEAX class IDs.
        assert np.all(mapped < NUM_FOVEAX_CLASSES)


# =========================================================================
# Mock predictor tests
# =========================================================================

class TestMockSemanticPredictor:
    """Tests for MockSemanticPredictor."""

    def test_output_length(self, sample_points: np.ndarray) -> None:
        pred = MockSemanticPredictor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = pred.predict(sample_points)
        assert len(result.class_ids) == len(sample_points)
        assert len(result.confidence) == len(sample_points)
        assert len(result.uncertainty) == len(sample_points)

    def test_source_label(self, sample_points: np.ndarray) -> None:
        pred = MockSemanticPredictor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = pred.predict(sample_points)
        assert result.source == "mock_geometry_baseline"

    def test_confidence_bounds(self, sample_points: np.ndarray) -> None:
        pred = MockSemanticPredictor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = pred.predict(sample_points)
        assert result.confidence.min() >= 0.0
        assert result.confidence.max() <= 1.0

    def test_uncertainty_bounds(self, sample_points: np.ndarray) -> None:
        pred = MockSemanticPredictor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = pred.predict(sample_points)
        assert result.uncertainty.min() >= 0.0
        assert result.uncertainty.max() <= 1.0

    def test_confidence_plus_uncertainty(self, sample_points: np.ndarray) -> None:
        pred = MockSemanticPredictor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = pred.predict(sample_points)
        total = result.confidence + result.uncertainty
        np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_deterministic(self, sample_points: np.ndarray) -> None:
        """Same input must produce identical output every time."""
        pred = MockSemanticPredictor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r1 = pred.predict(sample_points)
            r2 = pred.predict(sample_points)
        np.testing.assert_array_equal(r1.class_ids, r2.class_ids)
        np.testing.assert_array_equal(r1.confidence, r2.confidence)

    def test_height_rules(self, sample_points: np.ndarray) -> None:
        """Verify the basic height-based classification rules."""
        pred = MockSemanticPredictor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = pred.predict(sample_points)
        ids = result.class_ids
        # z < -1.5 → UNKNOWN (7)
        assert ids[0] == 7   # z = -2.0
        assert ids[1] == 7   # z = -1.6
        # z < 0.25 → DRIVABLE_GROUND (0)
        assert ids[2] == 0   # z = -0.5
        assert ids[3] == 0   # z =  0.0
        assert ids[4] == 0   # z =  0.2
        # 0.25 ≤ z < 1.8 → SOLID_OBSTACLE (4)
        assert ids[5] == 4   # z =  0.5
        assert ids[6] == 4   # z =  1.0
        assert ids[7] == 4   # z =  1.7
        # z ≥ 1.8 → BUILDING_WALL (3)
        assert ids[8] == 3   # z =  2.0
        assert ids[9] == 3   # z =  5.0

    def test_emits_warning(self, sample_points: np.ndarray) -> None:
        pred = MockSemanticPredictor()
        with pytest.warns(UserWarning, match="NOT an AI model"):
            pred.predict(sample_points)

    def test_rejects_wrong_shape(self) -> None:
        pred = MockSemanticPredictor()
        with pytest.raises(ValueError):
            pred.predict(np.ones((5, 3), dtype=np.float32))


# =========================================================================
# Ground-truth predictor tests
# =========================================================================

class TestGroundTruthSemanticPredictor:
    """Tests for GroundTruthSemanticPredictor."""

    def test_output_length(
        self,
        sample_points: np.ndarray,
        sample_labels_raw: np.ndarray,
    ) -> None:
        pred = GroundTruthSemanticPredictor(sample_labels_raw)
        result = pred.predict(sample_points)
        assert len(result.class_ids) == len(sample_points)

    def test_source_label(
        self,
        sample_points: np.ndarray,
        sample_labels_raw: np.ndarray,
    ) -> None:
        pred = GroundTruthSemanticPredictor(sample_labels_raw)
        result = pred.predict(sample_points)
        assert result.source == "semantic_kitti_ground_truth"

    def test_known_labels_have_full_confidence(
        self,
        sample_points: np.ndarray,
        sample_labels_raw: np.ndarray,
    ) -> None:
        pred = GroundTruthSemanticPredictor(sample_labels_raw)
        result = pred.predict(sample_points)
        known_mask = result.class_ids != 7
        np.testing.assert_array_equal(result.confidence[known_mask], 1.0)

    def test_unknown_labels_have_zero_confidence(
        self,
        sample_points: np.ndarray,
        sample_labels_raw: np.ndarray,
    ) -> None:
        pred = GroundTruthSemanticPredictor(sample_labels_raw)
        result = pred.predict(sample_points)
        unknown_mask = result.class_ids == 7
        np.testing.assert_array_equal(result.confidence[unknown_mask], 0.0)

    def test_confidence_plus_uncertainty(
        self,
        sample_points: np.ndarray,
        sample_labels_raw: np.ndarray,
    ) -> None:
        pred = GroundTruthSemanticPredictor(sample_labels_raw)
        result = pred.predict(sample_points)
        np.testing.assert_allclose(
            result.confidence + result.uncertainty, 1.0, atol=1e-6
        )

    def test_point_label_mismatch_raises(self) -> None:
        points = np.ones((5, 4), dtype=np.float32)
        labels = np.zeros(10, dtype=np.uint32)
        pred = GroundTruthSemanticPredictor(labels)
        with pytest.raises(ValueError, match="mismatch"):
            pred.predict(points)

    def test_requires_exactly_one_input(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            GroundTruthSemanticPredictor()

    def test_rejects_both_inputs(self, sample_labels_raw: np.ndarray) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            GroundTruthSemanticPredictor(
                sample_labels_raw,
                foveax_class_ids=np.zeros(len(sample_labels_raw), dtype=np.uint8),
            )


class TestGroundTruthSemanticPredictorPrecomputed:
    """Tests for the foveax_class_ids input path (used by rellis3d_loader)."""

    def test_output_length(self, sample_points: np.ndarray) -> None:
        class_ids = np.zeros(len(sample_points), dtype=np.uint8)
        pred = GroundTruthSemanticPredictor(foveax_class_ids=class_ids)
        result = pred.predict(sample_points)
        assert len(result.class_ids) == len(sample_points)

    def test_source_label(self, sample_points: np.ndarray) -> None:
        class_ids = np.zeros(len(sample_points), dtype=np.uint8)
        pred = GroundTruthSemanticPredictor(foveax_class_ids=class_ids)
        result = pred.predict(sample_points)
        assert result.source == "rellis3d_ground_truth"

    def test_class_ids_passed_through_unchanged(
        self, sample_points: np.ndarray
    ) -> None:
        class_ids = np.arange(len(sample_points), dtype=np.uint8) % NUM_FOVEAX_CLASSES
        pred = GroundTruthSemanticPredictor(foveax_class_ids=class_ids)
        result = pred.predict(sample_points)
        np.testing.assert_array_equal(result.class_ids, class_ids)

    def test_unknown_class_has_zero_confidence(
        self, sample_points: np.ndarray
    ) -> None:
        class_ids = np.full(len(sample_points), 7, dtype=np.uint8)  # UNKNOWN
        pred = GroundTruthSemanticPredictor(foveax_class_ids=class_ids)
        result = pred.predict(sample_points)
        np.testing.assert_array_equal(result.confidence, 0.0)

    def test_point_label_mismatch_raises(self) -> None:
        points = np.ones((5, 4), dtype=np.float32)
        class_ids = np.zeros(10, dtype=np.uint8)
        pred = GroundTruthSemanticPredictor(foveax_class_ids=class_ids)
        with pytest.raises(ValueError, match="mismatch"):
            pred.predict(points)


# =========================================================================
# Pretrained model placeholder test
# =========================================================================

class TestPretrainedModelPredictor:
    """Tests for PretrainedModelPredictor."""

    def test_raises_not_implemented(self, sample_points: np.ndarray) -> None:
        pred = PretrainedModelPredictor()
        with pytest.raises(NotImplementedError, match="SalsaNext"):
            pred.predict(sample_points)


# =========================================================================
# Validate prediction utility
# =========================================================================

class TestValidatePrediction:
    """Tests for validate_prediction."""

    def test_valid_prediction(self) -> None:
        n = 5
        pred = SemanticPrediction(
            class_ids=np.zeros(n, dtype=np.uint8),
            confidence=np.ones(n, dtype=np.float32),
            uncertainty=np.zeros(n, dtype=np.float32),
            source="test",
        )
        validate_prediction(pred, n)  # Should not raise.

    def test_wrong_class_ids_length(self) -> None:
        pred = SemanticPrediction(
            class_ids=np.zeros(3, dtype=np.uint8),
            confidence=np.ones(5, dtype=np.float32),
            uncertainty=np.zeros(5, dtype=np.float32),
            source="test",
        )
        with pytest.raises(ValueError, match="class_ids"):
            validate_prediction(pred, 5)

    def test_confidence_out_of_range(self) -> None:
        n = 3
        pred = SemanticPrediction(
            class_ids=np.zeros(n, dtype=np.uint8),
            confidence=np.array([0.5, 1.5, 0.3], dtype=np.float32),
            uncertainty=np.zeros(n, dtype=np.float32),
            source="test",
        )
        with pytest.raises(ValueError, match="confidence"):
            validate_prediction(pred, n)

    def test_uncertainty_out_of_range(self) -> None:
        n = 3
        pred = SemanticPrediction(
            class_ids=np.zeros(n, dtype=np.uint8),
            confidence=np.ones(n, dtype=np.float32),
            uncertainty=np.array([-0.1, 0.5, 0.3], dtype=np.float32),
            source="test",
        )
        with pytest.raises(ValueError, match="uncertainty"):
            validate_prediction(pred, n)
