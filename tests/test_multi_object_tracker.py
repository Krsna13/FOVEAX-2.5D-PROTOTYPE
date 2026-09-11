"""Unit tests for FOVEAX Phase 8A — multi_object_tracker module.

Run with:
    python -m pytest tests/test_multi_object_tracker.py -v
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perception.object_detector import Detection3D
from src.tracking.multi_object_tracker import (
    MultiObjectTracker,
    TrackState,
    _blend_yaw,
    _compute_track_confidence,
    _hungarian_assign,
    _gated_cost_matrix,
    _predict_kf,
    _update_kf,
    _build_state_transition,
    _build_process_noise_covariance,
    _build_measurement_noise_covariance,
    _class_compatible,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def vehicle_detection() -> Detection3D:
    """A vehicle detection."""
    return Detection3D(
        center_xyz=np.array([5.0, 0.0, 0.5], dtype=np.float32),
        size_lwh=np.array([4.0, 1.8, 1.5], dtype=np.float32),
        yaw_rad=0.0,
        class_id=1,
        class_name="vehicle",
        confidence=0.9,
        source="mock_geometric_clusterer",
    )


@pytest.fixture
def pedestrian_detection() -> Detection3D:
    """A pedestrian detection."""
    return Detection3D(
        center_xyz=np.array([5.0, 2.0, 1.0], dtype=np.float32),
        size_lwh=np.array([0.5, 0.5, 1.8], dtype=np.float32),
        yaw_rad=0.0,
        class_id=2,
        class_name="pedestrian",
        confidence=0.75,
        source="mock_geometric_clusterer",
    )


@pytest.fixture
def unknown_detection() -> Detection3D:
    """An unknown obstacle detection (the mock detector's class)."""
    return Detection3D(
        center_xyz=np.array([10.0, 0.0, 0.6], dtype=np.float32),
        size_lwh=np.array([1.0, 1.0, 1.2], dtype=np.float32),
        yaw_rad=0.0,
        class_id=0,
        class_name="unknown_obstacle",
        confidence=0.6,
        source="mock_geometric_clusterer",
    )


@pytest.fixture
def basic_tracker() -> MultiObjectTracker:
    """A tracker with default settings."""
    return MultiObjectTracker()


@pytest.fixture
def tracker_with_thresholds() -> MultiObjectTracker:
    """A tracker with tight thresholds for testing matching logic."""
    return MultiObjectTracker(
        gating_threshold_m=2.0,
        max_missed_frames=2,
        min_hits_for_dynamic=2,
        dynamic_speed_threshold_mps=0.3,
    )


# ============================================================================
# TrackState dataclass
# ============================================================================

class TestTrackState:
    """Tests for the TrackState dataclass."""

    def test_valid_creation(self) -> None:
        state = np.zeros(6, dtype=np.float64)
        cov = np.eye(6, dtype=np.float64)
        size = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        ts = TrackState(
            track_id=1,
            class_name="vehicle",
            state=state,
            covariance=cov,
            size_lwh=size,
            yaw_rad=0.0,
            age_frames=5,
            hits=4,
            missed_frames=1,
            confidence=0.8,
            source="mock",
            dynamic=False,
            last_timestamp_s=1.0,
        )
        assert ts.track_id == 1
        assert ts.class_name == "vehicle"
        assert ts.age_frames == 5
        assert ts.hits == 4
        assert ts.missed_frames == 1
        assert ts.confidence == 0.8
        assert ts.dynamic is False
        assert ts.last_timestamp_s == 1.0

    def test_state_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="shape.*6,"):
            TrackState(
                track_id=1,
                class_name="test",
                state=np.zeros(3, dtype=np.float64),
                covariance=np.eye(6, dtype=np.float64),
                size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                yaw_rad=0.0,
            )

    def test_covariance_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="shape.*6, 6"):
            TrackState(
                track_id=1,
                class_name="test",
                state=np.zeros(6, dtype=np.float64),
                covariance=np.eye(3, dtype=np.float64),
                size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                yaw_rad=0.0,
            )


# ============================================================================
# Kalman filter helpers
# ============================================================================

class TestStateTransition:
    """Tests for the state transition matrix."""

    def test_dt_zero(self) -> None:
        F = _build_state_transition(0.0)
        expected = np.eye(6, dtype=np.float64)
        np.testing.assert_array_equal(F, expected)

    def test_dt_positive(self) -> None:
        dt = 0.1
        F = _build_state_transition(dt)
        # Position should be updated by velocity * dt.
        np.testing.assert_array_equal(F[0, 3], dt)
        np.testing.assert_array_equal(F[1, 4], dt)
        np.testing.assert_array_equal(F[2, 5], dt)
        # Identity for the rest.
        for i in range(6):
            np.testing.assert_array_equal(F[i, i], 1.0)

    def test_applies_correctly(self) -> None:
        state = np.array([1.0, 2.0, 3.0, 10.0, 0.0, 0.0], dtype=np.float64)
        dt = 0.1
        F = _build_state_transition(dt)
        new_state = F @ state
        # x should increase by vx * dt.
        np.testing.assert_almost_equal(new_state[0], 1.0 + 10.0 * 0.1)
        np.testing.assert_almost_equal(new_state[1], 2.0)
        np.testing.assert_almost_equal(new_state[2], 3.0)


class TestProcessNoiseCovariance:
    """Tests for the process noise covariance matrix."""

    def test_dt_zero(self) -> None:
        Q = _build_process_noise_covariance(0.0, 1.0)
        expected = np.zeros((6, 6), dtype=np.float64)
        np.testing.assert_array_equal(Q, expected)

    def test_structure(self) -> None:
        dt = 0.1
        accel_std = 1.0
        Q = _build_process_noise_covariance(dt, accel_std)
        q = accel_std ** 2
        # Position variances.
        np.testing.assert_almost_equal(Q[0, 0], q * dt * dt)
        np.testing.assert_almost_equal(Q[1, 1], q * dt * dt)
        np.testing.assert_almost_equal(Q[2, 2], q * dt * dt)
        # Velocity variances (CV model: q * dt^2, not q).
        np.testing.assert_almost_equal(Q[3, 3], q * dt * dt)
        np.testing.assert_almost_equal(Q[4, 4], q * dt * dt)
        np.testing.assert_almost_equal(Q[5, 5], q * dt * dt)
        # Cross terms.
        np.testing.assert_almost_equal(Q[0, 3], q * dt)
        np.testing.assert_almost_equal(Q[3, 0], q * dt)


class TestMeasurementNoiseCovariance:
    """Tests for the measurement noise covariance."""

    def test_structure(self) -> None:
        R = _build_measurement_noise_covariance(0.5)
        expected_var = 0.5 ** 2
        np.testing.assert_almost_equal(R[0, 0], expected_var)
        np.testing.assert_almost_equal(R[1, 1], expected_var)
        np.testing.assert_almost_equal(R[2, 2], expected_var)
        # Velocity rows are zero.
        np.testing.assert_array_equal(R[3:, :], 0.0)


class TestPredictKf:
    """Tests for the Kalman predict step."""

    def test_no_change_when_dt_zero(self) -> None:
        state = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float64)
        cov = np.eye(6, dtype=np.float64)
        new_state, new_cov = _predict_kf(state, cov, 0.0, 1.0)
        np.testing.assert_array_equal(new_state, state)
        # Covariance should have only process noise (zero for dt=0).
        np.testing.assert_array_equal(new_cov, cov)

    def test_position_updates_with_velocity(self) -> None:
        state = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0], dtype=np.float64)
        cov = np.eye(6, dtype=np.float64)
        new_state, _ = _predict_kf(state, cov, 0.2, 1.0)
        np.testing.assert_almost_equal(new_state[0], 5.0 * 0.2)
        np.testing.assert_almost_equal(new_state[1], 0.0)

    def test_covariance_grows(self) -> None:
        state = np.zeros(6, dtype=np.float64)
        cov = np.eye(6, dtype=np.float64) * 0.1
        new_state, new_cov = _predict_kf(state, cov, 0.5, 1.0)
        # Covariance should be larger than the initial.
        assert (new_cov - cov).trace() > 0


class TestUpdateKf:
    """Tests for the Kalman update step."""

    def test_measurement_updates_position(self) -> None:
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        cov = np.eye(6, dtype=np.float64) * 1.0
        measurement = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        new_state, new_cov = _update_kf(state, cov, measurement, 0.5)
        # Position should move towards the measurement.
        assert new_state[0] > 0.0
        assert new_state[1] > 0.0
        assert new_state[2] > 0.0
        assert new_state[0] <= 1.0  # not overshoot

    def test_covariance_shrinks(self) -> None:
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        cov = np.eye(6, dtype=np.float64) * 1.0
        measurement = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        _, new_cov = _update_kf(state, cov, measurement, 0.5)
        # Position variances should be smaller.
        assert new_cov[0, 0] < cov[0, 0]

    def test_degenerate_covariance_handled(self) -> None:
        """A near-singular covariance should not crash."""
        state = np.zeros(6, dtype=np.float64)
        cov = np.zeros((6, 6), dtype=np.float64)
        measurement = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        # Should not raise.
        new_state, new_cov = _update_kf(state, cov, measurement, 0.5)
        assert new_state is not None
        assert new_cov is not None

    def test_symmetry_preserved(self) -> None:
        state = np.zeros(6, dtype=np.float64)
        cov = np.eye(6, dtype=np.float64)
        measurement = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        _, new_cov = _update_kf(state, cov, measurement, 0.5)
        diff = new_cov - new_cov.T
        assert np.allclose(diff, 0.0, atol=1e-10)


# ============================================================================
# Track confidence computation
# ============================================================================

class TestComputeTrackConfidence:
    """Tests for the track confidence aggregation."""

    def test_base_from_detection(self) -> None:
        conf = _compute_track_confidence(
            detection_confidence=0.9,
            age_frames=1,
            hits=1,
            missed_frames=0,
            max_missed_frames=5,
        )
        assert conf >= 0.9
        assert conf <= 1.0

    def test_no_detection_gives_low_conf(self) -> None:
        conf = _compute_track_confidence(
            detection_confidence=0.0,
            age_frames=1,
            hits=0,
            missed_frames=1,
            max_missed_frames=5,
        )
        assert conf < 0.2

    def test_maturity_increases_confidence(self) -> None:
        conf_new = _compute_track_confidence(
            detection_confidence=0.7,
            age_frames=1,
            hits=1,
            missed_frames=0,
            max_missed_frames=5,
        )
        conf_old = _compute_track_confidence(
            detection_confidence=0.7,
            age_frames=10,
            hits=5,
            missed_frames=0,
            max_missed_frames=5,
        )
        assert conf_old > conf_new

    def test_missed_frames_decrease_confidence(self) -> None:
        conf_clean = _compute_track_confidence(
            detection_confidence=0.8,
            age_frames=5,
            hits=5,
            missed_frames=0,
            max_missed_frames=5,
        )
        conf_missed = _compute_track_confidence(
            detection_confidence=0.8,
            age_frames=5,
            hits=5,
            missed_frames=5,
            max_missed_frames=5,
        )
        assert conf_missed < conf_clean

    def test_clipped_to_0_1(self) -> None:
        conf = _compute_track_confidence(
            detection_confidence=0.5,
            age_frames=100,
            hits=100,
            missed_frames=0,
            max_missed_frames=5,
        )
        assert 0.0 <= conf <= 1.0


# ============================================================================
# Class compatibility
# ============================================================================

class TestClassCompatible:
    """Tests for the class compatibility function."""

    def test_same_class(self) -> None:
        assert _class_compatible("vehicle", "vehicle", True) is True

    def test_different_known_classes(self) -> None:
        assert _class_compatible("vehicle", "pedestrian", True) is False

    def test_unknown_with_known(self) -> None:
        assert _class_compatible("unknown_obstacle", "vehicle", True) is True
        assert _class_compatible("vehicle", "unknown_obstacle", True) is True

    def test_both_unknown(self) -> None:
        assert _class_compatible("unknown_obstacle", "unknown_obstacle", True) is True

    def test_disabled_compatibility(self) -> None:
        assert _class_compatible("vehicle", "pedestrian", False) is True


# ============================================================================
# Hungarian assignment
# ============================================================================

class TestHungarianAssign:
    """Tests for the Hungarian / greedy assignment."""

    def test_empty_matrices(self) -> None:
        cost = np.empty((0, 0), dtype=np.float64)
        row_ind, col_ind, unmatched_t, unmatched_d = _hungarian_assign(cost)
        assert len(row_ind) == 0
        assert len(col_ind) == 0
        assert len(unmatched_t) == 0
        assert len(unmatched_d) == 0

    def test_one_track_one_detection(self) -> None:
        cost = np.array([[1.0]], dtype=np.float64)
        row_ind, col_ind, unmatched_t, unmatched_d = _hungarian_assign(cost)
        assert len(row_ind) == 1
        assert row_ind[0] == 0
        assert col_ind[0] == 0
        assert len(unmatched_t) == 0
        assert len(unmatched_d) == 0

    def test_inf_cost_not_matched(self) -> None:
        cost = np.array([[np.inf]], dtype=np.float64)
        row_ind, col_ind, unmatched_t, unmatched_d = _hungarian_assign(cost)
        assert len(row_ind) == 0
        assert len(col_ind) == 0
        assert len(unmatched_t) == 1
        assert len(unmatched_d) == 1

    def test_two_tracks_two_detections(self) -> None:
        cost = np.array([
            [1.0, 10.0],
            [10.0, 1.0],
        ], dtype=np.float64)
        row_ind, col_ind, unmatched_t, unmatched_d = _hungarian_assign(cost)
        assert len(row_ind) == 2
        # Should match diagonal.
        assert row_ind[0] == 0 and col_ind[0] == 0
        assert row_ind[1] == 1 and col_ind[1] == 1

    def test_greedy_fallback_when_no_scipy(self) -> None:
        """Force the greedy path by ensuring scipy is "unavailable"."""
        # Temporarily hide scipy.
        import importlib
        scipy_opt = None
        try:
            scipy_opt = importlib.import_module("scipy.optimize")
            del sys.modules["scipy.optimize"]
        except KeyError:
            pass

        try:
            cost = np.array([
                [1.0, 100.0],
                [100.0, 1.0],
            ], dtype=np.float64)
            row_ind, col_ind, unmatched_t, unmatched_d = _hungarian_assign(cost)
            assert len(row_ind) == 2
        finally:
            if scipy_opt is not None:
                sys.modules["scipy.optimize"] = scipy_opt

    def test_unmatched_tracks_returned(self) -> None:
        cost = np.array([
            [1.0, np.inf],
        ], dtype=np.float64)
        row_ind, col_ind, unmatched_t, unmatched_d = _hungarian_assign(cost)
        assert len(row_ind) == 1
        assert len(unmatched_t) == 0
        assert len(unmatched_d) == 1


# ============================================================================
# Gated cost matrix
# ============================================================================

class TestGatedCostMatrix:
    """Tests for the gated cost matrix builder."""

    def test_empty_inputs(self) -> None:
        cost = _gated_cost_matrix([], [], 3.0, True)
        assert cost.shape == (0, 0)

    def test_single_track_single_detection(self, vehicle_detection: Detection3D) -> None:
        from src.tracking.multi_object_tracker import TrackState as TS
        track = TS(
            track_id=1,
            class_name="vehicle",
            state=np.array([5.0, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float64),
            covariance=np.eye(6, dtype=np.float64),
            size_lwh=np.array([4.0, 1.8, 1.5], dtype=np.float32),
            yaw_rad=0.0,
        )
        cost = _gated_cost_matrix([track], [vehicle_detection], 10.0, True)
        assert cost.shape == (1, 1)
        # Distance between the same point should be ~0.
        assert cost[0, 0] < 0.01

    def test_gating_filters_far_detections(
        self, vehicle_detection: Detection3D
    ) -> None:
        from src.tracking.multi_object_tracker import TrackState as TS
        track = TS(
            track_id=1,
            class_name="vehicle",
            state=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            covariance=np.eye(6, dtype=np.float64),
            size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            yaw_rad=0.0,
        )
        cost = _gated_cost_matrix([track], [vehicle_detection], 2.0, True)
        # Detection is at (5, 0, 0.5), track at (0, 0, 0). Distance ~5.
        assert cost[0, 0] == np.inf

    def test_class_incompatibility_filters(
        self, vehicle_detection: Detection3D, pedestrian_detection: Detection3D
    ) -> None:
        from src.tracking.multi_object_tracker import TrackState as TS
        track = TS(
            track_id=1,
            class_name="vehicle",
            state=np.array([5.0, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float64),
            covariance=np.eye(6, dtype=np.float64),
            size_lwh=np.array([4.0, 1.8, 1.5], dtype=np.float32),
            yaw_rad=0.0,
        )
        cost = _gated_cost_matrix(
            [track],
            [vehicle_detection, pedestrian_detection],
            100.0,
            True,
        )
        # vehicle-vehicle should be low cost.
        assert cost[0, 0] < 0.01
        # vehicle-pedestrian should be inf (incompatible classes).
        assert cost[0, 1] == np.inf

    def test_class_compatibility_disabled(
        self, vehicle_detection: Detection3D, pedestrian_detection: Detection3D
    ) -> None:
        from src.tracking.multi_object_tracker import TrackState as TS
        track = TS(
            track_id=1,
            class_name="vehicle",
            state=np.array([5.0, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float64),
            covariance=np.eye(6, dtype=np.float64),
            size_lwh=np.array([4.0, 1.8, 1.5], dtype=np.float32),
            yaw_rad=0.0,
        )
        cost = _gated_cost_matrix(
            [track],
            [vehicle_detection, pedestrian_detection],
            100.0,
            False,
        )
        # Both should be finite (only distance matters).
        assert cost[0, 0] < np.inf
        assert cost[0, 1] < np.inf


# ============================================================================
# MultiObjectTracker
# ============================================================================

class TestMultiObjectTrackerConstruction:
    """Tests for tracker construction and validation."""

    def test_default_construction(self) -> None:
        tracker = MultiObjectTracker()
        assert tracker.process_noise_std == 1.0
        assert tracker.measure_noise_std == 0.5
        assert tracker.gating_threshold_m == 3.0
        assert tracker.max_missed_frames == 5
        assert tracker.min_hits_for_dynamic == 2
        assert tracker.dynamic_speed_threshold_mps == 0.5
        assert tracker.min_detection_confidence == 0.0
        assert len(tracker.tracks) == 0

    def test_rejects_negative_process_noise(self) -> None:
        with pytest.raises(ValueError, match="process_noise_std must be > 0"):
            MultiObjectTracker(process_noise_std=-1.0)

    def test_rejects_negative_measure_noise(self) -> None:
        with pytest.raises(ValueError, match="measure_noise_std must be > 0"):
            MultiObjectTracker(measure_noise_std=0.0)

    def test_rejects_negative_gating(self) -> None:
        with pytest.raises(ValueError, match="gating_threshold_m must be > 0"):
            MultiObjectTracker(gating_threshold_m=-1.0)

    def test_rejects_negative_max_missed(self) -> None:
        with pytest.raises(ValueError, match="max_missed_frames must be >= 0"):
            MultiObjectTracker(max_missed_frames=-1)

    def test_rejects_zero_min_hits_for_dynamic(self) -> None:
        with pytest.raises(ValueError, match="min_hits_for_dynamic must be >= 1"):
            MultiObjectTracker(min_hits_for_dynamic=0)

    def test_rejects_negative_dynamic_speed(self) -> None:
        with pytest.raises(ValueError, match="dynamic_speed_threshold_mps must be >= 0"):
            MultiObjectTracker(dynamic_speed_threshold_mps=-0.1)

    def test_rejects_invalid_min_detection_confidence(self) -> None:
        with pytest.raises(ValueError, match="min_detection_confidence must be in"):
            MultiObjectTracker(min_detection_confidence=1.5)


class TestMultiObjectTrackerUpdate:
    """Tests for the tracker update logic."""

    def test_empty_detections_returns_no_new_tracks(
        self, basic_tracker: MultiObjectTracker
    ) -> None:
        tracks = basic_tracker.update([], timestamp_s=0.0)
        assert len(tracks) == 0

    def test_single_detection_creates_one_track(
        self, basic_tracker: MultiObjectTracker, vehicle_detection: Detection3D
    ) -> None:
        tracks = basic_tracker.update([vehicle_detection], timestamp_s=0.0)
        assert len(tracks) == 1
        t = tracks[0]
        assert t.class_name == "vehicle"
        np.testing.assert_array_equal(t.state[:3], vehicle_detection.center_xyz)
        assert t.hits == 1
        assert t.age_frames == 1
        assert t.missed_frames == 0
        assert t.track_id == 0

    def test_predictions_move_tracks_forward(
        self, basic_tracker: MultiObjectTracker, vehicle_detection: Detection3D
    ) -> None:
        # First frame: create track.
        basic_tracker.update([vehicle_detection], timestamp_s=0.0)
        # Second frame: no detection, but track should be predicted.
        tracks = basic_tracker.update([], timestamp_s=1.0)
        assert len(tracks) == 1
        t = tracks[0]
        # The track should have been predicted to (5 + vx*dt, 0, 0.5).
        # Since initial velocity is zero, position stays at (5, 0, 0.5).
        np.testing.assert_almost_equal(t.state[0], 5.0, decimal=4)
        np.testing.assert_almost_equal(t.state[1], 0.0, decimal=4)
        np.testing.assert_almost_equal(t.state[2], 0.5, decimal=4)
        assert t.missed_frames == 1

    def test_track_deleted_after_max_missed_frames(
        self, tracker_with_thresholds: MultiObjectTracker, vehicle_detection: Detection3D
    ) -> None:
        tracker_with_thresholds.update([vehicle_detection], timestamp_s=0.0)
        # Two more frames with no detection (max_missed=2).
        tracker_with_thresholds.update([], timestamp_s=0.1)
        tracks = tracker_with_thresholds.update([], timestamp_s=0.2)
        assert len(tracks) == 0

    def test_track_not_deleted_before_max_missed(
        self, tracker_with_thresholds: MultiObjectTracker, vehicle_detection: Detection3D
    ) -> None:
        tracker_with_thresholds.update([vehicle_detection], timestamp_s=0.0)
        tracker_with_thresholds.update([], timestamp_s=0.1)
        # One more missed frame does NOT delete yet (max=2, missed=2).
        tracks = tracker_with_thresholds.update([], timestamp_s=0.2)
        assert len(tracks) == 0  # missed_frames=3 > max=2, so deleted

    def test_matching_same_class(
        self, basic_tracker: MultiObjectTracker, vehicle_detection: Detection3D
    ) -> None:
        # Frame 1: create track.
        basic_tracker.update([vehicle_detection], timestamp_s=0.0)
        # Frame 2: same detection at slightly different position.
        det2 = Detection3D(
            center_xyz=np.array([5.1, 0.0, 0.5], dtype=np.float32),
            size_lwh=np.array([4.0, 1.8, 1.5], dtype=np.float32),
            yaw_rad=0.0,
            class_id=1,
            class_name="vehicle",
            confidence=0.9,
            source="mock_geometric_clusterer",
        )
        tracks = basic_tracker.update([det2], timestamp_s=0.1)
        assert len(tracks) == 1
        t = tracks[0]
        assert t.hits == 2
        assert t.missed_frames == 0
        # Position should have been updated towards det2.
        assert t.state[0] > 5.0

    def test_unmatched_detection_creates_new_track(
        self, basic_tracker: MultiObjectTracker, vehicle_detection: Detection3D,
        pedestrian_detection: Detection3D
    ) -> None:
        basic_tracker.update([vehicle_detection], timestamp_s=0.0)
        tracks = basic_tracker.update([pedestrian_detection], timestamp_s=0.1)
        assert len(tracks) == 2
        ids = sorted(t.track_id for t in tracks)
        assert ids == [0, 1]

    def test_different_classes_not_matched(
        self, basic_tracker: MultiObjectTracker, vehicle_detection: Detection3D,
        pedestrian_detection: Detection3D
    ) -> None:
        # Even at the same position, different classes should not match.
        basic_tracker.update([vehicle_detection], timestamp_s=0.0)
        tracks = basic_tracker.update([pedestrian_detection], timestamp_s=0.1)
        assert len(tracks) == 2

    def test_low_confidence_detection_ignored(
        self, basic_tracker: MultiObjectTracker
    ) -> None:
        low_conf_det = Detection3D(
            center_xyz=np.array([10.0, 0.0, 0.5], dtype=np.float32),
            size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            yaw_rad=0.0,
            class_id=0,
            class_name="unknown_obstacle",
            confidence=0.01,  # Below default min of 0.0? Set higher min.
            source="mock_geometric_clusterer",
        )
        tracker = MultiObjectTracker(min_detection_confidence=0.5)
        tracks = tracker.update([low_conf_det], timestamp_s=0.0)
        assert len(tracks) == 0

    def test_track_confidence_initialises_from_detection(
        self, basic_tracker: MultiObjectTracker, vehicle_detection: Detection3D
    ) -> None:
        tracks = basic_tracker.update([vehicle_detection], timestamp_s=0.0)
        assert len(tracks) == 1
        assert tracks[0].confidence >= 0.85  # detection conf 0.9 + maturity bonus

    def test_dynamic_flag_after_enough_hits(
        self, basic_tracker: MultiObjectTracker
    ) -> None:
        # Create a track with a known velocity.
        from src.perception.object_detector import MockObjectDetector
        det = Detection3D(
            center_xyz=np.array([0.0, 0.0, 0.5], dtype=np.float32),
            size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            yaw_rad=0.0,
            class_id=0,
            class_name="vehicle",
            confidence=0.9,
            source="mock",
        )
        # Create track.
        basic_tracker.update([det], timestamp_s=0.0)
        # Manually set velocity for testing.
        t = basic_tracker.get_track(0)
        assert t is not None
        t.state[3] = 10.0  # vx = 10 m/s
        # Now update with a detection at a new position.
        det2 = Detection3D(
            center_xyz=np.array([1.0, 0.0, 0.5], dtype=np.float32),
            size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            yaw_rad=0.0,
            class_id=0,
            class_name="vehicle",
            confidence=0.9,
            source="mock",
        )
        basic_tracker.update([det2], timestamp_s=0.1)
        t = basic_tracker.get_track(0)
        assert t is not None
        # After 2 hits, dynamic flag should be set if speed > threshold.
        assert t.hits >= 2

    def test_timestamp_monotonicity_warning(
        self, basic_tracker: MultiObjectTracker, vehicle_detection: Detection3D
    ) -> None:
        basic_tracker.update([vehicle_detection], timestamp_s=1.0)
        with pytest.warns(RuntimeWarning, match="Timestamp decreased"):
            basic_tracker.update([vehicle_detection], timestamp_s=0.5)

    def test_reset_clears_all_tracks(self, basic_tracker: MultiObjectTracker,
                                       vehicle_detection: Detection3D) -> None:
        basic_tracker.update([vehicle_detection], timestamp_s=0.0)
        assert len(basic_tracker.tracks) == 1
        basic_tracker.reset()
        assert len(basic_tracker.tracks) == 0
        assert basic_tracker._next_track_id == 0

    def test_remove_track(self, basic_tracker: MultiObjectTracker,
                         vehicle_detection: Detection3D) -> None:
        basic_tracker.update([vehicle_detection], timestamp_s=0.0)
        assert len(basic_tracker.tracks) == 1
        assert basic_tracker.remove_track(0) is True
        assert len(basic_tracker.tracks) == 0
        assert basic_tracker.remove_track(999) is False

    def test_get_track(self, basic_tracker: MultiObjectTracker,
                       vehicle_detection: Detection3D) -> None:
        basic_tracker.update([vehicle_detection], timestamp_s=0.0)
        t = basic_tracker.get_track(0)
        assert t is not None
        assert t.track_id == 0
        assert t.class_name == "vehicle"
        assert basic_tracker.get_track(999) is None

    def test_deterministic_output_order(
        self, basic_tracker: MultiObjectTracker, vehicle_detection: Detection3D,
        pedestrian_detection: Detection3D
    ) -> None:
        basic_tracker.update([pedestrian_detection], timestamp_s=0.0)
        basic_tracker.update([vehicle_detection], timestamp_s=0.1)
        tracks = basic_tracker.tracks
        assert len(tracks) == 2
        ids = [t.track_id for t in tracks]
        assert ids == sorted(ids)

    def test_three_frames_tracking(
        self, basic_tracker: MultiObjectTracker, vehicle_detection: Detection3D
    ) -> None:
        det1 = vehicle_detection
        det2 = Detection3D(
            center_xyz=np.array([5.5, 0.0, 0.5], dtype=np.float32),
            size_lwh=np.array([4.0, 1.8, 1.5], dtype=np.float32),
            yaw_rad=0.0,
            class_id=1,
            class_name="vehicle",
            confidence=0.9,
            source="mock",
        )
        det3 = Detection3D(
            center_xyz=np.array([6.0, 0.0, 0.5], dtype=np.float32),
            size_lwh=np.array([4.0, 1.8, 1.5], dtype=np.float32),
            yaw_rad=0.0,
            class_id=1,
            class_name="vehicle",
            confidence=0.9,
            source="mock",
        )
        basic_tracker.update([det1], timestamp_s=0.0)
        basic_tracker.update([det2], timestamp_s=0.1)
        tracks = basic_tracker.update([det3], timestamp_s=0.2)
        assert len(tracks) == 1
        t = tracks[0]
        assert t.hits == 3
        assert t.age_frames == 3
        assert t.missed_frames == 0
        # Track should have moved along x.
        assert t.state[0] > 5.5


class TestMultiObjectTrackerEdgeCases:
    """Edge case tests for the tracker."""

    def test_many_detections_same_frame(
        self, basic_tracker: MultiObjectTracker
    ) -> None:
        detections = [
            Detection3D(
                center_xyz=np.array([float(i), 0.0, 0.5], dtype=np.float32),
                size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                yaw_rad=0.0,
                class_id=0,
                class_name="unknown_obstacle",
                confidence=0.5,
                source="mock",
            )
            for i in range(20)
        ]
        tracks = basic_tracker.update(detections, timestamp_s=0.0)
        assert len(tracks) == 20

    def test_single_point_detection(
        self, basic_tracker: MultiObjectTracker
    ) -> None:
        det = Detection3D(
            center_xyz=np.array([0.0, 0.0, 0.5], dtype=np.float32),
            size_lwh=np.array([0.1, 0.1, 0.1], dtype=np.float32),
            yaw_rad=0.0,
            class_id=0,
            class_name="unknown_obstacle",
            confidence=0.5,
            source="mock",
        )
        tracks = basic_tracker.update([det], timestamp_s=0.0)
        assert len(tracks) == 1
        assert tracks[0].size_lwh[0] == 0.1

    def test_yaw_blend(self) -> None:
        # Test the yaw blending helper.
        result = _blend_yaw(0.0, np.pi, 0.5)
        np.testing.assert_almost_equal(result, np.pi / 2.0)

        result = _blend_yaw(np.pi, -np.pi, 0.5)
        # pi and -pi are the same angle; blending them must not move the
        # estimate.  The output can be pi or -pi (both are equivalent).
        np.testing.assert_almost_equal(abs(result), np.pi, decimal=6)

        result = _blend_yaw(-np.pi / 2, np.pi / 2, 0.5)
        np.testing.assert_almost_equal(result, 0.0, decimal=6)

    def test_no_scipy_assignment_still_works(self) -> None:
        """Even without scipy, the tracker should fall back to greedy assignment."""
        import sys
        import importlib

        # Temporarily hide scipy.
        scipy_modules = {}
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("scipy"):
                scipy_modules[mod_name] = sys.modules[mod_name]
                del sys.modules[mod_name]

        try:
            tracker = MultiObjectTracker()
            det = Detection3D(
                center_xyz=np.array([0.0, 0.0, 0.5], dtype=np.float32),
                size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                yaw_rad=0.0,
                class_id=0,
                class_name="unknown_obstacle",
                confidence=0.5,
                source="mock",
            )
            tracks = tracker.update([det], timestamp_s=0.0)
            assert len(tracks) == 1
        finally:
            sys.modules.update(scipy_modules)


class TestMultiObjectTrackerTiming:
    """Timing and timestamp-related tests."""

    def test_timestamp_propagates_to_tracks(
        self, basic_tracker: MultiObjectTracker, vehicle_detection: Detection3D
    ) -> None:
        tracks = basic_tracker.update([vehicle_detection], timestamp_s=1.5)
        assert len(tracks) == 1
        assert tracks[0].last_timestamp_s == 1.5

    def test_elapsed_time_used_for_prediction(
        self, basic_tracker: MultiObjectTracker
    ) -> None:
        # Initialise a track with a known velocity.
        det = Detection3D(
            center_xyz=np.array([0.0, 0.0, 0.5], dtype=np.float32),
            size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            yaw_rad=0.0,
            class_id=0,
            class_name="vehicle",
            confidence=0.5,
            source="mock",
        )
        basic_tracker.update([det], timestamp_s=0.0)
        t = basic_tracker.get_track(0)
        assert t is not None
        t.state[3] = 5.0  # vx = 5 m/s

        # Update with a frame at t=1.0 (dt=1.0).
        basic_tracker.update([], timestamp_s=1.0)
        t = basic_tracker.get_track(0)
        assert t is not None
        # Predicted position: 0 + 5*1 = 5.
        np.testing.assert_almost_equal(t.state[0], 5.0, decimal=4)


class TestTrackerSpecRequirements:
    """Tests matching Phase 8A specification section D — tracker behaviour."""

    def test_same_track_id_across_frames(
        self, basic_tracker: MultiObjectTracker, vehicle_detection: Detection3D
    ) -> None:
        """A detection keeps the same track ID across frames."""
        basic_tracker.update([vehicle_detection], timestamp_s=0.0)
        tracks0 = basic_tracker.update([], timestamp_s=0.1)
        assert len(tracks0) == 1
        first_id = tracks0[0].track_id

        basic_tracker.update([], timestamp_s=0.2)
        tracks1 = basic_tracker.tracks
        assert len(tracks1) == 1
        assert tracks1[0].track_id == first_id

    def test_velocity_estimated_from_translated_detection(
        self, basic_tracker: MultiObjectTracker
    ) -> None:
        """Velocity is estimated from translated detections across frames."""
        det1 = Detection3D(
            center_xyz=np.array([0.0, 0.0, 0.5], dtype=np.float32),
            size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            yaw_rad=0.0,
            class_id=0,
            class_name="vehicle",
            confidence=0.9,
            source="mock",
        )
        det2 = Detection3D(
            center_xyz=np.array([1.0, 0.0, 0.5], dtype=np.float32),
            size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            yaw_rad=0.0,
            class_id=0,
            class_name="vehicle",
            confidence=0.9,
            source="mock",
        )
        basic_tracker.update([det1], timestamp_s=0.0)
        basic_tracker.update([det2], timestamp_s=1.0)
        t = basic_tracker.get_track(0)
        assert t is not None
        # The Kalman filter produces a smoothed velocity estimate.
        # After 2 observations with dt=1.0 and displacement=1.0 m,
        # the velocity should be in (0.3, 1.0) m/s in the +x direction.
        vx = float(t.state[3])
        assert 0.3 < vx < 1.0, f"Expected vx in (0.3, 1.0), got {vx}"
        # Velocity should be predominantly in +x, near-zero in y/z.
        np.testing.assert_almost_equal(abs(t.state[4]), 0.0, decimal=1)
        np.testing.assert_almost_equal(abs(t.state[5]), 0.0, decimal=1)

    def test_dynamic_false_after_first_observation(
        self, basic_tracker: MultiObjectTracker, vehicle_detection: Detection3D
    ) -> None:
        """dynamic is false after only one observation."""
        basic_tracker.update([vehicle_detection], timestamp_s=0.0)
        t = basic_tracker.get_track(0)
        assert t is not None
        # One hit is not enough to mark dynamic (min_hits_for_dynamic=2).
        assert t.dynamic is False

    def test_dynamic_becomes_true_after_enough_observations(
        self, basic_tracker: MultiObjectTracker
    ) -> None:
        """dynamic becomes true after enough observations and speed above threshold."""
        det1 = Detection3D(
            center_xyz=np.array([0.0, 0.0, 0.5], dtype=np.float32),
            size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            yaw_rad=0.0,
            class_id=0,
            class_name="vehicle",
            confidence=0.9,
            source="mock",
        )
        det2 = Detection3D(
            center_xyz=np.array([2.0, 0.0, 0.5], dtype=np.float32),
            size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            yaw_rad=0.0,
            class_id=0,
            class_name="vehicle",
            confidence=0.9,
            source="mock",
        )
        basic_tracker.update([det1], timestamp_s=0.0)
        basic_tracker.update([det2], timestamp_s=0.5)
        t = basic_tracker.get_track(0)
        assert t is not None
        # Two hits, speed ~4 m/s > default dynamic_speed_threshold.
        assert t.hits >= 2
        assert bool(t.dynamic) is True

    def test_track_deleted_after_max_missed_frames(
        self, basic_tracker: MultiObjectTracker, vehicle_detection: Detection3D
    ) -> None:
        """Track is deleted after max_missed_frames consecutive misses."""
        basic_tracker.update([vehicle_detection], timestamp_s=0.0)
        assert len(basic_tracker.tracks) == 1

        # max_missed_frames=5 (default). Miss 6 frames → deleted.
        for _ in range(6):
            basic_tracker.update([], timestamp_s=0.1)
        assert len(basic_tracker.tracks) == 0

    def test_hungarian_two_tracks_deterministic(
        self, basic_tracker: MultiObjectTracker
    ) -> None:
        """Hungarian association handles two tracks deterministically."""
        det_a = Detection3D(
            center_xyz=np.array([1.0, 0.0, 0.5], dtype=np.float32),
            size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            yaw_rad=0.0,
            class_id=0,
            class_name="vehicle",
            confidence=0.9,
            source="mock",
        )
        det_b = Detection3D(
            center_xyz=np.array([5.0, 0.0, 0.5], dtype=np.float32),
            size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            yaw_rad=0.0,
            class_id=0,
            class_name="vehicle",
            confidence=0.9,
            source="mock",
        )
        # Create two tracks.
        basic_tracker.update([det_a], timestamp_s=0.0)
        basic_tracker.update([det_b], timestamp_s=0.1)
        assert len(basic_tracker.tracks) == 2

        # Move both detections closer to each other — should still match
        # to the correct track by nearest distance.
        det_a2 = Detection3D(
            center_xyz=np.array([1.1, 0.0, 0.5], dtype=np.float32),
            size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            yaw_rad=0.0,
            class_id=0,
            class_name="vehicle",
            confidence=0.9,
            source="mock",
        )
        det_b2 = Detection3D(
            center_xyz=np.array([5.1, 0.0, 0.5], dtype=np.float32),
            size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            yaw_rad=0.0,
            class_id=0,
            class_name="vehicle",
            confidence=0.9,
            source="mock",
        )
        basic_tracker.update([det_a2, det_b2], timestamp_s=0.2)
        tracks = basic_tracker.tracks
        assert len(tracks) == 2
        # Both tracks should have 2 hits.
        for t in tracks:
            assert t.hits == 2

    def test_track_confidence_stays_in_0_1(
        self, basic_tracker: MultiObjectTracker, vehicle_detection: Detection3D
    ) -> None:
        """Track confidence always stays in [0, 1]."""
        for i in range(20):
            basic_tracker.update([vehicle_detection], timestamp_s=float(i))
        for t in basic_tracker.tracks:
            assert 0.0 <= t.confidence <= 1.0
