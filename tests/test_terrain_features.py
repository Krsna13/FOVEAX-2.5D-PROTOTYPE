"""Tests for src/perception/terrain_features.py -- real-geometry derived
terrain/object features (slope, overhang, pothole/bump, rock heuristic).

Every assertion here checks that a value is computed from the actual input
geometry (not a constant), and that "not enough real data" returns None
rather than a fabricated number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perception.terrain_features import (
    classify_hazard_kind,
    ground_clearance_m,
    is_overhang,
    is_probable_rock_heuristic,
    local_ground_z_percentile,
    plane_normal_pca,
    points_in_oriented_box,
    slope_angle_deg,
)


class TestPlaneNormalPCA:
    def test_flat_plane_normal_is_vertical(self):
        rng = np.random.default_rng(0)
        xy = rng.uniform(-2, 2, (200, 2))
        pts = np.column_stack([xy, np.full(200, 3.0)])
        normal = plane_normal_pca(pts)
        np.testing.assert_allclose(np.abs(normal), [0, 0, 1], atol=1e-6)

    def test_tilted_plane_normal_matches_known_tilt(self):
        """A plane z = x*tan(15deg) has a real, known normal angle from
        vertical of exactly 15 degrees -- not an assumed/rounded value."""
        rng = np.random.default_rng(1)
        x = rng.uniform(-2, 2, 300)
        y = rng.uniform(-2, 2, 300)
        tilt_rad = np.radians(15.0)
        z = x * np.tan(tilt_rad)
        pts = np.column_stack([x, y, z])
        normal = plane_normal_pca(pts)
        angle = np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1)))
        assert angle == pytest.approx(15.0, abs=0.05)

    def test_insufficient_points_returns_none(self):
        assert plane_normal_pca(np.array([[0, 0, 0], [1, 1, 1]])) is None

    def test_collinear_points_return_none(self):
        t = np.linspace(0, 1, 10)
        pts = np.column_stack([t, t, t])
        assert plane_normal_pca(pts) is None

    def test_coincident_points_return_none(self):
        pts = np.tile([1.0, 2.0, 3.0], (5, 1))
        assert plane_normal_pca(pts) is None


class TestSlopeAngleDeg:
    def test_flat_ground_is_zero(self):
        rng = np.random.default_rng(2)
        xy = rng.uniform(-1, 1, (50, 2))
        pts = np.column_stack([xy, np.zeros(50)])
        assert slope_angle_deg(pts) == pytest.approx(0.0, abs=1e-4)

    def test_steep_slope_reads_high_angle(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(-1, 1, 100)
        y = rng.uniform(-1, 1, 100)
        z = x * np.tan(np.radians(30.0))
        pts = np.column_stack([x, y, z])
        angle = slope_angle_deg(pts)
        assert angle == pytest.approx(30.0, abs=0.1)

    def test_different_geometry_gives_different_angle(self):
        """No hardcoded return value: two different real surfaces must
        produce two different real angles."""
        rng = np.random.default_rng(4)
        x = rng.uniform(-1, 1, 100)
        y = rng.uniform(-1, 1, 100)
        flat = slope_angle_deg(np.column_stack([x, y, np.zeros(100)]))
        tilted = slope_angle_deg(np.column_stack([x, y, x * np.tan(np.radians(20))]))
        assert flat != pytest.approx(tilted, abs=1.0)

    def test_none_when_underlying_normal_is_none(self):
        assert slope_angle_deg(np.array([[0, 0, 0]])) is None


class TestLocalGroundZPercentile:
    def test_uses_only_points_within_radius(self):
        near = np.array([[0.0, 0.0, 1.0, 0.0], [0.1, 0.0, 1.2, 0.0]])
        far = np.array([[50.0, 50.0, -99.0, 0.0]])
        pts = np.vstack([near, far])
        z = local_ground_z_percentile(pts, (0.0, 0.0), radius_m=1.0, percentile=50.0)
        assert z == pytest.approx(1.1, abs=0.01)
        assert z != pytest.approx(-99.0)

    def test_no_points_in_radius_returns_none(self):
        pts = np.array([[100.0, 100.0, 0.0, 0.0]])
        assert local_ground_z_percentile(pts, (0.0, 0.0), radius_m=1.0) is None

    def test_low_percentile_favors_lower_points(self):
        pts = np.array([[0, 0, 0.0, 0], [0, 0, 1.0, 0], [0, 0, 2.0, 0]])
        low = local_ground_z_percentile(pts, (0, 0), radius_m=1.0, percentile=10.0)
        high = local_ground_z_percentile(pts, (0, 0), radius_m=1.0, percentile=90.0)
        assert low < high


class TestOverhang:
    def test_ground_clearance_is_real_subtraction(self):
        assert ground_clearance_m(bbox_bottom_z=2.0, ground_z=0.5) == pytest.approx(1.5)
        assert ground_clearance_m(bbox_bottom_z=0.3, ground_z=0.5) == pytest.approx(-0.2)

    def test_overhang_threshold(self):
        assert is_overhang(0.6, "SOLID_OBSTACLE", threshold_m=0.5) is True
        assert is_overhang(0.5, "SOLID_OBSTACLE", threshold_m=0.5) is False
        assert is_overhang(0.4, "SOLID_OBSTACLE", threshold_m=0.5) is False
        assert is_overhang(-0.1, "SOLID_OBSTACLE", threshold_m=0.5) is False

    def test_vehicle_and_pedestrian_never_flagged_overhang(self):
        """Confirmed on real RELLIS-3D data: MockObjectDetector's ground
        rejection clips VEHICLE/PEDESTRIAN clusters to points above real
        ground, so a large real clearance number here reflects the sensor's
        elevated mount height and the detector's own filtering -- not an
        actual overhang -- for these two already-classified, ground-contact
        classes specifically (see OVERHANG_EXCLUDED_CLASSES)."""
        assert is_overhang(1.6, "VEHICLE", threshold_m=0.5) is False
        assert is_overhang(1.6, "PEDESTRIAN", threshold_m=0.5) is False

    def test_other_classes_still_eligible_at_the_same_clearance(self):
        assert is_overhang(1.6, "SOLID_OBSTACLE", threshold_m=0.5) is True
        assert is_overhang(1.6, "VEGETATION", threshold_m=0.5) is True
        assert is_overhang(1.6, "unclassified", threshold_m=0.5) is True


class TestPointsInOrientedBox:
    def test_selects_only_points_inside_padded_box(self):
        pts = np.array([
            [0.0, 0.0, 0.0, 1.0],   # inside
            [1.9, 0.0, 0.0, 1.0],   # inside (within margin of 4-length box)
            [10.0, 0.0, 0.0, 1.0],  # outside
        ], dtype=np.float32)
        mask = points_in_oriented_box(pts, np.array([0, 0, 0]), np.array([4, 2, 2]), margin=0.1)
        assert mask.tolist() == [True, True, False]

    def test_margin_expands_selection(self):
        pts = np.array([[2.05, 0.0, 0.0, 1.0]], dtype=np.float32)
        c, s = np.array([0, 0, 0]), np.array([4, 2, 2])
        assert not points_in_oriented_box(pts, c, s, margin=0.01)[0]
        assert points_in_oriented_box(pts, c, s, margin=0.1)[0]


class TestRockHeuristic:
    def test_small_dense_unclassified_cluster_is_flagged(self):
        assert is_probable_rock_heuristic(
            volume_m3=0.05, num_points=30, class_name="unclassified"
        ) is True

    def test_excluded_class_never_flagged_even_if_small_and_dense(self):
        assert is_probable_rock_heuristic(
            volume_m3=0.05, num_points=30, class_name="VEGETATION"
        ) is False

    def test_large_volume_not_flagged(self):
        assert is_probable_rock_heuristic(
            volume_m3=5.0, num_points=1000, class_name="unclassified"
        ) is False

    def test_low_density_not_flagged(self):
        assert is_probable_rock_heuristic(
            volume_m3=0.4, num_points=3, class_name="unclassified"
        ) is False

    def test_zero_volume_never_flagged(self):
        assert is_probable_rock_heuristic(
            volume_m3=0.0, num_points=100, class_name="unclassified"
        ) is False


class TestHazardKindClassification:
    def test_no_baseline_is_rough_not_a_guess(self):
        kind, extent = classify_hazard_kind(1.0, 0.0, baseline_z=None)
        assert kind == "Rough" and extent is None

    def test_real_dip_below_baseline_is_pothole(self):
        kind, depth = classify_hazard_kind(z_max_local=0.02, z_min_local=-0.3, baseline_z=0.0)
        assert kind == "Pothole"
        assert depth == pytest.approx(0.3)

    def test_real_rise_above_baseline_is_bump(self):
        kind, height = classify_hazard_kind(z_max_local=0.4, z_min_local=0.0, baseline_z=0.0)
        assert kind == "Bump"
        assert height == pytest.approx(0.4)

    def test_within_noise_threshold_is_rough(self):
        kind, extent = classify_hazard_kind(z_max_local=0.03, z_min_local=-0.03, baseline_z=0.0)
        assert kind == "Rough" and extent is None

    def test_deeper_dip_wins_over_smaller_bump(self):
        kind, extent = classify_hazard_kind(z_max_local=0.15, z_min_local=-0.5, baseline_z=0.0)
        assert kind == "Pothole"
        assert extent == pytest.approx(0.5)

    def test_taller_bump_wins_over_smaller_dip(self):
        kind, extent = classify_hazard_kind(z_max_local=0.5, z_min_local=-0.12, baseline_z=0.0)
        assert kind == "Bump"
        assert extent == pytest.approx(0.5)

    def test_nan_baseline_is_rough(self):
        kind, extent = classify_hazard_kind(1.0, 0.0, baseline_z=float("nan"))
        assert kind == "Rough" and extent is None
