"""Integration tests: DataStreamerThread._attach_terrain_features wires real
per-detection geometry (src/perception/terrain_features.py) into
Detection3D.metadata, and the hazard-cluster loop in _compute_real_extras
classifies real Pothole/Bump kinds -- using only synthetic-but-known
point clouds, not real datasets, so these run everywhere.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dashboard.data_streamer import DataStreamerThread
from src.perception.object_detector import Detection3D


def _streamer() -> DataStreamerThread:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return DataStreamerThread(source="sample", num_frames=1)


class TestAttachTerrainFeatures:
    def test_overhang_flagged_when_bbox_bottom_clears_ground(self):
        """A box floating 1m above a real, dense ground plane at z=0 must
        be flagged an overhang; its clearance must equal the real gap."""
        streamer = _streamer()
        rng = np.random.default_rng(0)
        ground_xy = rng.uniform(-3, 3, (500, 2))
        ground = np.column_stack([ground_xy, np.zeros(500), np.ones(500)]).astype(np.float32)

        det = Detection3D(
            center_xyz=np.array([0.0, 0.0, 1.5], dtype=np.float32),
            size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),  # bottom at z=1.0
            yaw_rad=0.0, class_id=4, class_name="SOLID_OBSTACLE",
            confidence=0.9, source="mock",
        )
        streamer._attach_terrain_features([det], ground)
        assert det.metadata["is_overhang"] is True
        assert det.metadata["ground_clearance_m"] == pytest.approx(1.0, abs=0.02)

    def test_no_overhang_when_box_sits_on_ground(self):
        streamer = _streamer()
        rng = np.random.default_rng(1)
        ground_xy = rng.uniform(-3, 3, (500, 2))
        ground = np.column_stack([ground_xy, np.zeros(500), np.ones(500)]).astype(np.float32)

        det = Detection3D(
            center_xyz=np.array([0.0, 0.0, 0.5], dtype=np.float32),
            size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),  # bottom at z=0.0
            yaw_rad=0.0, class_id=4, class_name="SOLID_OBSTACLE",
            confidence=0.9, source="mock",
        )
        streamer._attach_terrain_features([det], ground)
        assert det.metadata["is_overhang"] is False

    def test_slope_only_computed_for_ground_like_classes(self):
        """VEHICLE clusters aren't ground -- slope must not appear for them,
        even if their member points happen to be tilted."""
        streamer = _streamer()
        rng = np.random.default_rng(2)
        x = rng.uniform(-1, 1, 200)
        y = rng.uniform(-1, 1, 200)
        z = x * np.tan(np.radians(20.0)) + 0.5
        member = np.column_stack([x, y, z, np.ones(200)]).astype(np.float32)

        ground_det = Detection3D(
            center_xyz=np.array([0.0, 0.0, 0.5], dtype=np.float32),
            size_lwh=np.array([2.0, 2.0, 1.0], dtype=np.float32),
            yaw_rad=0.0, class_id=1, class_name="ROUGH_TERRAIN",
            confidence=0.9, source="mock",
        )
        vehicle_det = Detection3D(
            center_xyz=np.array([0.0, 0.0, 0.5], dtype=np.float32),
            size_lwh=np.array([2.0, 2.0, 1.0], dtype=np.float32),
            yaw_rad=0.0, class_id=5, class_name="VEHICLE",
            confidence=0.9, source="mock",
        )
        streamer._attach_terrain_features([ground_det, vehicle_det], member)
        assert ground_det.metadata["slope_deg"] == pytest.approx(20.0, abs=0.5)
        assert "slope_deg" not in vehicle_det.metadata

    def test_slope_absent_when_too_few_member_points_for_a_plane(self):
        """A near-empty cluster can't support a real plane fit -- no slope
        value must be fabricated for it."""
        streamer = _streamer()
        sparse = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        det = Detection3D(
            center_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            size_lwh=np.array([0.2, 0.2, 0.2], dtype=np.float32),
            yaw_rad=0.0, class_id=1, class_name="ROUGH_TERRAIN",
            confidence=0.9, source="mock",
        )
        streamer._attach_terrain_features([det], sparse)
        assert "slope_deg" not in det.metadata

    def test_rock_heuristic_flag_present_and_class_gated(self):
        streamer = _streamer()
        rng = np.random.default_rng(3)
        dense_small = rng.uniform(-0.1, 0.1, (200, 3)) + np.array([2.0, 0.0, 0.3])
        pts = np.hstack([dense_small, np.ones((200, 1))]).astype(np.float32)

        rock_like = Detection3D(
            center_xyz=np.array([2.0, 0.0, 0.3], dtype=np.float32),
            size_lwh=np.array([0.2, 0.2, 0.2], dtype=np.float32),
            yaw_rad=0.0, class_id=7, class_name="unclassified",
            confidence=0.9, source="mock",
        )
        veg = Detection3D(
            center_xyz=np.array([2.0, 0.0, 0.3], dtype=np.float32),
            size_lwh=np.array([0.2, 0.2, 0.2], dtype=np.float32),
            yaw_rad=0.0, class_id=2, class_name="VEGETATION",
            confidence=0.9, source="mock",
        )
        streamer._attach_terrain_features([rock_like, veg], pts)
        assert rock_like.metadata["is_probable_rock"] is True
        assert veg.metadata["is_probable_rock"] is False

    def test_vehicle_not_flagged_overhang_despite_large_real_clearance(self):
        """Regression for the real false-positive found on RELLIS-3D seq
        00001 frame 0 track 7: a VEHICLE detection sitting well above a
        real, dense, low ground cluster (simulating the sensor's elevated
        mount height) must not be flagged an overhang, even though the raw
        clearance number is large and real."""
        streamer = _streamer()
        rng = np.random.default_rng(7)
        ground_xy = rng.uniform(-3, 3, (500, 2))
        ground = np.column_stack(
            [ground_xy, np.full(500, -1.3), np.ones(500)]
        ).astype(np.float32)

        vehicle = Detection3D(
            center_xyz=np.array([0.0, 0.0, 0.4], dtype=np.float32),
            size_lwh=np.array([4.0, 1.8, 0.4], dtype=np.float32),  # bottom at z=0.2
            yaw_rad=0.0, class_id=5, class_name="VEHICLE",
            confidence=0.9, source="mock",
        )
        obstacle = Detection3D(
            center_xyz=np.array([0.0, 0.0, 0.4], dtype=np.float32),
            size_lwh=np.array([4.0, 1.8, 0.4], dtype=np.float32),
            yaw_rad=0.0, class_id=4, class_name="SOLID_OBSTACLE",
            confidence=0.9, source="mock",
        )
        streamer._attach_terrain_features([vehicle, obstacle], ground)
        assert vehicle.metadata["ground_clearance_m"] > 1.0  # real, large, computed
        assert vehicle.metadata["is_overhang"] is False
        assert obstacle.metadata["is_overhang"] is True  # same geometry, eligible class

    def test_ground_clearance_absent_when_no_nearby_points(self):
        """A detection whose neighbourhood has no real points at all
        (isolated far from everything else in the frame) gets no
        ground-clearance claim -- not a guessed one."""
        streamer = _streamer()
        far_points = np.array([[100.0, 100.0, 0.0, 1.0]], dtype=np.float32)
        det = Detection3D(
            center_xyz=np.array([0.0, 0.0, 0.5], dtype=np.float32),
            size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            yaw_rad=0.0, class_id=4, class_name="SOLID_OBSTACLE",
            confidence=0.9, source="mock",
        )
        streamer._attach_terrain_features([det], far_points)
        assert "ground_clearance_m" not in det.metadata
        assert "is_overhang" not in det.metadata


class TestHazardClusterKindViaRealGrid:
    def test_pothole_detected_from_real_dip_in_frame(self):
        """A localized dip surrounded by flat, dense ground must appear as
        a real Pothole hazard cluster with a real positive depth.

        Grid cells are 0.5m (self.grid_res); traversability blocks a cell
        when its real z_max-z_min height_range exceeds 0.6m. A 0.7m dip
        over a full-cell-sized (0.5m radius) patch guarantees every point
        in the affected cells is shifted -- not a mix of shifted/unshifted
        points that would dilute the real height_range below that
        threshold -- so it reliably registers as Blocked and forms a
        hazard cluster.
        """
        streamer = _streamer()
        rng = np.random.default_rng(4)
        xy = rng.uniform(-8, 8, (4000, 2))
        z = np.zeros(4000)

        dip_center = np.array([2.0, 2.0])
        in_dip = np.linalg.norm(xy - dip_center, axis=1) < 0.5
        z[in_dip] -= 0.7

        pts = np.column_stack([xy, z]).astype(np.float32)
        streamer._generate_maps(pts, [])
        extras = streamer._last_extras
        kinds = {c["kind"] for c in extras["hazard_clusters"]}
        assert "Pothole" in kinds
        pothole = next(c for c in extras["hazard_clusters"] if c["kind"] == "Pothole")
        assert pothole["extent_m"] > 0.1

    def test_bump_detected_from_real_rise_in_frame(self):
        streamer = _streamer()
        rng = np.random.default_rng(5)
        xy = rng.uniform(-8, 8, (4000, 2))
        z = np.zeros(4000)

        bump_center = np.array([-2.0, -2.0])
        in_bump = np.linalg.norm(xy - bump_center, axis=1) < 0.5
        z[in_bump] += 0.7

        pts = np.column_stack([xy, z]).astype(np.float32)
        streamer._generate_maps(pts, [])
        extras = streamer._last_extras
        kinds = {c["kind"] for c in extras["hazard_clusters"]}
        assert "Bump" in kinds
        bump = next(c for c in extras["hazard_clusters"] if c["kind"] == "Bump")
        assert bump["extent_m"] > 0.1

    def test_flat_frame_has_no_hazard_clusters(self):
        streamer = _streamer()
        rng = np.random.default_rng(6)
        xy = rng.uniform(-8, 8, (4000, 2))
        pts = np.column_stack([xy, np.zeros(4000)]).astype(np.float32)
        streamer._generate_maps(pts, [])
        assert streamer._last_extras["hazard_clusters"] == []
