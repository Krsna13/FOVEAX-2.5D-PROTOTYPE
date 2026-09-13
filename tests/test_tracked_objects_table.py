"""Tests for tracked_object_row_text (src/dashboard/track_labels.py), the
exact function FoveaXDashboardWindow.update_ui uses to build each TRACKED
OBJECTS list row. Focus: no-fabrication / no-desync -- the row's distance
and display class must come from the same functions
(track_distance_m / display_class_for_track) that build the 3D label text,
so the same track_id can never show two different values in the two places.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dashboard.track_labels import (
    display_class_for_track,
    displayed_speed_mps,
    track_distance_m,
    track_label_text,
    tracked_object_row_text,
)
from src.tracking.multi_object_tracker import TrackState

_ROW_RE = re.compile(
    r"^ID: (?P<id>-?\d+) \| (?P<cls>.+) \| (?P<status>Static|Dynamic) \| "
    r"(?P<speed>-?\d+\.\d+) m/s \| (?P<dist>-?\d+\.\d+)m$"
)


def _track(track_id, class_name, xyz, vxyz=(0.0, 0.0, 0.0), dynamic=False, metadata=None):
    return TrackState(
        track_id=track_id,
        class_name=class_name,
        state=np.array([*xyz, *vxyz], dtype=np.float64),
        covariance=np.eye(6, dtype=np.float64),
        size_lwh=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        yaw_rad=0.0,
        dynamic=dynamic,
        metadata=metadata or {},
    )


class TestRowFormat:
    @pytest.mark.parametrize(
        "t",
        [
            _track(1, "VEHICLE", (3.0, 4.0, 0.0), vxyz=(1.0, 0.0, 0.0), dynamic=True),
            _track(2, "VEGETATION", (0.0, 0.0, 0.0)),
            _track(3, "SOLID_OBSTACLE", (5.0, 0.0, 0.0), metadata={"is_overhang": True}),
        ],
    )
    def test_row_matches_expected_shape(self, t):
        assert _ROW_RE.match(tracked_object_row_text(t))


class TestNoDesyncWithLabel:
    """The exact no-desync requirement from the task: same track_id, same
    real distance and class in both the table row and the 3D label text."""

    @pytest.mark.parametrize(
        "t",
        [
            _track(7, "VEHICLE", (6.0, 8.0, 0.0), dynamic=True),
            _track(8, "VEGETATION", (-1.0, 1.0, 0.0)),
            _track(9, "SOLID_OBSTACLE", (0.0, 3.0, 0.0), metadata={"is_overhang": True}),
            _track(10, "ROUGH_TERRAIN", (4.0, 0.0, 0.0), metadata={"slope_deg": 11.0}),
        ],
    )
    def test_distance_matches(self, t):
        """Both displayed strings round the same real distance to 1 decimal
        place, so they must match each other and the real value within
        that rounding, never by more."""
        row = _ROW_RE.match(tracked_object_row_text(t))
        label_dist = float(re.search(r"- (-?\d+\.\d+)m$", track_label_text(t)).group(1))
        table_dist = float(row.group("dist"))
        real_dist = track_distance_m(t)
        assert table_dist == pytest.approx(real_dist, abs=0.05)
        assert table_dist == label_dist

    @pytest.mark.parametrize(
        "t",
        [
            _track(7, "VEHICLE", (6.0, 8.0, 0.0), dynamic=True),
            _track(9, "SOLID_OBSTACLE", (0.0, 3.0, 0.0), metadata={"is_overhang": True}),
            _track(10, "ROUGH_TERRAIN", (4.0, 0.0, 0.0), metadata={"slope_deg": 11.0}),
        ],
    )
    def test_display_class_matches(self, t):
        row = _ROW_RE.match(tracked_object_row_text(t))
        assert row.group("cls") == display_class_for_track(t)
        assert row.group("cls") in track_label_text(t)


class TestStaticSpeedZero:
    def test_static_track_shows_zero_speed_despite_real_nonzero_kalman_estimate(self):
        t = _track(1, "VEGETATION", (2.0, 0.0, 0.0), vxyz=(2.86, 0.0, 0.0), dynamic=False)
        assert displayed_speed_mps(t) == 0.0
        row = _ROW_RE.match(tracked_object_row_text(t))
        assert float(row.group("speed")) == 0.0

    def test_dynamic_track_shows_real_speed(self):
        t = _track(1, "VEHICLE", (2.0, 0.0, 0.0), vxyz=(3.0, 4.0, 0.0), dynamic=True)
        assert displayed_speed_mps(t) == pytest.approx(5.0)
