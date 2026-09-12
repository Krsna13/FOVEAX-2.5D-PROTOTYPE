import math
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.perception.centerline_profile import (
    compute_centerline_profile,
    find_corridor_markers,
)

# Grid: 50 rows (y: 0..50, 1m cells), 2 cols (x: -1..1, 1m cells).
# row r -> y-center = r + 0.5 ; col c -> x-center = c - 0.5.
GRID_RES = 1.0
BOUNDS = (-1.0, 1.0, 0.0, 50.0)
ROWS, COLS = 50, 2


def _empty_grids():
    elevation = np.full((ROWS, COLS), np.nan, dtype=np.float32)
    z_min = np.full((ROWS, COLS), np.nan, dtype=np.float32)
    point_count = np.zeros((ROWS, COLS), dtype=np.float32)
    return elevation, z_min, point_count


def test_flat_section_has_small_height_above_and_no_depression():
    elevation, z_min, point_count = _empty_grids()
    # Flat ground for the first 10m: z_max/z_min both ~0.05, well-observed.
    for r in range(10):
        elevation[r, :] = 0.06
        z_min[r, :] = 0.05
        point_count[r, :] = 20

    profile = compute_centerline_profile(
        elevation, z_min, point_count, BOUNDS, GRID_RES,
        ego_position=(0.0, 0.0), ego_heading_rad=math.pi / 2.0,
        width_m=2.0, max_range_m=15.0, bin_size_m=1.0,
    )

    flat_bins = [b for b in profile.bins if b.distance_m < 10.0]
    assert len(flat_bins) == 10
    for b in flat_bins:
        assert b.point_count > 0
        assert b.height_above_baseline_m is not None
        assert b.height_above_baseline_m < 0.05  # near-zero relief
        assert b.depth_below_baseline_m is None


def test_real_bump_is_reported_as_height_above_baseline():
    elevation, z_min, point_count = _empty_grids()
    for r in range(10):
        elevation[r, :] = 0.05
        z_min[r, :] = 0.05
        point_count[r, :] = 20
    # A pole at ~15m: base near ground, top at 2.0m.
    pole_row = 15
    elevation[pole_row, :] = 2.0
    z_min[pole_row, :] = 0.05
    point_count[pole_row, :] = 8

    profile = compute_centerline_profile(
        elevation, z_min, point_count, BOUNDS, GRID_RES,
        max_range_m=20.0, bin_size_m=1.0,
    )
    bump_bin = next(b for b in profile.bins if abs(b.distance_m - 15.5) < 1e-6)
    assert bump_bin.height_above_baseline_m is not None
    assert bump_bin.height_above_baseline_m > 1.5
    assert bump_bin.depth_below_baseline_m is None


def test_real_depression_is_reported_as_depth_below_baseline():
    elevation, z_min, point_count = _empty_grids()
    for r in range(10):
        elevation[r, :] = 0.05
        z_min[r, :] = 0.05
        point_count[r, :] = 20
    # A pothole at ~20m: rim near ground, bottom well below baseline.
    pothole_row = 20
    elevation[pothole_row, :] = 0.05
    z_min[pothole_row, :] = -0.45
    point_count[pothole_row, :] = 6

    profile = compute_centerline_profile(
        elevation, z_min, point_count, BOUNDS, GRID_RES,
        max_range_m=25.0, bin_size_m=1.0,
    )
    dip_bin = next(b for b in profile.bins if abs(b.distance_m - 20.5) < 1e-6)
    assert dip_bin.depth_below_baseline_m is not None
    assert dip_bin.depth_below_baseline_m < -0.3


def test_sparse_bin_has_real_low_point_count_not_hidden():
    elevation, z_min, point_count = _empty_grids()
    for r in range(10):
        elevation[r, :] = 0.05
        z_min[r, :] = 0.05
        point_count[r, :] = 20
    # A sparse bin at ~12m: only 1 real point backing it.
    sparse_row = 12
    elevation[sparse_row, 0] = 0.08
    z_min[sparse_row, 0] = 0.08
    point_count[sparse_row, 0] = 1
    # col 1 stays NaN/empty at this row.

    profile = compute_centerline_profile(
        elevation, z_min, point_count, BOUNDS, GRID_RES,
        max_range_m=15.0, bin_size_m=1.0,
    )
    sparse_bin = next(b for b in profile.bins if abs(b.distance_m - 12.5) < 1e-6)
    assert sparse_bin.point_count == 1
    assert sparse_bin.height_above_baseline_m is not None  # real data exists, just sparse


def test_bin_with_no_real_data_is_a_gap_not_interpolated():
    elevation, z_min, point_count = _empty_grids()
    for r in range(10):
        elevation[r, :] = 0.05
        z_min[r, :] = 0.05
        point_count[r, :] = 20
    # Rows 10-12 (10-13m) stay entirely empty -- a real occlusion gap.
    for r in range(13, 20):
        elevation[r, :] = 0.05
        z_min[r, :] = 0.05
        point_count[r, :] = 20

    profile = compute_centerline_profile(
        elevation, z_min, point_count, BOUNDS, GRID_RES,
        max_range_m=20.0, bin_size_m=1.0,
    )
    gap_bin = next(b for b in profile.bins if abs(b.distance_m - 11.5) < 1e-6)
    assert gap_bin.point_count == 0
    assert gap_bin.height_above_baseline_m is None
    assert gap_bin.depth_below_baseline_m is None
    assert gap_bin.baseline_z is None


def test_find_corridor_markers_keeps_only_objects_inside_corridor():
    objects = [
        {"x": 0.0, "y": 10.0, "distance_m": 10.0, "label": "on_centerline"},
        {"x": 5.0, "y": 10.0, "distance_m": 11.18, "label": "far_off_to_the_side"},
        {"x": 0.5, "y": 20.0, "distance_m": 20.01, "label": "just_inside_corridor"},
        {"x": 0.0, "y": -5.0, "distance_m": 5.0, "label": "behind_ego"},
    ]
    markers = find_corridor_markers(
        objects, BOUNDS, ego_position=(0.0, 0.0), ego_heading_rad=math.pi / 2.0,
        width_m=2.0, max_range_m=100.0,
    )
    labels = {m["label"] for m in markers}
    assert labels == {"on_centerline", "just_inside_corridor"}


def test_marker_distance_is_reused_verbatim_not_rederived():
    # The marker's distance_m must be the exact value passed in (matching
    # whatever the hazard table / tracked-objects panel already shows for
    # the same object), not a recomputed forward-projection distance.
    objects = [{"x": 0.3, "y": 8.0, "distance_m": 8.0056, "label": "hazard_1"}]
    markers = find_corridor_markers(objects, BOUNDS, width_m=2.0, max_range_m=100.0)
    assert len(markers) == 1
    assert markers[0]["distance_m"] == 8.0056
