import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.dashboard.export_web_dashboard_data import (
    GridResult,
    compute_ego_terrain_status,
)


def _make_grid_with_safe_patch(safe_row_range, safe_col_range, extent_m=(-20.0, 20.0, -20.0, 20.0)):
    """A synthetic GridResult, all Blocked (0.0) except a real Safe (1.0)
    patch at the given [row_range, col_range] -- lets us pin down exactly
    which real-world region compute_ego_terrain_status treats as "ahead"."""
    rows = cols = 80
    traversability = np.zeros((rows, cols), dtype=np.float32)
    point_density = np.full((rows, cols), 20, dtype=np.int32)
    r0, r1 = safe_row_range
    c0, c1 = safe_col_range
    traversability[r0:r1, c0:c1] = 1.0
    return GridResult(
        rows=rows, cols=cols, resolution_m=0.5, extent_m=extent_m,
        elevation=np.zeros((rows, cols), dtype=np.float32),
        traversability=traversability,
        semantic_class=np.full((rows, cols), -1, dtype=np.int16),
        point_density=point_density,
        overhead_gap=np.zeros((rows, cols), dtype=bool),
    )


def test_forward_axis_is_plus_y_not_plus_x():
    """Regression guard for the Bug 2 forward-axis fix: a real Safe patch
    directly ahead in +Y (forward, per data_streamer.py's own convention)
    must be picked up by the ROI; a Safe patch to the side in +X at the
    same real distance must NOT be picked up as "ahead".

    Grid: extent (-20,20,-20,20), res 0.5m -> row/col index i covers
    real coordinate [-20 + i*0.5, -20 + (i+1)*0.5). The ROI is real
    Y in [2,10] (forward) -- real row indices [44, 60).
    """
    # Safe patch directly AHEAD (+Y 2-10m, X near 0), widened a couple of
    # cells beyond the exact ROI so floor/ceil rounding at the ROI's own
    # edges can't pull in neighboring Blocked cells and muddy the result.
    ahead_row_range = (42, 62)   # covers y in [1, 11)
    ahead_col_range = (36, 44)   # covers x in [-2, 2)
    grid_ahead = _make_grid_with_safe_patch(ahead_row_range, ahead_col_range)
    result_ahead = compute_ego_terrain_status(grid_ahead)
    assert result_ahead["status"] == "DRIVABLE", (
        f"A real Safe patch directly ahead (+Y) was not recognized as "
        f"'ahead' -- forward-axis regression. Got: {result_ahead}"
    )

    # Safe patch to the SIDE (+X 2-10m, Y near 0) -- under the correct
    # +Y-forward convention this must NOT read as the ego's forward ROI.
    side_row_range = (36, 44)    # covers y in [-2, 2)
    side_col_range = (42, 62)    # covers x in [1, 11)
    grid_side = _make_grid_with_safe_patch(side_row_range, side_col_range)
    result_side = compute_ego_terrain_status(grid_side)
    assert result_side["status"] != "DRIVABLE", (
        f"A real Safe patch to the side (+X) was incorrectly treated as "
        f"the forward ROI -- this is the exact old (+X-forward) bug. "
        f"Got: {result_side}"
    )


def test_matches_data_streamer_forward_convention_on_a_real_point():
    """A real point 5m ahead (+Y) and near the centerline (X~0) must be
    classified as 'ahead' by compute_ego_terrain_status -- this is the
    same real point used to demonstrate the mismatch during diagnosis."""
    rows = cols = 80
    x_min, y_min = -20.0, -20.0
    res = 0.5
    real_x, real_y = 0.3, 5.0
    col = int((real_x - x_min) / res)
    row = int((real_y - y_min) / res)

    traversability = np.full((rows, cols), np.nan, dtype=np.float32)
    point_density = np.zeros((rows, cols), dtype=np.int32)
    traversability[row - 1:row + 2, col - 1:col + 2] = 1.0
    point_density[row - 1:row + 2, col - 1:col + 2] = 20

    grid = GridResult(
        rows=rows, cols=cols, resolution_m=res, extent_m=(-20.0, 20.0, -20.0, 20.0),
        elevation=np.zeros((rows, cols), dtype=np.float32),
        traversability=traversability,
        semantic_class=np.full((rows, cols), -1, dtype=np.int16),
        point_density=point_density,
        overhead_gap=np.zeros((rows, cols), dtype=bool),
    )
    result = compute_ego_terrain_status(grid)
    assert result["status"] == "DRIVABLE"
