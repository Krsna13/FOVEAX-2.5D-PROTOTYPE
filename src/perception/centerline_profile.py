"""Real forward-corridor height-profile computation for the FOVEAX dashboard.

Samples a corridor extending forward from the ego position along the ego
heading, in fixed-size distance bins, and reports each bin's REAL height
above (a bump: pole, vehicle, overhang) or depth below (a depression:
pothole, dip) a real LOCAL ground baseline -- a rolling median of nearby
z_min values, not one single global reference, so slopes read correctly
rather than as a spurious ramp against a far-away datum.

No value here is synthesized or interpolated: a distance bin with no real
grid cells backing it is reported as a gap (all fields None, point_count
0), never a guessed/interpolated height.

Note on ego heading: this pipeline has no existing ego heading/position
tracking (checked dashboard_state.py, data_streamer.py, and the tracker --
none exists; every FrameState is a single ego-centered point cloud with
no odometry). The caller-side default used by
src/dashboard/data_streamer.py is ego_position=(0,0), heading=+Y, matching
that module's own pre-existing forward-axis convention (its
elevation_profile/road_width already treat the column nearest x=0 as the
"centerline" and iterate over y as the forward axis). Note this disagrees
with src/dashboard/export_web_dashboard_data.py::compute_ego_terrain_status,
which treats +X as forward -- a pre-existing inconsistency between the two
dashboards' modules that this change does not silently paper over (see the
integration report).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# A bin's z_min must sit at least this far below the local baseline to
# count as a real depression rather than ordinary ground-texture noise.
DEFAULT_DEPRESSION_THRESHOLD_M = 0.05


@dataclass
class CenterlineBin:
    distance_m: float
    height_above_baseline_m: float | None  # real bump height, >=0, None = gap
    depth_below_baseline_m: float | None   # real depression depth, <0, None = no depression (or gap)
    baseline_z: float | None               # real local ground baseline used, None = gap
    point_count: int                       # real point density backing this bin (0 = gap)


@dataclass
class CenterlineProfile:
    bins: list[CenterlineBin] = field(default_factory=list)
    width_m: float = 2.0
    bin_size_m: float = 1.0
    max_range_m: float = 100.0


def compute_centerline_profile(
    elevation_grid: np.ndarray,
    z_min_grid: np.ndarray,
    point_count_grid: np.ndarray,
    grid_bounds: tuple[float, float, float, float],
    grid_res: float,
    ego_position: tuple[float, float] = (0.0, 0.0),
    ego_heading_rad: float = math.pi / 2.0,
    width_m: float = 2.0,
    max_range_m: float = 100.0,
    bin_size_m: float = 1.0,
    baseline_window_bins: int = 3,
    depression_threshold_m: float = DEFAULT_DEPRESSION_THRESHOLD_M,
) -> CenterlineProfile:
    """See module docstring.

    elevation_grid (real z_max per cell), z_min_grid (real z_min per cell)
    and point_count_grid (real point density per cell) are all (rows,
    cols) arrays over grid_bounds=(x_min, x_max, y_min, y_max) with
    row = floor((y - y_min) / grid_res), col = floor((x - x_min) / grid_res)
    -- the same pre-display-flip convention used internally by
    src/dashboard/data_streamer.py::_generate_maps. NaN in elevation_grid
    or z_min_grid marks an empty (no-point) cell.
    """
    x_min, x_max, y_min, y_max = grid_bounds
    rows, cols = elevation_grid.shape

    forward = np.array([math.cos(ego_heading_rad), math.sin(ego_heading_rad)])
    lateral = np.array([-forward[1], forward[0]])
    ego = np.array(ego_position, dtype=np.float64)

    n_bins = int(math.ceil(max_range_m / bin_size_m))

    row_idx, col_idx = np.indices((rows, cols))
    cell_x = x_min + (col_idx + 0.5) * grid_res
    cell_y = y_min + (row_idx + 0.5) * grid_res

    rel_x = cell_x - ego[0]
    rel_y = cell_y - ego[1]
    along = rel_x * forward[0] + rel_y * forward[1]
    across = rel_x * lateral[0] + rel_y * lateral[1]

    corridor_mask = (np.abs(across) <= width_m / 2.0) & (along >= 0.0) & (along < max_range_m)
    bin_idx = np.floor(along / bin_size_m).astype(np.int32)

    bins: list[CenterlineBin] = []
    trailing_z_min: list[float] = []

    for b in range(n_bins):
        distance_m = (b + 0.5) * bin_size_m
        bin_mask = corridor_mask & (bin_idx == b)
        valid_cells = bin_mask & ~np.isnan(elevation_grid) & ~np.isnan(z_min_grid)

        if not np.any(valid_cells):
            bins.append(CenterlineBin(distance_m, None, None, None, 0))
            continue

        n_points = int(np.nansum(np.where(valid_cells, point_count_grid, 0)))
        bin_z_max = float(np.nanmax(elevation_grid[valid_cells]))
        bin_z_min = float(np.nanmin(z_min_grid[valid_cells]))

        if trailing_z_min:
            baseline_z = float(np.median(trailing_z_min[-baseline_window_bins:]))
        else:
            baseline_z = bin_z_min  # first real bin: no trailing history yet

        height_above = max(0.0, bin_z_max - baseline_z)
        depth_below = None
        if (baseline_z - bin_z_min) >= depression_threshold_m:
            depth_below = -(baseline_z - bin_z_min)

        bins.append(CenterlineBin(
            distance_m=distance_m,
            height_above_baseline_m=height_above,
            depth_below_baseline_m=depth_below,
            baseline_z=baseline_z,
            point_count=n_points,
        ))
        trailing_z_min.append(bin_z_min)

    return CenterlineProfile(bins=bins, width_m=width_m, bin_size_m=bin_size_m, max_range_m=max_range_m)


def find_corridor_markers(
    objects: list[dict],
    grid_bounds: tuple[float, float, float, float],
    ego_position: tuple[float, float] = (0.0, 0.0),
    ego_heading_rad: float = math.pi / 2.0,
    width_m: float = 2.0,
    max_range_m: float = 100.0,
) -> list[dict]:
    """Filter real hazards/tracked objects to those inside the sampled
    corridor. Each input dict must have real "x", "y" world coordinates
    and a real "distance_m" (the SAME distance value already shown
    elsewhere in the dashboard for that object -- reused verbatim here,
    not re-derived, so the profile marker and e.g. the hazard table or
    nearest-dynamic-object panel always agree for the same object).
    """
    forward = np.array([math.cos(ego_heading_rad), math.sin(ego_heading_rad)])
    lateral = np.array([-forward[1], forward[0]])
    ego = np.array(ego_position, dtype=np.float64)

    markers = []
    for obj in objects:
        rel = np.array([obj["x"] - ego[0], obj["y"] - ego[1]])
        along = float(rel @ forward)
        across = float(rel @ lateral)
        if abs(across) <= width_m / 2.0 and 0.0 <= along < max_range_m:
            markers.append(obj)
    return markers
