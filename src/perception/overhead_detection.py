"""Overhead-clearance detection.

Distinguishes grid cells where a raised obstacle has real vertical
clearance to the ground beneath it (a branch, an overhanging structure)
from cells that are solid all the way down (a wall, a rock face), using
only each cell's real raw point z-values from the current frame.

Classification codes:
    EMPTY = -1            -- no points in this cell.
    GROUND = 0            -- occupied, but not tall enough to be an
                             "obstacle" at all (z-range < ground_noise_m).
                             Rendered with normal terrain coloring.
    SOLID_OBSTACLE = 1    -- a genuine obstacle column with no significant
                             vertical gap: points run continuously from
                             near-ground to the top of the cell.
    OVERHEAD_OBSTACLE = 2 -- a genuine vertical gap (>= clearance_threshold_m)
                             exists between the lowest point cluster and a
                             higher cluster in this cell -- real drivable
                             clearance underneath a hanging/elevated object.
    INDETERMINATE = 3     -- too few real points in this cell
                             (< min_points) to reliably tell solid from
                             overhead; never guessed either way.

Every classification is derived only from the current frame's raw point
z-values that actually fall in each cell. There is no synthetic data and
no class prior: a sparse cell is reported INDETERMINATE rather than
defaulted to either obstacle type.
"""
from __future__ import annotations

import numpy as np

EMPTY = -1
GROUND = 0
SOLID_OBSTACLE = 1
OVERHEAD_OBSTACLE = 2
INDETERMINATE = 3

# Matches typical vehicle height clearance, and the same clearance
# threshold already used for overhead-gap detection in
# src/dashboard/export_web_dashboard_data.py::build_grids.
DEFAULT_CLEARANCE_THRESHOLD_M = 2.0
DEFAULT_MIN_POINTS = 4
# Below this z-range, a cell reads as flat/low-relief ground, not tall
# enough to register as an "obstacle" column at all.
DEFAULT_GROUND_NOISE_M = 0.3


def classify_cell_z_values(
    z_values: np.ndarray,
    min_points: int = DEFAULT_MIN_POINTS,
    clearance_threshold_m: float = DEFAULT_CLEARANCE_THRESHOLD_M,
    ground_noise_m: float = DEFAULT_GROUND_NOISE_M,
) -> int:
    """Classify one cell's raw point z-values. See module docstring."""
    z_values = np.asarray(z_values)
    n = len(z_values)
    if n == 0:
        return EMPTY

    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    if (z_max - z_min) < ground_noise_m:
        return GROUND

    if n < min_points:
        return INDETERMINATE

    sorted_z = np.sort(z_values)
    gaps = np.diff(sorted_z)
    max_gap = float(np.max(gaps)) if len(gaps) > 0 else 0.0

    # Sorted ascending: the points below the largest gap are, by
    # construction, the lowest (near-ground) cluster in this cell. A real
    # empty span of at least clearance_threshold_m above them is genuine
    # drivable clearance underneath whatever point cluster sits above it.
    if max_gap >= clearance_threshold_m:
        return OVERHEAD_OBSTACLE
    return SOLID_OBSTACLE


def compute_overhead_clearance_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x_min: float,
    y_min: float,
    x_bins: int,
    y_bins: int,
    grid_res: float,
    min_points: int = DEFAULT_MIN_POINTS,
    clearance_threshold_m: float = DEFAULT_CLEARANCE_THRESHOLD_M,
    ground_noise_m: float = DEFAULT_GROUND_NOISE_M,
) -> np.ndarray:
    """Build a (y_bins, x_bins) int8 grid of the classification codes above.

    x, y, z must already be filtered to the grid extent (the same
    convention used by src/dashboard/data_streamer.py::_generate_maps and
    src/dashboard/export_web_dashboard_data.py::build_grids).
    """
    grid = np.full((y_bins, x_bins), EMPTY, dtype=np.int8)
    if len(x) == 0:
        return grid

    col = np.floor((x - x_min) / grid_res).astype(np.int32)
    row = np.floor((y - y_min) / grid_res).astype(np.int32)
    col = np.clip(col, 0, x_bins - 1)
    row = np.clip(row, 0, y_bins - 1)
    flat = row * x_bins + col

    # Fully vectorized replacement for a per-occupied-cell Python loop
    # calling classify_cell_z_values() (profiled at ~30% of a real
    # frame's total dashboard cost). Sorting by (cell, z) together puts
    # every cell's real z-values in one ascending, contiguous run, which
    # lets z_min/z_max/point-count/max-gap for ALL cells be computed with
    # vectorized NumPy reduceat operations instead of one Python call
    # (with its own internal np.sort/np.diff/np.max) per cell.
    order = np.lexsort((z, flat))
    sorted_flat = flat[order]
    sorted_z = z[order]

    unique_flat, group_start = np.unique(sorted_flat, return_index=True)
    n_total = len(sorted_flat)
    n_points = np.append(group_start[1:], n_total) - group_start

    z_min = np.minimum.reduceat(sorted_z, group_start)
    z_max = np.maximum.reduceat(sorted_z, group_start)

    # Per-point gap to the NEXT point in sorted_z, masked to -inf across
    # cell boundaries (same_group[i] is False where sorted_z[i], [i+1]
    # belong to different cells) so a boundary gap can never be mistaken
    # for a real intra-cell gap by the reduceat max below.
    max_gap = np.zeros(len(unique_flat), dtype=np.float64)
    if n_total > 1:
        diffs = np.diff(sorted_z)
        same_group = sorted_flat[1:] == sorted_flat[:-1]
        diffs_masked = np.where(same_group, diffs, -np.inf)

        # Singleton cells (n_points == 1) have no internal diff at all --
        # reduceat's "start index equals the next start index" fallback
        # (return the single element unreduced) does not apply cleanly
        # here since indices aren't guaranteed adjacent, so they are
        # excluded up front and simply keep max_gap = 0.0 (no gap to
        # measure), matching classify_cell_z_values's own `gaps = diff(...);
        # max_gap = max(gaps) if len(gaps) > 0 else 0.0` for a 1-point cell.
        multi_point = n_points > 1
        if np.any(multi_point):
            diff_starts = group_start[multi_point]
            # Each segment [diff_starts[k], diff_starts[k+1]) contains
            # exactly cell k's own real internal diffs plus zero or more
            # -inf boundary/skipped-singleton entries -- the -inf entries
            # can never win a max, so this is exact regardless of which
            # cells were excluded as singletons above.
            max_gap[multi_point] = np.maximum.reduceat(diffs_masked, diff_starts)

    height_range = z_max - z_min

    classification = np.full(len(unique_flat), SOLID_OBSTACLE, dtype=np.int8)
    classification[height_range < ground_noise_m] = GROUND
    tall_enough = height_range >= ground_noise_m
    classification[tall_enough & (n_points < min_points)] = INDETERMINATE
    classification[tall_enough & (n_points >= min_points) & (max_gap >= clearance_threshold_m)] = OVERHEAD_OBSTACLE

    grid.reshape(-1)[unique_flat] = classification
    return grid
