import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.perception.overhead_detection import (
    EMPTY,
    GROUND,
    SOLID_OBSTACLE,
    OVERHEAD_OBSTACLE,
    INDETERMINATE,
    classify_cell_z_values,
    compute_overhead_clearance_grid,
)


def test_empty_cell():
    assert classify_cell_z_values(np.array([])) == EMPTY


def test_flat_ground_low_relief():
    z = np.array([0.10, 0.12, 0.15, 0.11, 0.09])
    assert classify_cell_z_values(z) == GROUND


def test_sparse_tall_cell_is_indeterminate():
    # Only 2 points but with real vertical spread -- too few to trust
    # either classification, must not guess.
    z = np.array([0.1, 3.0])
    assert classify_cell_z_values(z, min_points=4) == INDETERMINATE


def test_clear_vertical_gap_is_overhead_obstacle():
    # A near-ground cluster (0.0-0.2m) and a higher cluster (2.5-2.7m):
    # 2.3m gap exceeds the 2.0m clearance threshold -- real drivable
    # clearance underneath a hanging/elevated obstacle (e.g. a branch).
    ground_cluster = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
    overhead_cluster = np.array([2.5, 2.55, 2.6, 2.65, 2.7])
    z = np.concatenate([ground_cluster, overhead_cluster])
    assert classify_cell_z_values(z, clearance_threshold_m=2.0) == OVERHEAD_OBSTACLE


def test_continuous_column_is_solid_obstacle():
    # Points continuous from near-ground to 1.5m with no gap anywhere
    # near clearance_threshold_m -- a solid wall/rock, not passable.
    z = np.linspace(0.0, 1.5, 20)
    assert classify_cell_z_values(z, clearance_threshold_m=2.0) == SOLID_OBSTACLE


def test_gap_just_under_threshold_is_solid_not_overhead():
    ground_cluster = np.array([0.0, 0.1, 0.2])
    near_cluster = np.array([1.9, 2.0, 2.1])  # gap = 1.7m < 2.0m threshold
    z = np.concatenate([ground_cluster, near_cluster])
    assert classify_cell_z_values(z, clearance_threshold_m=2.0) == SOLID_OBSTACLE


def test_compute_overhead_clearance_grid_matches_per_cell_classification():
    # 2x2 grid, resolution 1.0m, extent starting at (0,0).
    # Cell (row=0, col=0): overhead obstacle.
    # Cell (row=0, col=1): solid obstacle.
    # Cell (row=1, col=0): empty.
    # Cell (row=1, col=1): flat ground.
    x = np.array([0.1, 0.1, 0.1, 0.1,   1.1, 1.1, 1.1, 1.1,   1.1, 1.1])
    y = np.array([0.1, 0.1, 0.1, 0.1,   0.1, 0.1, 0.1, 0.1,   1.1, 1.1])
    z = np.array([0.0, 0.1, 2.5, 2.6,   0.0, 0.5, 1.0, 1.5,   0.10, 0.12])

    grid = compute_overhead_clearance_grid(
        x, y, z, x_min=0.0, y_min=0.0, x_bins=2, y_bins=2, grid_res=1.0,
        min_points=4, clearance_threshold_m=2.0, ground_noise_m=0.3,
    )

    assert grid[0, 0] == OVERHEAD_OBSTACLE
    assert grid[0, 1] == SOLID_OBSTACLE
    assert grid[1, 0] == EMPTY
    assert grid[1, 1] == GROUND


def test_never_guesses_indeterminate_stays_distinct_from_both_obstacle_types():
    # Regression guard: indeterminate must never collapse into solid or
    # overhead -- a caller checking `== SOLID_OBSTACLE` or
    # `== OVERHEAD_OBSTACLE` on a too-sparse cell must get False.
    z = np.array([0.1, 5.0])  # 2 points, huge spread, but n < min_points
    result = classify_cell_z_values(z, min_points=4, clearance_threshold_m=2.0)
    assert result == INDETERMINATE
    assert result != SOLID_OBSTACLE
    assert result != OVERHEAD_OBSTACLE


def _old_compute_overhead_clearance_grid(
    x, y, z, x_min, y_min, x_bins, y_bins, grid_res,
    min_points=4, clearance_threshold_m=2.0, ground_noise_m=0.3,
):
    """The exact pre-vectorization implementation (per-occupied-cell
    Python loop calling classify_cell_z_values), reimplemented here only
    as a reference oracle -- not reused from production code, since the
    point is to catch production code silently drifting from this
    known-correct baseline."""
    grid = np.full((y_bins, x_bins), EMPTY, dtype=np.int8)
    if len(x) == 0:
        return grid
    col = np.floor((x - x_min) / grid_res).astype(np.int32)
    row = np.floor((y - y_min) / grid_res).astype(np.int32)
    col = np.clip(col, 0, x_bins - 1)
    row = np.clip(row, 0, y_bins - 1)
    flat = row * x_bins + col
    sort_idx = np.argsort(flat)
    sorted_flat = flat[sort_idx]
    sorted_z = z[sort_idx]
    unique_flat, split_indices = np.unique(sorted_flat, return_index=True)
    z_groups = np.split(sorted_z, split_indices[1:])
    grid_flat = grid.reshape(-1)
    for flat_idx, cell_z in zip(unique_flat, z_groups):
        grid_flat[flat_idx] = classify_cell_z_values(
            cell_z, min_points=min_points,
            clearance_threshold_m=clearance_threshold_m, ground_noise_m=ground_noise_m,
        )
    return grid


class TestVectorizedGridMatchesOldPerCellLoop:
    """Regression guard for the compute_overhead_clearance_grid
    vectorization: the fully-vectorized (lexsort + reduceat) computation
    must produce cell-for-cell identical classifications to the original
    per-occupied-cell Python loop, on fixed synthetic multi-cell data
    covering every classification code.
    """

    def _fixed_synthetic_scene(self):
        """A 5x5 grid (resolution 1.0m) covering every classification:
        EMPTY, GROUND, SOLID_OBSTACLE, OVERHEAD_OBSTACLE, INDETERMINATE,
        plus cells with >1 point sharing identical z-values (zero-gap
        ties) and a cell whose single largest gap sits at the very start
        vs. very end of its sorted z-values."""
        cells = {
            # (row, col): list of z-values
            (0, 0): [0.0, 0.05, 0.08, 0.02],              # GROUND (low relief)
            (0, 1): [0.0, 0.3, 0.6, 0.9, 1.2],              # SOLID (continuous)
            (0, 2): [0.0, 0.05, 2.5, 2.6, 2.7],             # OVERHEAD (clear gap)
            (0, 3): [0.1, 4.0],                             # INDETERMINATE (n<4, tall)
            (1, 0): [],                                     # EMPTY (no points)
            (1, 1): [1.0, 1.0, 1.0, 1.0, 1.0],              # GROUND (zero relief, ties)
            (1, 2): [0.0, 0.1, 0.2, 2.5],                   # gap at the END of sorted z
            (1, 3): [0.0, 2.5, 2.6, 2.7],                   # gap at the START of sorted z
            (2, 0): [0.0, 0.5, 1.0, 1.5, 1.9999],           # SOLID (gap just under 2.0m threshold)
            (2, 1): [0.0, 0.5, 1.0, 1.5, 2.0],              # OVERHEAD (gap exactly at threshold)
            (2, 2): [3.3, 3.3],                             # INDETERMINATE (n=2, tall, ties)
            (3, 3): [0.0, 0.01],                            # GROUND (n=2 but low relief -> not indeterminate)
        }
        xs, ys, zs = [], [], []
        for (row, col), z_list in cells.items():
            for zv in z_list:
                xs.append(col + 0.3)
                ys.append(row + 0.3)
                zs.append(zv)
        return np.array(xs), np.array(ys), np.array(zs)

    def test_full_grid_matches_old_implementation_exactly(self):
        x, y, z = self._fixed_synthetic_scene()
        kwargs = dict(x_min=0.0, y_min=0.0, x_bins=5, y_bins=5, grid_res=1.0)

        old_grid = _old_compute_overhead_clearance_grid(x, y, z, **kwargs)
        new_grid = compute_overhead_clearance_grid(x, y, z, **kwargs)

        np.testing.assert_array_equal(new_grid, old_grid)

        # Sanity: the fixed scene actually exercises every classification
        # code, otherwise this test could trivially pass.
        codes_present = set(np.unique(old_grid).tolist())
        assert codes_present == {EMPTY, GROUND, SOLID_OBSTACLE, OVERHEAD_OBSTACLE, INDETERMINATE}

    def test_random_synthetic_grid_matches_old_implementation(self):
        rng = np.random.default_rng(7)
        n_points = 4000
        x = rng.uniform(0.0, 20.0, n_points)
        y = rng.uniform(0.0, 20.0, n_points)
        # Mixture of low-relief and tall points so all classification
        # branches get real coverage across many real-ish cells.
        z = np.where(
            rng.random(n_points) < 0.5,
            rng.uniform(0.0, 0.2, n_points),
            rng.uniform(0.0, 6.0, n_points),
        )
        kwargs = dict(x_min=0.0, y_min=0.0, x_bins=20, y_bins=20, grid_res=1.0)

        old_grid = _old_compute_overhead_clearance_grid(x, y, z, **kwargs)
        new_grid = compute_overhead_clearance_grid(x, y, z, **kwargs)

        np.testing.assert_array_equal(new_grid, old_grid)
