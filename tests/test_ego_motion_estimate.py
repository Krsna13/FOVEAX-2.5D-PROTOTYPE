import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pytest

pytest.importorskip("open3d", reason="open3d not installed")

from src.perception.ego_motion_estimate import estimate_relative_displacement_m


def _make_structured_cloud(n=2000, seed=0):
    """A real-ish structured point cloud (not a degenerate flat plane, so
    ICP has real geometry to register against)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-10, 10, n)
    y = rng.uniform(-10, 10, n)
    z = 0.1 * np.sin(x) + 0.1 * np.cos(y) + rng.normal(0, 0.02, n)
    return np.column_stack([x, y, z])


def test_identical_clouds_report_near_zero_displacement():
    cloud = _make_structured_cloud()
    result = estimate_relative_displacement_m(cloud, cloud.copy())
    assert result is not None
    assert result < 0.05


def test_known_translation_is_recovered_approximately():
    cloud = _make_structured_cloud()
    shifted = cloud.copy()
    shifted[:, 0] += 1.0  # real, known 1.0m shift in x
    result = estimate_relative_displacement_m(cloud, shifted)
    assert result is not None
    # Coarse ICP on downsampled points -- allow real tolerance, not exact.
    assert 0.5 < result < 1.5


def test_too_few_points_returns_none_not_a_guess():
    tiny_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    tiny_b = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = estimate_relative_displacement_m(tiny_a, tiny_b)
    assert result is None


def test_empty_cloud_returns_none():
    cloud = _make_structured_cloud()
    empty = np.zeros((0, 3))
    assert estimate_relative_displacement_m(empty, cloud) is None
    assert estimate_relative_displacement_m(cloud, empty) is None
