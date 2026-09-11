"""Unit tests for FOVEAX Phase 4 -- Adaptive Variable-Resolution 2.5D Mapping.

Tests foveated zone definitions, zone map generation, and the
Problem Statement (PS-26053) memory reduction metric vs uniform 3D voxels.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import importlib
_mod = importlib.import_module("src.07_adaptive_2point5d_map")
AdaptiveZone = _mod.AdaptiveZone
SAMPLE_ZONES = _mod.SAMPLE_ZONES
SPEC_ZONES_100M = _mod.SPEC_ZONES_100M
compute_memory_reduction_metrics = _mod.compute_memory_reduction_metrics
create_zone_map = _mod.create_zone_map



class TestAdaptiveZonesConfig:
    def test_spec_zones_cover_up_to_100m(self) -> None:
        """PS specification requires foveated range up to 100m."""
        assert len(SPEC_ZONES_100M) == 3
        near, mid, far = SPEC_ZONES_100M

        assert near.name == "Near"
        assert near.r_min == 0.0
        assert near.r_max == 15.0
        assert near.resolution == 0.05  # 5 cm

        assert mid.name == "Middle"
        assert mid.r_min == 15.0
        assert mid.r_max == 35.0
        assert mid.resolution == 0.20  # 20 cm

        assert far.name == "Far"
        assert far.r_min == 35.0
        assert far.r_max == 100.0  # 100m
        assert far.resolution == 0.50  # 50 cm

    def test_sample_zones_validity(self) -> None:
        assert len(SAMPLE_ZONES) == 3
        for z in SAMPLE_ZONES:
            assert z.r_max > z.r_min
            assert z.resolution > 0


class TestMemoryReductionMetric:
    def test_memory_reduction_vs_3d_voxels(self) -> None:
        """Verify >99% reduction vs uniform 3D voxel grid as requested by PS-26053."""
        metrics = compute_memory_reduction_metrics(SPEC_ZONES_100M)

        assert metrics["total_3d_voxels"] > 1_000_000_000
        assert metrics["cell_savings_vs_3d_pct"] > 99.9
        assert metrics["byte_savings_vs_3d_pct"] > 99.0
        assert metrics["cell_savings_vs_uniform_2d_pct"] > 90.0

    def test_memory_reduction_custom_extent(self) -> None:
        metrics = compute_memory_reduction_metrics(
            SPEC_ZONES_100M,
            extent_x=(-50.0, 50.0),
            extent_y=(-50.0, 50.0),
            height_m=5.0,
        )
        assert metrics["cell_savings_vs_3d_pct"] > 99.0


class TestCreateZoneMap:
    def test_zone_filtering_and_layers(self) -> None:
        """Test points inside zone are mapped to z_min, z_max, point_count."""
        # 10 points at distance 10m (inside Near zone 0-15m)
        rng = np.random.default_rng(42)
        n = 10
        x = np.full(n, 6.0, dtype=np.float32)
        y = np.full(n, 8.0, dtype=np.float32)  # distance = 10.0m
        z = np.linspace(0.5, 2.5, n, dtype=np.float32)
        pts = np.column_stack([x, y, z])

        zone = SPEC_ZONES_100M[0]  # Near (0-15m, 5cm res)
        res = create_zone_map(pts, zone, x_min=-20.0, x_max=20.0, y_min=-20.0, y_max=20.0)

        assert res["occupied"].sum() == 1  # all in same cell
        assert np.nansum(res["point_count"]) == n
        assert np.nanmin(res["z_min"]) == pytest.approx(0.5)
        assert np.nanmax(res["z_max"]) == pytest.approx(2.5)
        assert np.nanmax(res["height_range"]) == pytest.approx(2.0)

    def test_points_outside_zone_are_excluded(self) -> None:
        # Point at 50m distance (outside Near zone 0-15m)
        pts = np.array([[30.0, 40.0, 1.0]], dtype=np.float32)
        zone = SPEC_ZONES_100M[0]
        res = create_zone_map(pts, zone, x_min=-100.0, x_max=100.0, y_min=-100.0, y_max=100.0)
        assert res["occupied"].sum() == 0
        assert np.nansum(res["point_count"]) == 0
