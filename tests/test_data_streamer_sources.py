"""Unit tests for DataStreamer sources (sample, SemanticKITTI, RELLIS-3D)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dashboard.data_streamer import DataStreamerThread
from src.perception.object_detector import MockObjectDetector


class TestDataStreamerSources:
    """Test data streamer loading and processing across data sources."""

    def test_sample_source_streaming(self):
        streamer = DataStreamerThread(source="sample", num_frames=3)
        streamer._setup_sources()

        state0 = streamer.process_frame(0, timestamp_s=0.0)
        assert state0.frame_idx == 0
        assert len(state0.points) > 0
        assert "elevation" in state0.grid_maps
        assert "traversability" in state0.grid_maps
        assert "roi" in state0.grid_maps

        state1 = streamer.process_frame(1, timestamp_s=0.1)
        assert state1.frame_idx == 1
        assert len(state1.points) == len(state0.points)

    def test_semantickitti_source_streaming_if_available(self):
        kitti_path = _PROJECT_ROOT / "data/semantic_kitti/dataset/sequences/00/velodyne"
        if not kitti_path.exists():
            pytest.skip("SemanticKITTI data not found.")

        streamer = DataStreamerThread(source="semantickitti", sequence="00", num_frames=2)
        streamer._setup_sources()

        state = streamer.process_frame(0, timestamp_s=0.0)
        assert len(state.points) > 10000
        assert state.points.shape[1] == 4

    def test_rellis3d_source_streaming_if_available(self):
        rellis_root = Path("C:/dev/data/rellis3d")
        if not rellis_root.exists():
            rellis_root = _PROJECT_ROOT / "data/rellis3d"
        if not rellis_root.exists():
            pytest.skip("RELLIS-3D data not found.")

        streamer = DataStreamerThread(source="rellis3d", sequence="00000", num_frames=2)
        streamer._setup_sources()

        state = streamer.process_frame(0, timestamp_s=0.0)
        assert len(state.points) > 10000
        assert state.points.shape[1] == 4
