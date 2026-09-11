"""Unit tests for FOVEAX Phase 7 -- rellis3d_loader module.

Uses small synthetic .bin/.label fixtures written to tmp_path, matching the
real RELLIS-3D directory layout and file formats confirmed against the real
downloaded dataset -- not the actual 15GB dataset itself, so this suite stays
fast and doesn't require C:\\dev\\data\\rellis3d to exist.

Run with:
    python -m pytest tests/test_rellis3d_loader.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perception.rellis3d_loader import load_rellis3d_frame


def _write_frame(
    root: Path,
    sequence: str,
    frame: str,
    points: np.ndarray,
    raw_semantic_ids: np.ndarray,
    instance_ids: np.ndarray | None = None,
) -> None:
    """Write a synthetic frame matching the real RELLIS-3D layout."""
    seq_dir = root / "Rellis-3D" / sequence
    bin_dir = seq_dir / "os1_cloud_node_kitti_bin"
    label_dir = seq_dir / "os1_cloud_node_semantickitti_label_id"
    bin_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    points.astype(np.float32).tofile(bin_dir / f"{frame}.bin")

    if instance_ids is None:
        instance_ids = np.zeros_like(raw_semantic_ids)
    packed = (instance_ids.astype(np.uint32) << 16) | (
        raw_semantic_ids.astype(np.uint32) & 0xFFFF
    )
    packed.tofile(label_dir / f"{frame}.label")


@pytest.fixture
def rellis_root(tmp_path: Path) -> Path:
    return tmp_path


class TestLoadRellis3DFrame:
    def test_happy_path_shapes(self, rellis_root: Path) -> None:
        points = np.array(
            [[0.0, 0.0, 0.0, 0.1]] * 5, dtype=np.float32
        )
        raw_ids = np.array([3, 4, 17, 18, 8], dtype=np.uint32)  # grass, tree, person, fence, vehicle
        _write_frame(rellis_root, "00000", "000000", points, raw_ids)

        result = load_rellis3d_frame(rellis_root, "00000", "000000")

        assert result.points.shape == (5, 4)
        assert result.foveax_class_ids.shape == (5,)
        assert result.n_points_total == 5
        assert result.n_excluded_sky == 0

    def test_sky_points_are_excluded(self, rellis_root: Path) -> None:
        points = np.array(
            [[0.0, 0.0, 0.0, 0.1]] * 4, dtype=np.float32
        )
        raw_ids = np.array([3, 7, 7, 8], dtype=np.uint32)  # grass, sky, sky, vehicle
        _write_frame(rellis_root, "00000", "000000", points, raw_ids)

        result = load_rellis3d_frame(rellis_root, "00000", "000000")

        assert result.n_points_total == 4
        assert result.n_excluded_sky == 2
        assert result.points.shape == (2, 4)
        assert result.foveax_class_ids.shape == (2,)

    def test_all_sky_raises(self, rellis_root: Path) -> None:
        points = np.array([[0.0, 0.0, 0.0, 0.1]] * 3, dtype=np.float32)
        raw_ids = np.array([7, 7, 7], dtype=np.uint32)
        _write_frame(rellis_root, "00000", "000000", points, raw_ids)

        with pytest.raises(ValueError, match="every"):
            load_rellis3d_frame(rellis_root, "00000", "000000")

    def test_unknown_class_id_raises(self, rellis_root: Path) -> None:
        points = np.array([[0.0, 0.0, 0.0, 0.1]] * 2, dtype=np.float32)
        raw_ids = np.array([3, 999], dtype=np.uint32)  # 999 is not a real ontology ID
        _write_frame(rellis_root, "00000", "000000", points, raw_ids)

        with pytest.raises(ValueError, match="not present"):
            load_rellis3d_frame(rellis_root, "00000", "000000")

    def test_missing_bin_file_raises(self, rellis_root: Path) -> None:
        # Only write the label file, not the point cloud.
        label_dir = (
            rellis_root / "Rellis-3D" / "00000" / "os1_cloud_node_semantickitti_label_id"
        )
        label_dir.mkdir(parents=True)
        np.zeros(3, dtype=np.uint32).tofile(label_dir / "000000.label")

        with pytest.raises(FileNotFoundError, match="point cloud"):
            load_rellis3d_frame(rellis_root, "00000", "000000")

    def test_missing_label_file_raises(self, rellis_root: Path) -> None:
        bin_dir = rellis_root / "Rellis-3D" / "00000" / "os1_cloud_node_kitti_bin"
        bin_dir.mkdir(parents=True)
        np.zeros((3, 4), dtype=np.float32).tofile(bin_dir / "000000.bin")

        with pytest.raises(FileNotFoundError, match="label"):
            load_rellis3d_frame(rellis_root, "00000", "000000")

    def test_point_label_count_mismatch_raises(self, rellis_root: Path) -> None:
        points = np.zeros((5, 4), dtype=np.float32)
        raw_ids = np.array([3, 4, 8], dtype=np.uint32)  # only 3, not 5
        _write_frame(rellis_root, "00000", "000000", points, raw_ids)

        with pytest.raises(ValueError, match="mismatch"):
            load_rellis3d_frame(rellis_root, "00000", "000000")

    def test_malformed_bin_size_raises(self, rellis_root: Path) -> None:
        seq_dir = rellis_root / "Rellis-3D" / "00000"
        bin_dir = seq_dir / "os1_cloud_node_kitti_bin"
        label_dir = seq_dir / "os1_cloud_node_semantickitti_label_id"
        bin_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        # 5 float32 values -- not divisible by 4.
        np.zeros(5, dtype=np.float32).tofile(bin_dir / "000000.bin")
        np.zeros(1, dtype=np.uint32).tofile(label_dir / "000000.label")

        with pytest.raises(ValueError, match="divisible by 4"):
            load_rellis3d_frame(rellis_root, "00000", "000000")

    def test_semantic_id_extracted_from_lower_16_bits(self, rellis_root: Path) -> None:
        points = np.array([[0.0, 0.0, 0.0, 0.1]] * 2, dtype=np.float32)
        raw_ids = np.array([3, 8], dtype=np.uint32)  # grass, vehicle
        instance_ids = np.array([42, 7], dtype=np.uint32)  # nonzero instance bits
        _write_frame(rellis_root, "00000", "000000", points, raw_ids, instance_ids)

        result = load_rellis3d_frame(rellis_root, "00000", "000000")

        # grass=3 -> ROUGH_TERRAIN(1), vehicle=8 -> VEHICLE(5); instance bits
        # in the upper 16 must not corrupt the extracted semantic class.
        assert result.foveax_class_ids.tolist() == [1, 5]
