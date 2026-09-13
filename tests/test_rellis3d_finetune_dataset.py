"""Tests for the RELLIS-3D fine-tuning split and data pipeline.

Covers the two failure modes that would silently invalidate a fine-tune:
train/val/test leakage, and training preprocessing drifting away from the
inference preprocessing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.perception.range_projection import (
    project_points_to_range_image,
    rescale_intensity_to_unit_range,
)
from src.perception.rellis3d_loader import load_rellis3d_frame
from src.perception.semantic_labels import NUM_FOVEAX_CLASSES
from src.training.rellis3d_dataset import (
    IGNORE_INDEX,
    OUSTER_FOV_DOWN,
    OUSTER_FOV_UP,
    SPLIT_BUFFER_FRAMES,
    TEST_SEQUENCE,
    TRAIN_VAL_SEQUENCES,
    build_splits,
    build_training_sample,
    class_weights_from_counts,
    list_sequence_frames,
)

RELLIS_ROOT = Path("C:/dev/data/rellis3d")
ARCH_CFG_PATH = Path("models/salsanext/pretrained/pretrained/arch_cfg.yaml")


def _require_dataset():
    if not (RELLIS_ROOT / "Rellis-3D").is_dir():
        pytest.skip("RELLIS-3D data not found.")


def _require_arch_cfg():
    if not ARCH_CFG_PATH.is_file():
        pytest.skip("SalsaNext arch_cfg.yaml not found.")
    return yaml.safe_load(open(ARCH_CFG_PATH))


class TestSplits:
    def test_test_sequence_is_fully_held_out(self):
        _require_dataset()
        splits = build_splits(RELLIS_ROOT)
        train_val_seqs = {r.sequence for r in splits["train"] + splits["val"]}
        assert TEST_SEQUENCE not in train_val_seqs
        assert {r.sequence for r in splits["test"]} == {TEST_SEQUENCE}
        # Every frame of the test sequence is present -- it is held out whole,
        # not sampled.
        assert len(splits["test"]) == len(
            list_sequence_frames(RELLIS_ROOT, TEST_SEQUENCE)
        )

    def test_no_frame_appears_in_two_splits(self):
        _require_dataset()
        splits = build_splits(RELLIS_ROOT)
        keys = {
            name: {(r.sequence, r.frame) for r in refs}
            for name, refs in splits.items()
        }
        assert not keys["train"] & keys["val"]
        assert not keys["train"] & keys["test"]
        assert not keys["val"] & keys["test"]

    def test_buffer_gap_separates_train_from_val(self):
        """Adjacent ~10 Hz frames are near-duplicates, so the train block must
        end well before the val block begins."""
        _require_dataset()
        splits = build_splits(RELLIS_ROOT)
        for seq in TRAIN_VAL_SEQUENCES:
            frames = list_sequence_frames(RELLIS_ROOT, seq)
            index = {f: i for i, f in enumerate(frames)}
            train_idx = [index[r.frame] for r in splits["train"] if r.sequence == seq]
            val_idx = [index[r.frame] for r in splits["val"] if r.sequence == seq]
            assert train_idx and val_idx
            assert min(val_idx) - max(train_idx) >= SPLIT_BUFFER_FRAMES

    def test_val_block_is_contiguous_tail(self):
        _require_dataset()
        splits = build_splits(RELLIS_ROOT)
        for seq in TRAIN_VAL_SEQUENCES:
            frames = list_sequence_frames(RELLIS_ROOT, seq)
            val = [r.frame for r in splits["val"] if r.sequence == seq]
            assert val[-1] == frames[-1]
            # contiguous run at stride 1
            start = frames.index(val[0])
            assert val == frames[start:]

    def test_stride_subsamples_without_crossing_boundary(self):
        _require_dataset()
        full = build_splits(RELLIS_ROOT)
        strided = build_splits(RELLIS_ROOT, train_stride=2, val_stride=10)
        full_train = {(r.sequence, r.frame) for r in full["train"]}
        full_val = {(r.sequence, r.frame) for r in full["val"]}
        assert {(r.sequence, r.frame) for r in strided["train"]} <= full_train
        assert {(r.sequence, r.frame) for r in strided["val"]} <= full_val
        assert len(strided["train"]) < len(full["train"])

    def test_splits_are_deterministic(self):
        _require_dataset()
        a = build_splits(RELLIS_ROOT, train_stride=3)
        b = build_splits(RELLIS_ROOT, train_stride=3)
        assert [(r.sequence, r.frame) for r in a["train"]] == [
            (r.sequence, r.frame) for r in b["train"]
        ]

    def test_rejects_invalid_stride(self):
        with pytest.raises(ValueError):
            build_splits(RELLIS_ROOT, train_stride=0)


class TestTrainingSample:
    def test_shapes_and_label_domain(self):
        _require_dataset()
        arch_cfg = _require_arch_cfg()
        from src.training.rellis3d_dataset import FrameRef

        image, labels = build_training_sample(
            RELLIS_ROOT, FrameRef("00001", "000000"), arch_cfg
        )
        h = arch_cfg["dataset"]["sensor"]["img_prop"]["height"]
        w = arch_cfg["dataset"]["sensor"]["img_prop"]["width"]
        assert image.shape == (5, h, w)
        assert image.dtype == np.float32
        assert labels.shape == (h, w)
        assert labels.dtype == np.int64
        assert labels.min() >= 0
        assert labels.max() < NUM_FOVEAX_CLASSES

    def test_unsupervised_pixels_are_ignored_not_fabricated(self):
        """Pixels no point won must be IGNORE_INDEX, never a real class --
        otherwise the loss trains on invented labels for empty space."""
        _require_dataset()
        arch_cfg = _require_arch_cfg()
        from src.training.rellis3d_dataset import FrameRef

        ref = FrameRef("00001", "000000")
        _, labels = build_training_sample(RELLIS_ROOT, ref, arch_cfg)

        frame = load_rellis3d_frame(RELLIS_ROOT, ref.sequence, ref.frame)
        points = frame.points.copy()
        rescale_intensity_to_unit_range(points)
        proj = project_points_to_range_image(
            points, arch_cfg, fov_up=OUSTER_FOV_UP, fov_down=OUSTER_FOV_DOWN
        )
        unsupervised = proj.proj_idx <= 0
        assert unsupervised.any()
        assert np.all(labels[unsupervised] == IGNORE_INDEX)

    def test_supervised_labels_match_winning_point(self):
        _require_dataset()
        arch_cfg = _require_arch_cfg()
        from src.training.rellis3d_dataset import FrameRef

        ref = FrameRef("00002", "000100")
        _, labels = build_training_sample(RELLIS_ROOT, ref, arch_cfg)

        frame = load_rellis3d_frame(RELLIS_ROOT, ref.sequence, ref.frame)
        points = frame.points.copy()
        rescale_intensity_to_unit_range(points)
        proj = project_points_to_range_image(
            points, arch_cfg, fov_up=OUSTER_FOV_UP, fov_down=OUSTER_FOV_DOWN
        )
        supervised = proj.proj_idx > 0
        expected = frame.foveax_class_ids[proj.proj_idx[supervised]]
        assert np.array_equal(labels[supervised], expected)

    def test_training_uses_ouster_fov_not_velodyne_default(self):
        """The whole point of the fine-tune is the real sensor geometry; if
        training silently fell back to the checkpoint's HDL-64E FOV the
        projection would differ from inference."""
        _require_dataset()
        arch_cfg = _require_arch_cfg()
        from src.training.rellis3d_dataset import FrameRef

        ref = FrameRef("00001", "000000")
        image, _ = build_training_sample(RELLIS_ROOT, ref, arch_cfg)

        frame = load_rellis3d_frame(RELLIS_ROOT, ref.sequence, ref.frame)
        points = frame.points.copy()
        rescale_intensity_to_unit_range(points)
        ouster = project_points_to_range_image(
            points, arch_cfg, fov_up=OUSTER_FOV_UP, fov_down=OUSTER_FOV_DOWN
        )
        velodyne = project_points_to_range_image(points, arch_cfg)

        assert np.array_equal(image, ouster.image)
        assert not np.array_equal(ouster.image, velodyne.image)


class TestIntensityRescaleSharedWithInference:
    def test_rescales_only_nonzero_range_points(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0, 0.5],  # zero-range padding: must stay untouched
                [1.0, 0.0, 0.0, 0.001],
                [2.0, 0.0, 0.0, 0.003],
            ],
            dtype=np.float32,
        )
        rescale_intensity_to_unit_range(points)
        assert points[0, 3] == pytest.approx(0.5)
        assert points[1, 3] == pytest.approx(0.0)
        assert points[2, 3] == pytest.approx(1.0)

    def test_constant_intensity_does_not_divide_by_zero(self):
        points = np.array(
            [[1.0, 0.0, 0.0, 0.007], [2.0, 0.0, 0.0, 0.007]], dtype=np.float32
        )
        rescale_intensity_to_unit_range(points)
        assert np.all(points[:, 3] == 0.0)

    def test_all_zero_range_is_a_noop(self):
        points = np.zeros((3, 4), dtype=np.float32)
        points[:, 3] = 0.25
        rescale_intensity_to_unit_range(points)
        assert np.all(points[:, 3] == 0.25)


class TestClassWeights:
    def test_upstream_weight_formula(self):
        counts = np.array([1, 1, 2, 0, 0, 0, 0, 0], dtype=np.int64)
        w = class_weights_from_counts(counts, epsilon_w=0.001)
        # content = counts / total, w = 1/(content + eps)
        assert w[0] == pytest.approx(1.0 / (0.25 + 0.001), rel=1e-6)
        assert w[2] == pytest.approx(1.0 / (0.5 + 0.001), rel=1e-6)

    def test_ignore_index_weight_is_zero(self):
        counts = np.ones(NUM_FOVEAX_CLASSES, dtype=np.int64)
        w = class_weights_from_counts(counts)
        assert w[IGNORE_INDEX] == 0.0

    def test_rarer_class_gets_larger_weight(self):
        counts = np.array([1000, 10, 0, 0, 0, 0, 0, 0], dtype=np.int64)
        w = class_weights_from_counts(counts)
        assert w[1] > w[0]

    def test_empty_counts_rejected(self):
        with pytest.raises(ValueError):
            class_weights_from_counts(np.zeros(NUM_FOVEAX_CLASSES, dtype=np.int64))
