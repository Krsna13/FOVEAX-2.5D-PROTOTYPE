"""RELLIS-3D range-image dataset for fine-tuning SalsaNext in FOVEAX taxonomy.

Preprocessing here is deliberately *identical* to the inference path in
`src/perception/salsanext_predictor.py::predict()`:

  1. `load_rellis3d_frame()` -- same loader, same RELLIS3D_TO_FOVEAX mapping,
     same sky exclusion.
  2. Min-max intensity rescale to [0, 1] over non-zero-range points (RELLIS-3D
     raw intensities are ~0.0001-0.03; the checkpoint's training domain was
     ~0-1).
  3. `project_points_to_range_image()` with the real Ouster OS1-64 vertical FOV
     (fov_up=17.02, fov_down=-16.44) instead of the checkpoint's Velodyne
     HDL-64E assumption, and the checkpoint's own img_means/img_stds.

If training and inference preprocessing diverged, the fine-tune would not
transfer -- so both call the same two functions rather than reimplementing
the projection.

Per-pixel training targets come from the point that *won* each pixel in the
projection collision policy (closest point wins), read straight off
`RangeProjection.proj_idx`. Pixels no point won are set to IGNORE_INDEX.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.perception.range_projection import (
    project_points_to_range_image,
    rescale_intensity_to_unit_range,
)
from src.perception.rellis3d_loader import load_rellis3d_frame

# Real Ouster OS1-64 vertical FOV, as already used and validated at inference
# time (docs/validation_results.md section 2.2).
OUSTER_FOV_UP = 17.02
OUSTER_FOV_DOWN = -16.44

# FOVEAX class 7 is UNKNOWN. Unlike upstream SalsaNext (which ignores class 0,
# its own "unlabeled"), FOVEAX class 0 is DRIVABLE_GROUND -- a real, supervised
# class. Ignoring 0 here would silently discard the single largest and most
# safety-relevant class in the dataset.
IGNORE_INDEX = 7

# Sequence held out entirely from training and validation. Chosen as 00000
# because both published cross-domain baselines in docs/validation_results.md
# (6.65% un-adapted, 22.26% sensor-adapted) were measured on it, so the
# fine-tuned number is directly comparable against them on data the fine-tune
# never saw.
TEST_SEQUENCE = "00000"
TRAIN_VAL_SEQUENCES = ("00001", "00002", "00003", "00004")

# Contiguous tail fraction of each train/val sequence reserved for validation.
VAL_FRACTION = 0.15

# Frames dropped between the train block and the val block of each sequence.
# RELLIS-3D is recorded at ~10 Hz, so frames adjacent across the boundary are
# near-duplicates; without a gap the val set would effectively be visible
# during training.
SPLIT_BUFFER_FRAMES = 10


@dataclass(frozen=True)
class FrameRef:
    """One addressable RELLIS-3D frame."""

    sequence: str
    frame: str


def list_sequence_frames(rellis3d_root: Path | str, sequence: str) -> list[str]:
    """Return sorted 6-digit frame IDs present for a sequence."""
    bin_dir = Path(rellis3d_root) / "Rellis-3D" / sequence / "os1_cloud_node_kitti_bin"
    if not bin_dir.is_dir():
        raise FileNotFoundError(f"RELLIS-3D sequence directory not found: {bin_dir}")
    return sorted(p.stem for p in bin_dir.glob("*.bin"))


def build_splits(
    rellis3d_root: Path | str,
    train_stride: int = 1,
    val_stride: int = 1,
) -> dict[str, list[FrameRef]]:
    """Build the deterministic train/val/test split.

    Test is the whole of `TEST_SEQUENCE`. Within every other sequence the
    final `VAL_FRACTION` of frames (contiguous, i.e. a held-out tail segment
    rather than an interleaved sample) becomes validation, separated from the
    training block by `SPLIT_BUFFER_FRAMES` dropped frames.

    Strides subsample *after* splitting, so a stride never moves a frame
    across the train/val boundary.
    """
    if train_stride < 1 or val_stride < 1:
        raise ValueError("strides must be >= 1")

    train: list[FrameRef] = []
    val: list[FrameRef] = []

    for seq in TRAIN_VAL_SEQUENCES:
        frames = list_sequence_frames(rellis3d_root, seq)
        n = len(frames)
        n_val = int(round(n * VAL_FRACTION))
        val_start = n - n_val
        train_end = val_start - SPLIT_BUFFER_FRAMES
        if train_end <= 0:
            raise ValueError(f"sequence {seq} too short ({n} frames) to split")

        train += [FrameRef(seq, f) for f in frames[0:train_end:train_stride]]
        val += [FrameRef(seq, f) for f in frames[val_start::val_stride]]

    test_frames = list_sequence_frames(rellis3d_root, TEST_SEQUENCE)
    test = [FrameRef(TEST_SEQUENCE, f) for f in test_frames]

    return {"train": train, "val": val, "test": test}


def build_training_sample(
    rellis3d_root: Path | str,
    ref: FrameRef,
    arch_cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one frame and return (image (5,H,W) float32, labels (H,W) int64)."""
    frame = load_rellis3d_frame(rellis3d_root, ref.sequence, ref.frame)
    points = frame.points.copy()
    rescale_intensity_to_unit_range(points)

    projection = project_points_to_range_image(
        points, arch_cfg, fov_up=OUSTER_FOV_UP, fov_down=OUSTER_FOV_DOWN
    )

    # `mask` is upstream's `proj_idx > 0`, which the projection applies to the
    # input image. Loss must ignore exactly the pixels the model sees as
    # zeroed, so the same predicate selects the supervised pixels here.
    supervised = projection.proj_idx > 0
    labels = np.full(projection.proj_idx.shape, IGNORE_INDEX, dtype=np.int64)
    labels[supervised] = frame.foveax_class_ids[projection.proj_idx[supervised]]

    return projection.image, labels


class Rellis3DRangeDataset:
    """torch.utils.data.Dataset over RELLIS-3D range images.

    Kept free of a hard torch import at module scope so the split/frequency
    helpers remain usable (and testable) without torch installed.
    """

    def __init__(
        self,
        rellis3d_root: Path | str,
        refs: list[FrameRef],
        arch_cfg: dict,
    ):
        self.rellis3d_root = Path(rellis3d_root)
        self.refs = list(refs)
        self.arch_cfg = arch_cfg

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int):
        import torch

        image, labels = build_training_sample(
            self.rellis3d_root, self.refs[index], self.arch_cfg
        )
        return torch.from_numpy(image), torch.from_numpy(labels)


def compute_class_frequencies(
    rellis3d_root: Path | str,
    refs: list[FrameRef],
    num_classes: int,
    arch_cfg: dict,
    progress: bool = False,
) -> np.ndarray:
    """Count supervised pixels per FOVEAX class over `refs`.

    Counts are taken on the *projected range image*, not the raw point cloud,
    because that is what the loss actually sees.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for i, ref in enumerate(refs):
        _, labels = build_training_sample(rellis3d_root, ref, arch_cfg)
        valid = labels != IGNORE_INDEX
        counts += np.bincount(labels[valid], minlength=num_classes)
        if progress and (i + 1) % 50 == 0:
            print(f"  class-frequency scan: {i + 1}/{len(refs)} frames", flush=True)
    return counts


def class_weights_from_counts(counts: np.ndarray, epsilon_w: float = 0.001) -> np.ndarray:
    """Upstream SalsaNext weighting: w = 1 / (content + epsilon_w).

    `content` is each class's share of supervised pixels. IGNORE_INDEX gets
    weight 0 so it can never contribute to the loss even if a pixel slipped
    through.
    """
    total = counts.sum()
    if total == 0:
        raise ValueError("no supervised pixels found -- cannot derive class weights")
    content = counts.astype(np.float64) / float(total)
    weights = 1.0 / (content + epsilon_w)
    weights[IGNORE_INDEX] = 0.0
    return weights.astype(np.float32)
