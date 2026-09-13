"""Stage 6: convert real RELLIS-3D data into SalsaNext's expected on-disk
layout, for fine-tuning the official SalsaNext architecture on RELLIS-3D.

SalsaNext's own training code (external/SalsaNext/train/tasks/semantic/
dataset/kitti/parser.py) expects data as:

    <root>/sequences/<2-digit synthetic seq>/velodyne/<6-digit>.bin
    <root>/sequences/<2-digit synthetic seq>/labels/<6-digit>.label

and applies its own `learning_map` (from data_cfg.yaml) to the raw label ID
in each .label file's lower 16 bits at load time -- exactly like real
SemanticKITTI data. To fine-tune the pretrained checkpoint with ZERO
changes to its 20-class output head (so src/perception/salsanext_predictor.py
needs no changes), every RELLIS-3D raw label is remapped here to one
canonical real SemanticKITTI raw class ID before being written out, via:

    RELLIS-3D raw ID -> FOVEAX class (RELLIS3D_TO_FOVEAX, already trusted)
                      -> canonical real KITTI raw ID (_FOVEAX_TO_CANONICAL_KITTI, below)

The KITTI raw ID is what gets written into the .label file; SalsaNext's own
learning_map (data_cfg.yaml) then turns it into the same 0..19 train ID
space the pretrained checkpoint already uses.

Split (sequence-level test holdout, contiguous per-sequence train/valid
split to avoid interleaving near-identical adjacent frames):
  - Real RELLIS-3D sequence 00004 is held out ENTIRELY as the test set --
    never seen during training or validation.
  - Real RELLIS-3D sequences 00000-00003 are each split into a contiguous
    first-85%/last-15% train/valid block (not random-interleaved, so
    train and valid frames from the same sequence are never adjacent
    except at the single per-sequence cut point).

Point clouds are NOT hardlinked as-is: RELLIS-3D's real raw intensities
(~0.0001-0.01) are on a completely different scale from the SemanticKITTI
domain (~0-1) the pretrained checkpoint's img_means/img_stds were computed
against -- the exact same sensor-adaptation issue already solved at
inference time in src/perception/salsanext_predictor.py's
`rescale_intensity` option. SalsaNext's own training data loader
(external/SalsaNext/train/tasks/semantic/dataset/kitti/parser.py ->
common/laserscan.py) does NOT do that rescaling -- it just normalizes
whatever raw intensity is in the .bin file with img_means/img_stds. So the
same per-frame min-max-to-[0,1] rescale used at inference is applied here
at conversion time, baked into the written .bin files, using the identical
formula as SalsaNextPredictor.predict()'s rescale_intensity=True path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.perception.semantic_labels import RELLIS3D_TO_FOVEAX, RELLIS3D_EXCLUDED_CLASSES

# One canonical real SemanticKITTI raw class ID per FOVEAX class -- the
# bridge that lets a 20-class SalsaNext head be fine-tuned on RELLIS-3D
# without any architecture change. Chosen as the closest semantic match
# within each FOVEAX bucket.
_FOVEAX_TO_CANONICAL_KITTI: dict[int, int] = {
    0: 40,  # DRIVABLE_GROUND -> "road"
    1: 72,  # ROUGH_TERRAIN   -> "terrain"
    2: 70,  # VEGETATION      -> "vegetation"
    3: 50,  # BUILDING_WALL   -> "building"
    4: 80,  # SOLID_OBSTACLE  -> "pole"
    5: 10,  # VEHICLE         -> "car"
    6: 30,  # PEDESTRIAN      -> "person"
    7: 0,   # UNKNOWN         -> "unlabeled" (learning_ignore[0] = True)
}

# RELLIS-3D raw ID -> canonical real KITTI raw ID, composed from the two
# tables above. "sky" (excluded from the FOVEAX grid) maps to KITTI
# "unlabeled" (0), which SalsaNext's own learning_ignore treats as ignored
# in the loss -- the correct behavior for a non-terrain/non-obstacle class.
_RELLIS_RAW_TO_KITTI_RAW: dict[int, int] = {
    raw_id: _FOVEAX_TO_CANONICAL_KITTI[foveax_id]
    for raw_id, foveax_id in RELLIS3D_TO_FOVEAX.items()
}
for _excluded_id in RELLIS3D_EXCLUDED_CLASSES:
    _RELLIS_RAW_TO_KITTI_RAW[_excluded_id] = 0  # sky -> unlabeled

_MAX_RAW_ID = max(_RELLIS_RAW_TO_KITTI_RAW)
_LOOKUP = np.zeros(_MAX_RAW_ID + 1, dtype=np.uint32)
for _raw_id, _kitti_id in _RELLIS_RAW_TO_KITTI_RAW.items():
    _LOOKUP[_raw_id] = _kitti_id


def _rescale_intensity(points: np.ndarray) -> np.ndarray:
    """Per-frame min-max rescale of intensity to [0, 1] on valid (non-zero
    range) points -- identical formula to SalsaNextPredictor.predict()'s
    rescale_intensity=True path, so training and inference apply the same
    sensor adaptation."""
    out = points.copy()
    depth = np.linalg.norm(out[:, :3], axis=1)
    valid_idx = np.flatnonzero(depth > 0.0)
    if valid_idx.size > 0:
        intensities = out[valid_idx, 3]
        i_min = float(intensities.min())
        i_max = float(intensities.max())
        if i_max > i_min:
            out[valid_idx, 3] = (intensities - i_min) / (i_max - i_min)
        else:
            out[valid_idx, 3] = 0.0
    return out


def _convert_frame(bin_src: Path, label_src: Path, bin_dst: Path, label_dst: Path) -> None:
    if bin_dst.exists():
        return

    raw_points = np.fromfile(bin_src, dtype=np.float32).reshape(-1, 4)

    labels_raw = np.fromfile(label_src, dtype=np.uint32)
    semantic_ids = labels_raw & 0xFFFF
    unknown_mask = semantic_ids > _MAX_RAW_ID
    if unknown_mask.any():
        bad = sorted(set(int(x) for x in semantic_ids[unknown_mask]))
        raise ValueError(f"{label_src}: raw semantic IDs {bad} have no RELLIS3D_TO_FOVEAX/excluded mapping")

    # RELLIS-3D's Ouster scans are *organized* (fixed 64x2048 grid): no-return
    # beams are zero-padded to exactly (0,0,0,0). Official SalsaNext's own
    # training laserscan.py (external/SalsaNext/train/common/laserscan.py)
    # assumes every point is a real return (true for KITTI's sparse .bin
    # format) and does not guard against depth==0 -- arcsin(z/0) produces
    # NaN, which crashes range-image assignment. FOVEAX's own inference-time
    # projection (src/perception/range_projection.py) already excludes these
    # same zero-range points via `valid = depth_all > 0.0`; dropping them
    # here at conversion time keeps training data consistent with that and
    # avoids further changes to the external repo.
    depth = np.linalg.norm(raw_points[:, :3], axis=1)
    keep = depth > 0.0

    rescaled = _rescale_intensity(raw_points[keep])
    rescaled.astype(np.float32).tofile(bin_dst)

    kitti_ids = _LOOKUP[semantic_ids[keep]]
    kitti_ids.astype(np.uint32).tofile(label_dst)


def build_split(rellis_root: Path, out_root: Path) -> dict:
    src_root = rellis_root / "Rellis_3D_os1_cloud_node_kitti_bin" / "Rellis-3D"
    lbl_root = rellis_root / "Rellis_3D_os1_cloud_node_semantickitti_label_id_20210614" / "Rellis-3D"

    sequences = ["00000", "00001", "00002", "00003", "00004"]
    frame_lists = {}
    for seq in sequences:
        bins = sorted((src_root / seq / "os1_cloud_node_kitti_bin").glob("*.bin"))
        frame_lists[seq] = [p.stem for p in bins]

    synthetic_seq_map = {}  # synthetic 2-digit id -> (real_seq, frame_ids, split_name)
    next_id = 0

    def alloc(real_seq: str, frame_ids: list[str], split_name: str) -> str:
        nonlocal next_id
        sid = f"{next_id:02d}"
        next_id += 1
        synthetic_seq_map[sid] = (real_seq, frame_ids, split_name)
        return sid

    train_ids, valid_ids, test_ids = [], [], []
    for seq in ["00000", "00001", "00002", "00003"]:
        frames = frame_lists[seq]
        n_train = round(len(frames) * 0.85)
        sid_train = alloc(seq, frames[:n_train], "train")
        sid_valid = alloc(seq, frames[n_train:], "valid")
        train_ids.append(int(sid_train))
        valid_ids.append(int(sid_valid))

    sid_test = alloc("00004", frame_lists["00004"], "test")
    test_ids.append(int(sid_test))

    report = {"sequences": {}, "split": {"train": train_ids, "valid": valid_ids, "test": test_ids}}

    for sid, (real_seq, frame_ids, split_name) in synthetic_seq_map.items():
        vel_dir = out_root / "sequences" / sid / "velodyne"
        lbl_dir = out_root / "sequences" / sid / "labels"
        vel_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frame_ids):
            bin_src = src_root / real_seq / "os1_cloud_node_kitti_bin" / f"{frame}.bin"
            label_src = lbl_root / real_seq / "os1_cloud_node_semantickitti_label_id" / f"{frame}.label"
            out_name = f"{i:06d}"
            _convert_frame(bin_src, label_src, vel_dir / f"{out_name}.bin", lbl_dir / f"{out_name}.label")
        report["sequences"][sid] = {
            "real_sequence": real_seq,
            "split": split_name,
            "n_frames": len(frame_ids),
            "real_frame_range": [frame_ids[0], frame_ids[-1]] if frame_ids else None,
        }

    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rellis-root", default=r"C:\FOVEAX 2.5D\data\rellis3d")
    ap.add_argument("--out-root", default=r"C:\FOVEAX 2.5D\data\rellis3d_kitti_format")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report = build_split(Path(args.rellis_root), out_root)

    report_path = out_root / "conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nWrote conversion report to {report_path}")
