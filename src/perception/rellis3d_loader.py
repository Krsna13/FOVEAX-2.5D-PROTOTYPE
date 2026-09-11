"""RELLIS-3D dataset loader.

RELLIS-3D ships its LiDAR data in a SemanticKITTI-*compatible* file format
(.bin point clouds, .label per-point labels) but under a different directory
layout than SemanticKITTI itself. Confirmed directly against the real
downloaded dataset (2026-09-11), not assumed from documentation:

    <rellis3d_root>/Rellis-3D/<5-digit sequence>/os1_cloud_node_kitti_bin/<6-digit frame>.bin
    <rellis3d_root>/Rellis-3D/<5-digit sequence>/os1_cloud_node_semantickitti_label_id/<6-digit frame>.label

    .bin   : float32, N=131072 points x 4 columns [x, y, z, intensity]
             (Ouster OS1-64 organized scan, 64 x 2048).
    .label : uint32, one value per point, SemanticKITTI-style packing
             (lower 16 bits = semantic class ID, upper 16 bits = instance ID).

This differs from SemanticKITTI's own layout (`sequences/<2-digit>/{velodyne,labels}/`),
which is why this is a separate loader rather than a reuse of
`load_velodyne_scan` / `load_label_file` from `src/10_ai_semantic_2point5d_map.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.perception.semantic_labels import RELLIS3D_EXCLUDED_CLASSES, RELLIS3D_TO_FOVEAX

# Lookup table for vectorized raw-ID -> FOVEAX-ID remap. Sized to comfortably
# cover the real ontology's max ID (34) plus headroom; validated against
# RELLIS3D_TO_FOVEAX's keys at import time so it can never silently drift.
_MAX_KNOWN_RAW_ID = max(RELLIS3D_TO_FOVEAX)
_LOOKUP_SIZE = _MAX_KNOWN_RAW_ID + 1
_FOVEAX_LOOKUP = np.full(_LOOKUP_SIZE, -1, dtype=np.int16)
for _raw_id, _foveax_id in RELLIS3D_TO_FOVEAX.items():
    _FOVEAX_LOOKUP[_raw_id] = _foveax_id


@dataclass(frozen=True)
class Rellis3DFrame:
    """One loaded, sky-filtered, FOVEAX-remapped RELLIS-3D frame.

    Attributes
    ----------
    points : np.ndarray, shape (M, 4), dtype float32
        [x, y, z, intensity] for the M points that survived sky exclusion
        (M <= n_points_total).
    foveax_class_ids : np.ndarray, shape (M,), dtype uint8
        FOVEAX class ID for each surviving point, already remapped via
        RELLIS3D_TO_FOVEAX.
    n_points_total : int
        Point count before sky exclusion (== label count in the source file).
    n_excluded_sky : int
        Number of points dropped because their raw class was `sky`.
    """

    points: np.ndarray
    foveax_class_ids: np.ndarray
    n_points_total: int
    n_excluded_sky: int


def _sequence_dir(rellis3d_root: Path, sequence: str) -> Path:
    return Path(rellis3d_root) / "Rellis-3D" / sequence


def load_rellis3d_frame(
    rellis3d_root: Path | str,
    sequence: str,
    frame: str,
) -> Rellis3DFrame:
    """Load one RELLIS-3D frame's points + FOVEAX-remapped labels.

    Parameters
    ----------
    rellis3d_root : Path | str
        Directory containing the extracted `Rellis-3D/` folder
        (e.g. ``C:/dev/data/rellis3d``).
    sequence : str
        5-digit RELLIS-3D sequence ID, e.g. ``"00000"``.
    frame : str
        6-digit frame ID, e.g. ``"000000"``.

    Returns
    -------
    Rellis3DFrame

    Raises
    ------
    FileNotFoundError
        If the point cloud or label file doesn't exist.
    ValueError
        If a file is malformed, point/label counts don't match, or any
        label contains a raw semantic ID not present in RELLIS3D_TO_FOVEAX
        or RELLIS3D_EXCLUDED_CLASSES (refuses to silently fabricate a class
        for unmapped data).
    """
    seq_dir = _sequence_dir(rellis3d_root, sequence)
    bin_path = seq_dir / "os1_cloud_node_kitti_bin" / f"{frame}.bin"
    label_path = seq_dir / "os1_cloud_node_semantickitti_label_id" / f"{frame}.label"

    if not bin_path.exists():
        raise FileNotFoundError(f"RELLIS-3D point cloud not found: {bin_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"RELLIS-3D label file not found: {label_path}")

    raw_points = np.fromfile(bin_path, dtype=np.float32)
    if raw_points.size % 4 != 0:
        raise ValueError(
            f"{bin_path}: point file size ({raw_points.size} float32 values) "
            "is not divisible by 4 -- expected [x, y, z, intensity] layout."
        )
    points = raw_points.reshape(-1, 4)
    n_points_total = points.shape[0]

    labels_raw = np.fromfile(label_path, dtype=np.uint32)
    if len(labels_raw) != n_points_total:
        raise ValueError(
            f"Point-label mismatch for sequence {sequence} frame {frame}: "
            f"{n_points_total} points but {len(labels_raw)} labels "
            f"({bin_path.name} vs {label_path.name})."
        )

    semantic_ids = (labels_raw & 0xFFFF).astype(np.int64)

    known_ids = set(RELLIS3D_TO_FOVEAX) | set(RELLIS3D_EXCLUDED_CLASSES)
    unique_ids = np.unique(semantic_ids)
    unknown_ids = sorted(int(i) for i in unique_ids if int(i) not in known_ids)
    if unknown_ids:
        raise ValueError(
            f"{label_path}: found raw semantic class ID(s) {unknown_ids} not "
            "present in RELLIS3D_TO_FOVEAX or RELLIS3D_EXCLUDED_CLASSES -- "
            "refusing to silently fabricate a FOVEAX class for unmapped data. "
            "Update the ontology mapping in src/perception/semantic_labels.py "
            "if this is a legitimate new class."
        )

    excluded_mask = np.isin(semantic_ids, list(RELLIS3D_EXCLUDED_CLASSES))
    kept_points = points[~excluded_mask].astype(np.float32)
    kept_semantic_ids = semantic_ids[~excluded_mask]
    n_excluded_sky = int(excluded_mask.sum())

    if kept_points.shape[0] == 0:
        raise ValueError(
            f"Sequence {sequence} frame {frame}: every one of {n_points_total} "
            "points was excluded (sky) -- no usable points remain."
        )

    foveax_class_ids = _FOVEAX_LOOKUP[kept_semantic_ids].astype(np.uint8)

    return Rellis3DFrame(
        points=kept_points,
        foveax_class_ids=foveax_class_ids,
        n_points_total=n_points_total,
        n_excluded_sky=n_excluded_sky,
    )
