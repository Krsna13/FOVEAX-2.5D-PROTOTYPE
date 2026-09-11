"""FOVEAX web dashboard data export bridge.

Runs the real FOVEAX pipeline (real point cloud loading, real
MultiObjectTracker tracking, real semantic classification) over N real
frames and exports one JSON file per frame, for the astrafovea.html web
dashboard to load via fetch(). Every value in the exported JSON traces to
an actual computed output -- nothing here is synthetic or randomly jittered.

Reuses existing, already-tested components rather than reimplementing
pipeline logic:
    - src/perception/rellis3d_loader.py (RELLIS-3D point/label loading)
    - src/perception/semantic_predictor.py (GroundTruthSemanticPredictor,
      MockSemanticPredictor -- SemanticKITTI/RELLIS-3D ground truth + mock)
    - src/perception/salsanext_predictor.py (real SalsaNext inference)
    - src/tracking/multi_object_tracker.py (MultiObjectTracker -- the same
      tracker src/dashboard/data_streamer.py uses)
    - src/perception/object_detector.py (MockObjectDetector)
    - src/perception/semantic_labels.py (FOVEAX_CLASSES, canonical taxonomy)

Does NOT use src/dashboard/data_streamer.py's DataStreamerThread directly:
that class is a PyQt5 QThread, and pulling in PyQt5/Qt event-loop machinery
for a one-shot batch export is unnecessary overhead. Instead this reuses
the same underlying detector/tracker classes DataStreamerThread itself
uses, directly.

Canonical traversability convention (do not invert):
    1.0 = safe, 0.0 = blocked. Safe >= 0.70, Caution 0.40-0.70,
    Blocked < 0.40 (src/06_terrain_traversability.py). The grid formula
    used here is the same one already verified in
    src/dashboard/data_streamer.py::_generate_maps:
        traversability = 1.0 - clip(height_range / 1.0, 0.0, 1.0)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.perception.semantic_labels import FOVEAX_CLASSES, NUM_FOVEAX_CLASSES
from src.perception.object_detector import MockObjectDetector
from src.tracking.multi_object_tracker import MultiObjectTracker

# Real resolution zones from the Problem Statement spec (src/07_adaptive_2point5d_map.py
# SPEC_ZONES_100M), quoted directly -- not re-derived or guessed.
RESOLUTION_ZONES = [
    {"name": "Near", "r_min": 0.0, "r_max": 15.0, "resolution_m": 0.05},
    {"name": "Middle", "r_min": 15.0, "r_max": 35.0, "resolution_m": 0.20},
    {"name": "Far", "r_min": 35.0, "r_max": 100.0, "resolution_m": 0.50},
]

# Canonical traversability convention (src/06_terrain_traversability.py).
TRAVERSABILITY_SAFE_MIN = 0.70
TRAVERSABILITY_CAUTION_MIN = 0.40

# Hazard-cluster severity sub-tiers WITHIN the canonical "Blocked" band
# (traversability < 0.40). This subdivision is specific to the hazard-
# clusters feature (finer-grained triage of already-blocked regions), not
# a replacement of the canonical Safe/Caution/Blocked convention above.
HAZARD_CRITICAL_MAX = 0.20  # min_traversability < 0.20 -> Critical
HAZARD_CAUTION_MAX = 0.40   # 0.20 <= min_traversability < 0.40 -> Caution

# Export grid: covers the full 100m Far-zone range, downsampled to ~150x150
# cells for a compact JSON payload (per-cell values are still real, just
# coarser than the native point cloud resolution).
GRID_EXTENT_M = (-100.0, 100.0, -100.0, 100.0)  # x_min, x_max, y_min, y_max
GRID_TARGET_CELLS = 150


def _load_frame_points_and_gt(
    dataset: str, sequence: str, frame_id: str, data_root: Path | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load one real frame's points, plus ground-truth FOVEAX class ids if available.

    Returns (points (N,4) float32, gt_class_ids (N,) uint8 or None).
    """
    if dataset == "rellis3d":
        from src.perception.rellis3d_loader import load_rellis3d_frame

        root = data_root or Path("C:/dev/data/rellis3d")
        if not root.exists():
            root = Path("data/rellis3d")
        rf = load_rellis3d_frame(root, sequence, frame_id)
        return rf.points, rf.foveax_class_ids

    if dataset == "semantickitti":
        from src.perception.semantic_predictor import (
            extract_semantickitti_ids,
            remap_semantickitti_to_foveax,
        )

        root = data_root or Path("data/semantic_kitti/dataset/sequences")
        seq_dir = root / sequence
        point_path = seq_dir / "velodyne" / f"{frame_id}.bin"
        label_path = seq_dir / "labels" / f"{frame_id}.label"
        if not point_path.exists():
            raise FileNotFoundError(f"Point cloud not found: {point_path}")
        points = np.fromfile(point_path, dtype=np.float32).reshape(-1, 4)
        gt_class_ids = None
        if label_path.exists():
            labels_raw = np.fromfile(label_path, dtype=np.uint32)
            semantic_ids, _ = extract_semantickitti_ids(labels_raw)
            gt_class_ids = remap_semantickitti_to_foveax(semantic_ids)
        return points, gt_class_ids

    raise ValueError(f"Unknown dataset: {dataset}")


def _list_frame_ids(dataset: str, sequence: str, data_root: Path | None) -> list[str]:
    """Enumerate real available frame IDs for a dataset/sequence, sorted."""
    if dataset == "rellis3d":
        root = data_root or Path("C:/dev/data/rellis3d")
        if not root.exists():
            root = Path("data/rellis3d")
        bin_dir = root / "Rellis-3D" / sequence / "os1_cloud_node_kitti_bin"
        return sorted(p.stem for p in bin_dir.glob("*.bin"))

    if dataset == "semantickitti":
        root = data_root or Path("data/semantic_kitti/dataset/sequences")
        bin_dir = root / sequence / "velodyne"
        return sorted(p.stem for p in bin_dir.glob("*.bin"))

    raise ValueError(f"Unknown dataset: {dataset}")


@dataclass
class GridResult:
    rows: int
    cols: int
    resolution_m: float
    extent_m: tuple[float, float, float, float]
    elevation: np.ndarray       # (rows, cols) float, NaN = empty
    traversability: np.ndarray  # (rows, cols) float, NaN = empty
    semantic_class: np.ndarray  # (rows, cols) int16, -1 = empty
    point_density: np.ndarray   # (rows, cols) int32
    overhead_gap: np.ndarray    # (rows, cols) bool


def build_grids(
    points: np.ndarray, class_ids: np.ndarray | None
) -> GridResult:
    """Build elevation/traversability/semantic/density grids from real points.

    Traversability formula matches src/dashboard/data_streamer.py::_generate_maps
    exactly (already verified against the canonical 1.0=safe/0.0=blocked
    convention): traversability = 1.0 - clip(height_range / 1.0, 0.0, 1.0).
    """
    x_min, x_max, y_min, y_max = GRID_EXTENT_M
    cols = GRID_TARGET_CELLS
    rows = GRID_TARGET_CELLS
    res_x = (x_max - x_min) / cols
    res_y = (y_max - y_min) / rows
    resolution_m = (res_x + res_y) / 2.0

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    mask = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
    x, y, z = x[mask], y[mask], z[mask]
    cls = class_ids[mask] if class_ids is not None else None

    total_cells = rows * cols
    elevation = np.full((rows, cols), np.nan, dtype=np.float32)
    traversability = np.full((rows, cols), np.nan, dtype=np.float32)
    semantic_class = np.full((rows, cols), -1, dtype=np.int16)
    point_density = np.zeros((rows, cols), dtype=np.int32)

    overhead_gap = np.zeros((rows, cols), dtype=bool)

    if len(x) == 0:
        return GridResult(rows, cols, resolution_m, GRID_EXTENT_M, elevation,
                           traversability, semantic_class, point_density, overhead_gap)

    col_idx = np.floor((x - x_min) / res_x).astype(np.int32)
    row_idx = np.floor((y - y_min) / res_y).astype(np.int32)
    col_idx = np.clip(col_idx, 0, cols - 1)
    row_idx = np.clip(row_idx, 0, rows - 1)
    flat = row_idx * cols + col_idx

    point_density = np.bincount(flat, minlength=total_cells).reshape(rows, cols).astype(np.int32)

    z_max_arr = np.full(total_cells, -np.inf, dtype=np.float32)
    np.maximum.at(z_max_arr, flat, z)
    z_min_arr = np.full(total_cells, np.inf, dtype=np.float32)
    np.minimum.at(z_min_arr, flat, z)
    occupied = point_density.reshape(-1) > 0

    z_max_arr[~occupied] = np.nan
    z_min_arr[~occupied] = np.nan
    height_range = z_max_arr - z_min_arr
    trav = 1.0 - np.clip(height_range / 1.0, 0.0, 1.0)

    elevation = z_max_arr.reshape(rows, cols)
    traversability = trav.reshape(rows, cols)

    # --- Overhead Gap Detection ---
    clearance_threshold = 2.0
    suspect_mask = (height_range > clearance_threshold) & occupied
    suspect_indices = np.nonzero(suspect_mask)[0] # 1D flat indices

    if len(suspect_indices) > 0:
        # Group points by flat index
        sort_idx = np.argsort(flat)
        sorted_flat = flat[sort_idx]
        sorted_z = z[sort_idx]
        
        # Get split indices for each unique flat index
        unique_flat, split_indices = np.unique(sorted_flat, return_index=True)
        z_groups = np.split(sorted_z, split_indices[1:])
        
        # Map flat_index -> z_array
        z_dict = dict(zip(unique_flat, z_groups))
        
        for idx in suspect_indices:
            cell_z = z_dict.get(idx)
            if cell_z is not None and len(cell_z) >= 2:
                gaps = np.diff(cell_z)
                if np.max(gaps) > clearance_threshold:
                    overhead_gap.flat[idx] = True

    if cls is not None:
        # Dominant (mode) FOVEAX class per occupied cell.
        votes = np.zeros((total_cells, NUM_FOVEAX_CLASSES), dtype=np.int32)
        for c in range(NUM_FOVEAX_CLASSES):
            cmask = cls == c
            if cmask.any():
                np.add.at(votes[:, c], flat[cmask], 1)
        dominant = np.argmax(votes, axis=1).astype(np.int16)
        dominant[~occupied] = -1
        semantic_class = dominant.reshape(rows, cols)

    return GridResult(rows, cols, resolution_m, GRID_EXTENT_M, elevation,
                       traversability, semantic_class, point_density, overhead_gap)


def find_hazard_clusters(grid: GridResult) -> list[dict]:
    """Cluster contiguous traversability<0.40 (canonical Blocked) cells.

    Uses scipy.ndimage.label for connected-component labeling, matching
    the task's explicit request. Each cluster's severity comes from its
    real minimum traversability score; confidence from real mean point
    density, normalized the same way src/10_ai_semantic_2point5d_map.py's
    build_semantic_grid already does for map_confidence (density/20,
    clipped to [0,1]) -- reusing that existing convention rather than
    inventing a new density-to-confidence scale.
    """
    from scipy import ndimage

    x_min, x_max, y_min, y_max = grid.extent_m
    blocked = np.nan_to_num(grid.traversability, nan=1.0) < TRAVERSABILITY_CAUTION_MIN
    labeled, n_clusters = ndimage.label(blocked)

    clusters = []
    cell_area_m2 = grid.resolution_m ** 2
    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = labeled == cluster_id
        rows_idx, cols_idx = np.nonzero(cluster_mask)
        if len(rows_idx) == 0:
            continue

        min_trav = float(np.nanmin(grid.traversability[cluster_mask]))
        mean_density = float(grid.point_density[cluster_mask].mean())
        confidence = float(np.clip(mean_density / 20.0, 0.0, 1.0))

        centroid_row = float(rows_idx.mean())
        centroid_col = float(cols_idx.mean())
        world_x = x_min + (centroid_col + 0.5) * grid.resolution_m
        world_y = y_min + (centroid_row + 0.5) * grid.resolution_m
        distance_m = float(np.hypot(world_x, world_y))

        severity = "Critical" if min_trav < HAZARD_CRITICAL_MAX else "Caution"

        clusters.append({
            "cluster_id": cluster_id,
            "distance_m": round(distance_m, 2),
            "area_m2": round(len(rows_idx) * cell_area_m2, 2),
            "min_traversability": round(min_trav, 3),
            "severity": severity,
            "confidence": round(confidence, 3),
        })

    clusters.sort(key=lambda c: c["distance_m"])
    return clusters


def compute_ego_terrain_status(grid: GridResult) -> dict:
    """Compute real terrain status ahead of the vehicle."""
    x_min, x_max, y_min, y_max = grid.extent_m
    
    # Ego ROI: 2m to 10m ahead, -2m to 2m sideways
    roi_x_min, roi_x_max = 2.0, 10.0
    roi_y_min, roi_y_max = -2.0, 2.0
    
    res_x = grid.resolution_m
    res_y = grid.resolution_m
    
    col_min = int(max(0, np.floor((roi_x_min - x_min) / res_x)))
    col_max = int(min(grid.cols, np.ceil((roi_x_max - x_min) / res_x)))
    row_min = int(max(0, np.floor((roi_y_min - y_min) / res_y)))
    row_max = int(min(grid.rows, np.ceil((roi_y_max - y_min) / res_y)))
    
    roi_trav = grid.traversability[row_min:row_max, col_min:col_max]
    roi_density = grid.point_density[row_min:row_max, col_min:col_max]
    
    valid_mask = ~np.isnan(roi_trav)
    if not np.any(valid_mask):
        return {"status": "UNKNOWN", "confidence": "N/A", "avg_traversability": None}
        
    avg_trav = float(np.nanmean(roi_trav[valid_mask]))
    mean_density = float(roi_density[valid_mask].mean())
    confidence = float(np.clip(mean_density / 20.0, 0.0, 1.0))
    
    if avg_trav >= TRAVERSABILITY_SAFE_MIN:
        status = "DRIVABLE"
    elif avg_trav >= TRAVERSABILITY_CAUTION_MIN:
        status = "CAUTION"
    else:
        status = "NON-DRIVABLE"
        
    return {
        "status": status,
        "confidence": round(confidence, 3),
        "avg_traversability": round(avg_trav, 3)
    }


def make_semantic_predictor(predictor: str, args: argparse.Namespace):
    """Construct the requested semantic predictor, reusing existing classes."""
    if predictor == "mock":
        from src.perception.semantic_predictor import MockSemanticPredictor
        return MockSemanticPredictor()
    if predictor == "salsanext":
        from src.perception.salsanext_predictor import SalsaNextPredictor
        if not (args.salsanext_repo and args.checkpoint and args.config):
            raise ValueError(
                "--predictor salsanext requires --salsanext-repo, --checkpoint, and --config."
            )
        return SalsaNextPredictor(
            repo_path=Path(args.salsanext_repo),
            checkpoint_path=Path(args.checkpoint),
            config_path=Path(args.config),
            device=args.device,
            strict=True,
            fov_up=args.fov_up,
            fov_down=args.fov_down,
            rescale_intensity=args.rescale_intensity,
        )
    return None  # ground_truth: handled by loader-provided labels directly


def export_frames(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = _list_frame_ids(args.source, args.sequence, Path(args.data_root) if args.data_root else None)
    if not frame_ids:
        raise FileNotFoundError(
            f"No frames found for dataset={args.source} sequence={args.sequence}."
        )
    frame_ids = frame_ids[: args.frames]

    predictor = make_semantic_predictor(args.predictor, args)
    tracker = MultiObjectTracker()
    detector = MockObjectDetector()

    provenance = f"{args.predictor}_{args.source}"

    exported = []
    for i, frame_id in enumerate(frame_ids):
        t_start = time.perf_counter()

        points, gt_class_ids = _load_frame_points_and_gt(
            args.source, args.sequence, frame_id,
            Path(args.data_root) if args.data_root else None,
        )

        class_ids = None
        class_confidence = "Unavailable"
        if args.predictor == "ground_truth":
            class_ids = gt_class_ids
        elif predictor is not None:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                prediction = predictor.predict(points)
            class_ids = prediction.class_ids
            if hasattr(prediction, 'class_confidence') and prediction.class_confidence is not None:
                class_confidence = prediction.class_confidence

        detections = detector.detect(points, timestamp_s=float(i))
        tracks = tracker.update(detections, timestamp_s=float(i))

        grid = build_grids(points, class_ids)
        hazard_clusters = find_hazard_clusters(grid)
        ego_terrain_status = compute_ego_terrain_status(grid)

        latency_s = time.perf_counter() - t_start
        latency_ms = latency_s * 1000.0
        fps = 1.0 / latency_s if latency_s > 0 else 0.0

        tracked_objects = []
        for t in tracks:
            vx, vy = float(t.state[3]), float(t.state[4])
            tracked_objects.append({
                "id": t.track_id,
                "class_name": t.class_name,
                "x": round(float(t.state[0]), 3),
                "y": round(float(t.state[1]), 3),
                "vx": round(vx, 3),
                "vy": round(vy, 3),
                "speed_mps": round(float(np.hypot(vx, vy)), 3),
                "dynamic": bool(t.dynamic),
            })

        def _grid_to_list(arr: np.ndarray, nan_value=None):
            out = []
            for row in arr.tolist():
                out.append([nan_value if (v != v) else v for v in row])  # v!=v -> NaN check
            return out

        frame_json = {
            "frame_idx": i,
            "frame_id": frame_id,
            "dataset": args.source,
            "sequence": args.sequence,
            "predictor": args.predictor,
            "provenance": provenance,
            "point_count": int(len(points)),
            "fps": round(fps, 2),
            "latency_ms": round(latency_ms, 3),
            "resolution_zones": RESOLUTION_ZONES,
            "grid": {
                "rows": grid.rows,
                "cols": grid.cols,
                "resolution_m": round(grid.resolution_m, 4),
                "extent_m": list(grid.extent_m),
                "elevation": _grid_to_list(grid.elevation),
                "traversability": _grid_to_list(grid.traversability),
                "semantic_class": [[None if v == -1 else int(v) for v in row]
                                    for row in grid.semantic_class.tolist()],
                "point_density": grid.point_density.tolist(),
                "overhead_gap": grid.overhead_gap.tolist(),
            },
            "tracked_objects": tracked_objects,
            "hazard_clusters": hazard_clusters,
            "ego_terrain_status": ego_terrain_status,
            "class_confidence": class_confidence,
        }

        frame_path = out_dir / f"frame_{i:04d}.json"
        frame_path.write_text(json.dumps(frame_json), encoding="utf-8")
        exported.append(frame_path.name)
        print(f"[{i+1}/{len(frame_ids)}] {frame_path.name}: "
              f"{len(points):,} pts, {len(tracked_objects)} tracks, "
              f"{len(hazard_clusters)} hazard clusters, "
              f"{latency_ms:.1f} ms")

    manifest = {
        "dataset": args.source,
        "sequence": args.sequence,
        "predictor": args.predictor,
        "provenance": provenance,
        "num_frames": len(exported),
        "frame_files": exported,
        "grid": {
            "rows": GRID_TARGET_CELLS,
            "cols": GRID_TARGET_CELLS,
            "extent_m": list(GRID_EXTENT_M),
        },
        "resolution_zones": RESOLUTION_ZONES,
        "canonical_traversability_convention": (
            "1.0 = safe, 0.0 = blocked. Safe >= 0.70, Caution 0.40-0.70, "
            "Blocked < 0.40 (src/06_terrain_traversability.py). Hazard-cluster "
            f"severity sub-tiers the Blocked band: Critical < {HAZARD_CRITICAL_MAX}, "
            f"Caution {HAZARD_CRITICAL_MAX}-{HAZARD_CAUTION_MAX}."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest written: {out_dir / 'manifest.json'}")
    return out_dir


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export real FOVEAX pipeline output as JSON for the web dashboard."
    )
    parser.add_argument("--source", choices=["semantickitti", "rellis3d"], required=True,
                         help="Real dataset to load frames from.")
    parser.add_argument("--sequence", default=None,
                         help="Sequence ID (default: '00' for semantickitti, '00000' for rellis3d).")
    parser.add_argument("--frames", type=int, default=5, help="Number of real frames to export.")
    parser.add_argument("--predictor", choices=["ground_truth", "mock", "salsanext"],
                         default="ground_truth", help="Semantic class source (default: ground_truth).")
    parser.add_argument("--data-root", default=None, help="Override dataset root directory.")
    parser.add_argument("--output-dir", default="outputs/phase9/web_export",
                         help="Directory to write frame_NNNN.json + manifest.json.")
    # SalsaNext-specific (only used when --predictor salsanext)
    parser.add_argument("--salsanext-repo", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fov-up", type=float, default=None)
    parser.add_argument("--fov-down", type=float, default=None)
    parser.add_argument("--rescale-intensity", action="store_true", default=False)
    args = parser.parse_args(argv)
    if args.sequence is None:
        args.sequence = "00000" if args.source == "rellis3d" else "00"
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    export_frames(args)


if __name__ == "__main__":
    main()
