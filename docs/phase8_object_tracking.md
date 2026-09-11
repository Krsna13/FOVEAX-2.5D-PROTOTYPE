# FOVEAX Phase 8A — 3D Object Detection and Multi-Object Tracking

## Overview

Phase 8A adds a **model-agnostic 3D object detection and multi-object tracking** pipeline to FOVEAX, separate from the terrain analysis branch.

```
Terrain branch:
    Slope, roughness, potholes, rocks
    → Traversability map

Object branch (Phase 8A):
    Geometric obstacle clusters
    → 3D boxes [class, x, y, z, l, w, h, yaw, confidence]
    → Track ID + velocity + direction + dynamic flag
    → Dynamic-object visual output (BEV images, JSONL, metrics)
```

**This is an engineering and visualization foundation. It is NOT real AI object detection.**

## Pipeline

```
Input point cloud (Nx4: x, y, z, intensity)
    ↓
MockObjectDetector — deterministic Euclidean clustering above ground
    ↓
3D bounding boxes per cluster (axis-aligned, class_name="unknown_obstacle")
    ↓
MultiObjectTracker — Kalman filter + Hungarian matching
    ↓
Track ID, velocity, direction, track confidence, dynamic flag
    ↓
BEV visual output + JSONL logs + metrics
```

## Current Components

### src/perception/object_detector.py

- **`Detection3D`** dataclass: `center_xyz`, `size_lwh`, `yaw_rad`, `class_id`, `class_name`, `confidence`, `source`, `metadata`
- **`ObjectDetector`** abstract interface: `detect(points_xyzi, timestamp_s) -> list[Detection3D]`
- **`GroundTruthBoxProvider`**: placeholder — raises NotImplementedError (SemanticKITTI has no 3D boxes)
- **`MockObjectDetector`**: deterministic geometric clustering baseline (NOT AI)
- **`OpenPCDetDetector`**: Phase 8B placeholder (raises NotImplementedError)

### src/tracking/multi_object_tracker.py

- **`TrackState`** dataclass: track_id, class_name, state vector, covariance, size, yaw, age, hits, missed_frames, confidence, dynamic flag, source
- **`MultiObjectTracker`**: constant-velocity 3D Kalman filter + Hungarian assignment (SciPy) / greedy fallback
- Configurable: gating threshold, max missed frames, dynamic speed threshold, process/measurement noise

### src/11_object_detection_tracking.py

- CLI entry point with `--source sample` and `--source semantickitti`
- Sample mode: synthetic cloud with moving obstacle cluster
- SemanticKITTI mode: reads .bin frames if available
- Saves: BEV images, detections.jsonl, tracks.jsonl, phase8_metrics.txt, phase8_summary.png

## MockObjectDetector — Geometric Clustering Baseline

**This is NOT an AI model.** It performs:

1. Ground removal (points below `ground_threshold_m`)
2. Voxel pre-filtering for performance
3. BFS clustering within `cluster_max_dist_m`
4. Axis-aligned 3D bounding box per cluster
5. Deterministic confidence from point count and compactness
6. Class assignment: `class_name="unknown_obstacle"`

Confidence is a deterministic function of point count saturation and cluster compactness, bounded in [0, 1].

## Future Phase 8B — Real AI Detection

Phase 8B will integrate **OpenPCDet** with either:

- **PointPillars** (fast BEV/pillar-based LiDAR detector)
- **CenterPoint** (stronger 3D detection/tracking, joint detection/tracking)

### Recommended data for real 3D box detection

- **nuScenes** — 3D detection GT available, CenterPoint native support
- **Waymo Open Dataset** — 3D detection GT available
- **KITTI 3D Object Detection** — classic benchmark, PointPillars native support
- **CARLA** — synthetic with labelled dynamic actors

### Why not SemanticKITTI?

SemanticKITTI provides **per-point semantic and instance labels** but NOT standard 3D object detection boxes (class, x, y, z, length, width, height, yaw). Converting instance segments to 3D boxes is a separate non-trivial step. SemanticKITTI is primarily used for point-level semantic segmentation, not 3D detection-box annotations.

## Usage

### Sample mode (synthetic test)

```bash
python src/11_object_detection_tracking.py \
    --source sample \
    --detector mock \
    --frames 10 \
    --ground-threshold 0.25 \
    --cluster-distance 0.75 \
    --min-cluster-points 15 \
    --track-gate 2.0 \
    --dynamic-speed-threshold 0.25
```

Output:
```
outputs/phase8/sample/
    frames/
        frame_0000.png
        frame_0001.png
        ...
    detections.jsonl
    tracks.jsonl
    phase8_metrics.txt
    phase8_summary.png
```

### SemanticKITTI mode (if dataset available)

```bash
python src/11_object_detection_tracking.py \
    --source semantickitti \
    --sequence 00 \
    --start-frame 000000 \
    --frames 5 \
    --dt 0.1
```

Requires SemanticKITTI dataset at `data/semantic_kitti/dataset/sequences/`.

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--source` | `sample` | `sample` or `semantickitti` |
| `--detector` | `mock` | Detector type (only `mock` in Phase 8A) |
| `--frames` | `10` | Number of frames to process |
| `--sequence` | `00` | SemanticKITTI sequence ID |
| `--start-frame` | `000000` | First SemanticKITTI frame ID |
| `--dt` | `0.1` | Time delta between frames (seconds) |
| `--ground-threshold` | `0.25` | Ground threshold for mock detector (metres) |
| `--cluster-distance` | `0.75` | Maximum cluster distance (metres) |
| `--min-cluster-points` | `15` | Minimum cluster points |
| `--track-gate` | `2.0` | Track-detection gating threshold (metres) |
| `--dynamic-speed-threshold` | `0.25` | Speed threshold for dynamic flag (m/s) |

## Important Notes

1. **MockObjectDetector is a geometric baseline, not AI.** Do not present sample cloud movement as real detection performance.
2. **No model weights are included.** Phase 8B will require external OpenPCDet installation and pretrained checkpoints.
3. **Deterministic and testable.** All calculations use NumPy/SciPy with fixed seeds where applicable.
4. **No ROS 2 or GPU required.** Runs on CPU with Python, NumPy, SciPy, Matplotlib, Open3D.
