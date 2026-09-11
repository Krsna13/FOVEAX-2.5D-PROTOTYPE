# FoveaX Trail: 2.5D Adaptive LiDAR Grid Mapping & Terrain Traversability

A deterministic, high-performance LiDAR perception and 2.5D adaptive elevation grid mapping pipeline designed for autonomous navigation and off-road/trail terrain understanding without requiring heavy end-to-end models in the real-time path.

---

## Overview

```
Raw LiDAR Point Cloud [x, y, z, intensity]
  │
  ├──► [Phase 1] Filtering, Voxelization & Spatial Bounds Selection
  │
  ├──► [Phase 2] 2.5D Elevation Mapping (Z_min, Z_max, Height Range, Point Density)
  │
  ├──► [Phase 3] Terrain Traversability: Slope, Roughness & Step-Height Analysis
  │
  ├──► [Phase 4] Adaptive Multi-Resolution Foveated Grids (Near / Mid / Far Zones)
  │
  ├──► [Phase 5] Spatial Importance ROI (Distance Priority + Terrain Risk + Uncertainty)
  │
  ├──► [Phase 6+] SemanticKITTI Integration & Multi-Class Semantic 2.5D Elevation Grid
  │
  └──► [Phase 8A] 3D Object Detection & Multi-Object Tracking
```

---

## Repository Structure

```
├── .gitignore
├── README.md
├── src/
│   ├── 01_view_open3d_sample.py         # Open3D point cloud viewer
│   ├── 02_color_by_height.py            # Point cloud colorized by elevation (Z)
│   ├── 03_filter_point_cloud.py         # Voxel downsampling & statistical outlier removal
│   ├── 04_save_filtered_cloud.py        # Point cloud export and verification
│   ├── 05_create_2point5d_map.py        # 2.5D elevation grid generation (min/max/range/density)
│   ├── 06_terrain_traversability.py     # Traversability scoring, slope & roughness estimation
│   ├── 07_adaptive_2point5d_map.py      # Multi-zone adaptive grid resolution
│   ├── 08_spatial_importance_roi.py     # Spatial ROI importance & foveated focus maps
│   ├── 09_semantickitti_semantic_map.py # SemanticKITTI loader, projection & semantic 2.5D grid
│   ├── 10_ai_semantic_2point5d_map.py   # AI inference integration (SalsaNext, Mock, GT)
│   ├── 11_object_detection_tracking.py  # 3D object detection & multi-object tracking
│   ├── perception/                      # Perception modules (ObjectDetectors, etc.)
│   └── tracking/                        # Tracking modules (MultiObjectTracker, etc.)
└── outputs/
    ├── phase1_*.png                     # Visualizations of raw, height-colored, and filtered clouds
    ├── phase2/                          # 2.5D elevation layers (z_min, z_max, height_range, density)
    ├── phase3/                          # Slope, roughness, step height, and traversability score maps
    ├── phase4/                          # Adaptive resolution zones (near 5cm, mid 20cm, far 50cm)
    ├── phase5/                          # Spatial importance, uncertainty, and ROI classification
    ├── phase6/                          # Semantic 2.5D maps from KITTI ground truth
    ├── phase7/                          # AI semantic segmentations and model reports
    └── phase8/                          # 3D object detection and tracking visualizations
```

---

## Pipeline Phases

### Phase 1: Point Cloud Ingestion & Filtering
- **Input:** 3D LiDAR point clouds (Open3D sample datasets / KITTI `.bin`).
- **Processing:** Statistical outlier removal (`nb_neighbors=20`, `std_ratio=2.0`), voxel downsampling, and elevation-based colormaps.
- **Output:** Cleaned point clouds and visual inspection figures.

### Phase 2: 2.5D Elevation Grid Generation
- Converts unstructured 3D point clouds into structured, memory-efficient 2.5D grid arrays:
  - `z_min`: Ground / bottom surface elevation.
  - `z_max`: Maximum surface / obstacle elevation.
  - `height_range`: Difference (`z_max - z_min`) representing obstacle presence.
  - `point_density`: LiDAR hit count per cell for confidence estimation.

### Phase 3: Terrain Traversability Estimation
- Computes geometrical hazard layers:
  - **Slope (degrees):** Sobel spatial gradients in X and Y directions.
  - **Roughness:** Standard deviation of height in local kernel windows.
  - **Step Height:** Maximum height discrepancy across neighboring grid cells.
- Evaluates composite traversability cost maps:
  - `FREE` (0.0 - 0.3)
  - `CAUTION` (0.3 - 0.7)
  - `LETHAL_OBSTACLE` (0.7 - 1.0)

### Phase 4: Adaptive Multi-Resolution Grids (Foveated Zones)
- Allocates grid resolution dynamically based on proximity (extending to 100 m as per Problem Statement PS-26053):
  - **Near Zone (0 - 15m):** High resolution (5 cm cells) for reactive collision avoidance.
  - **Middle Zone (15 - 35m):** Medium resolution (20 cm cells) for path planning.
  - **Far Zone (35 - 100m):** Coarse resolution (50 cm cells) for situational awareness.
- **Memory Reduction**: Achieves **>99.9% cell count reduction** and **>99.7% memory byte savings** compared to an equivalent uniform 3D voxel grid at 5 cm resolution (471k cells vs 2.56B voxels over a 200m x 200m x 8m volume).


### Phase 5: Spatial Importance & ROI Computation
- Dynamically blends:
  - **Distance Priority:** Higher attention near the sensor.
  - **Terrain Risk:** Prioritizes steep or rough regions.
  - **Uncertainty:** Identifies sparse or unobserved cells.
- Produces prioritized Regions of Interest (ROI) for targeted compute allocation.

### Phase 6: SemanticKITTI Semantic 2.5D Grid
- Maps raw SemanticKITTI point cloud labels (`.bin` and `.label`) into consolidated FoveaX navigational classes:
  - `DRIVABLE_GROUND`, `ROUGH_TERRAIN`, `VEGETATION`, `BUILDING_WALL`, `SOLID_OBSTACLE`, `VEHICLE`, `PEDESTRIAN`.
- Produces 2.5D semantic cost and elevation maps.

### Phase 7: AI Semantic Segmentation
- Replaces the ground truth labels from Phase 6 with an AI inference backend.
- Supports pluggable SemanticPredictors:
  - **Ground Truth**: Uses dataset annotations (Phase 6 logic). Also works against RELLIS-3D via
    `--dataset-type rellis3d` (see below), not just SemanticKITTI.
  - **Mock**: Geometry baseline heuristic (not AI).
  - **SalsaNext**: Real pretrained AI inference (spherical range-image projection, real model forward
    pass, reverse re-projection, SemanticKITTI-to-FOVEAX remap) using the official
    [SalsaNext repository](https://github.com/TiagoCortinhal/SalsaNext). **Verified 93.3% per-point
    agreement with ground truth on SemanticKITTI.** Currently **not usable on RELLIS-3D** (11.1%
    agreement) due to a sensor FOV + intensity-scale mismatch between the checkpoint's training sensor
    and RELLIS-3D's Ouster OS1-64 — see `docs/rellis3d_integration.md` for the full diagnosis.
- See `docs/salsanext_setup.md` for instructions on cloning the repository and downloading the official checkpoint to `models/salsanext/`.

### Phase 7 (extension): RELLIS-3D dataset support
- `--dataset-type rellis3d` on `10_ai_semantic_2point5d_map.py` reads RELLIS-3D's own directory layout
  (`Rellis-3D/<5-digit sequence>/os1_cloud_node_kitti_bin/` + `os1_cloud_node_semantickitti_label_id/`),
  distinct from SemanticKITTI's `sequences/<2-digit>/{velodyne,labels}/`.
- `src/perception/rellis3d_loader.py` excludes invalid no-return points (`x=y=z=0.0`, which RELLIS-3D's
  own ground truth sometimes mislabels with a real semantic class rather than tagging void/unlabeled —
  see `docs/rellis3d_integration.md`) and remaps its 20-class ontology to FOVEAX classes via
  `RELLIS3D_TO_FOVEAX` in `src/perception/semantic_labels.py`.

### Phase 8A: 3D Object Detection & Multi-Object Tracking (Baseline)
- Implements a model-agnostic 3D object detection and multi-object tracking pipeline.
- Detectors: Mock geometric clustering (for validation and tracking testing without AI).
- Tracker: AB3DMOT (A Baseline for 3D Multi-Object Tracking) style Kalman Filter tracking with 3D IoU / Euclidean distance data association.

### Phase 8B: Real AI Object Detection
- Replaces mock obstacle clusters with PointPillars AI predictions using the OpenPCDet toolbox.
- Gives FOVEAX real 3D vehicle/pedestrian boxes and confidence scores for tracking, risk analysis, and local map refinement.
- **Dataset Guidance**: SemanticKITTI is recommended for semantic terrain mapping, while KITTI Detection / nuScenes / CARLA are recommended for box-based object detection.

### Phase 9: Real-time Dashboard & Session Analytics
- A multi-panel PyQt5 and Open3D real-time dashboard visualizing both 2D mapping artifacts and 3D point cloud tracks.
- Includes telemetry logging (FPS, latency, resource usage) and session recording into JSONL format.

### Phase 10: ROS 2 Integration
- Hooks the FOVEAX core pipeline into standard `sensor_msgs/PointCloud2` streams using a bounded queue and strict single-threaded execution to prevent callbacks from blocking.
- Translates FOVEAX output maps and tracks to serialized JSON strings for downstream consumers.

### Phase 11: Deployment & Optimization (RTX 5050 / 8GB Target)
- Restructures the system for deployment on memory-constrained (8GB VRAM) laptop GPUs like the RTX 5050.
- Introduces ONNX exporters and TensorRT `fp16` compilation engines to aggressively minimize VRAM footprint and latency.
- Incorporates dynamic `vram_budget.py` checks that measure actual free VRAM at runtime (via `pynvml` or `torch.cuda`) and prevent concurrent execution OOMs before they happen.

---

## Setup & Usage

### Prerequisites
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install numpy scipy open3d opencv-python matplotlib
```

### Running the Stages
```bash
# Run 2.5D elevation map generation
python src/05_create_2point5d_map.py

# Run terrain traversability analysis
python src/06_terrain_traversability.py

# Run adaptive multi-resolution mapping
python src/07_adaptive_2point5d_map.py

# Run spatial importance ROI calculation
python src/08_spatial_importance_roi.py

# Run SemanticKITTI mapping with ground truth annotations
python src/09_semantickitti_semantic_map.py

# Run Phase 7 AI Segmentation with mock geometry predictor
python src/10_ai_semantic_2point5d_map.py --source mock

# Run Phase 7 AI Segmentation with ground truth annotations
python src/10_ai_semantic_2point5d_map.py --source ground_truth

# Run Phase 7 AI Segmentation with SalsaNext pretrained model (Requires setup!)
# NOTE: --config points inside the downloaded checkpoint folder, NOT into the
# cloned repo -- arch_cfg.yaml/data_cfg.yaml ship alongside the weights, not
# in external/SalsaNext. See docs/salsanext_setup.md.
python src/10_ai_semantic_2point5d_map.py --source salsanext \
    --salsanext-repo external/SalsaNext \
    --checkpoint models/salsanext/pretrained/pretrained/SalsaNext \
    --config models/salsanext/pretrained/pretrained/arch_cfg.yaml \
    --device auto

# Run Phase 7 AI Segmentation on RELLIS-3D instead of SemanticKITTI
python src/10_ai_semantic_2point5d_map.py --source ground_truth \
    --dataset-type rellis3d --sequence 00000 --frame 000000

# Run Phase 8A: 3D Object Detection & Tracking (Synthetic Sample)
python src/11_object_detection_tracking.py --source sample --detector mock --frames 10

# Run Phase 8A: 3D Object Detection & Tracking (SemanticKITTI)
python src/11_object_detection_tracking.py --source semantickitti --sequence 00 --start-frame 000000 --frames 5

# Set up OpenPCDet for Phase 8B
git clone https://github.com/open-mmlab/OpenPCDet.git external/OpenPCDet
# (Then manually install dependencies as per docs/openpcdet_pointpillars_setup.md)

# Run Phase 8B: Real AI Object Detection with PointPillars (Requires setup!)
python src/11_object_detection_tracking.py --source <compatible_data_source> --detector pointpillars \
    --openpcdet-repo external/OpenPCDet \
    --checkpoint models/openpcdet/<OFFICIAL_CHECKPOINT> \
    --config external/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml \
    --device auto

# Phase 11: Deploy & Optimize (TensorRT & 8GB VRAM)
# Check performance and memory limits dynamically
python src/13_optimize_and_deploy.py --profile
# Example TRT build (requires ONNX and TensorRT API/trtexec)
# python src/13_optimize_and_deploy.py --build-engine --semantic-model model.onnx --precision fp16
```