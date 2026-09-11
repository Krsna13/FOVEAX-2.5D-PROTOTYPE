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
- Evaluates composite traversability cost maps using the **canonical convention** (`1.0 = SAFE`, `0.0 = BLOCKED/LETHAL`):
  - `SAFE` (score ≥ 0.70, colored green)
  - `CAUTION` (0.40 ≤ score < 0.70, colored yellow)
  - `BLOCKED / LETHAL` (score < 0.40, colored red with hatched pattern overlay)

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
  - **Ground Truth**: Uses dataset annotations (Phase 6 logic). Works against SemanticKITTI and RELLIS-3D via `--dataset-type rellis3d`.
  - **Mock**: Geometry baseline heuristic (not AI).
  - **SalsaNext**: Real pretrained AI inference (spherical range-image projection, real model forward pass, reverse re-projection, SemanticKITTI-to-FOVEAX remap) using the official [SalsaNext repository](https://github.com/TiagoCortinhal/SalsaNext).
  - **Domain Adaptation on RELLIS-3D**: Sensor-adapted vertical FOV (+17.02°/-16.44°) and intensity rescaling triples agreement (11.08% → 37.10% overall, 47.72% in middle zone), removing projection collapse.
- See `docs/salsanext_setup.md` and `docs/rellis3d_integration.md`.

### Phase 8: 3D Object Detection & Multi-Object Tracking
- Implements a model-agnostic 3D object detection and multi-object tracking pipeline.
- Detectors: Mock geometric clustering (deterministic baseline) and OpenPCDet PointPillars.
- Tracker: Kalman Filter multi-object tracking with 3D IoU and Euclidean distance data association.

### Phase 9: Real-time Dashboard & Session Analytics
- Multi-panel PyQt5 and Open3D real-time dashboard visualizing both 2D mapping artifacts and 3D point cloud tracks.
- Live data streaming supporting synthetic samples, SemanticKITTI, and RELLIS-3D datasets at >17 FPS.
- Telemetry recording into JSONL format (`outputs/phase9/session/dashboard_session.jsonl`).
- Publication-quality snapshot generation (`outputs/phase9/dashboard_semantickitti_overview.png`, `outputs/phase9/dashboard_rellis3d_overview.png`).

### Phase 10: ROS 2 Integration
- Hooks the FOVEAX core pipeline into standard `sensor_msgs/PointCloud2` streams using a bounded queue and strict single-threaded execution to prevent callbacks from blocking.
- Translates FOVEAX output maps and tracks to serialized JSON strings for downstream consumers.

### Phase 11: Deployment & Optimization (RTX 5050 / 8GB Target)
- Restructures the system for deployment on memory-constrained (8GB VRAM) laptop GPUs like the RTX 5050.
- Introduces ONNX exporters and TensorRT `fp16` compilation engines to aggressively minimize VRAM footprint and latency.
- Dynamic `vram_budget.py` checks measure actual free VRAM at runtime and prevent concurrent execution OOMs.

---

## Setup & Usage

### Prerequisites
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Running the Stages
```bash
# Run 2.5D elevation map generation
python src/05_create_2point5d_map.py

# Run terrain traversability analysis
python src/06_terrain_traversability.py

# Run adaptive multi-resolution mapping (with 100m PS-spec metrics)
python src/07_adaptive_2point5d_map.py

# Run spatial importance ROI calculation
python src/08_spatial_importance_roi.py

# Run SemanticKITTI mapping with ground truth annotations
python src/09_semantickitti_semantic_map.py

# Run Phase 7 AI Segmentation with mock geometry predictor
python src/10_ai_semantic_2point5d_map.py --source mock

# Run Phase 7 AI Segmentation with ground truth annotations
python src/10_ai_semantic_2point5d_map.py --source ground_truth

# Run Phase 7 AI Segmentation with SalsaNext pretrained model
python src/10_ai_semantic_2point5d_map.py --source salsanext \
    --salsanext-repo external/SalsaNext \
    --checkpoint models/salsanext/pretrained/pretrained/SalsaNext \
    --config models/salsanext/pretrained/pretrained/arch_cfg.yaml \
    --device auto

# Run Phase 7 AI Segmentation on RELLIS-3D with sensor adaptation
python src/10_ai_semantic_2point5d_map.py --source salsanext \
    --dataset-type rellis3d --sequence 00000 --frame 000000 \
    --salsanext-repo external/SalsaNext \
    --checkpoint models/salsanext/pretrained/pretrained/SalsaNext \
    --config models/salsanext/pretrained/pretrained/arch_cfg.yaml \
    --device auto

# Evaluate distance-bucketed accuracy + confusion matrix across Near (0-15m), Mid (15-35m), Far (35-100m) zones
python src/10b_eval_distance_metrics.py --dataset-type semantickitti --sequence 00 --frame 000000 \
    --salsanext-repo external/SalsaNext \
    --checkpoint models/salsanext/pretrained/pretrained/SalsaNext \
    --config models/salsanext/pretrained/pretrained/arch_cfg.yaml \
    --device auto

# Same, on RELLIS-3D with sensor-domain adaptation (Ouster OS1-64 FOV + intensity rescale)
python src/10b_eval_distance_metrics.py --dataset-type rellis3d --sequence 00000 --frame 000000 \
    --salsanext-repo external/SalsaNext \
    --checkpoint models/salsanext/pretrained/pretrained/SalsaNext \
    --config models/salsanext/pretrained/pretrained/arch_cfg.yaml \
    --device auto

# Run Phase 8: 3D Object Detection & Tracking (Synthetic Sample)
python src/11_object_detection_tracking.py --source sample --detector mock --frames 10

# Run Phase 8: 3D Object Detection & Tracking (SemanticKITTI)
python src/11_object_detection_tracking.py --source semantickitti --sequence 00 --start-frame 000000 --frames 5

# Run Phase 9: Real-Time Dashboard (Interactive GUI)
python src/13_realtime_dashboard.py --source semantickitti --frames 50

# Run Phase 9: Headless Replay & Telemetry Logging (SemanticKITTI or RELLIS-3D)
python src/13_realtime_dashboard.py --source semantickitti --frames 10 --headless
python src/13_realtime_dashboard.py --source rellis3d --frames 10 --headless

# Export Publication-Quality Dashboard Visualizations
python src/dashboard/export_dashboard_snapshots.py

# Run Phase 11: Deployment Profiling & Benchmarking
python src/13_optimize_and_deploy.py --profile
python src/13_optimize_and_deploy.py --benchmark

# Run complete automated test suite (302 tests)
pytest
```