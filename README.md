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
  └──► [Phase 6+] SemanticKITTI Integration & Multi-Class Semantic 2.5D Elevation Grid
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
│   └── 09_semantickitti_semantic_map.py # SemanticKITTI loader, projection & semantic 2.5D grid
└── outputs/
    ├── phase1_*.png                     # Visualizations of raw, height-colored, and filtered clouds
    ├── phase2/                          # 2.5D elevation layers (z_min, z_max, height_range, density)
    ├── phase3/                          # Slope, roughness, step height, and traversability score maps
    ├── phase4/                          # Adaptive resolution zones (near 5cm, mid 20cm, far 50cm)
    └── phase5/                          # Spatial importance, uncertainty, and ROI classification
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
- Allocates grid resolution dynamically based on proximity:
  - **Near Zone (0 - 15m):** High resolution (5 cm cells) for reactive collision avoidance.
  - **Middle Zone (15 - 35m):** Medium resolution (20 cm cells) for path planning.
  - **Far Zone (35 - 50m):** Coarse resolution (50 cm cells) for situational awareness.

### Phase 5: Spatial Importance & ROI Computation
- Dynamically blends:
  - **Distance Priority:** Higher attention near the sensor.
  - **Terrain Risk:** Prioritizes steep or rough regions.
  - **Uncertainty:** Identifies sparse or unobserved cells.
- Produces prioritized Regions of Interest (ROI) for targeted compute allocation.

### Phase 6+: SemanticKITTI Semantic 2.5D Grid
- Maps raw SemanticKITTI point cloud labels (`.bin` and `.label`) into consolidated FoveaX navigational classes:
  - `DRIVABLE_GROUND`, `ROUGH_TERRAIN`, `VEGETATION`, `BUILDING_WALL`, `SOLID_OBSTACLE`, `VEHICLE`, `PEDESTRIAN`.
- Produces 2.5D semantic cost and elevation maps.

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

# Run SemanticKITTI mapping (requires KITTI sequence in data/)
python src/09_semantickitti_semantic_map.py
```