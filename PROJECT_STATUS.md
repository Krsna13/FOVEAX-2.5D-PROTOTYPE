# FOVEAX: Adaptive Variable-Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception

**Project Status Document**
**Addressing:** SIH Problem Statement PS-26053 (DRDO/IDEX — Smart Vehicles)

> **Note on sourcing:** No `PS-26053.pdf` was found anywhere in this repository. The problem-statement
> analysis below is reconstructed from the codebase's own design rationale (comments, docstrings,
> `README.md`, and the phase-by-phase implementation choices), not from the official PDF text. Where a
> claim is architectural fact (a file exists, a function does X), it was verified directly against the
> repository. Every phase status below reflects what code, tests, and generated `outputs/` artifacts
> actually exist as of this document — not roadmap intentions.

---

## 1. Executive Summary

FOVEAX is a LiDAR perception pipeline that converts raw 3D point clouds into a **2.5D adaptive elevation
and traversability map**, layered with semantic terrain classification and 3D object detection/tracking,
for autonomous ground vehicles operating in unstructured (off-road/trail) terrain. The system is built
around one central design bet: **classical geometric processing does the real-time-critical work, and AI
models plug in behind swappable interfaces** rather than being load-bearing for every frame. This keeps
the core pipeline deterministic, debuggable, and cheap to run, while still allowing state-of-the-art
segmentation (SalsaNext) and detection (PointPillars/OpenPCDet) to be dropped in where accuracy is
needed.

**Current status:** Phases 0 through 8A are complete with working code, passing tests, and real generated
outputs on disk. **Phase 7B (real AI segmentation) is now genuinely exercised, not just implemented**:
the official SalsaNext repository is cloned, the pretrained checkpoint is downloaded and loads cleanly
(312/312 weights matched), and real inference has been run and measured — **93.3% per-point agreement
with ground truth on SemanticKITTI**, the checkpoint's own training domain. Real cross-dataset work has
also started: RELLIS-3D (an off-road dataset) has a working loader, class-taxonomy mapping, and CLI
integration, but running the SalsaNext checkpoint against it exposed a genuine, root-caused
**sensor/intensity domain-transfer failure (11.1% agreement)** — see Phase 7B and
`docs/rellis3d_integration.md`. Phase 8B (real AI detection) has its target repository (OpenPCDet)
cloned but no checkpoint downloaded yet, and was deliberately deprioritized in favor of the segmentation
work above, since the SIH problem statement calls for segmentation. Phase 9 (dashboard) and Phase 10
(ROS 2 integration) have substantial, tested code but have **never been run end-to-end** — no session
telemetry or ROS output has ever been generated. Phase 11 (deployment/optimization) has real code,
passing tests, and real profiling/benchmark output on disk, though its ONNX export path is an intentional
stub for the proprietary model architectures. Phase 12 (validation/benchmarking) has real, measured
accuracy numbers as a byproduct of the Phase 7B work above but no formal per-class metric script yet.
Phase 13 (documentation/demo packaging) has not been started.

**Repository state:** 283 automated tests pass (0 failures), up from 207 at the start of the RELLIS-3D/
SalsaNext work. Only 4 git commits exist, covering Phases 0–5 and README documentation — the substantial
Phase 6–11 work (semantic mapping, detection/tracking, dashboard, ROS 2, deployment, RELLIS-3D
integration, and the full `tests/` suite) exists in the working tree but is **not yet committed to
version control**.

---

## 2. Problem Statement Analysis

### 2.1 Why 3D LiDAR processing is computationally expensive
A single LiDAR sweep produces tens of thousands to hundreds of thousands of 3D points per frame (the
project's own benchmark harness tests densities of 50,000 / 100,000 / 150,000 / 250,000 points — see
`outputs/phase11/benchmark/benchmark_summary.txt`). Running dense 3D convolutions, voxel-based deep
networks, or full point-wise neural inference over that volume, every frame, at real-time rates, is
expensive in both compute and memory — and gets worse the higher the sensor's point density and range.

### 2.2 Why 2D occupancy grids lose critical height information
A conventional 2D occupancy grid collapses the entire vertical (Z) column of a cell into a single
occupied/free bit. That discards exactly the information an off-road vehicle needs most: whether a raised
region is a curb, a rock, an overhanging branch, or a traversable slope. FOVEAX's answer is the "2.5D"
grid — each cell stores `z_min`, `z_max`, `height_range`, and `point_density` (see
`src/05_create_2point5d_map.py`), which is cheap to compute and store (unlike full 3D voxels) but keeps
enough vertical structure to distinguish flat ground from an obstacle or a step.

### 2.3 The three-way tradeoff: Accuracy vs. Latency vs. Memory
Every design decision in this codebase is visibly shaped by this tradeoff:
- **Accuracy** — solved by keeping AI models (SalsaNext, PointPillars) available as pluggable, swappable
  predictors behind common interfaces (`SemanticPredictor`, `ObjectDetector`), so accuracy can be
  dialed up without redesigning the pipeline.
- **Latency** — solved by doing the geometric heavy lifting (filtering, gridding, slope/roughness/
  step-height, tracking) in cheap, deterministic NumPy/SciPy operations that don't require a GPU pass,
  and by Phase 11's TensorRT/FP16 compilation for the parts that do.
- **Memory** — solved by the foveated multi-resolution grid (Phase 4) and, at the deployment layer, by
  explicit VRAM budgeting (`src/deployment/vram_budget.py`) that refuses to load models concurrently if
  they won't fit in an 8 GB budget, rather than crashing with an out-of-memory error mid-run.

### 2.4 The foveated mapping approach
Modeled on biological foveal vision — sharp detail at the center of gaze, coarser detail in the
periphery — Phase 4 (`src/07_adaptive_2point5d_map.py`) allocates grid resolution by distance from the
vehicle:

| Zone | Range | Resolution | Purpose |
|---|---|---|---|
| Near | 0–15 m | 5 cm cells | Reactive collision avoidance |
| Middle | 15–35 m | 20 cm cells | Path planning |
| Far | 35–100 m | 50 cm cells | Situational awareness |

This means compute and memory scale with what actually matters for immediate vehicle safety, not with
raw sensor range. Over a 200m x 200m x 8m corridor, this achieves >99.9% cell count reduction and >99.7%
memory savings compared to an equivalent uniform 3D voxel grid at 5cm resolution.

---

## 3. Completed Work (Phases 0–10)

Status legend: **✅ Complete** (code + tests + generated outputs all exist), **🟡 Implemented, Unverified**
(code + tests exist, but never run against real data/hardware it targets), **⚠️ Partial** (interface/
scaffolding exists, core logic is a documented stub), **❌ Not Started**.

### Phase 0 — Environment Setup — ✅ Complete
- **Deliverable:** `src/12_env_inspector.py`, `requirements.txt`, `pyproject.toml`.
- **Technical detail:** Dependencies are formally pinned in `requirements.txt` and `pyproject.toml` (including `numpy`, `scipy`, `open3d`, `opencv-python`, `matplotlib`, `PyQt5`, `torch==2.14.0+cu130`, `pytest`). PyTorch specifically requires the `cu130` build on this hardware target — the `cu126` build installs and imports fine but has no compiled kernels for the RTX 5050's Blackwell (`sm_120`) architecture.

### Phase 1 — Point Cloud Loading — ✅ Complete
- **Deliverables:** `src/01_view_open3d_sample.py`, `src/02_color_by_height.py`,
  `src/03_filter_point_cloud.py`, `src/04_save_filtered_cloud.py`.
- **Activities:** Open3D-based point cloud loading and visualization, elevation-based (height) coloring,
  statistical outlier removal (`nb_neighbors=20`, `std_ratio=2.0`) and voxel downsampling, filtered-cloud
  export.
- **Generated outputs on disk:** `outputs/phase1_raw_pointcloud.png`,
  `outputs/phase1_height_colored.png`, `outputs/phase1_filtered_pointcloud.png`,
  `outputs/filtered_sample_cloud.ply`.

### Phase 2 — Basic 2.5D Mapping — ✅ Complete
- **Deliverable:** `src/05_create_2point5d_map.py`.
- **Technical detail:** Converts unstructured point clouds into a structured grid with four layers per
  cell: `z_min` (ground), `z_max` (obstacle top), `height_range` (`z_max - z_min`), `point_density` (hit
  count, used as a confidence proxy).
- **Generated outputs on disk:** `outputs/phase2/z_min.png`, `z_max.png`, `height_range.png`,
  `point_density.png`, `map_layers.npz`.

### Phase 3 — Terrain Analysis — ✅ Complete
- **Deliverable:** `src/06_terrain_traversability.py` — this file is the **canonical source** for the
  traversability convention used across the entire codebase (dashboard, ROI, grid overlay all defer to
  it): `1.0 = safe`, `0.0 = blocked`, with thresholds Safe ≥ 0.70, Caution 0.40–0.70, Blocked/Lethal < 0.40.
- **Technical detail:** Slope via Sobel spatial gradients (X/Y), roughness via local-window standard
  deviation of height, step-height via max neighbor-cell height discrepancy; these three combine into a
  composite traversability score/class.
- **Generated outputs on disk:** `outputs/phase3/slope_degrees.png`, `roughness.png`, `step_height.png`,
  `traversability_score.png`, `traversability_classes.png`, `terrain_layers.npz`, `phase3_metrics.txt`.

### Phase 4 — Distance-Based Adaptation — ✅ Complete
- **Deliverable:** `src/07_adaptive_2point5d_map.py`.
- **Technical detail:** Implements the near (0–15m, 5cm), middle (15–35m, 20cm), and far (35–100m, 50cm)
  foveated zone system specified in the Problem Statement. Features automated theoretical memory-reduction
  metrics against an equivalent uniform 3D voxel grid (>99.9% cell reduction, >99.7% memory byte savings).
- **Generated outputs on disk:** `outputs/phase4/near_zone_map.npz`, `middle_zone_map.npz`,
  `far_zone_map.npz`, `adaptive_resolution_zones.png`, `phase4_metrics.txt`.

### Phase 5 — Spatial Importance & ROI Manager — ✅ Complete
- **Deliverable:** `src/08_spatial_importance_roi.py`.
- **Technical detail:** Blends three signals into a per-cell importance score: distance priority (closer
  = higher attention), terrain risk (from Phase 3), and uncertainty (sparse/unobserved cells).
- **Generated outputs on disk:** `outputs/phase5/distance_priority.png`, `terrain_risk.png`,
  `uncertainty_map.png`, `importance_score.png`, `roi_classes.png`, `recommended_resolution.png`,
  `roi_layers.npz`, `phase5_metrics.txt`.

### Phase 6 — SemanticKITTI Integration — ✅ Complete
- **Deliverable:** `src/09_semantickitti_semantic_map.py`, `src/perception/semantic_labels.py`.
- **Technical detail:** Loads real SemanticKITTI `.bin` (Velodyne) and `.label` files, remaps raw
  SemanticKITTI class IDs to a consolidated FOVEAX taxonomy (`DRIVABLE_GROUND`, `ROUGH_TERRAIN`,
  `VEGETATION`, `BUILDING_WALL`, `SOLID_OBSTACLE`, `VEHICLE`, `PEDESTRIAN`), builds a semantic 2.5D grid
  via majority-vote per cell.
- **Real data present:** `data/semantic_kitti/dataset/sequences/00/velodyne/` contains **4,541** real
  `.bin` scan files with a matching `labels/` directory — this is actual SemanticKITTI sequence 00 data,
  not synthetic.
- **Generated outputs on disk:** `outputs/phase6/semantic_2point5d_map.png`, `semantic_z_max.png`,
  `semantic_confidence.png`, `semantic_map_layers.npz`.

### Phase 7A — AI-Ready Interface — ✅ Complete
- **Deliverable:** `src/perception/semantic_predictor.py` — defines the `SemanticPredictor` abstract
  interface with three concrete implementations: `GroundTruthSemanticPredictor` (replays Phase 6
  annotations), `MockSemanticPredictor` (deterministic geometric heuristic, explicitly not AI, emits a
  `UserWarning` saying so), and `SalsaNextPredictor` (real inference wrapper).
- **Deliverable:** `src/10_ai_semantic_2point5d_map.py` — CLI that switches between the three via
  `--source {mock, ground_truth, salsanext}`.
- **Generated outputs on disk:** `outputs/phase7/ground_truth/*` and `outputs/phase7/mock/*` both have
  full output sets (semantic maps, confidence, uncertainty, layers). `outputs/phase7/model_environment_report.txt`.

### Phase 7B — Real AI Segmentation — ✅ Complete on SemanticKITTI; ⚠️ Domain-transfer failure on RELLIS-3D
- **Deliverable:** `src/perception/salsanext_predictor.py` — real, working inference (not a stub):
  spherical range-image projection (`src/perception/range_projection.py`, transcribed from the official
  `train/common/laserscan.py::do_range_projection`) → model forward pass → reverse re-projection →
  SemanticKITTI-to-FOVEAX remap via the existing `SEMANTICKITTI_TO_FOVEAX` table.
- **What's actually present (2026-09-11):** `external/SalsaNext` cloned (HEAD `7548c124`), official
  pretrained checkpoint downloaded to `models/salsanext/pretrained/pretrained/` (weights + `arch_cfg.yaml`
  + `data_cfg.yaml`). Three real bugs found and fixed while wiring it up (all independently
  regression-tested): a wrong module import path, a `torch.load` pickle-format incompatibility with
  PyTorch 2.6+'s `weights_only` default, and an `nn.DataParallel` `module.` key-prefix mismatch. The
  checkpoint loads cleanly and completely: 312/312 weights matched, correct CUDA device placement.
- **Measured accuracy, real data, both directions:**
  - **SemanticKITTI (the checkpoint's own training domain): 93.3% per-point agreement** with ground
    truth on seq 00/frame 000000 — predicted vs. ground-truth class distributions track closely across
    all 8 FOVEAX classes (e.g. DRIVABLE_GROUND 52.6% vs 52.1%, VEHICLE 3.8% vs 3.4%).
  - **RELLIS-3D: 11.1% per-point agreement — unusable as-is.** 85.8% of points predicted VEHICLE in an
    off-road scene with zero real vehicles. Root-caused, not just observed: the checkpoint was trained
    on a Velodyne HDL-64E's FOV (`+3°`/`-25°`); RELLIS-3D's Ouster OS1-64 actually spans `-16.44°` to
    `+17.02°`, putting 17.5% of points outside the assumed FOV; RELLIS-3D's raw intensity scale
    (0.00015–0.01169) is also ~30–100x smaller than what the checkpoint's normalization expects,
    collapsing the signal channel to a near-constant value. Full diagnosis in
    `docs/rellis3d_integration.md`. Not yet resolved — reprojecting with the Ouster's true FOV and
    rescaling intensity is the next concrete step if RELLIS-3D segmentation accuracy is required.
- **Test coverage:** 29 new tests across `tests/test_range_projection.py` (projection geometry with
  hand-computed expected pixel coordinates, collision policy, invalid-point handling) and
  `tests/test_salsanext_predict_pipeline.py` (end-to-end `predict()` on a stub model, real PyTorch ops,
  no GPU/checkpoint needed), plus `tests/test_salsanext_predictor.py`'s existing 9 (error paths, config
  parsing, `sys.path`/import fix, `weights_only`, prefix-stripping). All passing.

### Phase 8A — Detection/Tracking Foundation — ✅ Complete
- **Deliverables:** `src/11_object_detection_tracking.py`, `src/perception/object_detector.py`
  (`ObjectDetector` abstract base + `MockObjectDetector`, a deterministic geometric clustering baseline
  explicitly documented as "NOT an AI / trained object detector"), `src/tracking/multi_object_tracker.py`
  (`MultiObjectTracker`, `TrackState` — Kalman-filter tracking-by-detection with a constant-velocity
  model over `[x, y, z, vx, vy, vz]`, 3D IoU / Euclidean-distance data association, in the style of
  AB3DMOT).
- **Generated outputs on disk:** `outputs/phase8/sample/` — real per-frame outputs including
  `detections.jsonl`, `tracks.jsonl`, `phase8_summary.png`, 10 rendered BEV frames
  (`frame_0000.png`–`frame_0009.png`), plus a `mock/` subfolder with the same artifacts and
  `model_provenance.txt` and per-frame dynamic-object overlays (`.npz` + `.png`).
- **Docs:** `docs/phase8_object_tracking.md` (154 lines).

### Phase 8B — Real Object AI — ⚠️ Partial (interface complete, model not runnable here)
- **Deliverable:** `src/perception/openpcdet_predictor.py` — `OpenPCDetPointPillarsDetector`,
  `OpenPCDetDetector` wired against the official [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
  toolbox (setup guide `docs/openpcdet_pointpillars_setup.md`), including 3D IoU box computation and
  OpenPCDet→FOVEAX class-name mapping utilities.
- **What's actually present:** `external/OpenPCDet` is cloned (pinned `v0.5.2`, the highest real tag —
  an earlier reference to a `v0.6.0`/specific commit hash in `docs/openpcdet_pointpillars_setup.md` was
  found to not exist upstream and was corrected). No PointPillars checkpoint has been downloaded, and no
  real box-based detection has been produced yet — a KITTI Detection + PointPillars real-inference plan
  was deliberately superseded by the RELLIS-3D semantic-segmentation work above, since the SIH problem
  statement calls for segmentation, not box detection. This code path remains untouched and passing.
- **Test coverage:** `tests/test_openpcdet_predictor.py` — passes, but (per its own `UserWarning`
  output at test time) exercises the interface/config-validation path, not real trained-model inference.
- **VRAM integration verified real:** `check_concurrent_fit()` from `src/deployment/vram_budget.py` is
  actually called (not dead code) from `src/deployment/optimized_inference.py` before engine loads.

### Phase 9 — Real-Time Dashboard — 🟡 Implemented, Never Run
- **Deliverables:** `src/13_realtime_dashboard.py` (entry point), `src/dashboard/dashboard_state.py`
  (`FrameState`, `HardwareMetrics`, plus `assert_coordinate_frame_consistency()` — a non-blocking
  cross-panel coordinate-sanity check), `src/dashboard/data_streamer.py` (`DataStreamerThread`, a
  `QThread` producing frames at a target rate with live CPU/RAM/GPU/VRAM telemetry via `psutil`/`pynvml`),
  `src/dashboard/qt_main_window.py` (PyQt5 2D map panel with `RdYlGn` traversability colormap and hatched
  blocked-cell overlay), `src/dashboard/open3d_viewer.py` (separate-process Open3D 3D point-cloud/track
  viewer).
- **What's actually present:** `outputs/phase9/` **does not exist as a directory at all** — no session
  JSONL telemetry has ever been written, meaning the dashboard has never been launched end-to-end in
  this environment. All verification of its logic (traversability sign convention, colormap threshold,
  coordinate-frame warnings) has been done through targeted unit tests
  (`tests/test_dashboard_state.py`, `tests/test_dashboard_traversability_convention.py`), not a live run.
- **Known-fixed issues during development** (all verified via passing tests): a matplotlib API removal
  (`cm.get_cmap` → `mpl.colormaps[...]`) that would have crashed on first render; an unreshaped-array bug
  in the mock traversability computation that would have thrown `ValueError` on the first frame with
  real points; a PyQt5/PyTorch Windows DLL load-order conflict (fixed via `tests/conftest.py` and an
  import-order guard in `src/13_realtime_dashboard.py`) that would crash the process the moment a real
  torch-based detector (Phase 8B) is wired into the live dashboard.

### Phase 10 — Live ROS 2 Integration — 🟡 Implemented, Never Run
- **Deliverables:** `ros2_ws/src/foveax_ros/foveax_ros/lidar_node.py` (a plain `rclpy.Node` — explicitly
  documented in `docs/phase10_ros2_integration.md` as a deliberate choice, deferring `LifecycleNode`
  management to a future hardening pass), `diagnostics.py`, `ros2_ws/src/foveax_ros/launch/foveax.launch.py`,
  `src/integrations/ros2_pointcloud_adapter.py` (bidirectional `PointCloud2` ↔ numpy conversion, with
  graceful handling of missing intensity fields), `src/integrations/ros2_tf_adapter.py` (`TF2Adapter`
  for coordinate frame transforms).
- **Architecture:** Documented decoupled threading model — a single-threaded ROS executor does O(1) work
  enqueuing raw messages onto a thread-safe queue; a separate FOVEAX worker thread dequeues, transforms
  via `tf2_ros` (10-second buffer, graceful `ExtrapolationException` handling), and runs the
  detection/tracking pipeline. QoS explicitly set to `sensor_data` profile (BEST_EFFORT/VOLATILE/depth 5)
  to match real LiDAR driver conventions.
- **What's actually present:** No ROS 2 environment (`ROS_DISTRO`) is set up in this workspace; the node
  has never been built with `colcon` or launched against a live or bagged sensor stream in this
  environment. No `outputs/phase10/` directory exists.

---

## 4. Remaining Work (Phases 11–13)

### Phase 11 — Deployment and Optimization — 🟡 Substantially Implemented
Unlike Phases 9–10, this phase has **real generated output on disk**, meaning its CLI paths have actually
been executed, at least in profiling/benchmarking mode:
- **Deliverables:** `src/13_optimize_and_deploy.py` (CLI), `src/deployment/vram_budget.py` (real-time
  VRAM polling via `torch.cuda.mem_get_info()`/`pynvml`, `check_concurrent_fit()`), `deployment_config.py`
  (`HardwareProfile` registry, `rtx_5050_laptop` profile: 8 GB hard VRAM cap, 6 GB budgeted for
  concurrent engines), `model_exporter.py` (ONNX export path), `tensorrt_builder.py` (engine
  compilation, falls back to printing the exact `trtexec` command if the Python TensorRT API is
  unavailable), `calibration.py` (`FOVEAXInt8Calibrator` for INT8 quantization), `profiler.py`.
- **Generated outputs on disk (real, not placeholder):**
  `outputs/phase11/benchmark/benchmark_summary.txt` (tested at 50k/100k/150k/250k point densities against
  the `rtx_5050_laptop` target), `outputs/phase11/profiling/profiling_summary.txt` (measured
  preprocessing latency: 10.65 ms; measured peak VRAM: 254 MB / 3.1% of 8151 MB),
  `outputs/phase11/tensorrt/engine_build_log.txt` + `engine_metadata.json`,
  `outputs/phase11/optimized/optimized_session.jsonl`.
- **Documented, intentional limitation:** the ONNX exporter deliberately raises `ValueError` for the
  semantic/detection modules rather than re-implementing SalsaNext/PointPillars' proprietary
  architectures — real export requires running the *official* repos' own export scripts once those
  repos and checkpoints are in place (currently they are not — see Phase 7B/8B above).
- **Remaining work:** run the exporter/builder against a real SalsaNext or PointPillars checkpoint once
  one is downloaded; validate the FP16 engine's actual accuracy delta vs. native PyTorch; exercise the
  INT8 calibration path against a representative dataset (currently untested against real data — the
  5–10 mAP degradation risk documented in `docs/phase11_deployment_and_optimization.md` is unverified
  in either direction).

### Phase 12 — Validation and Benchmarking — 🟡 Started (informal), formal metric not yet built
No `outputs/phase12/` directory or formal scored-metric code exists yet, but real cross-dataset
accuracy numbers now exist as a byproduct of Phase 7B's real-checkpoint work:
- **SalsaNext vs. SemanticKITTI ground truth: 93.3% per-point agreement** (seq 00/frame 000000) —
  informal (a qualitative per-point-equality check, not a formal per-class IoU/mIoU metric), but real,
  not estimated.
- **SalsaNext vs. RELLIS-3D ground truth: 11.1% agreement**, root-caused to a sensor FOV + intensity
  scale mismatch, not a code bug (see Phase 7B and `docs/rellis3d_integration.md`).
- **RELLIS-3D loader/mapping/CLI wiring is done and unit-tested** (`src/perception/rellis3d_loader.py`,
  `RELLIS3D_TO_FOVEAX` in `src/perception/semantic_labels.py`, `--dataset-type rellis3d` on
  `src/10_ai_semantic_2point5d_map.py`) — this is real, working cross-dataset infrastructure, not just a
  referenced future dataset. nuScenes and CARLA remain unaddressed — no loader or adapter code exists
  for either.
- **What's still missing:** a formal per-class IoU/mIoU scoring script (the 93.3%/11.1% numbers above
  are whole-scene per-point agreement, not a proper confusion-matrix-based metric); PointPillars vs.
  KITTI/nuScenes ground-truth boxes remains fully blocked (no checkpoint downloaded).
- End-to-end latency/FPS benchmarking of the full pipeline (grid + semantic + detection + tracking
  together), as distinct from Phase 11's per-component profiling.
- Traversability accuracy validation against labeled hazard ground truth (none currently exists in-repo).

### Phase 13 — Documentation, Packaging and Demo Preparation — ❌ Not Started
- `README.md` exists and is substantial (covers Phases 1–11 CLI usage) but has not been updated for the
  Phase 9/10/11 code added since. `CLAUDE.md` (developer/architecture-oriented, not judge-facing) was
  updated separately.
- No packaging (Docker, installable wheel, or one-command setup script) exists.
- No demo video, slide deck, or judge-facing walkthrough exists in the repository.
- Git history is minimal (4 commits) and does not reflect Phases 6–11 — a commit/tagging pass is needed
  before this can be presented as "the submitted state."

---

## 5. Project Architecture

### 5.1 System Overview (Pipeline Flow)

```
Raw LiDAR Point Cloud [x, y, z, intensity]
        │
        ▼
┌───────────────────────┐
│ Phase 1: Filtering     │  voxel downsample, statistical outlier removal
└───────────┬────────────┘
        ▼
┌───────────────────────┐
│ Phase 2: 2.5D Grid      │  z_min, z_max, height_range, point_density
└───────────┬────────────┘
        ▼
┌───────────────────────┐
│ Phase 3: Traversability │  slope, roughness, step-height → 1.0=safe .. 0.0=blocked
└───────────┬────────────┘
        ▼
┌───────────────────────┐
│ Phase 4: Foveated Grid  │  near 5cm / mid 20cm / far 50cm zones
└───────────┬────────────┘
        ▼
┌───────────────────────┐
│ Phase 5: ROI Importance │  distance + terrain risk + uncertainty
└───────────┬────────────┘
        │
        ├──────────────────────────────┐
        ▼                              ▼
┌────────────────────┐      ┌─────────────────────────┐
│ Phase 6/7: Semantic  │      │ Phase 8A/8B: Detection    │
│ GT / Mock / SalsaNext│      │ Mock / PointPillars       │
└──────────┬───────────┘      └────────────┬─────────────┘
        │                              ▼
        │                    ┌─────────────────────────┐
        │                    │ Multi-Object Tracking     │
        │                    │ Kalman filter + IoU assoc │
        │                    └────────────┬─────────────┘
        │                              │
        └──────────────┬────────────────┘
                     ▼
        ┌─────────────────────────────┐
        │ Phase 9: FrameState          │  merged: points, grids, tracks, metrics
        │ → PyQt5 2D panel             │
        │ → Open3D 3D viewer           │
        │ → session JSONL telemetry    │
        └──────────────┬───────────────┘
                     │
     ┌────────────────┴────────────────┐
     ▼                                  ▼
┌───────────────────┐        ┌──────────────────────────┐
│ Phase 10: ROS 2     │        │ Phase 11: Deployment       │
│ live PointCloud2 in │        │ ONNX → TensorRT FP16/INT8  │
│ → JSON topics out    │        │ VRAM-budgeted concurrent   │
└─────────────────────┘        │ engine loading              │
                              └──────────────────────────┘
```

### 5.2 Directory Structure (as it actually exists)

```
FOVEAX 2.5D TRAIL/
├── CLAUDE.md                       Architecture notes for AI-assisted development
├── README.md                       Phase 1-11 overview and CLI usage
├── PROJECT_STATUS.md                This file
├── data/
│   └── semantic_kitti/dataset/sequences/00/   Real KITTI data: 4,541 velodyne .bin + labels
├── models/
│   ├── salsanext/pretrained/pretrained/   Real checkpoint (weights + arch_cfg.yaml + data_cfg.yaml)
│   └── openpcdet/                  Empty — no PointPillars checkpoint downloaded
├── external/
│   ├── SalsaNext/                  Cloned, HEAD 7548c124
│   └── OpenPCDet/                  Cloned, pinned v0.5.2
├── docs/
│   ├── phase8_object_tracking.md
│   ├── phase10_ros2_integration.md
│   ├── phase11_deployment_and_optimization.md
│   ├── openpcdet_pointpillars_setup.md
│   ├── salsanext_setup.md
│   └── rellis3d_integration.md     RELLIS-3D loader/mapping + SalsaNext domain-transfer diagnosis
├── src/
│   ├── 01_view_open3d_sample.py    Phase 1
│   ├── 02_color_by_height.py       Phase 1
│   ├── 03_filter_point_cloud.py    Phase 1
│   ├── 04_save_filtered_cloud.py   Phase 1
│   ├── 05_create_2point5d_map.py   Phase 2
│   ├── 06_terrain_traversability.py Phase 3 — canonical traversability convention
│   ├── 07_adaptive_2point5d_map.py Phase 4
│   ├── 08_spatial_importance_roi.py Phase 5
│   ├── 09_semantickitti_semantic_map.py  Phase 6
│   ├── 10_ai_semantic_2point5d_map.py    Phase 7 CLI
│   ├── 11_object_detection_tracking.py   Phase 8 CLI
│   ├── 12_env_inspector.py         Phase 0
│   ├── 13_optimize_and_deploy.py   Phase 11 CLI
│   ├── 13_realtime_dashboard.py    Phase 9 entry point
│   ├── perception/
│   │   ├── object_detector.py      ObjectDetector ABC, MockObjectDetector
│   │   ├── openpcdet_predictor.py  OpenPCDetPointPillarsDetector
│   │   ├── semantic_predictor.py   SemanticPredictor ABC, GT/Mock predictors
│   │   ├── salsanext_predictor.py  SalsaNextPredictor (real inference)
│   │   ├── range_projection.py     Spherical range-image projection for SalsaNext
│   │   ├── rellis3d_loader.py      RELLIS-3D file I/O, sky exclusion, class remap
│   │   ├── semantic_labels.py      FOVEAX taxonomy + KITTI remap + RELLIS3D_TO_FOVEAX
│   │   └── grid_overlay.py         Traversability overlay rendering
│   ├── tracking/
│   │   └── multi_object_tracker.py MultiObjectTracker, TrackState (Kalman)
│   ├── dashboard/
│   │   ├── dashboard_state.py      FrameState, HardwareMetrics, coord-frame check
│   │   ├── data_streamer.py        DataStreamerThread (QThread)
│   │   ├── qt_main_window.py       PyQt5 2D panel
│   │   └── open3d_viewer.py        Open3D 3D viewer process
│   ├── deployment/
│   │   ├── vram_budget.py          Real-time VRAM polling + concurrent-fit check
│   │   ├── deployment_config.py    HardwareProfile registry (rtx_5050_laptop)
│   │   ├── model_exporter.py       ONNX export
│   │   ├── tensorrt_builder.py     TensorRT engine compilation
│   │   ├── calibration.py          FOVEAXInt8Calibrator
│   │   └── profiler.py             Latency/memory profiling
│   └── integrations/
│       ├── ros2_pointcloud_adapter.py  PointCloud2 <-> numpy
│       └── ros2_tf_adapter.py          TF2Adapter
├── ros2_ws/src/foveax_ros/
│   ├── foveax_ros/{lidar_node.py, diagnostics.py}
│   ├── launch/foveax.launch.py
│   └── package.xml, setup.py
├── outputs/                        Real generated artifacts, phase1–phase8 + phase11,
│   plus outputs/phase7/rellis3d/<seq>_<frame>_<source>/ (RELLIS-3D runs, kept separate
│   from SemanticKITTI's outputs/phase7/<source>/ to avoid overwriting)
│   (no phase9/ or phase10/ — those phases have never been run)
└── tests/                          283 tests, 0 failures; conftest.py enforces
                                     torch-before-PyQt5 import order for the whole session
```

### 5.3 Module Descriptions

| Module | Responsibility |
|---|---|
| `src/perception/object_detector.py` | Defines the detector interface; `MockObjectDetector` is a deterministic geometric-clustering baseline (explicitly warns it is not AI). |
| `src/perception/openpcdet_predictor.py` | Real PointPillars 3D detection via the OpenPCDet toolbox; includes 3D IoU computation and class-name mapping. |
| `src/perception/semantic_predictor.py` | Defines the semantic-segmentation interface; ground-truth and mock implementations. |
| `src/perception/salsanext_predictor.py` | Real, working SalsaNext semantic segmentation (checkpoint loaded, verified 93.3% agreement on SemanticKITTI; not yet usable on RELLIS-3D — sensor/intensity domain mismatch). |
| `src/perception/range_projection.py` | Spherical range-image projection + reverse re-projection, transcribed from the official SalsaNext `laserscan.py`. |
| `src/perception/rellis3d_loader.py` | Loads RELLIS-3D's real (non-SemanticKITTI-layout) `.bin`/`.label` files, excludes sky points, remaps to FOVEAX classes. |
| `src/tracking/multi_object_tracker.py` | Kalman-filter multi-object tracking (constant-velocity, IoU/Euclidean association). |
| `src/dashboard/*` | Live PyQt5 2D map panel + Open3D 3D viewer, driven by a background `QThread` that assembles `FrameState` each tick. |
| `src/deployment/*` | Hardware-aware model export, TensorRT compilation, VRAM budgeting, and profiling for constrained-GPU deployment. |
| `src/integrations/*` | ROS 2 message ↔ numpy adapters and TF2 coordinate-frame transforms. |
| `ros2_ws/src/foveax_ros/` | The actual ROS 2 package: a lidar-subscribing node with a decoupled executor/worker threading model. |

---

## 6. Technical Stack

**Core:** Python 3.11, NumPy, SciPy, Open3D, OpenCV, Matplotlib, PyQt5 (dashboard UI).

**AI / ML:** PyTorch `2.14.0+cu130` (CUDA 13.0 — required for RTX 5050 Blackwell/`sm_120` kernel support;
the more common `cu126` build installs but has no usable kernels on this GPU). TensorRT (via `trtexec`
CLI fallback or Python API, not directly verified installed in this environment). ONNX (export target
format).

**AI Models:**
- **SalsaNext** — real-time semantic segmentation on range-image projections of LiDAR scans. Checkpoint
  downloaded, loaded, and run for real: 93.3% per-point agreement with ground truth on SemanticKITTI;
  11.1% on RELLIS-3D (domain-transfer failure, diagnosed, not yet fixed — see `docs/rellis3d_integration.md`).
- **PointPillars** (via OpenPCDet) — pillar-based 3D object detection. Repo cloned, no checkpoint
  downloaded yet — real inference not yet run (deliberately deprioritized in favor of the SalsaNext/
  RELLIS-3D segmentation work above, per the SIH problem statement's segmentation focus).

**Datasets:**
- **SemanticKITTI** — real data present (`data/semantic_kitti/dataset/sequences/00/`, 4,541 scans),
  actively used by Phases 6 and 7's ground-truth path.
- **RELLIS-3D, nuScenes, CARLA** — referenced as future/target datasets for off-road and box-detection
  validation; **no loader, adapter, or data currently exists in-repo for any of these three.**

**Robotics:** ROS 2 (Jazzy or Humble, per `docs/phase10_ros2_integration.md`) — package scaffolding
present, no ROS 2 install verified in this environment.

**Testing:** pytest, 207 tests across 17 test files, 0 failures.

---

## 7. Performance Metrics

### 7.1 Current (measured, from real Phase 11 output)
From `outputs/phase11/profiling/profiling_summary.txt`:

| Metric | Measured Value |
|---|---|
| Preprocessing latency | 10.65 ms (mean, min, max — single run) |
| Peak VRAM used | 254 MB (3.1% of 8,151 MB total) |

From `outputs/phase11/benchmark/benchmark_summary.txt`: benchmark harness exercised at point densities of
50,000 / 100,000 / 150,000 / 250,000 against the `rtx_5050_laptop` target profile (full per-density
timing breakdown is in that file; only the density sweep itself is summarized here to avoid
over-precision on numbers not independently re-verified for this document).

### 7.2 Expected (post-optimization, per `docs/phase11_deployment_and_optimization.md`)
These are documented engineering targets, **not yet independently measured against a real compiled
engine** in this environment (no SalsaNext/PointPillars checkpoint has been compiled to TensorRT here):

| Metric | Target |
|---|---|
| FP16 TensorRT vs. native PyTorch throughput | 2.5×–4× |
| VRAM savings vs. native PyTorch | 1–3 GB |
| Concurrent engine VRAM budget | ≤ 6 GB (of 8 GB total, 2 GB reserved for OS/dashboard/CUDA context) |

### 7.3 Target Hardware
- **Primary:** NVIDIA RTX 5050 Laptop GPU — Blackwell GB207, 2,560 CUDA cores, 5th-gen Tensor Cores,
  8 GB GDDR7 (hard VRAM constraint).
- **Future (documented, not yet targeted in code):** Jetson Orin Nano (8 GB unified), Jetson Orin NX
  (16 GB unified), Jetson AGX Orin (32 GB unified).

---

## 8. Next Steps

### Week 1 — Phase 11 Implementation + RELLIS-3D fix
- [x] ~~Clone OpenPCDet and SalsaNext into `external/`; download official pretrained checkpoints into `models/`.~~ Done — both cloned, SalsaNext checkpoint downloaded and loading cleanly.
- [ ] Fix the RELLIS-3D domain-transfer failure: reproject with the Ouster OS1-64's real FOV (`-16.44°`..`+17.02°`, not the assumed HDL-64E `-25°`..`+3°`) and rescale/renormalize intensity; re-measure per-point agreement against the current 11.1% baseline.
- [ ] Run the ONNX exporter against the real SalsaNext checkpoint (currently only the intentional-stub path is exercised).
- [ ] Compile at least one real FP16 TensorRT engine and re-measure latency/VRAM against the Phase 11 targets in §7.2.
- [ ] Exercise the INT8 calibration path (`calibration.py`) against a representative sample and measure actual accuracy delta (currently unverified in either direction).
- [ ] Download a PointPillars checkpoint and run real OpenPCDet inference (deprioritized so far in favor of the segmentation work above).

### Week 2 — Phase 11 Completion + Phase 12 Start
- [ ] Validate `check_concurrent_fit()` against real dual-engine loading (semantic + detection simultaneously) on the actual RTX 5050 target.
- [ ] Launch `src/13_realtime_dashboard.py` end-to-end for the first time in this environment; confirm `outputs/phase9/session/*.jsonl` telemetry is actually produced and the coordinate-frame warning path is exercised with real data.
- [ ] Build and launch the `ros2_ws` package for the first time (`colcon build`, `ros2 launch`); confirm `outputs/phase10/` telemetry is produced against either a live sensor or a bagged replay.
- [ ] Formalize Phase 12: replace the current informal per-point-agreement check (93.3% KITTI / 11.1% RELLIS-3D) with a proper per-class IoU/mIoU metric script.

### Week 3 — Phase 12 Completion + Phase 13 Start
- [ ] Run the formal accuracy metric against SemanticKITTI and RELLIS-3D (once the domain-transfer fix above lands) and PointPillars detection once a checkpoint exists.
- [ ] Run full end-to-end pipeline latency/FPS benchmark (distinct from Phase 11's per-component profiling).
- [ ] Update `README.md` to cover Phases 6–11 (currently documents through Phase 11's CLI but predates Phase 9/10 dashboard and ROS integration work).
- [ ] Commit the substantial uncommitted Phase 6–11 work to git with meaningful, phase-scoped commits (currently only 4 commits exist, covering Phases 0–5).

### Week 4 — Final Polish and Submission
- [ ] Record a demo run covering the full pipeline: raw point cloud → dashboard → tracked objects → optimized inference.
- [ ] Package for reproducibility (pin a `requirements.txt`/`pyproject.toml` — none currently exists; document the exact `torch==2.14.0+cu130` requirement and the OneDrive/venv-location gotcha discovered during development).
- [ ] Final review of all `docs/*.md` for accuracy against the as-built code.
- [ ] Prepare judge-facing summary materials (this document plus a demo video/slide deck — neither currently exists).

---

## 9. Success Criteria

### 9.1 Technical Success Metrics
- [ ] End-to-end pipeline sustains real-time frame rate (target FPS to be fixed once Phase 12 benchmarking exists; Phase 9's `HardwareMetrics` already tracks `target_fps` vs. `achieved_fps` per frame, but no live run has recorded this yet).
- [ ] TensorRT FP16 engine achieves the documented 2.5×–4× throughput improvement over native PyTorch, measured (not just targeted).
- [ ] Concurrent semantic + detection engines fit within the 6 GB budget on the RTX 5050 target, verified with `check_concurrent_fit()` against real engine sizes.
- [x] Semantic segmentation validated against ground truth with a real AI model on at least one dataset — done: 93.3% per-point agreement, SalsaNext vs. SemanticKITTI ground truth. A second dataset (RELLIS-3D) has working infrastructure but a diagnosed, unresolved domain-transfer failure (11.1%); object detection accuracy is still outstanding (no PointPillars checkpoint downloaded).
- [ ] 283/283 existing unit tests continue to pass through all remaining phases; new phases (12, 13) get their own test coverage where they produce testable logic.

### 9.2 SIH Submission Requirements
- [ ] GitHub repository with clean, phase-scoped commit history (currently 4 commits cover only Phases 0–5; a catch-up commit pass is required — this now includes the entire RELLIS-3D/SalsaNext work from this session).
- [ ] Working end-to-end demo (dashboard + ROS 2 integration both currently implemented but never run — this is the single largest gap between "code exists" and "demo-ready").
- [ ] Complete documentation: architecture (`CLAUDE.md` ✅), phase-by-phase usage (`README.md`, needs a Phase 9–11 update), and this project-status document (✅, this file).
- [ ] Reproducible environment setup (`requirements.txt`/`pyproject.toml` — currently missing; the exact PyTorch CUDA build requirement is non-obvious and must be documented, since the default `pip install torch` selects a build with no kernels for this hardware).
