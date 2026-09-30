# FOVEAX: Adaptive Variable-Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception

**Project Status Document**
**Addressing:** SIH Problem Statement PS-26053 (DRDO/IDEX — Smart Vehicles)

> **Note on sourcing:** No `PS-26053.pdf` was found anywhere in this repository. The problem-statement
> analysis below is reconstructed from the codebase's own design rationale (comments, docstrings,
> `README.md`, and the phase-by-phase implementation choices), not from the official PDF text. Where a
> claim is architectural fact (a file exists, a function does X), it was verified directly against the
> repository. Every phase status below reflects what code, tests, and generated `outputs/` artifacts
> actually exist as of this document — not roadmap intentions.

> **Last reviewed: 2026-09-30**, against the code committed on top of `f174f08`.
> Measured numbers live in `docs/validation_results.md` (the source of truth); this document
> summarizes them and tracks phase status.

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

**Current status:** Phases 0–9, 11 and 12 (including the Stage 6 RELLIS-3D fine-tune) are complete, with
working code, tests, and generated outputs on disk. Phase 8B (PointPillars) is interface-only, and Phase 10
(ROS 2) is implemented but has never run in a ROS 2 environment.

- **Semantic segmentation (Phase 7B / Stage 6):** the official SalsaNext checkpoint loads cleanly (312/312
  weights) and scores **94.98%** on SemanticKITTI (seq 00/frame 000000, UNKNOWN excluded). On off-road
  RELLIS-3D the held-out sequence `00000` goes **6.65%** (pretrained) → **22.26%** (Ouster FOV + intensity
  adaptation) → **81.53%** (fine-tuned on RELLIS-3D, 8-class FOVEAX head). The fine-tuned model forgets
  SemanticKITTI (23.97%), so both checkpoints are kept and chosen by domain.
- **Dashboard (Phase 9):** the PyQt5 + embedded Open3D dashboard runs end to end on real RELLIS-3D and
  SemanticKITTI data, with a pull-based playback controller (step/pause/restart/environment switch), 3D box
  labels, per-object terrain features, overhead/centerline panels and live accuracy. Throughput is below
  real-time targets: `process_frame()` takes 128 ms/frame (≈7.8 FPS) in the 2026-09-30 `--profile` run.
- **Phase 11:** profiling, benchmarking and VRAM budgeting for the RTX 5050 Laptop GPU; no real TensorRT
  engine has been compiled from a trained checkpoint yet.

**Repository state:** 505 tests, all passing on the committed code (`docs/validation_results.md` §7).
Git history has phase-scoped commits; the "OPT-n" dashboard speed-ups (about 36 ms/frame faster, but with
known bugs) are deliberately not committed.

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
| Near | 0–10 m | 5 cm cells | Reactive collision avoidance |
| Middle | 10–30 m | 20 cm cells | Path planning |
| Far | 30–100 m | 50 cm cells | Situational awareness |

This means compute and memory scale with what actually matters for immediate vehicle safety, not with
raw sensor range. Over a 200m x 200m x 8m corridor, this achieves >99.9% cell count reduction and >99.8%
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
- **Technical detail:** Implements the near (0–10m, 5cm), middle (10–30m, 20cm), and far (30–100m, 50cm)
  foveated zone system specified in the Problem Statement. Features automated theoretical memory-reduction
  metrics against an equivalent uniform 3D voxel grid (99.99% cell reduction, 99.81% memory byte savings).
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

### Phase 7B — Real AI Segmentation — ✅ Complete (SemanticKITTI in-domain; RELLIS-3D via Stage 6 fine-tune)

> **Update (2026-09-30):** the RELLIS-3D failure described below (as first diagnosed on 2026-09-11) was
> resolved in two steps — sensor adaptation (22.26% with UNKNOWN excluded) and fine-tuning (81.53% on
> held-out sequence `00000`); see Phase 12 below. The 93.3% / 11.1% figures in this section are the
> original per-point-agreement numbers, which include UNKNOWN ground truth; the current scorer
> (`10b_eval_distance_metrics.py`) excludes it (94.98% / 6.65%).

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

### Phase 9 — Real-Time Dashboard & Telemetry — ✅ Complete & Verified
- **Deliverables:** `src/13_realtime_dashboard.py` (entry point with live GUI and headless replay modes),
  `src/dashboard/dashboard_state.py` (`FrameState`, `HardwareMetrics`, plus `assert_coordinate_frame_consistency()`
  cross-panel coordinate-sanity check), `src/dashboard/data_streamer.py` (`DataStreamerThread`, supporting
  `sample`, `semantickitti`, and `rellis3d` data streams with live telemetry via `psutil`/`pynvml`),
  `src/dashboard/qt_main_window.py` (PyQt5 2D map panel with `RdYlGn` traversability colormap and hatched
  blocked-cell overlay), `src/dashboard/open3d_viewer.py` (separate-process Open3D 3D point-cloud/track
  viewer), `src/dashboard/export_dashboard_snapshots.py` (publication-quality overview renderer).
- **Generated outputs on disk:**
  - `outputs/phase9/session/dashboard_session.jsonl` (verified live telemetry across 10-frame replays at >17 FPS on 131k-point scans).
  - `outputs/phase9/dashboard_semantickitti_overview.png` (side-by-side Elevation, Traversability, ROI/Tracks for SemanticKITTI).
  - `outputs/phase9/dashboard_rellis3d_overview.png` (side-by-side Elevation, Traversability, ROI/Tracks for RELLIS-3D).
- **Added since the first dashboard milestone:** pull-based playback controller in
  `DashboardApplication` (QTimer + per-frame `FrameState` cache: STEP back replays without re-running
  the tracker, RESTART and environment switch reset the tracker), Open3D window embedded into the Qt
  window (`win32_embed.py`), 3D box class labels (`track_labels.py`), per-object terrain features
  (slope, clearance, overhang, pothole/bump; `perception/terrain_features.py`), overhead-clearance and
  forward-corridor panels, informational ICP ego-displacement, live accuracy vs. ground truth, and a
  `--profile` per-stage timing harness (`stage_timer.py`).
- **Committed 2026-09-30:** alert cards and threat toast moved from floating overlays into the left
  panel; default `--rate-hz` 14 → 30; `--ego-motion-every-n` flag (default 30); the app starts maximized.
- **Held back (not committed):** a set of "OPT-n" caches/throttles in `data_streamer.py`,
  `qt_main_window.py` and `open3d_viewer.py`. They cut `process_frame()` from 128 to about 92 ms but fail
  2 tests and have known correctness issues (see §8.1). They remain only in the local working tree.
- **Unit test coverage:** `tests/test_dashboard_state.py`, `test_dashboard_traversability_convention.py`,
  `test_data_streamer_sources.py`, `test_track_labels.py`, `test_tracked_objects_table.py`,
  `test_qt_dashboard_display_fixes.py`, `test_alert_card_placement.py`, `test_win32_zorder.py`,
  `test_terrain_features*.py`, `test_centerline_profile.py`, `test_overhead_detection.py`,
  `test_ego_motion_estimate.py`.

### Phase 10 — Live ROS 2 Integration — 🟡 Implemented (ROS 2 environment required to run)
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
- **Unit test coverage:** Validated via unit tests with mock ROS interfaces (`tests/test_pointcloud2_adapter.py`,
  `tests/test_transform_validation.py`).

---

## 4. Optimization & Validation (Phases 11–13)

### Phase 11 — Deployment and Optimization — ✅ Complete
- **Deliverables:** `src/13_optimize_and_deploy.py` (CLI), `src/deployment/vram_budget.py` (real-time
  VRAM polling via `torch.cuda.mem_get_info()`/`pynvml`, `check_concurrent_fit()`), `deployment_config.py`
  (`HardwareProfile` registry, `rtx_5050_laptop` profile: 8 GB hard VRAM cap, 6 GB budgeted for
  concurrent engines), `model_exporter.py` (ONNX export path with graceful dependency diagnostics),
  `tensorrt_builder.py` (engine compilation, falls back to printing the exact `trtexec` command if the
  Python TensorRT API is unavailable), `calibration.py` (`FOVEAXInt8Calibrator` for INT8 quantization), `profiler.py`.
- **Generated outputs on disk:**
  `outputs/phase11/benchmark/benchmark_summary.txt` (tested at 50k/100k/150k/250k point densities against
  the `rtx_5050_laptop` target), `outputs/phase11/profiling/profiling_summary.txt` (measured
  preprocessing latency: 10.65 ms; measured peak VRAM: 254 MB / 3.1% of 8151 MB),
  `outputs/phase11/tensorrt/engine_build_log.txt` + `engine_metadata.json`,
  `outputs/phase11/optimized/optimized_session.jsonl`.
- **Unit test coverage:** `tests/test_model_exporter.py`, `tests/test_optimized_pipeline_config.py`,
  `tests/test_profiler.py`, `tests/test_tensorrt_builder_config.py`, `tests/test_vram_budget.py`.

### Phase 12 — Validation and Distance-Bucketed Benchmarking — ✅ Complete
- **Deliverable:** `src/10b_eval_distance_metrics.py` — computes distance-bucketed accuracy and a full
  per-class confusion matrix matching FOVEAX's foveated zones: Near (0–10m), Mid (10–30m), Far (30–100m),
  with ground-truth UNKNOWN(7) points (e.g. SemanticKITTI's own "unlabeled"/"outlier" raw classes)
  excluded from scoring — matching SemanticKITTI's own official benchmark convention.
  (An earlier duplicate, `src/perception/distance_accuracy_eval.py`, computed the same underlying metric
  without that exclusion and carried a hardcoded "Summary & Engineering Findings" narrative that had
  drifted out of sync with its own live output — e.g. claiming ~11.1%/~37.1% for RELLIS-3D when the
  actual measured numbers were 6.65%/22.26%. It was deleted rather than fixed in place, since consolidating
  to one tested, always-live-computed source of truth was more valuable than patching a second one.)
- **Measured results** (SemanticKITTI seq 00/frame 000000, RELLIS-3D seq 00000/frame 000000):
  - **SemanticKITTI (Urban HDL-64E):** 93.31% overall agreement on all 124,668 points; with
    SemanticKITTI's own UNKNOWN ground truth excluded (`10b`'s convention): 94.98% overall on 122,480
    points (Near 97.70%, Mid 92.82%, Far 87.28%; re-measured 2026-09-30 with the 10/30 m zones).
  - **RELLIS-3D Unadapted Baseline:** 6.65% agreement (Near: 3.78%, Mid: 17.36%), suffering severe vehicle hallucination.
  - **RELLIS-3D Sensor-Adapted (+17.02°/-16.44° FOV + intensity rescale):** 22.26% overall agreement (Near: 15.07%, Mid: 49.57%),
    eliminating projection collapse and recovering trail rough terrain / vegetation geometry.
- **Unit test coverage:** `tests/test_eval_distance_metrics.py` (7 tests: bucket accuracy, UNKNOWN
  exclusion, mask restriction, empty-bucket handling, confusion-matrix contents — all pass).

### Phase 12 (Stage 6) — SalsaNext Fine-Tuning on RELLIS-3D — ✅ Complete
- **Deliverables:** `src/14_finetune_salsanext.py` (reuses upstream SalsaNext's network, Lovász-softmax,
  warmup scheduler, weighted NLL and SGD config; replaces the 20-class head with an 8-class FOVEAX head),
  `src/training/rellis3d_dataset.py` (deterministic splits and range-image samples with preprocessing
  identical to `SalsaNextPredictor.predict()`), `src/prepare_rellis3d_for_salsanext_training.py`
  (alternative export into upstream's `sequences/` layout), `SalsaNextPredictor(taxonomy="foveax")`.
- **Split:** sequence `00000` held out entirely as test; train 4,532 / val 162 frames from `00001`–`00004`
  with a 15% contiguous val tail and a 10-frame gap.
- **Results:** best val accuracy 89.05% (epoch 9); **81.53% on 100 held-out test frames** (Near 78.83%,
  Mid 89.42%, Far 69.85%); SemanticKITTI forgetting 94.98% → 23.97%. 101 min of training compute, 3.77 GB
  peak VRAM at batch size 2. Logs: `outputs/phase12/train_log*.txt`, `class_frequencies.json`.
- **Checkpoints:** `models/salsanext/rellis3d_finetuned/` (`best.pt` = epoch 9); the original checkpoint
  is unchanged (MD5-verified).
- **Test coverage:** `tests/test_rellis3d_finetune_dataset.py`.

### Phase 13 — Documentation, Packaging and Presentation Preparation — ✅ Complete
- `README.md` and `PROJECT_STATUS.md` fully updated with complete CLI commands, architecture rationale,
  hardware profiles, and benchmark numbers.
- Pinned `requirements.txt` and `pyproject.toml` with detailed Blackwell / `cu130` installation notes.
- Detailed technical documentation in `docs/` (`salsanext_setup.md`, `rellis3d_integration.md`,
  `phase8_object_tracking.md`, `phase10_ros2_integration.md`, `phase11_deployment_and_optimization.md`).
- Version control history organized into clean, phase-scoped commits.
- Measured-results source of truth: `docs/validation_results.md`. (A `walkthrough.md` was referenced
  here previously; no such file exists in the repository.)

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
├── README.md                       Overview, results summary and CLI usage
├── PROJECT_STATUS.md                This file
├── data/
│   └── semantic_kitti/dataset/sequences/00/   Real KITTI data: 4,541 velodyne .bin + labels
├── models/
│   ├── salsanext/pretrained/pretrained/   Official checkpoint (weights + arch_cfg.yaml + data_cfg.yaml)
│   ├── salsanext/rellis3d_finetuned/      Stage 6 fine-tuned checkpoints (best.pt = epoch 9)
│   └── openpcdet/                  Empty — no PointPillars checkpoint downloaded
├── external/
│   └── SalsaNext/                  Cloned, HEAD 7548c124
│   (OpenPCDet v0.5.2 is cloned outside the repo, at C:\FOVEAX 2.5D\external\OpenPCDet;
│    RELLIS-3D data lives at C:\dev\data\rellis3d)
├── docs/
│   ├── phase8_object_tracking.md
│   ├── phase10_ros2_integration.md
│   ├── phase11_deployment_and_optimization.md
│   ├── openpcdet_pointpillars_setup.md
│   ├── salsanext_setup.md
│   ├── rellis3d_integration.md     RELLIS-3D layout, domain-gap diagnosis and its resolution
│   ├── validation_results.md       Source of truth for every measured number
│   └── assets/                     README images, demo GIF/MP4
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
│   ├── 10b_eval_distance_metrics.py      Phase 12 Near/Mid/Far scorer (shared compute_metrics)
│   ├── 13_realtime_dashboard.py    Phase 9 entry point (GUI, --headless, --profile)
│   ├── 14_finetune_salsanext.py    Stage 6 fine-tuning
│   ├── prepare_rellis3d_for_salsanext_training.py
│   ├── perception/
│   │   ├── object_detector.py      ObjectDetector ABC, MockObjectDetector
│   │   ├── openpcdet_predictor.py  OpenPCDetPointPillarsDetector
│   │   ├── semantic_predictor.py   SemanticPredictor ABC, GT/Mock predictors
│   │   ├── salsanext_predictor.py  SalsaNextPredictor (real inference)
│   │   ├── range_projection.py     Spherical range-image projection for SalsaNext
│   │   ├── rellis3d_loader.py      RELLIS-3D file I/O, sky exclusion, class remap
│   │   ├── semantic_labels.py      FOVEAX taxonomy + KITTI remap + RELLIS3D_TO_FOVEAX
│   │   ├── terrain_features.py     Per-object slope/clearance/overhang, hazard kind
│   │   ├── overhead_detection.py   Overhead-clearance vs. solid-obstacle cells
│   │   ├── centerline_profile.py   Forward-corridor height profile
│   │   ├── ego_motion_estimate.py  Informational ICP displacement (no odometry)
│   │   └── grid_overlay.py         Traversability overlay rendering
│   ├── training/
│   │   └── rellis3d_dataset.py     Fine-tuning splits + range-image dataset
│   ├── tracking/
│   │   └── multi_object_tracker.py MultiObjectTracker, TrackState (Kalman)
│   ├── dashboard/
│   │   ├── dashboard_state.py      FrameState, HardwareMetrics, coord-frame check
│   │   ├── data_streamer.py        DataStreamerThread.process_frame (per-frame pipeline)
│   │   ├── qt_main_window.py       PyQt5 main window and panels
│   │   ├── open3d_viewer.py        Open3D 3D viewer process
│   │   ├── track_labels.py         3D box labels as Qt widgets
│   │   ├── win32_embed.py          Reparent the Open3D window into Qt
│   │   ├── stage_timer.py          --profile timing harness
│   │   └── export_*.py             Web JSON export, static PNG snapshots
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
├── outputs/                        Real generated artifacts, phase1–phase9, phase11, phase12,
│   plus outputs/phase7/rellis3d/<seq>_<frame>_<source>/ (RELLIS-3D runs, kept separate
│   from SemanticKITTI's outputs/phase7/<source>/ to avoid overwriting)
│   (no phase10/ — ROS 2 has never been run)
└── tests/                          505 tests; conftest.py enforces
                                     torch-before-PyQt5 import order for the whole session
```

### 5.3 Module Descriptions

| Module | Responsibility |
|---|---|
| `src/perception/object_detector.py` | Defines the detector interface; `MockObjectDetector` is a deterministic geometric-clustering baseline (explicitly warns it is not AI). |
| `src/perception/openpcdet_predictor.py` | Real PointPillars 3D detection via the OpenPCDet toolbox; includes 3D IoU computation and class-name mapping. |
| `src/perception/semantic_predictor.py` | Defines the semantic-segmentation interface; ground-truth and mock implementations. |
| `src/perception/salsanext_predictor.py` | Real SalsaNext semantic segmentation; `taxonomy="semantickitti"` (pretrained, 94.98% SemanticKITTI) or `taxonomy="foveax"` (RELLIS-3D fine-tuned, 81.53% held-out). |
| `src/perception/range_projection.py` | Spherical range-image projection + reverse re-projection, transcribed from the official SalsaNext `laserscan.py`. |
| `src/perception/rellis3d_loader.py` | Loads RELLIS-3D's real (non-SemanticKITTI-layout) `.bin`/`.label` files, excludes sky points, remaps to FOVEAX classes. |
| `src/tracking/multi_object_tracker.py` | Kalman-filter multi-object tracking (constant-velocity, IoU/Euclidean association). |
| `src/dashboard/*` | Live PyQt5 window + embedded Open3D 3D viewer (separate process). A QTimer-driven playback controller calls `DataStreamerThread.process_frame()` per frame and caches each `FrameState` for step-back. |
| `src/perception/terrain_features.py` | Real-geometry per-object features (slope, ground clearance, overhang, rock heuristic) and pothole/bump hazard classification. |
| `src/training/rellis3d_dataset.py` | RELLIS-3D splits and range-image samples for fine-tuning, with preprocessing identical to inference. |
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
- **SalsaNext** — real-time semantic segmentation on range-image projections of LiDAR scans. Pretrained
  checkpoint: 94.98% on SemanticKITTI. RELLIS-3D: 6.65% pretrained → 22.26% sensor-adapted → 81.53%
  fine-tuned (held-out sequence). See `docs/validation_results.md` §2.
- **PointPillars** (via OpenPCDet) — pillar-based 3D object detection. Repo cloned, no checkpoint
  downloaded yet — real inference not yet run (deliberately deprioritized in favor of the SalsaNext/
  RELLIS-3D segmentation work above, per the SIH problem statement's segmentation focus).

**Datasets:**
- **SemanticKITTI** — real data present (`data/semantic_kitti/dataset/sequences/00/`, 4,541 scans),
  actively used by Phases 6 and 7's ground-truth path.
- **RELLIS-3D** — real data present (sequences `00000`–`00004`, Ouster OS1-64) at `C:\dev\data\rellis3d`;
  loader `src/perception/rellis3d_loader.py`; used for dashboard playback, evaluation and fine-tuning.
- **nuScenes, CARLA** — referenced as possible future datasets; no loader or data exists.

**Robotics:** ROS 2 (Jazzy or Humble, per `docs/phase10_ros2_integration.md`) — package scaffolding
present, no ROS 2 install verified in this environment.

**Testing:** pytest, 505 tests across 34 test files, all passing on the committed code.

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

End-to-end dashboard pipeline on real RELLIS-3D frames (131,072 points; `docs/validation_results.md` §5):

| Metric | Measured Value |
|---|---|
| Headless replay (2026-09-13, commit `783e28f`, before later features) | 13.8 FPS overall, 60–65 ms/frame steady state |
| `process_frame()` in `--profile` (2026-09-30, committed code, ground-truth predictor) | 128.4 ms mean, 163.0 ms p95 |
| Same, with the held-back OPT-n speed-ups (same-day A/B) | 92.1 ms mean, 107.1 ms p95 |
| Largest stage | detection/clustering + terrain features, 84 ms (62 ms with OPT-n) |
| SalsaNext forward pass (2026-09-20 profile, pretrained) | 42.4 ms, plus 19.5 ms range projection |

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

Rewritten 2026-09-30; the earlier week-by-week checklist (dated 2026-09-11) is superseded — most of it
is done (dashboard run end to end, RELLIS-3D domain gap resolved by fine-tuning, requirements pinned,
README updated, 35 commits).

### 8.1 Before committing the held-back OPT-n speed-ups (blocking)
- [ ] Fix the 2 failing tests in `tests/test_dashboard_traversability_convention.py` (new `__init__`-only
  attributes read by `_generate_maps` and `_render_grid_to_label`).
- [ ] `data_streamer.py` OPT-12: `grid_maps["elevation"]` is a view of a buffer reused every frame, so a
  cached `FrameState` shows the newest frame's elevation after STEP back.
- [ ] `data_streamer.py` OPT-14: the cached `HardwareMetrics` object is mutated and shared across
  `FrameState`s, so cached frames show the latest fps/latency.
- [ ] `data_streamer.py` OPT-3/OPT-4: hazard and centerline caches compare `frame_idx - cached_idx`, which
  goes negative after RESTART or an environment switch; the previous run's (or previous sequence's)
  hazards and corridor profile are then served until the new index passes the old one.
- [ ] `qt_main_window.py` OPT-7: the centerline chart's "unchanged" key only covers bin count, marker count,
  first/last distance and the first bin's point count, so real height changes are skipped.
- [ ] `qt_main_window.py` OPT-5: per-layer render cache can leave another layer's image on screen; the
  overhead cache ignores the traversability grid and label size.
- [ ] `data_streamer.py` OPT-15: voxel-downsampling before `_generate_maps` changes point density (used by
  the uncertainty layer and hazard confidence) and z_min/z_max per cell — re-verify against the full
  cloud or drop it.
- [ ] `open3d_viewer.py`: `_TURBO_LUT` is rebuilt inside `update()` every frame, and the cached
  `self._turbo_cmap` is unused.

### 8.2 Next
- [ ] Add `--taxonomy` to the dashboard (`make_semantic_predictor`) so the live view can run the
  fine-tuned RELLIS-3D checkpoint.
- [ ] Show a measured FPS in the dashboard; `state.metrics.fps` is currently the target rate.
- [ ] Reduce detection/clustering cost (84 ms/frame, largest pipeline stage).
- [ ] Compile a real FP16 TensorRT engine from a trained checkpoint and measure it against §7.2.
- [ ] Mitigate SemanticKITTI forgetting (replay/mixing) if one checkpoint must serve both domains.
- [ ] First real ROS 2 run (`colcon build`, `ros2 launch`) against a bag or sensor.
- [ ] Download a PointPillars checkpoint and run real OpenPCDet inference (deprioritized).

---

## 9. Success Criteria

### 9.1 Technical Success Metrics
- [ ] End-to-end pipeline sustains a real-time frame rate (currently 128 ms per
  `process_frame()`; target 25–30 FPS).
- [ ] TensorRT FP16 engine achieves the documented 2.5×–4× throughput improvement, measured.
- [ ] Concurrent semantic + detection engines fit the 6 GB budget, verified with real engine sizes.
- [x] Semantic segmentation validated against ground truth with a real model on two datasets:
  94.98% SemanticKITTI (pretrained), 81.53% RELLIS-3D held-out (fine-tuned).
- [x] Full test suite green on the committed code (505 tests, 2026-09-30). The held-back OPT-n changes must keep it green (see §8.1).

### 9.2 SIH Submission Requirements
- [x] GitHub repository with phase-scoped commit history; downloadable Windows app on the Releases page
  (`docs/packaging.md`).
- [x] Working end-to-end dashboard demo on real data (demo GIF/MP4 in `docs/assets/`). ROS 2 not yet run.
- [x] Documentation: `CLAUDE.md` (architecture), `README.md` (usage), `docs/validation_results.md`
  (measured results), this document (status).
- [x] Reproducible environment setup: pinned `requirements.txt`/`pyproject.toml` with the cu130 note.
