# FoveaX Trail

## Project Description

FoveaX Trail is a deterministic, high-performance LiDAR perception and 2.5D adaptive elevation-grid
mapping pipeline for autonomous navigation and off-road/trail terrain understanding. The design goal
throughout is to keep the real-time path lean — classical geometric processing (voxel grids, slope/
roughness/step-height math, foveated multi-resolution grids) carries as much of the load as possible,
with AI models (SalsaNext for semantic segmentation, PointPillars/OpenPCDet for 3D object detection)
plugged in behind swappable interfaces rather than baked into the core pipeline.

The pipeline runs end to end from a raw LiDAR point cloud to: a 2.5D elevation + traversability map,
a semantic terrain classification, tracked 3D object detections, a live PyQt5/Open3D dashboard, a
ROS 2 integration for live sensor streams, and a deployment/optimization layer (ONNX + TensorRT,
INT8 calibration, VRAM budgeting) targeting memory-constrained GPUs — developed and verified against
an RTX 5050 Laptop GPU (8GB VRAM, Blackwell/sm_120, requires PyTorch cu130+).

Full phase-by-phase description and CLI usage: see `README.md`. This file is oriented at how the
pieces connect, for architecture/codebase questions.

## Pipeline Flow

```mermaid
flowchart TD
    A["Raw LiDAR Point Cloud\n(x, y, z, intensity)"] --> B["Phase 1\nFiltering, voxel downsampling,\nstatistical outlier removal"]
    B --> C["Phase 2\n2.5D Elevation Grid\n(z_min, z_max, height_range, density)"]
    C --> D["Phase 3\nTerrain Traversability\nslope, roughness, step-height\n1.0=safe .. 0.0=blocked"]
    D --> E["Phase 4\nAdaptive Multi-Resolution Grids\nnear 5cm / mid 20cm / far 50cm"]
    E --> F["Phase 5\nSpatial Importance ROI\ndistance + terrain risk + uncertainty"]

    C --> G["Phase 6\nSemanticKITTI ground-truth\nsemantic 2.5D grid"]
    G --> H["Phase 7\nAI Semantic Segmentation\nGroundTruth / Mock / SalsaNext\n(src/perception/semantic_predictor.py)"]
    R["RELLIS-3D raw scans\n(Rellis-3D/<seq>/os1_cloud_node_*)"] --> RL["rellis3d_loader.py\nsky exclusion, RELLIS3D_TO_FOVEAX remap"]
    RL --> H

    A --> I["Phase 8A\n3D Object Detection\nMockObjectDetector (geometric)\n(src/perception/object_detector.py)"]
    I --> J["Phase 8B\nReal AI Detection\nOpenPCDet PointPillars\n(src/perception/openpcdet_predictor.py)"]
    J --> K["Multi-Object Tracking\nKalman filter, IoU/Euclidean assoc.\n(src/tracking/multi_object_tracker.py)"]

    H --> L
    F --> L
    K --> L["Phase 9\nReal-Time Dashboard\nPyQt5 2D panel + Open3D 3D viewer\n(src/dashboard/, src/13_realtime_dashboard.py)"]
    L --> M["Session Telemetry\nFPS, latency, coordinate-frame\nwarnings -> outputs/phase9/session/*.jsonl"]

    N["Live sensor_msgs/PointCloud2\n(ROS 2 topic)"] --> O["Phase 10\nROS 2 Integration\nlidar_node.py, tf2 adapter,\nbounded queue, single-thread exec\n(ros2_ws/, src/integrations/)"]
    O --> I

    H --> P["Phase 11\nDeployment & Optimization\nONNX export, TensorRT fp16/INT8,\nVRAM budget checks\n(src/deployment/)"]
    J --> P
    P --> Q["Optimized Inference\nruntime engine, 8GB VRAM target\n(src/deployment/optimized_inference.py)"]
```

### Data flow in one line
`Point cloud → filtered/voxelized → 2.5D elevation+traversability grid → semantic classification
(GT/Mock/SalsaNext) → object detection (Mock/PointPillars) → tracking → merged FrameState →
dashboard (PyQt5 + Open3D) / ROS 2 output topics, with TensorRT-optimized inference substituted
in for deployment.`

### Key interfaces (the swap points)
- `SemanticPredictor` (`src/perception/semantic_predictor.py`) — `GroundTruthSemanticPredictor`,
  `MockSemanticPredictor`, `SalsaNextPredictor` all implement this; swap via `--source` on
  `src/10_ai_semantic_2point5d_map.py` and via `--predictor` on `src/13_realtime_dashboard.py`.
  `GroundTruthSemanticPredictor` accepts either raw SemanticKITTI-packed `labels_raw` or
  already-remapped `foveax_class_ids` (the latter from `rellis3d_loader.py`) — mutually exclusive
  constructor args.
- `--dataset-type {semantickitti, rellis3d}` on `src/10_ai_semantic_2point5d_map.py` selects which
  dataset root/layout `--sequence`/`--frame` resolve against (SemanticKITTI 2-digit sequences vs.
  RELLIS-3D 5-digit sequences under a different root entirely — see `docs/rellis3d_integration.md`).
- `ObjectDetector` (`src/perception/object_detector.py`) — `MockObjectDetector` (geometric baseline,
  not AI) vs `OpenPCDetPointPillarsDetector` (real AI); swap via `--detector`.
- `SalsaNextPredictor.predict()` is real, working inference (spherical range-image projection ->
  model forward pass -> reverse re-projection -> SemanticKITTI-to-FOVEAX remap), not a stub — see
  `src/perception/range_projection.py`. Two checkpoints, selected by the `taxonomy` constructor arg:
  - `taxonomy="semantickitti"` — official pretrained checkpoint
    (`models/salsanext/pretrained/pretrained/`). 94.98% on SemanticKITTI (UNKNOWN excluded). On
    RELLIS-3D it scores 6.65% raw / 22.26% with sensor adaptation (Ouster FOV `fov_up=17.02,
    fov_down=-16.44` + `rescale_intensity`) — a Velodyne-vs-Ouster domain gap, diagnosed in
    `docs/rellis3d_integration.md`.
  - `taxonomy="foveax"` — fine-tuned on RELLIS-3D with an 8-class FOVEAX head
    (`models/salsanext/rellis3d_finetuned/best.pt`, trained by `src/14_finetune_salsanext.py`).
    81.53% on held-out sequence `00000`, but forgets SemanticKITTI (23.97%). Its argmax is already
    a FOVEAX ID — never pass it through `learning_map_inv`/`SEMANTICKITTI_TO_FOVEAX`.
  - Numbers and methodology: `docs/validation_results.md` (source of truth). The dashboard's
    `make_semantic_predictor` (`export_web_dashboard_data.py`) has no `--taxonomy` flag, so the
    live dashboard can currently only load the pretrained (`semantickitti`) checkpoint.
- `FrameState` / `HardwareMetrics` (`src/dashboard/dashboard_state.py`) — the single struct that
  carries points, grid_maps, tracks, and metrics from `DataStreamerThread` into both the 2D (Qt) and
  3D (Open3D) dashboard panels. Also carries `coordinate_warnings` from
  `assert_coordinate_frame_consistency()`, a non-blocking cross-panel sanity check.
- Traversability convention is canonical from `src/06_terrain_traversability.py` and
  `src/perception/grid_overlay.py`: **1.0 = safe, 0.0 = blocked**, thresholds Safe ≥ 0.70,
  Caution 0.40–0.70, Blocked < 0.40. Any new traversability-consuming code (colormaps, hatching,
  mock data) must match this or it silently inverts safe/unsafe in the UI.

## Repository Map

```
src/
├── 01-05_*.py                    Phase 1-2: point cloud I/O, filtering, 2.5D elevation grid
├── 06_terrain_traversability.py  Phase 3: canonical traversability convention lives here
├── 07_adaptive_2point5d_map.py   Phase 4: foveated multi-resolution grids
├── 08_spatial_importance_roi.py  Phase 5: ROI importance blending
├── 09_semantickitti_semantic_map.py  Phase 6: ground-truth semantic grid
├── 10_ai_semantic_2point5d_map.py    Phase 7: pluggable AI semantic segmentation CLI
├── 10b_eval_distance_metrics.py      Near/Mid/Far accuracy + confusion matrix; its
│                                      compute_metrics() is the ONE shared scorer (dashboard too)
├── 11_object_detection_tracking.py   Phase 8A/8B: detection + tracking CLI
├── 12_env_inspector.py           Environment/hardware capability check
├── 13_optimize_and_deploy.py     Phase 11: CLI for profiling / TensorRT engine build
├── 13_realtime_dashboard.py      Phase 9: dashboard entry point (imports torch before PyQt5 —
│                                  see comment in file, DLL load-order constraint on Windows);
│                                  modes: live GUI (default), --headless, --profile
├── 14_finetune_salsanext.py      Stage 6: fine-tune SalsaNext on RELLIS-3D (FOVEAX 8-class head)
├── prepare_rellis3d_for_salsanext_training.py  Converts RELLIS-3D into upstream SalsaNext's
│                                  sequences/ layout (alternative path to training/)
├── perception/                   ObjectDetector, SemanticPredictor and their implementations,
│                                  rellis3d_loader.py (RELLIS-3D file I/O + class remap),
│                                  range_projection.py (spherical projection for SalsaNext),
│                                  terrain_features.py (per-detection slope/clearance/overhang,
│                                  hazard kind), overhead_detection.py, centerline_profile.py,
│                                  ego_motion_estimate.py (informational ICP only, no odometry)
├── training/                     rellis3d_dataset.py: splits (seq 00000 = held-out test) and
│                                  range-image samples, preprocessing identical to predict()
├── tracking/                     MultiObjectTracker, TrackState (Kalman filter tracking)
├── dashboard/                    FrameState/HardwareMetrics, DataStreamerThread.process_frame,
│                                  Qt main window, Open3D viewer process, track_labels.py (3D box
│                                  labels as Qt widgets), win32_embed.py (reparent Open3D window),
│                                  stage_timer.py (--profile harness), export_* (web JSON, PNGs)
├── deployment/                   vram_budget, model_exporter (ONNX), tensorrt_builder,
│                                  calibration (INT8), optimized_inference, profiler
└── integrations/                 ROS 2 PointCloud2 <-> numpy adapter, tf2 adapter

ros2_ws/src/foveax_ros/foveax_ros/   Phase 10: lidar_node.py (plain rclpy.Node, not
                                      LifecycleNode — see docs/phase10_ros2_integration.md),
                                      diagnostics.py

docs/                              Phase-specific setup/implementation notes
packaging/                         PyInstaller spec + build_app.py: standalone Windows app with a bundled
                                    RELLIS-3D sample, shipped on GitHub Releases (docs/packaging.md)
outputs/phaseN/                    Generated artifacts per phase (git-ignored where large)
tests/                             pytest suite; conftest.py enforces torch-before-PyQt5
                                    import order for the whole session (see below)
```

## Dashboard Runtime Model (read before changing dashboard code)

- **The GUI does not use `DataStreamerThread.run()`.** `DashboardApplication` (in
  `src/13_realtime_dashboard.py`) is a pull-based playback controller: a `QTimer` on the Qt thread
  calls `streamer.process_frame(idx, ...)` directly and stores each `FrameState` in
  `_frame_cache[idx]`. STEP back replays cached states (the Kalman tracker is never run backward);
  RESTART / environment switch replace the tracker and clear the cache, and frame indices restart
  at 0. `run()` (the QThread loop) is only a legacy path.
- **Cached `FrameState`s must own their data.** Anything placed in a `FrameState` (grid arrays,
  `HardwareMetrics`) must not be a view of, or the same object as, a buffer reused on the next
  frame — step-back replays it later.
- **Any per-frame throttle cache in `DataStreamerThread` must handle frame indices going
  backward** (restart/environment switch) — a `frame_idx - cached_idx < N` check alone serves the
  previous run's data.
- The Open3D 3D view runs in a **separate process** fed through `o3d_queue` (drop-oldest), and its
  GLFW window is reparented into the Qt window via `win32_embed.py` (`--separate-windows` opts out).
  Alert cards and the threat toast are laid out inside the left panel, not floated over the 3D
  view — floating overlays lost z-order fights with the native Open3D child window
  (`tests/test_alert_card_placement.py`).
- `state.metrics.fps` is the rate passed into `process_frame` (the target playback rate), **not a
  measured frame rate**. Measure with `--profile` (per-stage mean/p50/p95 via
  `src/dashboard/stage_timer.py`, CUDA-synchronized for GPU stages, 1 Hz `nvidia-smi` log to
  `outputs/phase12/gpu_util_profile.csv`). The profile runs Open3D in-process and off-screen, so its
  `open3d_update` cost is not what the separate-process live GUI pays per frame.
- Tests often build `DataStreamerThread` / `FoveaXDashboardWindow` via `__new__` (skipping
  `__init__`), so attributes introduced only in `__init__` break them.

## Environment Notes (read before touching dependencies)

- **Verify the venv path before using it — it has moved more than once.** It was originally at
  `.venv/` inside this project folder while the project lived on OneDrive-synced `Desktop` (that
  caused real, reproducible corruption during large package installs — OneDrive syncing mid-write).
  It was then moved to `C:\dev\foveax_venv`, and then again to `C:\FOVEAX 2.5D\foveax_venv`
  (sibling of this repo, not inside it) after the whole project was relocated off OneDrive. **As of
  2026-09-11 (re-confirmed 2026-09-30) the real venv is at `C:\FOVEAX 2.5D\foveax_venv`** (Python
  3.11) — `C:\dev\foveax_venv` no longer
  exists. Do not trust a hardcoded path from an old session/transcript; run
  `Test-Path <candidate>\Scripts\python.exe` first. Same caution applies to this repo's own root —
  it has also moved multiple times (currently `C:\FOVEAX 2.5D\FOVEAX_2.5D_TRAIL`).
- **Data and third-party locations:** RELLIS-3D lives at `C:\dev\data\rellis3d\Rellis-3D\<00000-00004>\`
  (the loaders' default root; `data/rellis3d` is only a fallback). SemanticKITTI seq `00` is at
  `data/semantic_kitti/dataset/sequences/00/`. `external/SalsaNext` is cloned at `7548c12`;
  OpenPCDet lives at `C:\FOVEAX 2.5D\external\OpenPCDet` (outside this repo).
- **PyTorch must be the `cu130` build** (`torch==2.14.0+cu130`), not `cu126`. The RTX 5050 (Blackwell,
  compute capability `sm_120`) has no kernels in `cu126` — `cuda.is_available()` reports `True` but
  any real op fails with `no kernel image is available for execution on the device`. Always verify
  with an actual op (matmul), not just `is_available()`.
- **Import order: torch before PyQt5, in the same process.** PyQt5's bundled Qt DLLs and torch's
  bundled CUDA DLLs conflict on Windows if PyQt5 loads first — crashes as `OSError: WinError 1114`
  loading `c10.dll`, or a harder native `access violation`. `tests/conftest.py` imports torch first
  for the whole pytest session; `src/13_realtime_dashboard.py` does the same at its entry point. Any
  new entry point or test file that touches both PyQt5 and a torch-based predictor needs this too.

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST extraction is local and free)
- `.graphifyignore` keeps `external/`, `models/`, `data/`, `outputs/` out of the graph (third-party clones, checkpoints, datasets, generated artifacts)
- Caveat: the installed graphify build (`ai.py`) always sends community node labels (function/class names, up to 20 per cluster) to pollinations.ai to generate cluster summaries — no source code, but there is no opt-out flag
- `src/graphify-out/` is an older src-only copy from 2026-09-10; `graphify-out/` at the repo root is the current graph
