# FOVEAX 2.5D — Validation Results

**All numbers on this page were freshly re-measured against current source
files and live runs on the date below — none are carried forward from
memory or prior conversation reports without re-verification.**

Verified: 2026-09-13, against commit `783e28f` (interpreter:
`C:\FOVEAX 2.5D\foveax_venv\Scripts\python.exe`, Python 3.11.16).
On 2026-09-30 the foveated zones moved from 0–15 / 15–35 / 35–100 m to
the problem statement's 0–10 / 10–30 / 30–100 m (cell sizes unchanged).
§1–§3 were re-measured with the new zones the same day: every overall
(0–100 m) accuracy reproduced exactly, only the per-bucket splits and
the memory figures changed. §5.1 and §7 were also re-measured on 2026-09-30
against the code that is committed on top of `f174f08` (the unfinished
"OPT-n" dashboard speed-ups are not part of it); each section states which
run it comes from.

---

## 1. Problem Statement (PS-26053) Compliance

| Requirement | Spec | Real measured value | Status |
|---|---|---|---|
| Near-zone resolution (0–10 m) | 5 cm | 5 cm (`SPEC_ZONES_100M`, `src/07_adaptive_2point5d_map.py`) | ✅ Met |
| Middle-zone resolution (10–30 m) | 20 cm | 20 cm | ✅ Met |
| Far-zone resolution (30–100 m) | 50 cm | 50 cm | ✅ Met |
| Memory reduction vs. uniform 3D voxel grid | Significant reduction | **99.99%** cell reduction, **99.81%** byte reduction (see §3) | ✅ Met |
| Real-time dashboard operation | Live playback | **128 ms/frame** pipeline time (≈7.8 FPS), real RELLIS-3D data (see §5) | ⚠️ Below typical automotive real-time targets (25–30 FPS); see §6 |
| Semantic segmentation (in-domain) | Usable accuracy | 94.98% overall (SemanticKITTI, real SalsaNext checkpoint, see §2) | ✅ Met |
| Semantic segmentation (cross-domain, RELLIS-3D) | N/A (stretch) | 22.26% overall after sensor adaptation only, **81.53% after fine-tuning** (see §2.3) | ✅ Met after fine-tuning (held-out test sequence, never seen in training) |

## 2. Deep Learning (SalsaNext) Results

Real inference via `src/perception/salsanext_predictor.py` against the
official pretrained SalsaNext checkpoint
(`models/salsanext/pretrained/pretrained/SalsaNext`), scored with
`src/10b_eval_distance_metrics.py::compute_metrics()` (the single,
shared implementation — not reimplemented anywhere else in this
codebase; the live dashboard's accuracy card calls this exact function
too).

### 2.1 SemanticKITTI (in-domain — the checkpoint's native training domain)

Sequence `00`, frame `000000`:

| Bucket | Points evaluated | Accuracy |
|---|---:|---:|
| Near (0–10 m) | 62,325 | **97.70%** |
| Mid (10–30 m) | 52,916 | **92.82%** |
| Far (30–100 m) | 7,239 | **87.28%** |
| **Overall (0–100 m)** | **122,480** | **94.98%** |

### 2.2 RELLIS-3D (cross-domain — off-road, different LiDAR: Ouster OS1-64 vs. training-domain Velodyne HDL-64E)

Sequence `00000`, frame `000000`:

| Bucket | Points evaluated | Accuracy, no adaptation | Accuracy, with sensor adaptation* |
|---|---:|---:|---:|
| Near (0–10 m) | 103,534 | 3.78% | **15.07%** |
| Mid (10–30 m) | 26,810 | 17.36% | **49.57%** |
| Far (30–100 m) | 726 | 20.25% | **38.43%** |
| **Overall (0–100 m)** | **131,070** | **6.65%** | **22.26%** |

\* Sensor adaptation = real Ouster OS1-64 vertical FOV override
(`fov_up=17.02°, fov_down=-16.44°`, vs. the checkpoint's assumed Velodyne
HDL-64E FOV of `-25°..+3°`) plus intensity rescaling to `[0,1]` (RELLIS-3D
intensities are ~0.0001–0.01, the training domain's are ~0–1). Both real,
documented, measured adaptations — not a guess. Sensor adaptation alone
(22.26%) is a real, honest domain-transfer failure, not a usable
production number on its own — see §2.3 for the fine-tuned result that
resolves this.

### 2.3 RELLIS-3D fine-tuned (Stage 6 — SalsaNext fine-tuned directly on real RELLIS-3D data)

Real fine-tuning run (`src/14_finetune_salsanext.py`), starting from the
same official SemanticKITTI-pretrained checkpoint used in §2.1/§2.2, with
its 20-class head replaced by an 8-class FOVEAX head (see
`src/perception/salsanext_predictor.py`'s `taxonomy="foveax"` path) and
fine-tuned on real RELLIS-3D data using the same real Ouster OS1-64 sensor
adaptation as §2.2. Full detail (split, config, per-epoch trajectory,
forgetting check) in the Stage 6 report; summarized here.

**Split** (deterministic, `src/training/rellis3d_dataset.py::build_splits`):
sequence `00000` held out **entirely** as test (2,847 frames, never seen
during training); train/val drawn from sequences `00001`-`00004` with a
15% contiguous tail per sequence as val and a 10-frame gap between train
and val to prevent near-duplicate adjacent-frame leakage. Real counts:
train 4,532 frames (stride 2), val 162 frames (stride 10), test 2,847
frames (full sequence, stride 20 for evaluation = 100 frames actually
scored below).

**Config**: 10 epochs, batch size 2, SGD lr=0.01 (upstream SalsaNext's own
warmup+decay schedule), class-weighted `NLLLoss` (`w = 1/(content +
0.001)`, computed on the real training split's own class frequencies) +
`Lovasz_softmax`, both reused unmodified from the official SalsaNext repo.
Real per-epoch validation accuracy (held-out val split, 8 classes):

| Epoch | Train loss | Val accuracy |
|---:|---:|---:|
| 0 | 0.8431 | 84.99% |
| 1 | 0.5350 | 86.55% |
| 2 | 0.4908 | 86.30% |
| 3 | 0.4109 | 87.77% |
| 4 | 0.3913 | 88.87% |
| 5 | 0.3659 | 87.76% |
| 6 | 0.3806 | 87.52% |
| 7 | 0.4147 | 88.41% |
| 8 | 0.3744 | 88.47% |
| **9 (best)** | **0.3220** | **89.05%** |

Real training time: 101 minutes total training compute (sum of real
per-epoch `train_seconds`; wall-clock spanned longer across this session
due to two real interruptions — see the Stage 6 report — both resumed
cleanly from checkpoint, no lost epochs). Peak real GPU memory: 3,767 MB
(RTX 5050 Laptop, 8,151 MB total) at batch size 2.

**Held-out test-sequence result** (sequence `00000`, 100 frames, stride
20, real Ouster sensor adaptation applied, scored by the same
`compute_metrics()` used everywhere else on this page):

| Bucket | Points evaluated | Accuracy |
|---|---:|---:|
| Near (0–10 m) | 7,103,965 | 78.83% |
| Mid (10–30 m) | 2,522,815 | 89.42% |
| Far (30–100 m) | 61,051 | 69.85% |
| **Overall (0–100 m)** | **9,687,831** | **81.53%** |

Real progression on this held-out sequence: **6.65%** (pretrained,
no adaptation) → **22.26%** (pretrained + sensor adaptation) →
**81.53%** (fine-tuned on RELLIS-3D). The Far bucket's lower accuracy
(69.85%, only 61,051 points) reflects real sparse far-range coverage on
this off-road LiDAR, not a training defect.

**SemanticKITTI forgetting check**: because the fine-tuned head emits
FOVEAX classes directly (not SemanticKITTI's 20), its accuracy is
evaluated against SemanticKITTI ground truth *mapped into the same FOVEAX
space* the original pretrained checkpoint is scored in throughout this
page (frame `00/000000`, `--taxonomy foveax`):

| | Overall FOVEAX-space accuracy |
|---|---:|
| Original pretrained checkpoint (§2.1) | 94.98% |
| **Fine-tuned on RELLIS-3D** | **23.97%** |

This is real, substantial forgetting, not a near-zero collapse — expected
given 10 epochs of RELLIS-3D-only fine-tuning with no SemanticKITTI replay
or mixing. Flagged honestly as a known limitation (§6) rather than
omitted: this checkpoint should be used for off-road RELLIS-3D-like
scenes, not as a drop-in replacement for the original urban-domain
checkpoint.

**Checkpoints**: original SemanticKITTI checkpoint preserved unchanged at
`models/salsanext/pretrained/pretrained/SalsaNext` (confirmed via MD5,
matches its pre-fine-tuning hash). New fine-tuned checkpoints at
`models/salsanext/rellis3d_finetuned/` (`epoch_000.pt`...`epoch_009.pt`,
`last.pt`, `best.pt` = epoch 9).

## 3. Grid Engine Efficiency (Phase B)

Computed live via
`src/07_adaptive_2point5d_map.py::compute_memory_reduction_metrics()`
against the real `SPEC_ZONES_100M` config (Near 0–10m/5cm, Middle
10–30m/20cm, Far 30–100m/50cm) — a fixed architectural property of the
current zone config, not a per-frame measurement:

| Metric | Value |
|---|---:|
| Finest resolution | 0.05 m |
| Uniform 3D voxel grid (same finest resolution, 8 m height) | 2,560,000,000 voxels / 2,441.4 MB |
| Uniform 2.5D grid (same finest resolution) | 16,000,000 cells / 244.1 MB |
| Adaptive foveated 2.5D grid (actual FOVEAX design) | 302,850 cells / 4.6 MB |
| **Cell reduction vs. uniform 3D voxels** | **99.99%** |
| **Byte reduction vs. uniform 3D voxels** | **99.81%** |
| Cell reduction vs. uniform 2.5D grid | 98.11% |

## 4. Tracking / Detection Results

- Clustering: `MockObjectDetector` (deterministic geometric baseline, not
  a trained detector) using vectorized `cKDTree.query_pairs` +
  `scipy.sparse.csgraph.connected_components` (replaced a per-point BFS
  loop; verified bit-identical clustering output against the original
  BFS on real RELLIS-3D frames before/after).
- Tracking: constant-velocity Kalman filter + Hungarian assignment
  (`src/tracking/multi_object_tracker.py`).
- **Known, fixed false-positive class**: "Dynamic" classification is
  gated by real-world class semantics — classes that cannot physically
  move (VEGETATION, BUILDING_WALL, SOLID_OBSTACLE, DRIVABLE_GROUND,
  ROUGH_TERRAIN) can never be flagged Dynamic, regardless of measured
  speed. Root cause: this pipeline has no ego-motion compensation
  (no odometry), so a real, long-duration vehicle motion (translation
  and/or rotation) makes every real static object appear to drift
  through the ego-relative sensor frame. Verified over a real 200-frame
  RELLIS-3D run (sequence `00002`): **0** Dynamic-VEGETATION events
  across the full run (down from 12 tracks falsely flagged before the
  fix), while genuinely mobile classes still correctly reported Dynamic
  (VEHICLE: 176 events / 10 tracks; PEDESTRIAN: 185 events / 6 tracks).

## 5. Dashboard Performance

**Current numbers are in §5.1** (per-stage profile of the committed code:
128.4 ms/frame). The headless replay below is an older measurement, taken
on commit `783e28f` before the per-object terrain features, 3D labels and
overhead/corridor panels were added, so it no longer describes the code:

Real headless replay, RELLIS-3D sequence `00001`, 30 real frames
(`src/13_realtime_dashboard.py --headless`), 2026-09-13:

| Metric | Value |
|---|---:|
| Steady-state per-frame latency (frames 5–30) | 60.1–64.8 ms |
| Steady-state FPS (from latency) | ~15.5–16.6 FPS |
| Overall measured FPS (incl. one-time first-frame warmup) | **13.8 FPS** |
| Real points per frame | 131,072 |

This reflects the current state after two rounds of real profiling-driven
optimization (clustering vectorization + overhead-grid vectorization),
which together took real per-frame processing time from ~214 ms down to
~62–65 ms on this same real data (a ~3.3x improvement). See §6 for the
honest gap to a 25–30 FPS target.

### 5.1 Per-stage profile of the committed code (`--profile`, 2026-09-30)

`python src/13_realtime_dashboard.py --profile --profile-warmup 20
--profile-frames 200 --source rellis3d --sequence 00001` (default
`--predictor ground_truth`, so no SalsaNext inference). This harness
drives the same objects as the live GUI, frame by frame, with the Qt
window and Open3D viewer real but off-screen and **in-process** (see
`src/dashboard/stage_timer.py`). 200 measured frames:

| Stage | mean ms | p95 ms | % of frame |
|---|---:|---:|---:|
| load (RELLIS-3D `.bin` + labels) | 9.4 | 14.7 | 2.4% |
| live_accuracy_metrics | 4.3 | 6.8 | 1.1% |
| clustering_heuristics (detect + classify + terrain features) | 84.3 | 109.1 | 21.4% |
| kalman_tracking | 1.4 | 2.2 | 0.4% |
| grid_maps_hazards | 23.5 | 30.4 | 6.0% |
| metrics_state_logging | 5.4 | 7.6 | 1.4% |
| ego_motion_icp (every 5th frame in `--profile`) | 29.3 | 40.3 | 1.5% |
| open3d_update (in-process) | 184.5 | 221.9 | 46.9% |
| qt_ui_update + qt_event_loop_paint | 74.4 | — | 18.9% |
| **frame_total** | **393.8** | **479.2** | 100% |

- **`process_frame()` alone: 128.4 ms mean (p95 163.0 ms), ≈7.8 FPS
  ceiling for the pipeline itself.** The dashboard's on-screen latency
  (`state.metrics.latency_ms`) averaged 122.9 ms in the same run; it
  measures `process_frame()` only and excludes drawing.
- **Held-back speed-ups, same-day A/B.** The working tree contains
  unfinished dashboard speed-ups (caches and throttles, not committed
  because of known bugs, see `PROJECT_STATUS.md` §8.1). Profiled
  back to back under the same conditions they give **92.1 ms** per
  frame (clustering 61.8 ms, grids 14.4 ms), about 36 ms (28%) faster.
  Two earlier runs of that code gave 86.5 and 92.1 ms, so expect
  roughly ±10% between runs on this laptop.
- The in-process `open3d_update` cost is a profiling artifact: the live
  GUI runs Open3D in a separate process, so it is not added to the Qt
  thread's per-frame time.
- **The on-screen FPS figure is the target rate, not a measurement.**
  `state.metrics.fps` carried exactly `30.0` (the `--rate-hz` default)
  on every measured frame. Do not quote the dashboard's FPS label as
  achieved throughput.
- GPU utilisation during this run averaged 1.2% (no model inference).
  An earlier 2026-09-20 run with `--predictor salsanext` (pretrained
  checkpoint, before the dashboard performance changes) measured
  `salsanext_forward` at 42.4 ms and `range_image_projection` at 19.5 ms
  per frame (`outputs/phase12/profile_run_raw.txt`).

**Do not quote the "145–171 FPS" MockObjectDetector figure as
representative of real dashboard performance** — that number was
measured on the ~2,150-point synthetic `sample` source, not a real
~131,000-point RELLIS-3D/SemanticKITTI scan; it does not reflect this
project's real-data throughput.

## 6. Known Limitations

1. **No ego-motion compensation.** This pipeline has no odometry/SLAM.
   Over a long real playback, genuine vehicle motion makes static real
   objects appear to move in the ego-relative frame. Mitigated for
   physically-static classes (see §4) but not solved for genuinely
   mobile classes (VEHICLE/PEDESTRIAN velocity readings may still be
   partially contaminated by uncompensated ego motion).
2. **The fine-tuned RELLIS-3D checkpoint (81.53% held-out overall, §2.3)
   substantially forgets the original SemanticKITTI domain (94.98% →
   23.97% overall, FOVEAX-space).** Real, measured, not a near-zero
   collapse, but a real tradeoff: this checkpoint is not a drop-in
   replacement for the original — use `models/salsanext/pretrained/`
   for urban/SemanticKITTI-like scenes and
   `models/salsanext/rellis3d_finetuned/` for off-road/RELLIS-3D-like
   scenes. Resolving this would need replaying real SemanticKITTI
   samples during fine-tuning (not done here).
3. **Dashboard throughput is well below typical 25–30 FPS real-time
   targets.** `process_frame()` takes 128 ms per real 131k-point frame
   (≈7.8 FPS) in the 2026-09-30 profile (§5.1); an earlier headless run
   measured 13.8 FPS before later features were added. Detection,
   clustering and per-object terrain features (84 ms/frame) are the
   largest stage. Unfinished speed-ups that cut this to about 92 ms are
   held back until their bugs are fixed.
4. **`MockObjectDetector` is a deterministic geometric baseline, not a
   trained AI detector** — it clusters points above a ground-height
   threshold; it does not perform learned object recognition. Real
   semantic classification (VEGETATION/VEHICLE/PEDESTRIAN/etc.) only
   applies when real ground-truth or SalsaNext labels are cross-
   referenced against a cluster.

## 7. Test Suite Status

Fresh run, re-confirmed 2026-09-13 (post Stage 6 SalsaNext RELLIS-3D
fine-tuning + terrain-feature/distance-display dashboard work -- 138 new
regression tests added since this doc's original 352-test baseline):

```
490 passed, 33 warnings in 8.60s
```

No skipped tests, no failures, no regressions against any prior baseline
recorded during this project's development.

**Fresh run, 2026-09-30** (the committed code, with the 0–10 / 10–30 /
30–100 m zones, the packaging changes and the new
`tests/test_alert_card_placement.py`; SemanticKITTI data and the SalsaNext
checkpoints present):

```
505 passed, 32 warnings in 5.46s
```

The unfinished "OPT-n" dashboard speed-ups (kept out of the commit) fail 2
tests in `tests/test_dashboard_traversability_convention.py`: those tests
build `DataStreamerThread` and `FoveaXDashboardWindow` via `__new__`, and the
speed-up code reads attributes that only `__init__` sets. They also have
correctness bugs listed in `PROJECT_STATUS.md` §8.1.
