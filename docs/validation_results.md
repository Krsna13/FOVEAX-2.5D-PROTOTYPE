# FOVEAX 2.5D — Validation Results

**All numbers on this page were freshly re-measured against current source
files and live runs on the date below — none are carried forward from
memory or prior conversation reports without re-verification.**

Verified: 2026-09-13, against commit `783e28f` (interpreter:
`C:\FOVEAX 2.5D\foveax_venv\Scripts\python.exe`, Python 3.11.16).

---

## 1. Problem Statement (PS-26053) Compliance

| Requirement | Spec | Real measured value | Status |
|---|---|---|---|
| Near-zone resolution (0–15 m) | 5 cm | 5 cm (`SPEC_ZONES_100M`, `src/07_adaptive_2point5d_map.py`) | ✅ Met |
| Middle-zone resolution (15–35 m) | 20 cm | 20 cm | ✅ Met |
| Far-zone resolution (35–100 m) | 50 cm | 50 cm | ✅ Met |
| Memory reduction vs. uniform 3D voxel grid | Significant reduction | **99.98%** cell reduction, **99.71%** byte reduction (see §3) | ✅ Met |
| Real-time dashboard operation | Live playback | **13.8 FPS** measured, real RELLIS-3D data (see §5) | ⚠️ Below typical automotive real-time targets (25–30 FPS); see §6 |
| Semantic segmentation (in-domain) | Usable accuracy | 94.98% overall (SemanticKITTI, real SalsaNext checkpoint, see §2) | ✅ Met |
| Semantic segmentation (cross-domain, RELLIS-3D) | N/A (stretch) | 22.26% overall after sensor adaptation, 6.65% without (see §2) | ⚠️ Known limitation, not met — see §6 |

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
| Near (0–15 m) | 90,608 | **96.03%** |
| Mid (15–35 m) | 26,947 | **92.96%** |
| Far (35–100 m) | 4,925 | **86.58%** |
| **Overall (0–100 m)** | **122,480** | **94.98%** |

### 2.2 RELLIS-3D (cross-domain — off-road, different LiDAR: Ouster OS1-64 vs. training-domain Velodyne HDL-64E)

Sequence `00000`, frame `000000`:

| Bucket | Points evaluated | Accuracy, no adaptation | Accuracy, with sensor adaptation* |
|---|---:|---:|---:|
| Near (0–15 m) | 114,780 | 4.99% | **18.71%** |
| Mid (15–35 m) | 16,035 | 18.38% | **47.72%** |
| Far (35–100 m) | 255 | 12.94% | **17.65%** |
| **Overall (0–100 m)** | **131,070** | **6.65%** | **22.26%** |

\* Sensor adaptation = real Ouster OS1-64 vertical FOV override
(`fov_up=17.02°, fov_down=-16.44°`, vs. the checkpoint's assumed Velodyne
HDL-64E FOV of `-25°..+3°`) plus intensity rescaling to `[0,1]` (RELLIS-3D
intensities are ~0.0001–0.01, the training domain's are ~0–1). Both real,
documented, measured adaptations — not a guess. Even after adaptation,
22.26% overall is a real, honest domain-transfer failure, not a
usable production number — see §6.

## 3. Grid Engine Efficiency (Phase B)

Computed live via
`src/07_adaptive_2point5d_map.py::compute_memory_reduction_metrics()`
against the real `SPEC_ZONES_100M` config (Near 0–15m/5cm, Middle
15–35m/20cm, Far 35–100m/50cm) — a fixed architectural property of the
current zone config, not a per-frame measurement:

| Metric | Value |
|---|---:|
| Finest resolution | 0.05 m |
| Uniform 3D voxel grid (same finest resolution, 8 m height) | 2,560,000,000 voxels / 2,441.4 MB |
| Uniform 2.5D grid (same finest resolution) | 16,000,000 cells / 244.1 MB |
| Adaptive foveated 2.5D grid (actual FOVEAX design) | 471,554 cells / 7.2 MB |
| **Cell reduction vs. uniform 3D voxels** | **99.98%** |
| **Byte reduction vs. uniform 3D voxels** | **99.71%** |
| Cell reduction vs. uniform 2.5D grid | 97.05% |

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

Real headless replay, RELLIS-3D sequence `00001`, 30 real frames
(`src/13_realtime_dashboard.py --headless`):

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
2. **RELLIS-3D cross-domain semantic accuracy is low (22.26% overall,
   even after sensor adaptation).** This is a real, unresolved
   domain-transfer failure, not a production-ready result. Root-caused
   to real sensor differences (Ouster OS1-64 vs. Velodyne HDL-64E FOV
   and intensity scale) — full resolution would need fine-tuning on
   RELLIS-3D or an equivalent off-road training set.
3. **Dashboard FPS (13.8, real) is below typical 25–30 FPS real-time
   targets.** Two rounds of profiling-driven vectorization improved this
   ~3.3x from an original 4.2 FPS; the next real bottleneck (identified
   via profiling, not fixed here) is `scipy.sparse.csgraph
   .connected_components`'s internal CSR conversion overhead
   (~14 ms/frame).
4. **`MockObjectDetector` is a deterministic geometric baseline, not a
   trained AI detector** — it clusters points above a ground-height
   threshold; it does not perform learned object recognition. Real
   semantic classification (VEGETATION/VEHICLE/PEDESTRIAN/etc.) only
   applies when real ground-truth or SalsaNext labels are cross-
   referenced against a cluster.

## 7. Test Suite Status

Fresh run, same commit as this document:

```
341 passed, 33 warnings in 2.61s
```

No skipped tests, no failures, no regressions against any prior baseline
recorded during this project's development.
