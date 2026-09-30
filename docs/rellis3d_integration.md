# RELLIS-3D Integration (Phase A)

This document covers integration notes for RELLIS-3D as an additional ground-truth/semantic-segmentation dataset, alongside the existing SemanticKITTI path.

**Layout** (default root `C:\dev\data\rellis3d`, fallback `data/rellis3d`): `Rellis-3D/<5-digit sequence 00000-00004>/os1_cloud_node_kitti_bin/<6-digit>.bin` (131,072 points × float32 x/y/z/intensity, Ouster OS1-64 organized scan) and `.../os1_cloud_node_semantickitti_label_id/<6-digit>.label` (uint32, SemanticKITTI-style packed semantic+instance). Loaded by `src/perception/rellis3d_loader.py`, which excludes sky points and remaps the 20 RELLIS-3D classes to FOVEAX via `RELLIS3D_TO_FOVEAX` (`src/perception/semantic_labels.py`).

**Status (2026-09-30):** the domain gap described below has been addressed in two steps — sensor adaptation (22.26%) and fine-tuning on RELLIS-3D (81.53% on held-out sequence `00000`). See [Resolution](#resolution-sensor-adaptation-then-fine-tuning) and `docs/validation_results.md` §2.

## SalsaNext transfers poorly to RELLIS-3D: sensor + intensity domain gap

The SemanticKITTI-pretrained SalsaNext checkpoint produces **unusable** predictions on RELLIS-3D as-is. On seq 00000/frame 000000 it labels **85.8% of points VEHICLE** in an off-road trail scene whose ground truth contains **zero** vehicles (11.1% per-point agreement).

This is a domain-transfer failure, not an implementation bug. The same pipeline, unchanged, run on a real SemanticKITTI frame (seq 00/frame 000000) reaches **93.3% per-point agreement** with a class distribution tracking ground truth closely across all eight FOVEAX classes. Two concrete, measured causes:

1. **Vertical FOV mismatch.** `arch_cfg.yaml` describes a Velodyne HDL-64E (`fov_up: 3`, `fov_down: -25`). RELLIS-3D's Ouster OS1-64 actually spans **-16.44° to +17.02°**. Projecting Ouster data through HDL-64E FOV parameters puts 17.5% of points outside the assumed FOV (clamped into edge rows) and misplaces the rest vertically, so the network sees a structurally wrong range image.
2. **Intensity scale mismatch.** RELLIS-3D intensity ranges **0.00015–0.01169** (mean 0.0036), while the checkpoint's normalization assumes KITTI remission on a ~0–1 scale (`img_means[4]=0.21`, `img_stds[4]=0.16`). After normalization the signal channel is a near-constant ≈ -1.29, carrying no information and sitting far outside the trained distribution.

Anyone continuing this work should treat these as the two things to address before expecting meaningful RELLIS-3D predictions -- e.g. projecting with the Ouster's true FOV, and rescaling/renormalizing intensity -- and should re-validate on KITTI afterward to confirm the pipeline itself hasn't regressed.

## Resolution: sensor adaptation, then fine-tuning

1. **Sensor adaptation** (inference-time only): project with the Ouster's real FOV (`fov_up=17.02`, `fov_down=-16.44`) and rescale intensity to `[0,1]` (`rescale_intensity_to_unit_range`, shared by inference and training so they cannot drift). `src/10b_eval_distance_metrics.py` applies this automatically for `--dataset-type rellis3d` unless `--no-adapt-sensor` is passed. Seq `00000`/frame `000000`: 6.65% → 22.26% overall. Better, but not usable.
2. **Fine-tuning** (`src/14_finetune_salsanext.py`): the official checkpoint's 20-class head is replaced with an 8-class FOVEAX head and trained on RELLIS-3D with the same sensor adaptation. The split is built by `src/training/rellis3d_dataset.py::build_splits`: sequence `00000` is held out entirely as test, and train/val come from `00001`-`00004` with a 15% contiguous val tail and a 10-frame gap. Result: **81.53%** overall on 100 held-out frames. Checkpoints: `models/salsanext/rellis3d_finetuned/` (`best.pt` = epoch 9). Use `taxonomy="foveax"` / `--taxonomy foveax` to load it.
3. **Trade-off:** the fine-tuned model forgets SemanticKITTI (94.98% → 23.97%), so keep both checkpoints and choose by domain.

## Data quality: invalid-return points labeled as real classes

RELLIS-3D's own ground-truth labels assign real semantic classes (e.g. person, bush) to some invalid LiDAR returns (points at exactly x=y=z=0.0, i.e. no-return padding), rather than tagging them void/unlabeled. Verified in seq 00000/frame 000000: 22,642 of 35,589 "person"-labeled points (64%) were invalid-return padding, not real detections. The ground-truth loader passes labels through as-is (correct behavior for a ground-truth path), but this means raw ground-truth class distributions should not be treated as a clean accuracy baseline without first filtering invalid-return points. When comparing SalsaNext predictions against ground truth in Phase C, filter out (x,y,z) == (0,0,0) points from both sides before computing any distribution comparison or accuracy metric, or the comparison will be measuring agreement on padding artifacts, not real terrain/object classification.
