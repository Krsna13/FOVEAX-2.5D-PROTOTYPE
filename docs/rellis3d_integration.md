# RELLIS-3D Integration (Phase A)

This document covers integration notes for RELLIS-3D as an additional ground-truth/semantic-segmentation dataset, alongside the existing SemanticKITTI path. See `CLAUDE.md` for the real, verified directory layout and file formats (`Rellis-3D/<5-digit sequence>/{os1_cloud_node_kitti_bin,os1_cloud_node_semantickitti_label_id}/`).

## SalsaNext transfers poorly to RELLIS-3D: sensor + intensity domain gap

The SemanticKITTI-pretrained SalsaNext checkpoint produces **unusable** predictions on RELLIS-3D as-is. On seq 00000/frame 000000 it labels **85.8% of points VEHICLE** in an off-road trail scene whose ground truth contains **zero** vehicles (11.1% per-point agreement).

This is a domain-transfer failure, not an implementation bug. The same pipeline, unchanged, run on a real SemanticKITTI frame (seq 00/frame 000000) reaches **93.3% per-point agreement** with a class distribution tracking ground truth closely across all eight FOVEAX classes. Two concrete, measured causes:

1. **Vertical FOV mismatch.** `arch_cfg.yaml` describes a Velodyne HDL-64E (`fov_up: 3`, `fov_down: -25`). RELLIS-3D's Ouster OS1-64 actually spans **-16.44° to +17.02°**. Projecting Ouster data through HDL-64E FOV parameters puts 17.5% of points outside the assumed FOV (clamped into edge rows) and misplaces the rest vertically, so the network sees a structurally wrong range image.
2. **Intensity scale mismatch.** RELLIS-3D intensity ranges **0.00015–0.01169** (mean 0.0036), while the checkpoint's normalization assumes KITTI remission on a ~0–1 scale (`img_means[4]=0.21`, `img_stds[4]=0.16`). After normalization the signal channel is a near-constant ≈ -1.29, carrying no information and sitting far outside the trained distribution.

Anyone continuing this work should treat these as the two things to address before expecting meaningful RELLIS-3D predictions -- e.g. projecting with the Ouster's true FOV, and rescaling/renormalizing intensity -- and should re-validate on KITTI afterward to confirm the pipeline itself hasn't regressed.

## Data quality: invalid-return points labeled as real classes

RELLIS-3D's own ground-truth labels assign real semantic classes (e.g. person, bush) to some invalid LiDAR returns (points at exactly x=y=z=0.0, i.e. no-return padding), rather than tagging them void/unlabeled. Verified in seq 00000/frame 000000: 22,642 of 35,589 "person"-labeled points (64%) were invalid-return padding, not real detections. The ground-truth loader passes labels through as-is (correct behavior for a ground-truth path), but this means raw ground-truth class distributions should not be treated as a clean accuracy baseline without first filtering invalid-return points. When comparing SalsaNext predictions against ground truth in Phase C, filter out (x,y,z) == (0,0,0) points from both sides before computing any distribution comparison or accuracy metric, or the comparison will be measuring agreement on padding artifacts, not real terrain/object classification.
