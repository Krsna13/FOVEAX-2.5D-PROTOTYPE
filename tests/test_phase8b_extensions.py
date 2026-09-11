"""Unit tests for Phase 8B extensions (Tasks I through N)."""

from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np
import pytest

from src.perception.openpcdet_predictor import (
    validate_point_cloud_range_and_intensity,
    compute_box3d_iou,
    check_duplicate_boxes_3d,
    map_openpcdet_class_to_foveax,
)
from src.perception.object_detector import Detection3D
from src.perception.grid_overlay import (
    compute_footprint_mask,
    create_frame_base_grids,
    generate_dynamic_object_overlay,
)
import importlib
_tracking_module = importlib.import_module("src.11_object_detection_tracking")
log_runtime_benchmark_to_report = _tracking_module.log_runtime_benchmark_to_report



# ============================================================================
# Task I: Range & Intensity Validation Tests
# ============================================================================

def test_range_validation_warns_on_out_of_bounds():
    # Point cloud range: [0, -40, -3, 70, 40, 1]
    pcr = [0.0, -40.0, -3.0, 70.0, 40.0, 1.0]
    
    # Points with x = -5 (below 0) and x = 80 (above 70)
    pts = np.array([
        [-5.0, 0.0, 0.0, 0.5],
        [80.0, 0.0, 0.0, 0.5],
    ], dtype=np.float32)

    with pytest.warns(UserWarning, match="Point cloud spatial bounds exceed PointPillars expected range"):
        range_valid, intensity_valid = validate_point_cloud_range_and_intensity(pts, pcr)

    assert not range_valid
    assert intensity_valid  # intensity was 0.5, valid


def test_range_validation_silent_when_within_bounds():
    pcr = [0.0, -40.0, -3.0, 70.0, 40.0, 1.0]
    pts = np.array([
        [10.0, -5.0, -1.0, 0.5],
        [50.0, 15.0, 0.2, 0.8],
    ], dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        range_valid, intensity_valid = validate_point_cloud_range_and_intensity(pts, pcr)

    assert range_valid
    assert intensity_valid


def test_intensity_validation_warns_on_unnormalized():
    pcr = [0.0, -40.0, -3.0, 70.0, 40.0, 1.0]
    # Points with 0-255 un-normalized intensity
    pts = np.array([
        [10.0, 0.0, 0.0, 150.0],
        [20.0, 0.0, 0.0, 255.0],
    ], dtype=np.float32)

    with pytest.warns(UserWarning, match="Point cloud intensity range"):
        range_valid, intensity_valid = validate_point_cloud_range_and_intensity(pts, pcr)

    assert range_valid
    assert not intensity_valid


# ============================================================================
# Task M: Class Taxonomy Reconciliation Tests
# ============================================================================

def test_taxonomy_mapping():
    # Car -> VEHICLE (5)
    name, cid = map_openpcdet_class_to_foveax("Car")
    assert name == "VEHICLE"
    assert cid == 5

    # Pedestrian -> PEDESTRIAN (6)
    name, cid = map_openpcdet_class_to_foveax("Pedestrian")
    assert name == "PEDESTRIAN"
    assert cid == 6

    # Cyclist -> VEHICLE (5)
    name, cid = map_openpcdet_class_to_foveax("Cyclist")
    assert name == "VEHICLE"
    assert cid == 5

    # Unmapped classes -> SOLID_OBSTACLE (4)
    name, cid = map_openpcdet_class_to_foveax("Cone")
    assert name == "SOLID_OBSTACLE"
    assert cid == 4

    name, cid = map_openpcdet_class_to_foveax("unknown_box")
    assert name == "SOLID_OBSTACLE"
    assert cid == 4


# ============================================================================
# Task N: 3D IoU and Duplicate Box Sanity Check Tests
# ============================================================================

def test_3d_iou_identical_boxes():
    b1 = Detection3D(
        center_xyz=np.array([10.0, 5.0, 0.0], dtype=np.float32),
        size_lwh=np.array([4.0, 2.0, 1.5], dtype=np.float32),
        yaw_rad=0.0,
        class_id=5,
        class_name="VEHICLE",
        confidence=0.9,
        source="test",
    )
    b2 = Detection3D(
        center_xyz=np.array([10.0, 5.0, 0.0], dtype=np.float32),
        size_lwh=np.array([4.0, 2.0, 1.5], dtype=np.float32),
        yaw_rad=0.0,
        class_id=5,
        class_name="VEHICLE",
        confidence=0.85,
        source="test",
    )
    iou = compute_box3d_iou(b1, b2)
    assert abs(iou - 1.0) < 1e-4


def test_3d_iou_disjoint_boxes():
    b1 = Detection3D(
        center_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        size_lwh=np.array([2.0, 2.0, 1.0], dtype=np.float32),
        yaw_rad=0.0,
        class_id=5,
        class_name="VEHICLE",
        confidence=0.9,
        source="test",
    )
    b2 = Detection3D(
        center_xyz=np.array([10.0, 10.0, 0.0], dtype=np.float32),
        size_lwh=np.array([2.0, 2.0, 1.0], dtype=np.float32),
        yaw_rad=0.0,
        class_id=5,
        class_name="VEHICLE",
        confidence=0.8,
        source="test",
    )
    iou = compute_box3d_iou(b1, b2)
    assert iou == 0.0


def test_duplicate_box_check_flags_high_overlap():
    b1 = Detection3D(
        center_xyz=np.array([10.0, 0.0, 0.0], dtype=np.float32),
        size_lwh=np.array([4.0, 2.0, 1.5], dtype=np.float32),
        yaw_rad=0.0,
        class_id=5,
        class_name="VEHICLE",
        confidence=0.9,
        source="test",
    )
    # Slight shift, ~85% overlap
    b2 = Detection3D(
        center_xyz=np.array([10.2, 0.1, 0.0], dtype=np.float32),
        size_lwh=np.array([4.0, 2.0, 1.5], dtype=np.float32),
        yaw_rad=0.0,
        class_id=5,
        class_name="VEHICLE",
        confidence=0.85,
        source="test",
    )
    b3 = Detection3D(
        center_xyz=np.array([30.0, 0.0, 0.0], dtype=np.float32),
        size_lwh=np.array([4.0, 2.0, 1.5], dtype=np.float32),
        yaw_rad=0.0,
        class_id=5,
        class_name="VEHICLE",
        confidence=0.7,
        source="test",
    )
    duplicates = check_duplicate_boxes_3d([b1, b2, b3], iou_threshold=0.70)
    assert len(duplicates) == 1
    assert duplicates[0][0] == 0
    assert duplicates[0][1] == 1
    assert duplicates[0][2] >= 0.70


# ============================================================================
# Task J: ROI & Traversability Feedback Loop Tests
# ============================================================================

def test_compute_footprint_mask():
    # Grid centered at origin, 20x20 cells with 0.5m resolution: span [-5, 5]
    xs = np.linspace(-5, 5, 21)
    ys = np.linspace(-5, 5, 21)
    gx, gy = np.meshgrid(xs, ys)

    box = Detection3D(
        center_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        size_lwh=np.array([2.0, 2.0, 1.0], dtype=np.float32),
        yaw_rad=0.0,
        class_id=5,
        class_name="VEHICLE",
        confidence=0.9,
        source="test",
    )
    mask = compute_footprint_mask(gx, gy, [box])
    
    # Origin cell must be covered
    assert mask[10, 10]
    # Far cell must not be covered
    assert not mask[0, 0]
    assert np.sum(mask) > 0


def test_generate_dynamic_object_overlay(tmp_path):
    # Dummy point cloud
    pts = np.random.uniform(-5, 5, (100, 4)).astype(np.float32)
    pts[:, 3] = 0.5

    box = Detection3D(
        center_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        size_lwh=np.array([2.0, 2.0, 1.0], dtype=np.float32),
        yaw_rad=0.0,
        class_id=5,
        class_name="VEHICLE",
        confidence=0.9,
        source="test",
    )

    overlay_dir = tmp_path / "dynamic_overlay"
    npz_path = generate_dynamic_object_overlay(
        points_xyzi=pts,
        dynamic_objects=[box],
        output_dir=overlay_dir,
        frame_idx=0,
        resolution_m=0.2,
    )

    assert npz_path.exists()
    assert (overlay_dir / "frame_0000_overlay.png").exists()

    data = np.load(npz_path)
    assert "dynamic_footprint_mask" in data
    assert "traversability_overlay" in data
    assert "importance_score_overlay" in data

    mask = data["dynamic_footprint_mask"]
    trav_overlay = data["traversability_overlay"]
    imp_overlay = data["importance_score_overlay"]

    if np.any(mask):
        # Cells covered by dynamic objects must have traversability 0.0 and importance 1.0
        assert np.all(trav_overlay[mask] == 0.0)
        assert np.all(imp_overlay[mask] == 1.0)


# ============================================================================
# Task K: Runtime Benchmark Logger Tests
# ============================================================================

def test_log_runtime_benchmark(tmp_path):
    report_file = tmp_path / "model_report.txt"
    report_file.write_text("=== Header ===\nInitial content\n", encoding="utf-8")

    times = [0.010, 0.020, 0.030]  # 10ms, 20ms, 30ms -> mean 20ms = 50 FPS
    text = log_runtime_benchmark_to_report(
        detector_name="TestDetector",
        device_name="TestGPU",
        detection_times=times,
        report_path=report_file,
    )

    assert "mean=20.00 ms" in text
    assert "50.00 FPS" in text
    assert "TestGPU" in text
    assert "Measured benchmark numbers only" in text

    saved = report_file.read_text(encoding="utf-8")
    assert "=== Runtime Benchmark Measurements ===" in saved
    assert "Initial content" in saved


def test_log_runtime_benchmark_append_history(tmp_path):
    """Confirm that calling the benchmark logger multiple times preserves history."""
    report_file = tmp_path / "model_report.txt"
    report_file.write_text("=== Environment Inspection ===\nPython: 3.11\n", encoding="utf-8")

    # Run 1
    log_runtime_benchmark_to_report(
        detector_name="DetectorRun1",
        device_name="CPU",
        detection_times=[0.010],
        report_path=report_file,
    )

    # Run 2
    log_runtime_benchmark_to_report(
        detector_name="DetectorRun2",
        device_name="CUDA",
        detection_times=[0.005],
        report_path=report_file,
    )

    saved = report_file.read_text(encoding="utf-8")
    assert "DetectorRun1" in saved
    assert "DetectorRun2" in saved
    assert "Run timestamp:" in saved
    assert saved.count("=== Runtime Benchmark Measurements ===") == 2
    assert "=== Environment Inspection ===" in saved
