"""Unit tests for distance-bucketed accuracy evaluation.

Tests:
    - Distance bucket assignment (Near: 0-15m, Mid: 15-35m, Far: 35-100m)
    - Agreement calculation per zone
    - Class recall and prediction distribution
    - Report text formatting
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.perception.distance_accuracy_eval import (
    compute_distance_metrics,
    format_report_text,
    EvaluationReport,
    DEFAULT_DISTANCE_ZONES,
)


class TestDistanceAccuracyEval:
    """Test suite for distance-bucketed accuracy metrics."""

    def test_bucket_assignment_and_accuracy(self):
        # 3 points:
        # p0 at (5, 0) -> dist = 5m (Near)
        # p1 at (20, 0) -> dist = 20m (Middle)
        # p2 at (50, 0) -> dist = 50m (Far)
        points = np.array([
            [5.0, 0.0, 0.0, 0.5],
            [20.0, 0.0, 0.0, 0.5],
            [50.0, 0.0, 0.0, 0.5],
        ], dtype=np.float32)

        # Ground truth: 0 (drivable), 1 (rough), 2 (vegetation)
        gt = np.array([0, 1, 2], dtype=np.uint8)

        # Preds: 0 (correct), 0 (wrong, predicted drivable instead of rough), 2 (correct)
        preds = np.array([0, 0, 2], dtype=np.uint8)

        report = compute_distance_metrics(points, preds, gt)

        assert report.total_points == 3
        assert report.total_agreement == 2
        assert pytest.approx(report.overall_accuracy_pct, 0.01) == 66.67

        near_z = report.zone_metrics[0]
        assert near_z.zone_name.startswith("Near")
        assert near_z.point_count == 1
        assert near_z.agreement_count == 1
        assert near_z.overall_accuracy_pct == 100.0
        assert near_z.class_accuracies[0] == 100.0

        mid_z = report.zone_metrics[1]
        assert mid_z.zone_name.startswith("Middle")
        assert mid_z.point_count == 1
        assert mid_z.agreement_count == 0
        assert mid_z.overall_accuracy_pct == 0.0
        assert mid_z.class_accuracies[1] == 0.0
        assert mid_z.pred_distribution[0] == 100.0  # Predicted drivable

        far_z = report.zone_metrics[2]
        assert far_z.zone_name.startswith("Far")
        assert far_z.point_count == 1
        assert far_z.agreement_count == 1
        assert far_z.overall_accuracy_pct == 100.0
        assert far_z.class_accuracies[2] == 100.0

    def test_empty_zone_handling(self):
        # All points in near zone
        points = np.array([
            [2.0, 2.0, 0.0],
            [4.0, 3.0, 0.0],
        ], dtype=np.float32)
        gt = np.array([0, 0], dtype=np.uint8)
        preds = np.array([0, 0], dtype=np.uint8)

        report = compute_distance_metrics(points, preds, gt)
        assert report.zone_metrics[0].point_count == 2
        assert report.zone_metrics[1].point_count == 0  # Mid empty
        assert report.zone_metrics[2].point_count == 0  # Far empty

    def test_mismatched_lengths_raise(self):
        points = np.zeros((5, 3), dtype=np.float32)
        preds = np.zeros(4, dtype=np.uint8)
        gt = np.zeros(5, dtype=np.uint8)

        with pytest.raises(ValueError, match="Array length mismatch"):
            compute_distance_metrics(points, preds, gt)

    def test_format_report_text(self):
        points = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
        gt = np.array([0], dtype=np.uint8)
        preds = np.array([0], dtype=np.uint8)

        report = compute_distance_metrics(points, preds, gt)
        report.dataset_name = "TestDataset"
        report.frame_id = "000000"
        report.adaptation_name = "TestAdaptation"

        text = format_report_text([report])
        assert "TestDataset" in text
        assert "TestAdaptation" in text
        assert "Overall Agreement: 100.00%" in text
