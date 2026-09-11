"""Unit tests for src/10b_eval_distance_metrics.py's compute_metrics().

This is the surviving distance-bucketed evaluation tool (consolidated from
a duplicate, src/perception/distance_accuracy_eval.py, which was deleted --
see docs/rellis3d_integration.md for why). Covers bucket accuracy, the
GT-UNKNOWN(7) exclusion policy, and confusion-matrix contents.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

eval_module = importlib.import_module("src.10b_eval_distance_metrics")
compute_metrics = eval_module.compute_metrics


class TestComputeMetrics:
    def test_basic_accuracy(self) -> None:
        gt = np.array([0, 1, 2, 0], dtype=np.uint8)
        pred = np.array([0, 0, 2, 0], dtype=np.uint8)  # index 1 wrong
        mask = np.ones(4, dtype=bool)

        result = compute_metrics(gt, pred, mask, "test-bucket")

        assert result["n_points"] == 4
        assert result["accuracy"] == pytest.approx(0.75)
        assert result["bucket_name"] == "test-bucket"

    def test_unknown_ground_truth_excluded_from_scoring(self) -> None:
        """GT==7 (UNKNOWN) must not count as a scored point either way.

        3 real points (all correct) + 2 UNKNOWN-labeled points where the
        model "disagrees" -- if UNKNOWN were counted, accuracy would drop
        to 60%; it must stay 100% since those 2 points have no real label
        to score against.
        """
        gt = np.array([0, 0, 0, 7, 7], dtype=np.uint8)
        pred = np.array([0, 0, 0, 3, 5], dtype=np.uint8)
        mask = np.ones(5, dtype=bool)

        result = compute_metrics(gt, pred, mask, "test-bucket")

        assert result["n_points"] == 3
        assert result["accuracy"] == pytest.approx(1.0)

    def test_mask_restricts_to_bucket(self) -> None:
        gt = np.array([0, 1, 2, 3], dtype=np.uint8)
        pred = np.array([0, 1, 2, 3], dtype=np.uint8)
        mask = np.array([True, True, False, False])  # only first 2 points

        result = compute_metrics(gt, pred, mask, "test-bucket")

        assert result["n_points"] == 2

    def test_all_unknown_returns_none(self) -> None:
        gt = np.array([7, 7], dtype=np.uint8)
        pred = np.array([0, 1], dtype=np.uint8)
        mask = np.ones(2, dtype=bool)

        result = compute_metrics(gt, pred, mask, "empty-bucket")

        assert result is None

    def test_empty_mask_returns_none(self) -> None:
        gt = np.array([0, 1], dtype=np.uint8)
        pred = np.array([0, 1], dtype=np.uint8)
        mask = np.zeros(2, dtype=bool)

        result = compute_metrics(gt, pred, mask, "empty-bucket")

        assert result is None

    def test_confusion_matrix_contents(self) -> None:
        # 3 ground-truth DRIVABLE_GROUND(0) points: 2 correct, 1 predicted VEHICLE(5).
        gt = np.array([0, 0, 0], dtype=np.uint8)
        pred = np.array([0, 0, 5], dtype=np.uint8)
        mask = np.ones(3, dtype=bool)

        result = compute_metrics(gt, pred, mask, "test-bucket")

        assert result["confusion"][0] == {0: 2, 5: 1}

    def test_confusion_matrix_excludes_unknown_gt_class(self) -> None:
        gt = np.array([7], dtype=np.uint8)
        pred = np.array([0], dtype=np.uint8)
        mask = np.ones(1, dtype=bool)

        # Mix in one real point so the bucket isn't empty.
        gt = np.array([7, 0], dtype=np.uint8)
        pred = np.array([0, 0], dtype=np.uint8)
        mask = np.ones(2, dtype=bool)

        result = compute_metrics(gt, pred, mask, "test-bucket")

        assert 7 not in result["confusion"]
        assert result["n_points"] == 1
