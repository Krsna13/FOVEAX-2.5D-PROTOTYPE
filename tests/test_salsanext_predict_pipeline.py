"""End-to-end tests for SalsaNextPredictor.predict().

Uses real PyTorch on CPU with a tiny stub model standing in for the real
SalsaNext network -- so this needs no GPU, no cloned repo, and no 103MB
checkpoint, while still exercising the genuine projection -> inference ->
reverse-projection -> taxonomy-remap path end to end.

(Kept separate from tests/test_salsanext_predictor.py, which replaces the
module's `torch` with a MagicMock for the whole module and so cannot run
real tensor ops.)

Run with:
    python -m pytest tests/test_salsanext_predict_pipeline.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perception.salsanext_predictor import SalsaNextPredictor
from src.perception.semantic_labels import SEMANTICKITTI_TO_FOVEAX

# Real values from models/salsanext/pretrained/pretrained/arch_cfg.yaml,
# but with a small image so tests stay fast. fov/means/stds are the real ones.
_TEST_ARCH_CFG = {
    "dataset": {
        "sensor": {
            "fov_up": 3,
            "fov_down": -25,
            "img_prop": {"width": 64, "height": 16},
            "img_means": [12.12, 10.88, 0.23, -1.04, 0.21],
            "img_stds": [12.32, 11.47, 6.91, 0.86, 0.16],
        }
    }
}

# The real learning_map_inv from data_cfg.yaml, quoted exactly.
_REAL_LEARNING_MAP_INV = {
    0: 0, 1: 10, 2: 11, 3: 15, 4: 18, 5: 20, 6: 30, 7: 31, 8: 32, 9: 40,
    10: 44, 11: 48, 12: 49, 13: 50, 14: 51, 15: 70, 16: 71, 17: 72, 18: 80,
    19: 81,
}
_NUM_CLASSES = len(_REAL_LEARNING_MAP_INV)


class _ConstantClassModel(nn.Module):
    """Stub network that predicts one fixed class for every pixel."""

    def __init__(self, class_id: int, n_classes: int = _NUM_CLASSES) -> None:
        super().__init__()
        self.class_id = class_id
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        logits = torch.zeros(batch, self.n_classes, height, width)
        logits[:, self.class_id, :, :] = 10.0  # dominant -> argmax + high softmax
        return logits


def _make_predictor(model: nn.Module, tmp_path: Path) -> SalsaNextPredictor:
    """Build a predictor with the model pre-injected (no repo/checkpoint I/O)."""
    repo_path = tmp_path / "SalsaNext"
    repo_path.mkdir()
    checkpoint_path = tmp_path / "ckpt"
    checkpoint_path.touch()
    config_path = tmp_path / "arch_cfg.yaml"
    config_path.touch()

    predictor = SalsaNextPredictor(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device="cpu",
        strict=True,
    )
    # Inject what _load_model() would normally populate, so predict() runs
    # without needing the real repo or checkpoint on disk.
    predictor.model = model.eval()
    predictor._arch_cfg = _TEST_ARCH_CFG
    predictor._learning_map_inv = _REAL_LEARNING_MAP_INV
    return predictor


def _sample_points(n: int = 32) -> np.ndarray:
    rng = np.random.default_rng(42)
    xyz = rng.uniform(-20.0, 20.0, size=(n, 3)).astype(np.float32)
    # Guarantee non-zero range so nothing is filtered as invalid padding.
    xyz[np.linalg.norm(xyz, axis=1) < 1.0] = np.array([5.0, 5.0, 0.0], dtype=np.float32)
    intensity = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    return np.hstack([xyz, intensity]).astype(np.float32)


class TestPredictOutputContract:
    def test_output_shape_and_dtype(self, tmp_path: Path) -> None:
        points = _sample_points(32)
        predictor = _make_predictor(_ConstantClassModel(class_id=9), tmp_path)

        prediction = predictor.predict(points)

        assert prediction.class_ids.shape == (32,)
        assert prediction.class_ids.dtype == np.uint8
        assert prediction.confidence.shape == (32,)
        assert prediction.confidence.dtype == np.float32
        assert prediction.uncertainty.shape == (32,)
        assert prediction.uncertainty.dtype == np.float32

    def test_source_label(self, tmp_path: Path) -> None:
        predictor = _make_predictor(_ConstantClassModel(class_id=9), tmp_path)
        prediction = predictor.predict(_sample_points(16))
        assert prediction.source == "salsanext_pretrained"

    def test_confidence_and_uncertainty_in_unit_range(self, tmp_path: Path) -> None:
        predictor = _make_predictor(_ConstantClassModel(class_id=9), tmp_path)
        prediction = predictor.predict(_sample_points(32))

        assert prediction.confidence.min() >= 0.0
        assert prediction.confidence.max() <= 1.0
        assert prediction.uncertainty.min() >= 0.0
        assert prediction.uncertainty.max() <= 1.0


class TestTaxonomyRemapping:
    @pytest.mark.parametrize(
        "train_id,expected_kitti_id",
        [
            (1, 10),   # car
            (6, 30),   # person
            (9, 40),   # road
            (15, 70),  # vegetation
            (17, 72),  # terrain
        ],
    )
    def test_train_id_maps_through_to_expected_foveax_class(
        self, tmp_path: Path, train_id: int, expected_kitti_id: int
    ) -> None:
        """Model train ID -> SemanticKITTI ID -> FOVEAX class.

        Uses the real learning_map_inv and the existing, already-tested
        SEMANTICKITTI_TO_FOVEAX table (no duplicate mapping introduced).
        """
        points = _sample_points(24)
        predictor = _make_predictor(_ConstantClassModel(class_id=train_id), tmp_path)

        prediction = predictor.predict(points)

        expected_foveax = SEMANTICKITTI_TO_FOVEAX[expected_kitti_id]
        # Every projected point should carry the single predicted class.
        assert set(np.unique(prediction.class_ids).tolist()) == {expected_foveax}

    def test_unlabeled_train_class_becomes_foveax_unknown(self, tmp_path: Path) -> None:
        """Train class 0 -> SemanticKITTI 0 ("unlabeled") -> FOVEAX UNKNOWN."""
        predictor = _make_predictor(_ConstantClassModel(class_id=0), tmp_path)
        prediction = predictor.predict(_sample_points(16))

        assert set(np.unique(prediction.class_ids).tolist()) == {7}
        # UNKNOWN points must carry no usable confidence.
        assert np.all(prediction.confidence == 0.0)
        assert np.all(prediction.uncertainty == 1.0)


class TestInvalidPointHandling:
    def test_zero_range_points_labeled_unknown_not_fabricated(
        self, tmp_path: Path
    ) -> None:
        """(0,0,0) padding must come back UNKNOWN, never a real class."""
        real = _sample_points(8)
        padding = np.zeros((4, 4), dtype=np.float32)
        points = np.vstack([real, padding]).astype(np.float32)

        predictor = _make_predictor(_ConstantClassModel(class_id=9), tmp_path)
        prediction = predictor.predict(points)

        assert prediction.class_ids.shape == (12,)
        # Last 4 are the padding points.
        assert np.all(prediction.class_ids[8:] == 7)
        assert np.all(prediction.confidence[8:] == 0.0)
        # The real points got the predicted (road -> DRIVABLE_GROUND) class.
        assert np.all(prediction.class_ids[:8] == SEMANTICKITTI_TO_FOVEAX[40])


class TestDomainAdaptation:
    def test_predict_with_rescale_intensity(self, tmp_path: Path) -> None:
        """Intensity rescaling normalizes input intensity without altering output contract."""
        points = _sample_points(16)
        # Simulate tiny RELLIS-3D scale
        points[:, 3] = points[:, 3] * 0.01

        predictor = _make_predictor(_ConstantClassModel(class_id=9), tmp_path)
        predictor.rescale_intensity = True

        prediction = predictor.predict(points)
        assert prediction.class_ids.shape == (16,)
        assert np.all(prediction.class_ids == SEMANTICKITTI_TO_FOVEAX[40])

    def test_predict_with_fov_override(self, tmp_path: Path) -> None:
        """FOV override applies custom sensor bounds (e.g. Ouster OS1-64)."""
        points = _sample_points(16)
        predictor = _make_predictor(_ConstantClassModel(class_id=9), tmp_path)
        predictor.fov_up = 17.02
        predictor.fov_down = -16.44

        prediction = predictor.predict(points)
        assert prediction.class_ids.shape == (16,)
        assert np.all(prediction.class_ids == SEMANTICKITTI_TO_FOVEAX[40])

