"""FOVEAX Phase 7A — Model-agnostic semantic prediction interface.

Defines an abstract SemanticPredictor and concrete implementations:
    - GroundTruthSemanticPredictor : uses SemanticKITTI .label ground truth
    - MockSemanticPredictor        : geometry-only heuristic (not AI)
    - PretrainedModelPredictor     : placeholder for SalsaNext / RandLA-Net
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.perception.semantic_labels import (
    FOVEAX_CLASSES,
    FOVEAX_COLORS,
    NUM_FOVEAX_CLASSES,
    SEMANTICKITTI_TO_FOVEAX,
)


# ---------------------------------------------------------------------------
# SemanticPrediction dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SemanticPrediction:
    """Per-point semantic prediction returned by any predictor.

    Attributes
    ----------
    class_ids : np.ndarray, shape (N,), dtype uint8
        Predicted FOVEAX class ID for every point.
    confidence : np.ndarray, shape (N,), dtype float32, values in [0, 1]
        How confident the predictor is in the assigned class.
    uncertainty : np.ndarray, shape (N,), dtype float32, values in [0, 1]
        Semantic uncertainty per point (1 - confidence for simple cases).
    source : str
        Human-readable label for the prediction source.
    """

    class_ids: np.ndarray
    confidence: np.ndarray
    uncertainty: np.ndarray
    source: str
    class_confidence: dict[str, float] | None = None


def validate_prediction(prediction: SemanticPrediction, n_points: int) -> None:
    """Validate that a SemanticPrediction is consistent with *n_points*.

    Raises
    ------
    ValueError
        If any array has the wrong length or values are out of range.
    """
    if prediction.class_ids.shape != (n_points,):
        raise ValueError(
            f"class_ids has shape {prediction.class_ids.shape}, "
            f"expected ({n_points},)."
        )
    if prediction.confidence.shape != (n_points,):
        raise ValueError(
            f"confidence has shape {prediction.confidence.shape}, "
            f"expected ({n_points},)."
        )
    if prediction.uncertainty.shape != (n_points,):
        raise ValueError(
            f"uncertainty has shape {prediction.uncertainty.shape}, "
            f"expected ({n_points},)."
        )
    if prediction.confidence.min() < 0.0 or prediction.confidence.max() > 1.0:
        raise ValueError(
            "confidence values must be in [0, 1]; got range "
            f"[{prediction.confidence.min():.4f}, "
            f"{prediction.confidence.max():.4f}]."
        )
    if prediction.uncertainty.min() < 0.0 or prediction.uncertainty.max() > 1.0:
        raise ValueError(
            "uncertainty values must be in [0, 1]; got range "
            f"[{prediction.uncertainty.min():.4f}, "
            f"{prediction.uncertainty.max():.4f}]."
        )


def validate_points_xyzi(points: np.ndarray) -> None:
    """Validate that *points* has shape (N, 4) and is finite.

    Raises
    ------
    ValueError
        If shape is wrong or the array contains NaN/Inf.
    """
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(
            f"points_xyzi must have shape (N, 4), got {points.shape}."
        )
    if points.shape[0] == 0:
        raise ValueError("points_xyzi must contain at least one point.")


# ---------------------------------------------------------------------------
# Abstract predictor
# ---------------------------------------------------------------------------

class SemanticPredictor(ABC):
    """Abstract interface for per-point semantic prediction.

    All concrete predictors must implement :meth:`predict`.
    """

    @abstractmethod
    def predict(self, points_xyzi: np.ndarray) -> SemanticPrediction:
        """Predict a FOVEAX semantic class for every input point.

        Parameters
        ----------
        points_xyzi : np.ndarray, shape (N, 4), dtype float32
            Columns are [x, y, z, intensity].

        Returns
        -------
        SemanticPrediction
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Ground-truth predictor (SemanticKITTI labels)
# ---------------------------------------------------------------------------

def remap_semantickitti_to_foveax(raw_semantic_ids: np.ndarray) -> np.ndarray:
    """Map raw SemanticKITTI class IDs to simplified FOVEAX class IDs.

    Parameters
    ----------
    raw_semantic_ids : np.ndarray, shape (N,)
        Lower-16-bit semantic IDs extracted from SemanticKITTI label file.

    Returns
    -------
    np.ndarray, shape (N,), dtype uint8
        FOVEAX class IDs.
    """
    mapped = np.full(len(raw_semantic_ids), 7, dtype=np.uint8)
    for raw_id, foveax_id in SEMANTICKITTI_TO_FOVEAX.items():
        mapped[raw_semantic_ids == raw_id] = foveax_id
    return mapped


def extract_semantickitti_ids(
    labels_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract semantic and instance IDs from raw SemanticKITTI labels.

    Parameters
    ----------
    labels_raw : np.ndarray, shape (N,), dtype uint32
        Raw 32-bit labels from a .label file.

    Returns
    -------
    semantic_ids : np.ndarray, shape (N,), uint32
        Lower 16 bits — raw SemanticKITTI semantic class.
    instance_ids : np.ndarray, shape (N,), uint32
        Upper 16 bits — instance ID.
    """
    semantic_ids = labels_raw & 0xFFFF
    instance_ids = labels_raw >> 16
    return semantic_ids, instance_ids


class GroundTruthSemanticPredictor(SemanticPredictor):
    """Wrap known-correct ground-truth per-point labels as a predictor.

    Supports two mutually exclusive input forms, selected by which keyword
    is provided:

    Parameters
    ----------
    labels_raw : np.ndarray, shape (N,), dtype uint32, optional
        Raw 32-bit SemanticKITTI-packed label values read from a ``.label``
        file. Remapped internally via SEMANTICKITTI_TO_FOVEAX. This is the
        original SemanticKITTI code path and is unchanged.
    foveax_class_ids : np.ndarray, shape (N,), dtype uint8, optional
        Already-remapped FOVEAX class IDs, e.g. produced by
        ``src/perception/rellis3d_loader.py`` (which does its own
        raw-ID -> FOVEAX remap via RELLIS3D_TO_FOVEAX before returning,
        since RELLIS-3D's raw IDs are not SemanticKITTI IDs and must not be
        run back through ``remap_semantickitti_to_foveax``).

    Exactly one of the two must be given.
    """

    def __init__(
        self,
        labels_raw: np.ndarray | None = None,
        *,
        foveax_class_ids: np.ndarray | None = None,
    ) -> None:
        if (labels_raw is None) == (foveax_class_ids is None):
            raise ValueError(
                "Provide exactly one of labels_raw (raw SemanticKITTI-packed "
                "uint32 labels) or foveax_class_ids (already-remapped FOVEAX "
                "class IDs, e.g. from rellis3d_loader.load_rellis3d_frame)."
            )
        if labels_raw is not None:
            if labels_raw.ndim != 1:
                raise ValueError(
                    f"labels_raw must be 1-D, got shape {labels_raw.shape}."
                )
            self._labels_raw = labels_raw.astype(np.uint32)
            self._foveax_class_ids = None
        else:
            if foveax_class_ids.ndim != 1:
                raise ValueError(
                    f"foveax_class_ids must be 1-D, got shape "
                    f"{foveax_class_ids.shape}."
                )
            self._labels_raw = None
            self._foveax_class_ids = foveax_class_ids.astype(np.uint8)

    def predict(self, points_xyzi: np.ndarray) -> SemanticPrediction:
        """Return ground-truth FOVEAX classes with confidence = 1.0.

        Parameters
        ----------
        points_xyzi : np.ndarray, shape (N, 4)

        Returns
        -------
        SemanticPrediction
        """
        validate_points_xyzi(points_xyzi)
        n = len(points_xyzi)

        if self._foveax_class_ids is not None:
            if len(self._foveax_class_ids) != n:
                raise ValueError(
                    f"Point-label mismatch: {n} points but "
                    f"{len(self._foveax_class_ids)} labels."
                )
            class_ids = self._foveax_class_ids
            source = "rellis3d_ground_truth"
        else:
            if len(self._labels_raw) != n:
                raise ValueError(
                    f"Point-label mismatch: {n} points but "
                    f"{len(self._labels_raw)} labels."
                )
            semantic_ids, _ = extract_semantickitti_ids(self._labels_raw)
            class_ids = remap_semantickitti_to_foveax(semantic_ids)
            source = "semantic_kitti_ground_truth"

        # Known mapped labels get confidence 1.0; UNKNOWN (7) gets 0.0.
        is_known = class_ids != 7
        confidence = np.where(is_known, 1.0, 0.0).astype(np.float32)
        uncertainty = (1.0 - confidence).astype(np.float32)

        prediction = SemanticPrediction(
            class_ids=class_ids,
            confidence=confidence,
            uncertainty=uncertainty,
            source=source,
        )
        validate_prediction(prediction, n)
        return prediction


# ---------------------------------------------------------------------------
# Mock predictor (geometry-only heuristic — NOT AI)
# ---------------------------------------------------------------------------

class MockSemanticPredictor(SemanticPredictor):
    """Geometry-only heuristic predictor for pipeline testing.

    ⚠️  WARNING: This is NOT an AI model and must NOT be used for
    safety-critical decisions. It assigns FOVEAX classes using only
    height thresholds and intensity. It exists solely to validate
    the Phase 7 pipeline before a real pretrained model is integrated.
    """

    _SOURCE = "mock_geometry_baseline"

    def predict(self, points_xyzi: np.ndarray) -> SemanticPrediction:
        """Predict FOVEAX classes using simple height / intensity rules.

        Heuristic rules (deterministic, no randomness):
            z < -1.5  OR  non-finite  → UNKNOWN  (7)
            z < 0.25                  → DRIVABLE_GROUND (0)
            0.25 ≤ z < 1.8           → SOLID_OBSTACLE  (4)
            z ≥ 1.8                  → BUILDING_WALL   (3)

        Confidence is a deterministic function of z and intensity.

        ⚠️  This is NOT AI inference. Do not cite these results as
        machine-learning predictions.

        Parameters
        ----------
        points_xyzi : np.ndarray, shape (N, 4)

        Returns
        -------
        SemanticPrediction
        """
        validate_points_xyzi(points_xyzi)
        warnings.warn(
            "MockSemanticPredictor is a geometry heuristic, NOT an AI model. "
            "Do not use for safety claims.",
            UserWarning,
            stacklevel=2,
        )

        n = len(points_xyzi)
        z = points_xyzi[:, 2]
        intensity = points_xyzi[:, 3]

        class_ids = np.full(n, 7, dtype=np.uint8)       # default UNKNOWN
        confidence = np.zeros(n, dtype=np.float32)

        # Normalise intensity to [0, 1] for confidence modulation.
        i_min, i_max = np.nanmin(intensity), np.nanmax(intensity)
        if i_max > i_min:
            intensity_norm = (intensity - i_min) / (i_max - i_min)
        else:
            intensity_norm = np.full(n, 0.5, dtype=np.float32)

        # Rule 1: invalid / very low points → UNKNOWN
        invalid = ~np.isfinite(z) | (z < -1.5)
        class_ids[invalid] = 7
        confidence[invalid] = 0.0

        # Rule 2: low points → DRIVABLE_GROUND
        ground = (~invalid) & (z < 0.25)
        class_ids[ground] = 0
        # Confidence higher near z=0 (expected ground), modulated by intensity.
        ground_z = np.abs(z[ground])
        confidence[ground] = np.clip(
            0.5 + 0.3 * (1.0 - ground_z / 0.25) + 0.2 * intensity_norm[ground],
            0.0,
            1.0,
        ).astype(np.float32)

        # Rule 3: medium height → SOLID_OBSTACLE
        obstacle = (~invalid) & (z >= 0.25) & (z < 1.8)
        class_ids[obstacle] = 4
        # Confidence higher in the mid-range.
        obs_z = z[obstacle]
        confidence[obstacle] = np.clip(
            0.4 + 0.3 * ((obs_z - 0.25) / 1.55) + 0.2 * intensity_norm[obstacle],
            0.0,
            1.0,
        ).astype(np.float32)

        # Rule 4: tall points → BUILDING_WALL
        building = (~invalid) & (z >= 1.8)
        class_ids[building] = 3
        confidence[building] = np.clip(
            0.6 + 0.2 * intensity_norm[building],
            0.0,
            1.0,
        ).astype(np.float32)

        uncertainty = (1.0 - confidence).astype(np.float32)

        prediction = SemanticPrediction(
            class_ids=class_ids,
            confidence=confidence,
            uncertainty=uncertainty,
            source=self._SOURCE,
        )
        validate_prediction(prediction, n)
        return prediction


# ---------------------------------------------------------------------------
# Placeholder for pretrained model (Phase 7B / 7C)
# ---------------------------------------------------------------------------

class PretrainedModelPredictor(SemanticPredictor):
    """Placeholder for a real pretrained LiDAR segmentation model.

    This class will be completed in Phase 7B/C when SalsaNext or
    RandLA-Net inference is integrated.
    """

    def predict(self, points_xyzi: np.ndarray) -> SemanticPrediction:
        """Not yet implemented — raises ``NotImplementedError``.

        Raises
        ------
        NotImplementedError
            Always.  Connect pretrained model weights here in Phase 7B/C.
        """
        raise NotImplementedError(
            "Connect pretrained SalsaNext or RandLA-Net inference here. "
            "Model weights are intentionally not included in Phase 7A."
        )
