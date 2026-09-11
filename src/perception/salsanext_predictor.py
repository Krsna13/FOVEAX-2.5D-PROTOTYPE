"""FOVEAX adapter for the official SalsaNext LiDAR segmentation model.

This module provides the SalsaNextPredictor class, which implements the
SemanticPredictor interface to allow FOVEAX to use the official SalsaNext
pretrained model without duplicating its code.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

from src.perception.semantic_labels import (
    NUM_FOVEAX_CLASSES,
    SEMANTICKITTI_TO_FOVEAX,
)
from src.perception.range_projection import (
    project_points_to_range_image,
    reproject_labels_to_points,
)
from src.perception.semantic_predictor import (
    SemanticPrediction,
    SemanticPredictor,
    validate_points_xyzi,
    validate_prediction,
)

# SalsaNext's own training class 0 is "unlabeled"; learning_map_inv maps it
# back to SemanticKITTI 0, which SEMANTICKITTI_TO_FOVEAX maps to FOVEAX
# UNKNOWN. Used for points that never reached a pixel.
_UNKNOWN_TRAIN_ID = 0
_FOVEAX_UNKNOWN = 7

# Optional PyTorch import. We don't fail immediately to allow the test suite
# to run without PyTorch installed.
try:
    import torch
    import yaml
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SalsaNextPredictor(SemanticPredictor):
    """Adapter for the official SalsaNext model.

    This class loads the official SalsaNext architecture from an external
    repository directory and runs inference on raw SemanticKITTI point clouds.
    """

    def __init__(
        self,
        repo_path: Path,
        checkpoint_path: Path,
        config_path: Path,
        device: str = "auto",
        learning_map_path: Path | None = None,
        strict: bool = True,
        fov_up: float | None = None,
        fov_down: float | None = None,
        rescale_intensity: bool = False,
    ):
        """Initialise the SalsaNext adapter.

        Args:
            repo_path: Path to the cloned official SalsaNext repository.
            checkpoint_path: Path to the official pretrained checkpoint.
            config_path: Path to the official architecture config YAML.
            device: 'cuda', 'cpu', or 'auto'.
            learning_map_path: Optional path to semantic-kitti.yaml for class mappings.
            strict: If True, raise FileNotFoundError immediately if files are missing.
            fov_up: Optional override for vertical FOV up in degrees (e.g. 17.02 for Ouster OS1-64).
            fov_down: Optional override for vertical FOV down in degrees (e.g. -16.44 for Ouster OS1-64).
            rescale_intensity: If True, min-max normalize non-zero point intensities to [0, 1].
        """
        self.repo_path = Path(repo_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.learning_map_path = (
            Path(learning_map_path) if learning_map_path else None
        )
        self.fov_up = fov_up
        self.fov_down = fov_down
        self.rescale_intensity = rescale_intensity

        if strict:
            self._verify_dependencies()

        # Resolve device
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                if TORCH_AVAILABLE:
                    warnings.warn("CUDA unavailable. SalsaNext inference will run on CPU and may be slow.")
        else:
            self.device = torch.device(device)

        self.model = None

    def _verify_dependencies(self) -> None:
        """Verify that all required files and dependencies exist."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. SalsaNext requires torch, torchvision, and pyyaml."
            )

        if not self.repo_path.is_dir():
            raise FileNotFoundError(
                f"SalsaNext repository not found at {self.repo_path}. "
                "Please clone it as instructed in docs/salsanext_setup.md."
            )

        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"SalsaNext config not found at {self.config_path}."
            )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SalsaNext checkpoint not found at {self.checkpoint_path}. "
                "Please download the official pretrained weights."
            )

    def _load_model(self) -> None:
        """Dynamically load the SalsaNext model from the external repo."""
        if self.model is not None:
            return

        self._verify_dependencies()

        # Add the repository to sys.path so we can import its modules
        repo_str = str(self.repo_path.resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        # SalsaNext.py itself does `import __init__ as booger` -- a bare,
        # non-relative import of the literal module name "__init__", which
        # only resolves if train/tasks/semantic/modules/ is *directly* on
        # sys.path (the official eval.sh gets this for free by `cd`-ing into
        # train/tasks/semantic/ before running infer.py, since Python then
        # auto-adds that directory to sys.path[0]). Verified minimal: only
        # this one directory is needed -- SalsaNext.py imports nothing else
        # from the repo besides this and stdlib/torch.
        modules_str = str(
            (self.repo_path / "train" / "tasks" / "semantic" / "modules").resolve()
        )
        if modules_str not in sys.path:
            sys.path.insert(0, modules_str)

        try:
            # We must load the architecture configuration first
            with open(self.config_path, "r") as f:
                arch_cfg = yaml.safe_load(f)
            # Kept on the instance: predict() needs the sensor block for the
            # spherical projection (fov, img_prop, img_means, img_stds).
            self._arch_cfg = arch_cfg

            # num_classes is NOT present anywhere in arch_cfg.yaml -- verified
            # directly against a real downloaded checkpoint's arch_cfg.yaml
            # (top-level keys: train/post/dataset, with dataset.sensor holding
            # only spherical-projection params like fov/img_prop/img_means).
            # The official SalsaNext code (train/tasks/semantic/dataset/kitti/
            # parser.py:97) derives it instead from data_cfg.yaml, which ships
            # alongside arch_cfg.yaml in the same downloaded model directory:
            #     self.nclasses = len(self.learning_map_inv)
            data_cfg_path = self.config_path.parent / "data_cfg.yaml"
            if not data_cfg_path.is_file():
                raise FileNotFoundError(
                    f"SalsaNext data_cfg.yaml not found at {data_cfg_path} "
                    "(expected as a sibling of config_path -- this is where "
                    "the official pretrained-model download places it "
                    "alongside arch_cfg.yaml)."
                )
            with open(data_cfg_path, "r") as f:
                data_cfg = yaml.safe_load(f)
            num_classes = len(data_cfg["learning_map_inv"])
            # learning_map_inv turns the model's 0..19 training classes back
            # into real SemanticKITTI IDs (10=car, 30=person, ...), which is
            # what SEMANTICKITTI_TO_FOVEAX is keyed on.
            self._learning_map_inv = data_cfg["learning_map_inv"]

            # Import the SalsaNext network definition. Verified directly
            # against the real cloned repository: the class lives in
            # train/tasks/semantic/modules/SalsaNext.py, not "SalsaNextAnet.py"
            # (that module does not exist anywhere in the official repo).
            from train.tasks.semantic.modules.SalsaNext import SalsaNext

            self.model = SalsaNext(num_classes)
            
            # Load weights
            # The checkpoint might be a file or a directory depending on how it was saved.
            # Often it's a directory containing 'SalsaNext' state dict.
            #
            # weights_only=False is required for this legacy (pre-2.6-default) SalsaNext
            # checkpoint, which contains pickled numpy scalar types not in PyTorch's
            # default safe-unpickling allowlist. This re-enables arbitrary-code-execution
            # risk inherent to unpickling; acceptable here because the checkpoint's
            # provenance is the paper authors' own official GitHub-linked Google Drive
            # download (docs/salsanext_setup.md), not an untrusted third-party source.
            # Do not apply this to any other checkpoint without the same provenance
            # verification.
            checkpoint = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=False
            )
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            # Checkpoint was saved from an nn.DataParallel-wrapped model, so every key
            # carries a "module." prefix the bare SalsaNext model doesn't expect.
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)

            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            raise RuntimeError(
                f"Failed to load SalsaNext model. Ensure the repository structure matches expectations. "
                f"Underlying error: {e}"
            ) from e

    def predict(
        self,
        points: np.ndarray,
        fov_up: float | None = None,
        fov_down: float | None = None,
        rescale_intensity: bool | None = None,
    ) -> SemanticPrediction:
        """Predict semantic classes for the given points using SalsaNext.

        Args:
            points: Array of shape (N, 4) containing [x, y, z, intensity].
            fov_up: Optional override for vertical FOV up in degrees.
            fov_down: Optional override for vertical FOV down in degrees.
            rescale_intensity: Optional override to enable/disable intensity rescaling.

        Returns:
            SemanticPrediction containing class IDs, confidence, and uncertainty.
        """
        validate_points_xyzi(points)
        num_points = points.shape[0]

        if not TORCH_AVAILABLE or self.model is None:
            # If not initialized, try to load. If it fails due to missing files
            # (which we allow in non-strict mode until predict is called), we raise.
            self._load_model()

        eff_fov_up = fov_up if fov_up is not None else self.fov_up
        eff_fov_down = fov_down if fov_down is not None else self.fov_down
        eff_rescale = (
            rescale_intensity if rescale_intensity is not None else self.rescale_intensity
        )

        points_to_project = points.copy()
        if eff_rescale:
            # Rescale intensity on valid (non-zero range) points to [0, 1]
            depth = np.linalg.norm(points_to_project[:, :3], axis=1)
            valid_idx = np.flatnonzero(depth > 0.0)
            if valid_idx.size > 0:
                intensities = points_to_project[valid_idx, 3]
                i_min = float(intensities.min())
                i_max = float(intensities.max())
                if i_max > i_min:
                    points_to_project[valid_idx, 3] = (intensities - i_min) / (i_max - i_min)
                else:
                    points_to_project[valid_idx, 3] = 0.0

        # --- 1. Spherical range-image projection (upstream-faithful) ---
        projection = project_points_to_range_image(
            points_to_project, self._arch_cfg, fov_up=eff_fov_up, fov_down=eff_fov_down
        )

        # --- 2. Inference: (5,H,W) -> (1,5,H,W) -> logits (1,C,H,W) ---
        with torch.no_grad():
            proj_in = torch.from_numpy(projection.image).unsqueeze(0).to(self.device)
            logits = self.model(proj_in)
            probs = torch.softmax(logits, dim=1)

            pixel_train_ids = probs.argmax(dim=1)[0]
            pixel_confidence = probs.max(dim=1).values[0]

            # Normalized entropy in [0,1] as the uncertainty measure, so it
            # stays comparable with the other predictors' uncertainty field.
            eps = 1e-10
            entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)[0]
            pixel_uncertainty = entropy / float(np.log(probs.shape[1]))

            pixel_train_ids = pixel_train_ids.cpu().numpy()
            pixel_confidence = pixel_confidence.cpu().numpy()
            pixel_uncertainty = pixel_uncertainty.cpu().numpy()

        # --- 3. Reverse projection: pixels -> original point order ---
        # UNKNOWN_TRAIN_ID is the model's own "unlabeled" training class (0),
        # which learning_map_inv maps back to SemanticKITTI 0 -> FOVEAX
        # UNKNOWN. Points with no pixel therefore end up UNKNOWN without
        # fabricating a class for them.
        point_train_ids = reproject_labels_to_points(
            pixel_train_ids, projection, invalid_label=_UNKNOWN_TRAIN_ID
        )
        point_confidence = reproject_labels_to_points(
            pixel_confidence.astype(np.float32), projection, invalid_label=0.0
        ).astype(np.float32)
        point_uncertainty = reproject_labels_to_points(
            pixel_uncertainty.astype(np.float32), projection, invalid_label=1.0
        ).astype(np.float32)

        # --- 4. Taxonomy: model train IDs -> SemanticKITTI IDs -> FOVEAX ---
        # learning_map_inv undoes the training-time class collapse (e.g.
        # train class 1 -> SemanticKITTI 10 "car"), then the existing,
        # already-tested SEMANTICKITTI_TO_FOVEAX table does the rest. No new
        # mapping table is introduced here.
        max_train_id = max(self._learning_map_inv)
        inv_lut = np.zeros(max_train_id + 1, dtype=np.int32)
        for train_id, kitti_id in self._learning_map_inv.items():
            inv_lut[train_id] = kitti_id
        point_kitti_ids = inv_lut[point_train_ids]

        max_kitti_id = max(SEMANTICKITTI_TO_FOVEAX)
        foveax_lut = np.full(max_kitti_id + 1, _FOVEAX_UNKNOWN, dtype=np.uint8)
        for kitti_id, foveax_id in SEMANTICKITTI_TO_FOVEAX.items():
            foveax_lut[kitti_id] = foveax_id
        class_ids = foveax_lut[point_kitti_ids]

        # Points that resolved to UNKNOWN carry no usable confidence, matching
        # GroundTruthSemanticPredictor's convention for its own UNKNOWN points.
        unknown = class_ids == _FOVEAX_UNKNOWN
        point_confidence[unknown] = 0.0
        point_uncertainty[unknown] = 1.0

        prediction = SemanticPrediction(
            class_ids=class_ids,
            confidence=np.clip(point_confidence, 0.0, 1.0).astype(np.float32),
            uncertainty=np.clip(point_uncertainty, 0.0, 1.0).astype(np.float32),
            source="salsanext_pretrained",
        )
        validate_prediction(prediction, num_points)
        return prediction
