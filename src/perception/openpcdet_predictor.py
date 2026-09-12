"""FOVEAX Phase 8B — Real AI Object Detection Adapter via OpenPCDet."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np

from .object_detector import Detection3D, ObjectDetector, validate_points_xyzi


# ============================================================================
# Class Taxonomy Mapping (Task M)
# ============================================================================

OPENPCDET_TO_FOVEAX: dict[str, tuple[str, int]] = {
    "car": ("VEHICLE", 5),
    "vehicle": ("VEHICLE", 5),
    "van": ("VEHICLE", 5),
    "truck": ("VEHICLE", 5),
    "bus": ("VEHICLE", 5),
    "pedestrian": ("PEDESTRIAN", 6),
    "person": ("PEDESTRIAN", 6),
    "cyclist": ("VEHICLE", 5),
    "bicycle": ("VEHICLE", 5),
}

DEFAULT_FALLBACK_CLASS: tuple[str, int] = ("SOLID_OBSTACLE", 4)


def map_openpcdet_class_to_foveax(native_class_name: str) -> tuple[str, int]:
    """Map an OpenPCDet native class name to the FOVEAX Phase 6 taxonomy.

    Parameters
    ----------
    native_class_name : str
        Native label name from OpenPCDet (e.g. 'Car', 'Pedestrian', 'Cyclist').

    Returns
    -------
    tuple[str, int]
        (foveax_class_name, foveax_class_id)
    """
    key = native_class_name.strip().lower()
    return OPENPCDET_TO_FOVEAX.get(key, DEFAULT_FALLBACK_CLASS)


# ============================================================================
# Coordinate & Intensity Validation (Task I)
# ============================================================================

def validate_point_cloud_range_and_intensity(
    points_xyzi: np.ndarray,
    point_cloud_range: Sequence[float],
) -> tuple[bool, bool]:
    """Validate LiDAR points spatial coordinates and normalized intensity.

    Emits warnings if points fall outside the expected PointPillars configuration
    bounds or if intensity deviates from the expected [0, 1] float range.
    Points are NEVER silently truncated or modified.

    Parameters
    ----------
    points_xyzi : np.ndarray, shape (N, 4)
        Columns: [x, y, z, intensity].
    point_cloud_range : Sequence[float]
        Expected bounds: [x_min, y_min, z_min, x_max, y_max, z_max].

    Returns
    -------
    tuple[bool, bool]
        (range_valid, intensity_valid)
    """
    if len(points_xyzi) == 0:
        return True, True

    p_min = np.min(points_xyzi, axis=0)
    p_max = np.max(points_xyzi, axis=0)

    cfg_xmin, cfg_ymin, cfg_zmin, cfg_xmax, cfg_ymax, cfg_zmax = point_cloud_range[:6]

    range_valid = True
    mismatches = []
    if p_min[0] < cfg_xmin or p_max[0] > cfg_xmax:
        range_valid = False
        mismatches.append(f"X range [{p_min[0]:.2f}, {p_max[0]:.2f}] vs expected [{cfg_xmin:.2f}, {cfg_xmax:.2f}]")
    if p_min[1] < cfg_ymin or p_max[1] > cfg_ymax:
        range_valid = False
        mismatches.append(f"Y range [{p_min[1]:.2f}, {p_max[1]:.2f}] vs expected [{cfg_ymin:.2f}, {cfg_ymax:.2f}]")
    if p_min[2] < cfg_zmin or p_max[2] > cfg_zmax:
        range_valid = False
        mismatches.append(f"Z range [{p_min[2]:.2f}, {p_max[2]:.2f}] vs expected [{cfg_zmin:.2f}, {cfg_zmax:.2f}]")

    if not range_valid:
        warnings.warn(
            f"[WARNING] Point cloud spatial bounds exceed PointPillars expected range: "
            f"{'; '.join(mismatches)}. Points outside the range will produce degraded "
            f"or zero detections in OpenPCDet voxelization. Points are NOT silently truncated.",
            UserWarning,
            stacklevel=2,
        )

    # Intensity validation: KITTI expects normalized float ~[0.0, 1.0]
    intensity_valid = True
    if p_min[3] < 0.0 or p_max[3] > 1.05:
        intensity_valid = False
        warnings.warn(
            f"[WARNING] Point cloud intensity range [{p_min[3]:.3f}, {p_max[3]:.3f}] "
            f"deviates from expected KITTI convention [0.0, 1.0]. Detection quality "
            f"may degrade. Intensities are NOT auto-rescaled silently.",
            UserWarning,
            stacklevel=2,
        )

    return range_valid, intensity_valid


# ============================================================================
# 3D IoU and Duplicate Box Sanity Check (Task N)
# ============================================================================

def _box2d_corners(cx: float, cy: float, l: float, w: float, yaw: float) -> np.ndarray:
    """Get 4 corners of an oriented 2D bounding box in counter-clockwise order."""
    cos_a = np.cos(yaw)
    sin_a = np.sin(yaw)
    dx = l / 2.0
    dy = w / 2.0
    # CCW order
    local_corners = np.array([
        [-dx, -dy],
        [ dx, -dy],
        [ dx,  dy],
        [-dx,  dy],
    ], dtype=np.float64)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    return local_corners @ rot.T + np.array([cx, cy], dtype=np.float64)


def _polygon_area(pts: np.ndarray) -> float:
    """Compute area of a 2D convex polygon using the Shoelace formula."""
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _clip_polygon(subject_polygon: np.ndarray, clip_polygon: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman polygon clipping for two convex 2D polygons."""
    output_list = list(subject_polygon)
    cp_len = len(clip_polygon)
    for i in range(cp_len):
        input_list = output_list
        output_list = []
        if not input_list:
            break
        c1 = clip_polygon[i]
        c2 = clip_polygon[(i + 1) % cp_len]

        def inside(p: np.ndarray) -> bool:
            return (c2[0] - c1[0]) * (p[1] - c1[1]) - (c2[1] - c1[1]) * (p[0] - c1[0]) >= -1e-9

        def intersection(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
            dc = [c1[0] - c2[0], c1[1] - c2[1]]
            dp = [p1[0] - p2[0], p1[1] - p2[1]]
            n1 = c1[0] * c2[1] - c1[1] * c2[0]
            n2 = p1[0] * p2[1] - p1[1] * p2[0]
            denom = dc[0] * dp[1] - dc[1] * dp[0]
            n3 = 1.0 / (denom if abs(denom) > 1e-12 else 1e-12)
            return np.array([(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3])

        s = input_list[-1]
        for e in input_list:
            if inside(e):
                if not inside(s):
                    output_list.append(intersection(s, e))
                output_list.append(e)
            elif inside(s):
                output_list.append(intersection(s, e))
            s = e
    return np.array(output_list, dtype=np.float64)


def compute_box3d_iou(d1: Detection3D, d2: Detection3D) -> float:
    """Compute exact 3D IoU between two oriented bounding boxes.

    Parameters
    ----------
    d1, d2 : Detection3D
        3D detections with center_xyz, size_lwh, and yaw_rad.

    Returns
    -------
    float
        3D Intersection over Union in [0.0, 1.0].
    """
    # 1. Height overlap along Z
    z1_min = float(d1.center_xyz[2] - d1.size_lwh[2] / 2.0)
    z1_max = float(d1.center_xyz[2] + d1.size_lwh[2] / 2.0)
    z2_min = float(d2.center_xyz[2] - d2.size_lwh[2] / 2.0)
    z2_max = float(d2.center_xyz[2] + d2.size_lwh[2] / 2.0)

    inter_z = max(0.0, min(z1_max, z2_max) - max(z1_min, z2_min))
    if inter_z <= 0.0:
        return 0.0

    # 2. BEV polygon intersection area
    p1 = _box2d_corners(float(d1.center_xyz[0]), float(d1.center_xyz[1]), float(d1.size_lwh[0]), float(d1.size_lwh[1]), float(d1.yaw_rad))
    p2 = _box2d_corners(float(d2.center_xyz[0]), float(d2.center_xyz[1]), float(d2.size_lwh[0]), float(d2.size_lwh[1]), float(d2.yaw_rad))

    clipped = _clip_polygon(p1, p2)
    inter_bev = _polygon_area(clipped)
    if inter_bev <= 0.0:
        return 0.0

    inter_vol = inter_bev * inter_z
    vol1 = float(d1.size_lwh[0] * d1.size_lwh[1] * d1.size_lwh[2])
    vol2 = float(d2.size_lwh[0] * d2.size_lwh[1] * d2.size_lwh[2])

    union_vol = vol1 + vol2 - inter_vol
    if union_vol <= 0.0:
        return 0.0

    return float(np.clip(inter_vol / union_vol, 0.0, 1.0))


def check_duplicate_boxes_3d(
    detections: list[Detection3D],
    iou_threshold: float = 0.70,
) -> list[tuple[int, int, float]]:
    """Check for high-overlap pairs of 3D detection boxes.

    Parameters
    ----------
    detections : list[Detection3D]
        List of detections in the current frame.
    iou_threshold : float
        Overlap threshold above which a warning is flagged.

    Returns
    -------
    list[tuple[int, int, float]]
        List of (idx_i, idx_j, iou) tuples for overlapping pairs.
    """
    duplicates = []
    n = len(detections)
    for i in range(n):
        for j in range(i + 1, n):
            iou = compute_box3d_iou(detections[i], detections[j])
            if iou >= iou_threshold:
                duplicates.append((i, j, iou))
    return duplicates


# ============================================================================
# OpenPCDetPointPillarsDetector Adapter
# ============================================================================

class OpenPCDetPointPillarsDetector(ObjectDetector):
    """Adapter for a pretrained OpenPCDet PointPillars model.

    This class dynamically loads the OpenPCDet library from an external
    repository path and uses a real checkpoint to generate predictions.
    """

    _SOURCE = "pointpillars_openpcdet_pretrained"

    def __init__(
        self,
        repo_path: Path,
        checkpoint_path: Path,
        config_path: Path,
        device: str = "auto",
        score_threshold: float = 0.20,
        duplicate_iou_threshold: float = 0.70,
        strict: bool = True,
    ) -> None:
        """Initialize the real OpenPCDet PointPillars detector.

        Parameters
        ----------
        repo_path : Path
            Path to the external OpenPCDet repository.
        checkpoint_path : Path
            Path to the .pth pretrained checkpoint file.
        config_path : Path
            Path to the OpenPCDet configuration file (YAML).
        device : str
            "auto", "cpu", or a specific cuda device.
        score_threshold : float
            Minimum confidence score.
        duplicate_iou_threshold : float
            Threshold for flagging overlapping duplicate boxes (Task N).
        strict : bool
            Reserved.
        """
        self.repo_path = repo_path.resolve()
        self.checkpoint_path = checkpoint_path.resolve()
        self.config_path = config_path.resolve()
        self.score_threshold = score_threshold
        self.duplicate_iou_threshold = duplicate_iou_threshold

        # 1. Verify paths
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"OpenPCDet repository not found at {self.repo_path}. "
                "Did you manually clone it?"
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {self.checkpoint_path}. "
                "Please download an official checkpoint."
            )
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config not found at {self.config_path}. "
                "Check the OpenPCDet tools/cfgs path."
            )

        # 2. Add repo to sys.path locally
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

        # 3. Import PyTorch and Resolve device
        import torch
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.device.type == "cpu":
            warnings.warn(
                "Inference running on CPU. This can be significantly slower. "
                "Ensure PyTorch was compiled with CUDA for optimal performance.",
                UserWarning,
                stacklevel=2,
            )

        # 4. Import OpenPCDet inside this adapter
        try:
            from pcdet.config import cfg, cfg_from_yaml_file
            from pcdet.models import build_network
            from pcdet.datasets import DatasetTemplate
        except ImportError as e:
            raise ImportError(
                "Failed to import pcdet. Ensure dependencies are installed."
            ) from e

        # 5. Load model and config
        self.cfg = cfg_from_yaml_file(str(self.config_path), cfg)

        # Default KITTI PointPillars spatial range if not present in config
        default_range = [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
        if hasattr(self.cfg, "DATA_CONFIG") and hasattr(self.cfg.DATA_CONFIG, "POINT_CLOUD_RANGE"):
            self.point_cloud_range = list(self.cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
        else:
            self.point_cloud_range = default_range

        self.dataset = DatasetTemplate(
            dataset_cfg=self.cfg.DATA_CONFIG,
            class_names=self.cfg.CLASS_NAMES,
            training=False,
            root_path=None,
            logger=None,
        )

        self.model = build_network(
            model_cfg=self.cfg.MODEL,
            num_class=len(self.cfg.CLASS_NAMES),
            dataset=self.dataset,
        )
        self.model.load_params_from_file(
            filename=str(self.checkpoint_path),
            logger=None,
            to_cpu=True,
        )
        self.model.to(self.device)
        self.model.eval()

        self.class_names = list(self.cfg.CLASS_NAMES)

        # Model domain validity warning
        warnings.warn(
            "A pretrained model is only valid for the dataset/domain it was trained on. "
            "Validate before safety or performance claims.",
            UserWarning,
            stacklevel=2,
        )

    def detect(
        self,
        points_xyzi: np.ndarray,
        timestamp_s: float | None = None,
        semantic_labels: np.ndarray | None = None,
        **kwargs: Any,
    ) -> list[Detection3D]:
        """Detect objects using the pretrained OpenPCDet model.

        Parameters
        ----------
        points_xyzi : np.ndarray, shape (N, 4)
            Input LiDAR points [x, y, z, intensity].
        timestamp_s : float | None
            Optional timestamp.

        Returns
        -------
        list[Detection3D]
            Deterministically sorted list of detections mapped to FOVEAX taxonomy.
        """
        validate_points_xyzi(points_xyzi)

        # Task I: Validate spatial bounds and intensity range before inference
        validate_point_cloud_range_and_intensity(points_xyzi, self.point_cloud_range)

        import torch
        from pcdet.models import load_data_to_gpu

        # Format input for OpenPCDet (batch format)
        num_points = points_xyzi.shape[0]
        pts = np.zeros((num_points, 5), dtype=np.float32)
        pts[:, 1:5] = points_xyzi

        data_dict = {
            "points": pts,
            "frame_id": 0,
        }

        data_dict = self.dataset.prepare_data(data_dict=data_dict)
        data_dict = self.dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        with torch.no_grad():
            pred_dicts, _ = self.model.forward(data_dict)

        pred_dict = pred_dicts[0]

        pred_boxes = pred_dict["pred_boxes"].cpu().numpy()
        pred_scores = pred_dict["pred_scores"].cpu().numpy()
        pred_labels = pred_dict["pred_labels"].cpu().numpy()

        detections = []
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            conf = float(np.clip(score, 0.0, 1.0))
            if conf < self.score_threshold:
                continue

            # Standard box convention in OpenPCDet: [x, y, z, dx, dy, dz, heading]
            center = box[0:3].astype(np.float32)
            size = box[3:6].astype(np.float32)
            yaw = float(box[6])

            # 1-based native label
            native_class_id = int(label)
            if 1 <= native_class_id <= len(self.class_names):
                native_class_name = self.class_names[native_class_id - 1]
            else:
                native_class_name = f"unknown_{native_class_id}"

            # Task M: Map native class to FOVEAX taxonomy while keeping native metadata
            foveax_class_name, foveax_class_id = map_openpcdet_class_to_foveax(native_class_name)

            metadata = {
                "config_path": str(self.config_path),
                "checkpoint_name": self.checkpoint_path.name,
                "device": str(self.device),
                "native_label": native_class_id,
                "native_class_name": native_class_name,
                "foveax_semantic_class": foveax_class_name,
                "foveax_class_id": foveax_class_id,
            }

            d = Detection3D(
                center_xyz=center,
                size_lwh=size,
                yaw_rad=yaw,
                class_id=foveax_class_id,
                class_name=foveax_class_name,
                confidence=conf,
                source=self._SOURCE,
                metadata=metadata,
            )
            detections.append(d)

        # Task N: Duplicate box / NMS misconfiguration sanity check
        duplicate_pairs = check_duplicate_boxes_3d(detections, iou_threshold=self.duplicate_iou_threshold)
        if duplicate_pairs:
            warnings.warn(
                f"[WARNING] Detected {len(duplicate_pairs)} pair(s) of high-overlap 3D boxes "
                f"(IoU >= {self.duplicate_iou_threshold:.2f}). This may indicate duplicate detections "
                f"or NMS misconfiguration in OpenPCDet. Overlapping boxes are NOT silently deduplicated.",
                UserWarning,
                stacklevel=2,
            )

        # Sort deterministically
        detections.sort(
            key=lambda d: (-d.confidence, d.class_name, d.center_xyz[0], d.center_xyz[1])
        )

        return detections
