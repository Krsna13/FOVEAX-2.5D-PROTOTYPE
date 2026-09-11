import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np

# --- Resolve project root so the import works from any cwd ---------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.perception.semantic_labels import FOVEAX_CLASSES, NUM_FOVEAX_CLASSES
from src.perception.salsanext_predictor import SalsaNextPredictor
from src.perception.rellis3d_loader import load_rellis3d_frame
import importlib
ai_map_module = importlib.import_module("src.10_ai_semantic_2point5d_map")
load_velodyne_scan = ai_map_module.load_velodyne_scan
load_label_file = ai_map_module.load_label_file

_SEMANTICKITTI_ROOT = Path("data/semantic_kitti/dataset/sequences")
_RELLIS3D_ROOT = Path("C:/dev/data/rellis3d")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FOVEAX Phase 7 — Distance-bucketed accuracy and confusion metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--frame", default="000000",
        help="Frame ID, e.g. 000000 (default: 000000).",
    )
    parser.add_argument(
        "--sequence", default="00",
        help="Sequence ID (default: 00).",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["semantickitti", "rellis3d"],
        default="semantickitti",
        help="Dataset backend for --sequence/--frame (default: semantickitti).",
    )
    parser.add_argument(
        "--salsanext-repo", type=str, required=True,
        help="Path to official SalsaNext repository.",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to official pretrained checkpoint.",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to official architecture config YAML.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to run inference on: 'auto', 'cuda', 'cpu'.",
    )
    parser.add_argument(
        "--fov-up", type=float, default=None,
        help="Override vertical FOV up in degrees.",
    )
    parser.add_argument(
        "--fov-down", type=float, default=None,
        help="Override vertical FOV down in degrees.",
    )
    parser.add_argument(
        "--rescale-intensity", action="store_true", default=None,
        help="Rescale point intensity to [0, 1].",
    )
    parser.add_argument(
        "--no-adapt-sensor", action="store_true", default=False,
        help="Disable automatic sensor FOV override and intensity rescaling for RELLIS-3D.",
    )
    return parser.parse_args(argv)


def compute_metrics(
    gt_labels: np.ndarray, pred_labels: np.ndarray, mask: np.ndarray, bucket_name: str
) -> dict | None:
    """Compute (and print) accuracy + confusion matrix for one distance bucket.

    Ground-truth points labeled UNKNOWN (7, e.g. SemanticKITTI's own
    "unlabeled"/"outlier" raw classes) are excluded before scoring -- there
    is no real ground truth to compare a prediction against for those
    points, so counting them as "wrong" would penalize the model for
    disagreeing with an absent label. This matches SemanticKITTI's own
    official benchmark convention of ignoring its "unlabeled" class.

    Returns a plain dict (not a dataclass -- kept intentionally lightweight
    since this is primarily a print-report tool) so callers/tests can
    assert on the computed numbers directly instead of scraping stdout:
        {"bucket_name", "n_points", "accuracy", "confusion"}
    where confusion maps gt_class_id -> {pred_class_id: count}.
    Returns None if there are no valid (non-UNKNOWN) ground-truth points in
    this bucket.
    """
    valid_mask = mask & (gt_labels != 7)  # Ignore UNKNOWN (7) in ground truth
    n_points = int(valid_mask.sum())
    if n_points == 0:
        print(f"\n--- Bucket: {bucket_name} ---")
        print("No valid ground truth points in this bucket.")
        return None

    gt = gt_labels[valid_mask]
    pred = pred_labels[valid_mask]

    accuracy = (gt == pred).sum() / n_points

    print(f"\n--- Bucket: {bucket_name} ---")
    print(f"Points evaluated: {n_points:,}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nConfusion Matrix (Ground Truth -> Predicted):")
    confusion: dict[int, dict[int, int]] = {}
    for gt_id in sorted(FOVEAX_CLASSES.keys()):
        if gt_id == 7:
            continue
        gt_class_mask = (gt == gt_id)
        n_gt_class = gt_class_mask.sum()
        if n_gt_class == 0:
            continue

        pred_for_gt = pred[gt_class_mask]
        print(f"  GT {FOVEAX_CLASSES[gt_id]:<15} (n={n_gt_class:>6,}):")

        # Calculate distribution
        unique_preds, counts = np.unique(pred_for_gt, return_counts=True)
        pred_dist = sorted(zip(unique_preds, counts), key=lambda x: x[1], reverse=True)

        confusion[gt_id] = {int(p_id): int(count) for p_id, count in pred_dist}
        for p_id, count in pred_dist:
            pct = (count / n_gt_class) * 100
            print(f"    -> {FOVEAX_CLASSES[p_id]:<15}: {pct:>6.2f}% ({count:>6,})")

    return {
        "bucket_name": bucket_name,
        "n_points": n_points,
        "accuracy": float(accuracy),
        "confusion": confusion,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.dataset_type == "rellis3d":
        rellis3d_frame = load_rellis3d_frame(_RELLIS3D_ROOT, args.sequence, args.frame)
        points = rellis3d_frame.points
        gt_labels = rellis3d_frame.foveax_class_ids
    else:
        base_dir = _SEMANTICKITTI_ROOT / args.sequence
        point_path = base_dir / "velodyne" / f"{args.frame}.bin"
        label_path = base_dir / "labels" / f"{args.frame}.label"

        if not point_path.exists():
            raise FileNotFoundError(f"Point cloud not found: {point_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")

        points = load_velodyne_scan(point_path)
        labels_raw = load_label_file(label_path)
        
        # SemanticKITTI raw labels to FOVEAX classes mapping
        # We need the GroundTruthSemanticPredictor logic to map raw labels
        from src.perception.semantic_predictor import GroundTruthSemanticPredictor
        gt_predictor = GroundTruthSemanticPredictor(labels_raw)
        gt_labels = gt_predictor.predict(points).class_ids

    # Configuration for SalsaNext
    fov_up = args.fov_up
    fov_down = args.fov_down
    rescale_intensity = bool(args.rescale_intensity) if args.rescale_intensity is not None else False

    if args.dataset_type == "rellis3d" and not args.no_adapt_sensor:
        if fov_up is None:
            fov_up = 17.02
        if fov_down is None:
            fov_down = -16.44
        if args.rescale_intensity is None:
            rescale_intensity = True

    print(f"Loading SalsaNext predictor from {args.checkpoint}...")
    predictor = SalsaNextPredictor(
        repo_path=Path(args.salsanext_repo),
        checkpoint_path=Path(args.checkpoint),
        config_path=Path(args.config),
        device=args.device,
        strict=True,
        fov_up=fov_up,
        fov_down=fov_down,
        rescale_intensity=rescale_intensity,
    )

    print("Running inference...")
    prediction = predictor.predict(points)
    pred_labels = prediction.class_ids

    # Calculate distances (2D radial distance)
    x = points[:, 0]
    y = points[:, 1]
    distance = np.sqrt(x**2 + y**2)

    # Buckets: Near (0-15m), Mid (15-35m), Far (35-100m)
    mask_near = (distance >= 0.0) & (distance < 15.0)
    mask_mid = (distance >= 15.0) & (distance < 35.0)
    mask_far = (distance >= 35.0) & (distance <= 100.0)

    print(f"\n========================================================")
    print(f"Distance-Bucketed Evaluation: {args.dataset_type} {args.sequence}/{args.frame}")
    print(f"========================================================")

    compute_metrics(gt_labels, pred_labels, mask_near, "Near (0-15m)")
    compute_metrics(gt_labels, pred_labels, mask_mid, "Mid (15-35m)")
    compute_metrics(gt_labels, pred_labels, mask_far, "Far (35-100m)")
    
    # Overall
    mask_overall = (distance >= 0.0) & (distance <= 100.0)
    compute_metrics(gt_labels, pred_labels, mask_overall, "Overall (0-100m)")


if __name__ == "__main__":
    main()
