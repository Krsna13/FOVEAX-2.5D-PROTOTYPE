"""FOVEAX Phase 7 — AI-ready semantic 2.5D LiDAR map.

Replaces Phase 6's hard-coded ground-truth path with a pluggable
SemanticPredictor interface.  Supports three sources:

    --source ground_truth   Uses SemanticKITTI .label files
    --source mock           Geometry-only heuristic (NOT AI)
    --source pretrained     (placeholder — raises NotImplementedError)

Usage
-----
    python src/10_ai_semantic_2point5d_map.py --source mock
    python src/10_ai_semantic_2point5d_map.py --source ground_truth
    python src/10_ai_semantic_2point5d_map.py --source ground_truth --frame 000001
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- Resolve project root so the import works from any cwd ---------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.perception.semantic_labels import (  # noqa: E402
    FOVEAX_CLASSES,
    FOVEAX_COLORS,
    NUM_FOVEAX_CLASSES,
)
from src.perception.semantic_predictor import (  # noqa: E402
    GroundTruthSemanticPredictor,
    MockSemanticPredictor,
    PretrainedModelPredictor,
    SemanticPrediction,
    SemanticPredictor,
    validate_prediction,
)
from src.perception.salsanext_predictor import SalsaNextPredictor  # noqa: E402
from src.perception.rellis3d_loader import load_rellis3d_frame  # noqa: E402


# RELLIS-3D lives outside the repo tree (see docs on why: OneDrive-sync
# corruption made keeping large datasets inside the project unsafe). There is
# no --input flag for this -- SemanticKITTI's root is likewise a hardcoded
# constant below (_SEMANTICKITTI_ROOT), not a CLI flag, so this follows the
# same existing precedent rather than introducing a new one.
_SEMANTICKITTI_ROOT = Path("data/semantic_kitti/dataset/sequences")
_RELLIS3D_ROOT = Path("C:/dev/data/rellis3d")


# =========================================================================
# Grid configuration
# =========================================================================

@dataclass(frozen=True)
class GridConfig:
    """Axis-aligned 2.5D grid extent and resolution."""

    x_min: float = 0.0
    x_max: float = 50.0
    y_min: float = -25.0
    y_max: float = 25.0
    resolution: float = 0.20

    @property
    def cols(self) -> int:
        return int(np.ceil((self.x_max - self.x_min) / self.resolution))

    @property
    def rows(self) -> int:
        return int(np.ceil((self.y_max - self.y_min) / self.resolution))


# =========================================================================
# Data loading
# =========================================================================

def load_velodyne_scan(path: Path) -> np.ndarray:
    """Load a SemanticKITTI Velodyne .bin scan.

    Parameters
    ----------
    path : Path
        Absolute or relative path to the ``.bin`` file.

    Returns
    -------
    np.ndarray, shape (N, 4), dtype float32
        Columns: [x, y, z, intensity].
    """
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"Point file size ({raw.size}) is not divisible by 4.")
    return raw.reshape(-1, 4)


def load_label_file(path: Path) -> np.ndarray:
    """Load a SemanticKITTI ``.label`` file.

    Returns
    -------
    np.ndarray, shape (N,), dtype uint32
    """
    return np.fromfile(path, dtype=np.uint32)


# =========================================================================
# Confidence-weighted semantic 2.5D grid
# =========================================================================

@dataclass
class SemanticGridLayers:
    """All raster layers produced by the Phase 7 grid builder."""

    semantic: np.ndarray      # (R, C) uint8 — winning FOVEAX class
    confidence: np.ndarray    # (R, C) float32 — semantic confidence
    uncertainty: np.ndarray   # (R, C) float32 — semantic uncertainty
    map_confidence: np.ndarray  # (R, C) float32 — combined confidence
    z_min: np.ndarray         # (R, C) float32
    z_max: np.ndarray         # (R, C) float32
    point_count: np.ndarray   # (R, C) uint32
    n_points_roi: int         # total points that fell inside the ROI


def build_semantic_grid(
    points: np.ndarray,
    prediction: SemanticPrediction,
    config: GridConfig,
) -> SemanticGridLayers:
    """Create a confidence-weighted semantic 2.5D grid.

    For each occupied grid cell:
        - dominant class via confidence-weighted voting
        - semantic confidence  = weighted winner votes / total weight
        - semantic uncertainty = 1 - semantic confidence
        - map confidence       = blend of point density & semantic confidence
        - z_min, z_max, point_count

    Parameters
    ----------
    points : np.ndarray, shape (N, 4)
    prediction : SemanticPrediction
    config : GridConfig

    Returns
    -------
    SemanticGridLayers
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    inside = (
        (x >= config.x_min) & (x < config.x_max)
        & (y >= config.y_min) & (y < config.y_max)
    )
    xi, yi, zi = x[inside], y[inside], z[inside]
    labels = prediction.class_ids[inside]
    conf = prediction.confidence[inside]
    unc = prediction.uncertainty[inside]

    n_roi = int(inside.sum())
    if n_roi == 0:
        raise ValueError("No points inside the selected ROI.")

    cols, rows = config.cols, config.rows
    cell_count = rows * cols

    col_idx = np.floor((xi - config.x_min) / config.resolution).astype(np.int32)
    row_idx = np.floor((yi - config.y_min) / config.resolution).astype(np.int32)
    flat = row_idx * cols + col_idx

    # --- Point count ---
    point_count = np.bincount(flat, minlength=cell_count).astype(np.uint32)
    occupied = point_count > 0

    # --- Z extremes ---
    z_max = np.full(cell_count, -np.inf, dtype=np.float32)
    z_min_tmp = np.full(cell_count, np.inf, dtype=np.float32)
    np.maximum.at(z_max, flat, zi)
    np.minimum.at(z_min_tmp, flat, zi)
    z_max[~occupied] = np.nan
    z_min = np.full(cell_count, np.nan, dtype=np.float32)
    z_min[occupied] = z_min_tmp[occupied]

    # --- Confidence-weighted class voting ---
    weighted_votes = np.zeros((cell_count, NUM_FOVEAX_CLASSES), dtype=np.float64)
    for cid in range(NUM_FOVEAX_CLASSES):
        mask = labels == cid
        if mask.any():
            np.add.at(weighted_votes[:, cid], flat[mask], conf[mask])

    cell_label = np.argmax(weighted_votes, axis=1).astype(np.uint8)
    winner_weight = weighted_votes[np.arange(cell_count), cell_label]
    total_weight = weighted_votes.sum(axis=1)

    # --- Semantic confidence & uncertainty per cell ---
    sem_conf = np.zeros(cell_count, dtype=np.float32)
    sem_conf[occupied] = (
        winner_weight[occupied] / np.maximum(total_weight[occupied], 1e-9)
    ).astype(np.float32)

    sem_unc = np.zeros(cell_count, dtype=np.float32)
    sem_unc[occupied] = (1.0 - sem_conf[occupied]).astype(np.float32)

    cell_label[~occupied] = 7   # UNKNOWN for empty cells
    sem_conf[~occupied] = 0.0
    sem_unc[~occupied] = 1.0

    # --- Map confidence: blend density saturation + semantic confidence ---
    # density_score saturates at ~20 points per cell
    density_score = np.clip(
        point_count.astype(np.float32) / 20.0, 0.0, 1.0
    )
    map_conf = np.zeros(cell_count, dtype=np.float32)
    map_conf[occupied] = (
        0.5 * density_score[occupied] + 0.5 * sem_conf[occupied]
    )

    return SemanticGridLayers(
        semantic=cell_label.reshape(rows, cols),
        confidence=sem_conf.reshape(rows, cols),
        uncertainty=sem_unc.reshape(rows, cols),
        map_confidence=map_conf.reshape(rows, cols),
        z_min=z_min.reshape(rows, cols),
        z_max=z_max.reshape(rows, cols),
        point_count=point_count.reshape(rows, cols),
        n_points_roi=n_roi,
    )


# =========================================================================
# Visualisation & persistence
# =========================================================================

_SOURCE_BANNER = {
    "semantic_kitti_ground_truth": (
        "Source: SemanticKITTI GROUND-TRUTH labels (not AI predictions)"
    ),
    "rellis3d_ground_truth": (
        "Source: RELLIS-3D GROUND-TRUTH labels (not AI predictions)"
    ),
    "mock_geometry_baseline": (
        "[!] Source: MOCK geometry heuristic -- NOT AI inference"
    ),
}


def _add_source_subtitle(ax: plt.Axes, source: str) -> None:
    """Add a small subtitle below the title indicating prediction source."""
    banner = _SOURCE_BANNER.get(source, f"Source: {source}")
    ax.set_title(
        ax.get_title() + f"\n({banner})",
        fontsize=10,
    )


def save_phase7_outputs(
    layers: SemanticGridLayers,
    source: str,
    output_dir: Path,
) -> None:
    """Save all Phase 7 visualisation PNGs and the NPZ archive."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Semantic class map ----
    fig, ax = plt.subplots(figsize=(12, 8))
    rgb = FOVEAX_COLORS[layers.semantic]
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    ax.set_title("FOVEAX Phase 7 — Semantic 2.5D Map")
    _add_source_subtitle(ax, source)
    ax.set_xlabel("Forward X grid cell")
    ax.set_ylabel("Side Y grid cell")
    fig.tight_layout()
    fig.savefig(output_dir / "semantic_map.png", dpi=180)
    plt.close(fig)

    # ---- 2. Semantic confidence ----
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        layers.confidence, origin="lower", interpolation="nearest",
        cmap="viridis", vmin=0, vmax=1,
    )
    fig.colorbar(im, ax=ax, label="Semantic confidence")
    ax.set_title("FOVEAX Phase 7 — Semantic Confidence")
    _add_source_subtitle(ax, source)
    ax.set_xlabel("Forward X grid cell")
    ax.set_ylabel("Side Y grid cell")
    fig.tight_layout()
    fig.savefig(output_dir / "semantic_confidence.png", dpi=180)
    plt.close(fig)

    # ---- 3. Semantic uncertainty ----
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        layers.uncertainty, origin="lower", interpolation="nearest",
        cmap="inferno", vmin=0, vmax=1,
    )
    fig.colorbar(im, ax=ax, label="Semantic uncertainty")
    ax.set_title("FOVEAX Phase 7 — Semantic Uncertainty")
    _add_source_subtitle(ax, source)
    ax.set_xlabel("Forward X grid cell")
    ax.set_ylabel("Side Y grid cell")
    fig.tight_layout()
    fig.savefig(output_dir / "semantic_uncertainty.png", dpi=180)
    plt.close(fig)

    # ---- 4. Map confidence ----
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        layers.map_confidence, origin="lower", interpolation="nearest",
        cmap="viridis", vmin=0, vmax=1,
    )
    fig.colorbar(im, ax=ax, label="Map confidence")
    ax.set_title("FOVEAX Phase 7 — Map Confidence (density + semantic)")
    _add_source_subtitle(ax, source)
    ax.set_xlabel("Forward X grid cell")
    ax.set_ylabel("Side Y grid cell")
    fig.tight_layout()
    fig.savefig(output_dir / "map_confidence.png", dpi=180)
    plt.close(fig)

    # ---- 5. Z max ----
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        np.ma.masked_invalid(layers.z_max), origin="lower",
        interpolation="nearest", cmap="turbo",
    )
    fig.colorbar(im, ax=ax, label="Maximum elevation (m)")
    ax.set_title("FOVEAX Phase 7 — Maximum Elevation")
    _add_source_subtitle(ax, source)
    ax.set_xlabel("Forward X grid cell")
    ax.set_ylabel("Side Y grid cell")
    fig.tight_layout()
    fig.savefig(output_dir / "z_max.png", dpi=180)
    plt.close(fig)

    # ---- 6. Point density ----
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        layers.point_count, origin="lower", interpolation="nearest",
        cmap="hot",
    )
    fig.colorbar(im, ax=ax, label="Points per cell")
    ax.set_title("FOVEAX Phase 7 — Point Density")
    _add_source_subtitle(ax, source)
    ax.set_xlabel("Forward X grid cell")
    ax.set_ylabel("Side Y grid cell")
    fig.tight_layout()
    fig.savefig(output_dir / "point_density.png", dpi=180)
    plt.close(fig)

    # ---- 7. NPZ archive ----
    np.savez_compressed(
        output_dir / "semantic_layers.npz",
        semantic=layers.semantic,
        confidence=layers.confidence,
        uncertainty=layers.uncertainty,
        map_confidence=layers.map_confidence,
        z_min=layers.z_min,
        z_max=layers.z_max,
        point_count=layers.point_count,
    )


def write_metrics(
    layers: SemanticGridLayers,
    prediction: SemanticPrediction,
    n_total: int,
    output_dir: Path,
) -> str:
    """Write phase7_metrics.txt and return the text for console output."""
    occupied = layers.point_count > 0
    lines = [
        "=" * 60,
        "FOVEAX Phase 7 — AI Semantic 2.5D Map Metrics",
        "=" * 60,
        f"Prediction source : {prediction.source}",
        "",
    ]

    banner = _SOURCE_BANNER.get(prediction.source, prediction.source)
    lines.append(f"  >>> {banner}")
    lines.append("")

    lines += [
        f"Input points      : {n_total:,}",
        f"Points in ROI     : {layers.n_points_roi:,}",
        f"Grid shape        : {layers.semantic.shape}",
        f"Occupied cells    : {int(occupied.sum()):,}",
        "",
        f"Mean semantic confidence : {layers.confidence[occupied].mean():.4f}",
        f"Mean semantic uncertainty: {layers.uncertainty[occupied].mean():.4f}",
        f"Mean map confidence      : {layers.map_confidence[occupied].mean():.4f}",
        "",
        "Class distribution (points):",
    ]
    for cid, cname in FOVEAX_CLASSES.items():
        count = int((prediction.class_ids == cid).sum())
        lines.append(f"  {cname:20s}: {count:>10,}")

    lines += ["", f"Output folder: {output_dir}"]

    text = "\n".join(lines)
    (output_dir / "phase7_metrics.txt").write_text(text, encoding="utf-8")
    return text


# =========================================================================
# CLI entry point
# =========================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FOVEAX Phase 7 — AI semantic 2.5D LiDAR map.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python src/10_ai_semantic_2point5d_map.py --source mock
              python src/10_ai_semantic_2point5d_map.py --source ground_truth
              python src/10_ai_semantic_2point5d_map.py --source ground_truth --frame 000001
        """),
    )
    parser.add_argument(
        "--source",
        choices=["ground_truth", "mock", "pretrained", "salsanext"],
        default="mock",
        help="Semantic prediction source (default: mock).",
    )
    parser.add_argument(
        "--frame", default="000000",
        help="SemanticKITTI frame ID, e.g. 000000 (default: 000000).",
    )
    parser.add_argument(
        "--sequence", default="00",
        help=(
            "Sequence ID (default: 00). SemanticKITTI uses 2-digit IDs "
            "(e.g. 00); RELLIS-3D uses 5-digit IDs (e.g. 00000) -- see "
            "--dataset-type."
        ),
    )
    parser.add_argument(
        "--dataset-type",
        choices=["semantickitti", "rellis3d"],
        default="semantickitti",
        help=(
            "Dataset backend for --sequence/--frame (default: semantickitti). "
            "Required (not inferred from --sequence digit count) because the "
            "two datasets live under entirely different root directories "
            "and directory layouts -- SemanticKITTI under "
            "data/semantic_kitti/dataset/sequences/<seq>/{velodyne,labels}/, "
            "RELLIS-3D under its own root's "
            "Rellis-3D/<seq>/{os1_cloud_node_kitti_bin,"
            "os1_cloud_node_semantickitti_label_id}/ -- sequence-ID format "
            "alone can't safely pick between them."
        ),
    )
    parser.add_argument(
        "--resolution", type=float, default=0.20,
        help="Grid cell size in metres (default: 0.20).",
    )
    
    # SalsaNext specific arguments
    parser.add_argument(
        "--salsanext-repo", type=str,
        help="Path to official SalsaNext repository.",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        help="Path to official pretrained checkpoint.",
    )
    parser.add_argument(
        "--config", type=str,
        help="Path to official architecture config YAML.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to run inference on: 'auto', 'cuda', 'cpu'.",
    )
    parser.add_argument(
        "--fov-up", type=float, default=None,
        help="Override vertical FOV up in degrees for range projection (default: None, auto-set to 17.02 for rellis3d).",
    )
    parser.add_argument(
        "--fov-down", type=float, default=None,
        help="Override vertical FOV down in degrees for range projection (default: None, auto-set to -16.44 for rellis3d).",
    )
    parser.add_argument(
        "--rescale-intensity", action="store_true", default=None,
        help="Rescale point intensity to [0, 1] before range projection (auto-enabled for rellis3d).",
    )
    parser.add_argument(
        "--no-adapt-sensor", action="store_true", default=False,
        help="Disable automatic sensor FOV override and intensity rescaling for RELLIS-3D.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the Phase 7 semantic mapping pipeline."""
    args = parse_args(argv)

    rellis3d_frame = None  # populated only when dataset_type == "rellis3d"

    if args.dataset_type == "rellis3d":
        # Loader handles missing-file errors, point/label mismatch, sky
        # exclusion, and raw-ID -> FOVEAX remap internally; see
        # src/perception/rellis3d_loader.py.
        rellis3d_frame = load_rellis3d_frame(_RELLIS3D_ROOT, args.sequence, args.frame)
        points = rellis3d_frame.points
    else:
        base_dir = _SEMANTICKITTI_ROOT / args.sequence
        point_path = base_dir / "velodyne" / f"{args.frame}.bin"

        if not point_path.exists():
            raise FileNotFoundError(f"Point cloud not found: {point_path}")

        # --- Load points ---
        points = load_velodyne_scan(point_path)

    # --- Build predictor ---
    predictor: SemanticPredictor

    if args.source == "ground_truth":
        if args.dataset_type == "rellis3d":
            predictor = GroundTruthSemanticPredictor(
                foveax_class_ids=rellis3d_frame.foveax_class_ids
            )
        else:
            label_path = base_dir / "labels" / f"{args.frame}.label"
            if not label_path.exists():
                raise FileNotFoundError(f"Label file not found: {label_path}")
            labels_raw = load_label_file(label_path)
            if len(labels_raw) != len(points):
                raise ValueError(
                    f"Point-label mismatch: {len(points)} points, "
                    f"{len(labels_raw)} labels."
                )
            predictor = GroundTruthSemanticPredictor(labels_raw)

    elif args.source == "mock":
        predictor = MockSemanticPredictor()

    elif args.source == "pretrained":
        predictor = PretrainedModelPredictor()

    elif args.source == "salsanext":
        if not args.salsanext_repo or not args.checkpoint or not args.config:
            raise ValueError(
                "When --source salsanext is used, you must provide --salsanext-repo, "
                "--checkpoint, and --config. Please refer to docs/salsanext_setup.md."
            )

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

    else:
        raise ValueError(f"Unknown source: {args.source}")

    # --- Predict ---
    prediction = predictor.predict(points)
    validate_prediction(prediction, len(points))

    # --- Build grid ---
    config = GridConfig(
        x_min=0.0, x_max=50.0,
        y_min=-25.0, y_max=25.0,
        resolution=args.resolution,
    )
    layers = build_semantic_grid(points, prediction, config)

    # --- Save ---
    if args.dataset_type == "rellis3d":
        # Distinct path per sequence+frame: unlike SemanticKITTI (one fixed
        # dataset), multiple RELLIS-3D sequences/frames get validated in the
        # same session and must not overwrite each other's output.
        output_dir = (
            Path("outputs/phase7/rellis3d")
            / f"{args.sequence}_{args.frame}_{args.source}"
        )
    else:
        output_dir = Path("outputs/phase7") / args.source
    save_phase7_outputs(layers, prediction.source, output_dir)
    metrics_text = write_metrics(layers, prediction, len(points), output_dir)

    print(metrics_text)


if __name__ == "__main__":
    main()
