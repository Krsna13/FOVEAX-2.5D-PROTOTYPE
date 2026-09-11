from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class GridConfig:
    x_min: float = 0.0
    x_max: float = 50.0
    y_min: float = -25.0
    y_max: float = 25.0
    resolution: float = 0.20


import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.perception.semantic_labels import (
    FOVEAX_CLASSES,
    FOVEAX_COLORS,
    SEMANTICKITTI_TO_FOVEAX,
)


def load_semantickitti_scan(
    point_path: Path, label_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a single SemanticKITTI scan and its label file.

    Returns
    -------
    points : (N, 4) float32 – x, y, z, intensity
    semantic_ids : (N,) uint16-range – raw semantic class per point
    instance_ids : (N,) uint16-range – instance ID per point
    """
    points = np.fromfile(point_path, dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError("Point file size is not divisible by 4.")
    points = points.reshape(-1, 4)

    labels_raw = np.fromfile(label_path, dtype=np.uint32)
    if len(labels_raw) != len(points):
        raise ValueError(
            f"Point-label mismatch: {len(points)} points but "
            f"{len(labels_raw)} labels."
        )

    semantic_ids = labels_raw & 0xFFFF
    instance_ids = labels_raw >> 16
    return points, semantic_ids, instance_ids


def remap_to_foveax(semantic_ids: np.ndarray) -> np.ndarray:
    """Map raw SemanticKITTI class IDs to simplified FOVEAX class IDs."""
    mapped = np.full(len(semantic_ids), 7, dtype=np.uint8)
    for raw_id, foveax_id in SEMANTICKITTI_TO_FOVEAX.items():
        mapped[semantic_ids == raw_id] = foveax_id
    return mapped


def create_semantic_grid(
    points: np.ndarray,
    foveax_labels: np.ndarray,
    config: GridConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the semantic 2.5D grid via majority-vote per cell.

    Returns
    -------
    semantic_grid : (rows, cols) uint8 – winning FOVEAX class per cell
    confidence_grid : (rows, cols) float32 – fraction of votes for winner
    z_max_grid : (rows, cols) float32 – maximum elevation per cell
    """
    grid_cols = int(np.ceil((config.x_max - config.x_min) / config.resolution))
    grid_rows = int(np.ceil((config.y_max - config.y_min) / config.resolution))

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    inside = (
        (x >= config.x_min) & (x < config.x_max) &
        (y >= config.y_min) & (y < config.y_max)
    )
    x = x[inside]
    y = y[inside]
    z = z[inside]
    labels = foveax_labels[inside]

    if len(x) == 0:
        raise ValueError("No points inside selected ROI.")

    col = np.floor((x - config.x_min) / config.resolution).astype(np.int32)
    row = np.floor((y - config.y_min) / config.resolution).astype(np.int32)
    flat = row * grid_cols + col

    cell_count = grid_rows * grid_cols

    point_count = np.bincount(flat, minlength=cell_count).astype(np.float32)

    z_max = np.full(cell_count, -np.inf, dtype=np.float32)
    np.maximum.at(z_max, flat, z)
    occupied = point_count > 0
    z_max[~occupied] = np.nan

    # Class voting in every 2.5D cell.
    vote_counts = np.zeros((cell_count, len(FOVEAX_CLASSES)), dtype=np.uint16)
    for class_id in range(len(FOVEAX_CLASSES)):
        class_mask = labels == class_id
        vote_counts[:, class_id] = np.bincount(
            flat[class_mask], minlength=cell_count
        )

    cell_label = np.argmax(vote_counts, axis=1).astype(np.uint8)
    max_votes = vote_counts.max(axis=1).astype(np.float32)

    confidence = np.zeros(cell_count, dtype=np.float32)
    confidence[occupied] = (
        max_votes[occupied] / point_count[occupied]
    )
    cell_label[~occupied] = 7
    confidence[~occupied] = 0.0

    return (
        cell_label.reshape(grid_rows, grid_cols),
        confidence.reshape(grid_rows, grid_cols),
        z_max.reshape(grid_rows, grid_cols)
    )


def save_semantic_map(
    semantic_grid: np.ndarray,
    confidence_grid: np.ndarray,
    z_max_grid: np.ndarray,
    output_dir: Path
) -> None:
    """Render and save the three Phase 6 visualisation maps."""

    # --- Semantic colour map ---
    rgb_image = FOVEAX_COLORS[semantic_grid]
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb_image, origin="lower", interpolation="nearest")
    plt.title("FOVEAX — Semantic 2.5D LiDAR Map")
    plt.xlabel("Forward X grid cell")
    plt.ylabel("Side Y grid cell")
    plt.tight_layout()
    plt.savefig(output_dir / "semantic_2point5d_map.png", dpi=180)
    plt.close()

    # --- Confidence map ---
    plt.figure(figsize=(12, 8))
    image = plt.imshow(
        confidence_grid,
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        vmin=0,
        vmax=1
    )
    plt.colorbar(image, label="Semantic confidence")
    plt.title("FOVEAX — Semantic Map Confidence")
    plt.xlabel("Forward X grid cell")
    plt.ylabel("Side Y grid cell")
    plt.tight_layout()
    plt.savefig(output_dir / "semantic_confidence.png", dpi=180)
    plt.close()

    # --- Elevation map ---
    plt.figure(figsize=(12, 8))
    image = plt.imshow(
        np.ma.masked_invalid(z_max_grid),
        origin="lower",
        cmap="turbo",
        interpolation="nearest"
    )
    plt.colorbar(image, label="Maximum elevation (m)")
    plt.title("FOVEAX — Semantic Grid Maximum Elevation")
    plt.xlabel("Forward X grid cell")
    plt.ylabel("Side Y grid cell")
    plt.tight_layout()
    plt.savefig(output_dir / "semantic_z_max.png", dpi=180)
    plt.close()


def main():
    sequence = "00"
    frame = "000000"

    base_dir = Path("data/semantic_kitti/dataset/sequences") / sequence
    point_path = base_dir / "velodyne" / f"{frame}.bin"
    label_path = base_dir / "labels" / f"{frame}.label"

    output_dir = Path("outputs/phase6")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not point_path.exists():
        raise FileNotFoundError(f"Point cloud not found: {point_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    config = GridConfig(
        x_min=0.0,
        x_max=50.0,
        y_min=-25.0,
        y_max=25.0,
        resolution=0.20
    )

    points, semantic_ids, instance_ids = load_semantickitti_scan(
        point_path, label_path
    )
    foveax_labels = remap_to_foveax(semantic_ids)

    semantic_grid, confidence_grid, z_max_grid = create_semantic_grid(
        points, foveax_labels, config
    )

    print("FOVEAX Phase 6 — SemanticKITTI Integration")
    print(f"Sequence: {sequence}")
    print(f"Frame:    {frame}")
    print(f"Input points: {len(points):,}")
    print(f"Unique raw SemanticKITTI labels: {np.unique(semantic_ids).tolist()}")
    print(f"Grid shape: {semantic_grid.shape}")

    for class_id, class_name in FOVEAX_CLASSES.items():
        count = int(np.sum(foveax_labels == class_id))
        print(f"  {class_name}: {count:,} points")

    print(f"Unique instance IDs: {len(np.unique(instance_ids)):,}")

    save_semantic_map(
        semantic_grid, confidence_grid, z_max_grid, output_dir
    )

    np.savez_compressed(
        output_dir / "semantic_map_layers.npz",
        semantic_grid=semantic_grid,
        confidence_grid=confidence_grid,
        z_max_grid=z_max_grid,
        config=np.array([
            config.x_min, config.x_max,
            config.y_min, config.y_max,
            config.resolution
        ], dtype=np.float32)
    )

    print(f"Saved semantic maps to: {output_dir}")


if __name__ == "__main__":
    main()
