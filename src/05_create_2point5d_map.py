
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


@dataclass
class GridConfig:
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
    resolution: float = 0.02


def load_points(path: str) -> np.ndarray:
    point_cloud = o3d.io.read_point_cloud(path)
    points = np.asarray(point_cloud.points, dtype=np.float32)
    if len(points) == 0:
        raise ValueError(f"No points found in: {path}")
    return points


def create_2point5d_map(
    points_xyz: np.ndarray,
    config: GridConfig,
) -> dict[str, np.ndarray]:
    x_min, x_max = config.x_min, config.x_max
    y_min, y_max = config.y_min, config.y_max
    resolution = config.resolution

    width = int(np.ceil((x_max - x_min) / resolution))
    height = int(np.ceil((y_max - y_min) / resolution))

    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]

    valid = (
        (x >= x_min) &
        (x < x_max) &
        (y >= y_min) &
        (y < y_max)
    )
    x = x[valid]
    y = y[valid]
    z = z[valid]

    if len(z) == 0:
        raise ValueError("No points remain inside the selected ROI.")

    col = np.floor((x - x_min) / resolution).astype(np.int32)
    row = np.floor((y - y_min) / resolution).astype(np.int32)

    flat_index = row * width + col
    total_cells = height * width

    point_count = np.bincount(
        flat_index, minlength=total_cells
    ).astype(np.float32)

    z_sum = np.bincount(
        flat_index, weights=z, minlength=total_cells
    ).astype(np.float32)

    z_min = np.full(total_cells, np.inf, dtype=np.float32)
    z_max = np.full(total_cells, -np.inf, dtype=np.float32)

    np.minimum.at(z_min, flat_index, z)
    np.maximum.at(z_max, flat_index, z)

    occupied = point_count > 0

    z_mean = np.full(total_cells, np.nan, dtype=np.float32)
    z_mean[occupied] = z_sum[occupied] / point_count[occupied]

    z_min[~occupied] = np.nan
    z_max[~occupied] = np.nan

    height_range = z_max - z_min

    return {
        "z_min":       z_min.reshape(height, width),
        "z_max":       z_max.reshape(height, width),
        "z_mean":      z_mean.reshape(height, width),
        "height_range": height_range.reshape(height, width),
        "point_count": point_count.reshape(height, width),
        "occupied":    occupied.reshape(height, width),
    }


def save_layer_image(
    layer: np.ndarray,
    title: str,
    output_path: Path,
    colormap: str,
    colorbar_label: str,
) -> None:
    masked = np.ma.masked_invalid(layer)

    plt.figure(figsize=(8, 7))
    image = plt.imshow(
        masked,
        origin="lower",
        cmap=colormap,
        interpolation="nearest",
    )
    plt.colorbar(image, label=colorbar_label)
    plt.title(title)
    plt.xlabel("Grid X cell index")
    plt.ylabel("Grid Y cell index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    input_path = "outputs/filtered_sample_cloud.ply"
    output_dir = Path("outputs/phase2")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = GridConfig(
        x_min=0.5,
        x_max=4.0,
        y_min=0.8,
        y_max=2.5,
        resolution=0.02,
    )

    points = load_points(input_path)
    print(f"Loaded points: {len(points):,}")
    print(f"Grid resolution: {config.resolution:.3f} m")
    print(
        f"ROI: x=[{config.x_min}, {config.x_max}], "
        f"y=[{config.y_min}, {config.y_max}]"
    )

    # Diagnostic: print point ranges so ROI can be tuned if needed.
    print("\nPoint coordinate ranges:")
    print(f"  x: {points[:, 0].min():.3f} to {points[:, 0].max():.3f}")
    print(f"  y: {points[:, 1].min():.3f} to {points[:, 1].max():.3f}")
    print(f"  z: {points[:, 2].min():.3f} to {points[:, 2].max():.3f}")

    maps = create_2point5d_map(points, config)

    occupied_cells = int(np.sum(maps["occupied"]))
    total_cells = maps["occupied"].size

    print(f"\nOccupied cells: {occupied_cells:,} / {total_cells:,}")
    print(f"Grid occupancy: {100 * occupied_cells / total_cells:.2f}%")

    save_layer_image(
        maps["z_max"],
        "FOVEAX 2.5D Map: Maximum Elevation",
        output_dir / "z_max.png",
        "turbo",
        "Maximum height (m)",
    )
    save_layer_image(
        maps["z_min"],
        "FOVEAX 2.5D Map: Minimum Elevation",
        output_dir / "z_min.png",
        "turbo",
        "Minimum height (m)",
    )
    save_layer_image(
        maps["height_range"],
        "FOVEAX 2.5D Map: Height Range",
        output_dir / "height_range.png",
        "magma",
        "Height range (m)",
    )

    density = np.log1p(maps["point_count"])
    save_layer_image(
        density,
        "FOVEAX 2.5D Map: Log Point Density",
        output_dir / "point_density.png",
        "viridis",
        "log(1 + points per cell)",
    )

    np.savez_compressed(
        output_dir / "map_layers.npz",
        **maps,
    )
    print(f"Saved 2.5D tensor layers: {output_dir / 'map_layers.npz'}")


if __name__ == "__main__":
    main()
