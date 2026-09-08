from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

@dataclass(frozen=True)
class AdaptiveZone:
    name: str
    r_min: float
    r_max: float
    resolution: float
    color: str

def load_points(path: str) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(path)
    points = np.asarray(cloud.points, dtype=np.float32)
    if len(points) == 0:
        raise ValueError(f"Could not load points from: {path}")
    return points

def create_zone_map(
    points_xyz: np.ndarray,
    zone: AdaptiveZone,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float
) -> dict[str, np.ndarray]:
    resolution = zone.resolution
    grid_cols = int(np.ceil((x_max - x_min) / resolution))
    grid_rows = int(np.ceil((y_max - y_min) / resolution))
    
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    
    distance = np.sqrt(x**2 + y**2)
    in_zone = (
        (distance >= zone.r_min) & (distance < zone.r_max) &
        (x >= x_min) & (x < x_max) &
        (y >= y_min) & (y < y_max)
    )
    
    x = x[in_zone]
    y = y[in_zone]
    z = z[in_zone]
    
    total_cells = grid_rows * grid_cols
    point_count = np.zeros(total_cells, dtype=np.float32)
    z_min = np.full(total_cells, np.nan, dtype=np.float32)
    z_max = np.full(total_cells, np.nan, dtype=np.float32)
    z_mean = np.full(total_cells, np.nan, dtype=np.float32)
    height_range = np.full(total_cells, np.nan, dtype=np.float32)
    occupied = np.zeros(total_cells, dtype=bool)
    
    if len(z) == 0:
        return {
            "point_count": point_count.reshape(grid_rows, grid_cols),
            "z_min": z_min.reshape(grid_rows, grid_cols),
            "z_max": z_max.reshape(grid_rows, grid_cols),
            "z_mean": z_mean.reshape(grid_rows, grid_cols),
            "height_range": height_range.reshape(grid_rows, grid_cols),
            "occupied": occupied.reshape(grid_rows, grid_cols),
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
        }
        
    col = np.floor((x - x_min) / resolution).astype(np.int32)
    row = np.floor((y - y_min) / resolution).astype(np.int32)
    flat_index = row * grid_cols + col
    
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
    z_mean[occupied] = z_sum[occupied] / point_count[occupied]
    z_min[~occupied] = np.nan
    z_max[~occupied] = np.nan
    height_range = z_max - z_min
    
    return {
        "point_count": point_count.reshape(grid_rows, grid_cols),
        "z_min": z_min.reshape(grid_rows, grid_cols),
        "z_max": z_max.reshape(grid_rows, grid_cols),
        "z_mean": z_mean.reshape(grid_rows, grid_cols),
        "height_range": height_range.reshape(grid_rows, grid_cols),
        "occupied": occupied.reshape(grid_rows, grid_cols),
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
    }

def render_adaptive_map(
    points_xyz: np.ndarray,
    zones: list[AdaptiveZone],
    output_path: Path
) -> None:
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    distance = np.sqrt(x**2 + y**2)
    
    plt.figure(figsize=(10, 9))
    for zone in zones:
        mask = (distance >= zone.r_min) & (distance < zone.r_max)
        plt.scatter(
            x[mask], y[mask],
            c=z[mask], cmap="turbo", s=4, alpha=0.75,
            label=(
                f"{zone.name}: {zone.r_min:.1f}–{zone.r_max:.1f} m "
                f"({zone.resolution * 100:.0f} cm)"
            )
        )
        ring = plt.Circle(
            (0.0, 0.0), zone.r_max, fill=False, color=zone.color, linewidth=2
        )
        plt.gca().add_patch(ring)
        
    plt.scatter(
        [0], [0], marker="^", color="black", s=110, label="Sensor / Vehicle"
    )
    plt.colorbar(label="Point height z (m)")
    plt.title("FOVEAX Adaptive 2.5D Mapping — Resolution Zones")
    plt.xlabel("X coordinate (m)")
    plt.ylabel("Y coordinate (m)")
    plt.axis("equal")
    plt.grid(alpha=0.25)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved: {output_path}")

def save_metrics(
    zone_results: dict[str, dict],
    zones: list[AdaptiveZone],
    output_path: Path
) -> None:
    lines = [
        "FOVEAX Phase 4 — Adaptive Variable-Resolution 2.5D Mapping",
        "",
        "Zone metrics:"
    ]
    
    total_occupied = 0
    total_cells = 0
    
    for zone in zones:
        result = zone_results[zone.name]
        occupied_cells = int(np.sum(result["occupied"]))
        grid_cells = int(result["occupied"].size)
        point_count = int(np.nansum(result["point_count"]))
        
        total_occupied += occupied_cells
        total_cells += grid_cells
        
        lines.extend([
            "",
            f"Zone: {zone.name}",
            f"Distance range: {zone.r_min:.2f}–{zone.r_max:.2f} m",
            f"Resolution: {zone.resolution:.3f} m",
            f"Grid shape: {result['grid_rows']} rows × {result['grid_cols']} cols",
            f"Points in zone: {point_count:,}",
            f"Occupied cells: {occupied_cells:,} / {grid_cells:,}",
            f"Grid occupancy: {100 * occupied_cells / grid_cells:.2f}%"
        ])
        
    lines.extend([
        "",
        f"Total occupied cells: {total_occupied:,}",
        f"Total allocated cells across zones: {total_cells:,}",
        f"Overall occupancy: {100 * total_occupied / total_cells:.2f}%"
    ])
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {output_path}")

def main():
    input_path = "outputs/filtered_sample_cloud.ply"
    output_dir = Path("outputs/phase4")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # DEMO configuration for the small Open3D sample cloud.
    zones = [
        AdaptiveZone("Near", 0.0, 0.5, 0.02, "#e74c3c"),
        AdaptiveZone("Middle", 0.5, 1.2, 0.05, "#f1c40f"),
        AdaptiveZone("Far", 1.2, 3.0, 0.10, "#3498db"),
    ]
    
    # These bounds must cover the Open3D sample cloud.
    x_min, x_max = -3.0, 3.0
    y_min, y_max = -3.0, 3.0
    
    points = load_points(input_path)
    print(f"Loaded points: {len(points):,}")
    print(
        f"Point X range: {points[:, 0].min():.3f} to {points[:, 0].max():.3f} m"
    )
    print(
        f"Point Y range: {points[:, 1].min():.3f} to {points[:, 1].max():.3f} m"
    )
    
    zone_results = {}
    for zone in zones:
        result = create_zone_map(
            points, zone, x_min, x_max, y_min, y_max
        )
        zone_results[zone.name] = result
        print(
            f"{zone.name}: "
            f"{int(np.nansum(result['point_count'])):,} points, "
            f"{int(np.sum(result['occupied'])):,} occupied cells, "
            f"{zone.resolution * 100:.0f} cm resolution"
        )
        np.savez_compressed(
            output_dir / f"{zone.name.lower()}_zone_map.npz",
            point_count=result["point_count"],
            z_min=result["z_min"],
            z_max=result["z_max"],
            z_mean=result["z_mean"],
            height_range=result["height_range"],
            occupied=result["occupied"],
            resolution=np.array([zone.resolution], dtype=np.float32),
            r_min=np.array([zone.r_min], dtype=np.float32),
            r_max=np.array([zone.r_max], dtype=np.float32),
        )
        
    render_adaptive_map(
        points, zones, output_dir / "adaptive_resolution_zones.png"
    )
    save_metrics(
        zone_results, zones, output_dir / "phase4_metrics.txt"
    )

if __name__ == "__main__":
    main()
