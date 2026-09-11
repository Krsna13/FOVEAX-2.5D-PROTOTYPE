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


# Problem Statement (PS-26053) full-scale automotive specification (100m range)
SPEC_ZONES_100M = [
    AdaptiveZone("Near", 0.0, 15.0, 0.05, "#e74c3c"),     # 0-15m: 5cm cells (reactive safety)
    AdaptiveZone("Middle", 15.0, 35.0, 0.20, "#f1c40f"),  # 15-35m: 20cm cells (path planning)
    AdaptiveZone("Far", 35.0, 100.0, 0.50, "#3498db"),    # 35-100m: 50cm cells (situational awareness)
]

# Demo zones scaled for the small Open3D sample cloud
SAMPLE_ZONES = [
    AdaptiveZone("Near", 0.0, 0.5, 0.02, "#e74c3c"),
    AdaptiveZone("Middle", 0.5, 1.2, 0.05, "#f1c40f"),
    AdaptiveZone("Far", 1.2, 3.0, 0.10, "#3498db"),
]


def compute_memory_reduction_metrics(
    zones: list[AdaptiveZone] | None = None,
    extent_x: tuple[float, float] = (-100.0, 100.0),
    extent_y: tuple[float, float] = (-100.0, 100.0),
    height_m: float = 8.0,
    bytes_per_voxel: int = 1,
    bytes_per_cell: int = 16,
) -> dict[str, float]:
    """Compute theoretical memory reduction of foveated adaptive 2.5D grid vs uniform 3D voxels.

    Parameters
    ----------
    zones : list[AdaptiveZone], optional
        List of foveated zones. If None, uses SPEC_ZONES_100M.
    extent_x : tuple[float, float]
        Perception corridor X bounds in meters (width = extent_x[1] - extent_x[0]).
    extent_y : tuple[float, float]
        Perception corridor Y bounds in meters (height = extent_y[1] - extent_y[0]).
    height_m : float
        Vertical elevation span in meters (e.g. -3m to +5m = 8m).
    bytes_per_voxel : int
        Memory per 3D voxel (1 byte for occupancy bit/byte).
    bytes_per_cell : int
        Memory per 2.5D cell (16 bytes for 4 float32 layers: z_min, z_max, height_range, density).

    Returns
    -------
    dict with voxel/cell counts, memory usage, and percentage savings.
    """
    if zones is None:
        zones = SPEC_ZONES_100M

    finest_res = min(z.resolution for z in zones)
    width_x = extent_x[1] - extent_x[0]
    width_y = extent_y[1] - extent_y[0]

    # Uniform 3D voxel grid at finest resolution
    nx_fine = int(np.ceil(width_x / finest_res))
    ny_fine = int(np.ceil(width_y / finest_res))
    nz_fine = int(np.ceil(height_m / finest_res))
    total_3d_voxels = nx_fine * ny_fine * nz_fine
    mem_3d_bytes = total_3d_voxels * bytes_per_voxel

    # Uniform 2.5D grid at finest resolution
    total_uniform_2d_cells = nx_fine * ny_fine
    mem_uniform_2d_bytes = total_uniform_2d_cells * bytes_per_cell

    # Adaptive Variable-Resolution 2.5D Grid across zones (annular regions bounded by max range)
    adaptive_cells = 0
    for z in zones:
        annular_area = np.pi * (z.r_max**2 - z.r_min**2)
        cells_in_zone = int(np.ceil(annular_area / (z.resolution**2)))
        adaptive_cells += cells_in_zone

    mem_adaptive_bytes = adaptive_cells * bytes_per_cell

    cell_savings_vs_3d = (1.0 - (adaptive_cells / total_3d_voxels)) * 100.0
    byte_savings_vs_3d = (1.0 - (mem_adaptive_bytes / mem_3d_bytes)) * 100.0
    cell_savings_vs_uniform_2d = (1.0 - (adaptive_cells / total_uniform_2d_cells)) * 100.0

    return {
        "finest_resolution_m": finest_res,
        "total_3d_voxels": total_3d_voxels,
        "mem_3d_mb": mem_3d_bytes / (1024 * 1024),
        "total_uniform_2d_cells": total_uniform_2d_cells,
        "mem_uniform_2d_mb": mem_uniform_2d_bytes / (1024 * 1024),
        "total_adaptive_cells": adaptive_cells,
        "mem_adaptive_mb": mem_adaptive_bytes / (1024 * 1024),
        "cell_savings_vs_3d_pct": cell_savings_vs_3d,
        "byte_savings_vs_3d_pct": byte_savings_vs_3d,
        "cell_savings_vs_uniform_2d_pct": cell_savings_vs_uniform_2d,
    }


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
        "Zone metrics (Evaluated Cloud):"
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

    # Add Problem Statement (PS-26053) Specification Memory Reduction Analysis (100m Range)
    spec_metrics = compute_memory_reduction_metrics(SPEC_ZONES_100M)
    lines.extend([
        "",
        "============================================================",
        "Problem Statement Specification (100m Range Memory Reduction)",
        "============================================================",
        "Perception Volume: 200m x 200m x 8m (Range: 100m)",
        "Foveated Resolution Zones:",
        "  - Near   ( 0 - 15m):  5 cm resolution (0.05m)",
        "  - Middle (15 - 35m): 20 cm resolution (0.20m)",
        "  - Far    (35 - 100m): 50 cm resolution (0.50m)",
        "",
        f"Equivalent Uniform 3D Voxel Grid (at 5cm): {spec_metrics['total_3d_voxels']:,} voxels ({spec_metrics['mem_3d_mb']:.1f} MB at 1B/voxel)",
        f"Equivalent Uniform 2.5D Grid (at 5cm)     : {spec_metrics['total_uniform_2d_cells']:,} cells ({spec_metrics['mem_uniform_2d_mb']:.1f} MB at 16B/cell)",
        f"FOVEAX Adaptive 2.5D Foveated Grid        : {spec_metrics['total_adaptive_cells']:,} cells ({spec_metrics['mem_adaptive_mb']:.1f} MB at 16B/cell)",
        "",
        f"Theoretical Cell Count Reduction vs Uniform 3D Voxels : {spec_metrics['cell_savings_vs_3d_pct']:.4f}%",
        f"Theoretical Memory Byte Savings vs Uniform 3D Voxels  : {spec_metrics['byte_savings_vs_3d_pct']:.2f}%",
        f"Theoretical Cell Count Reduction vs Uniform 2.5D Grid : {spec_metrics['cell_savings_vs_uniform_2d_pct']:.2f}%",
    ])
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {output_path}")


def parse_args(argv: list[str] | None = None):
    import argparse
    parser = argparse.ArgumentParser(description="FOVEAX Phase 4 — Adaptive 2.5D Mapping")
    parser.add_argument(
        "--mode", choices=["sample", "100m"], default="sample",
        help="Run mode: 'sample' for Open3D sample cloud, '100m' for 100m automotive spec (default: sample)."
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Optional input point cloud file path (.ply or .bin)."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    output_dir = Path("outputs/phase4")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "100m":
        zones = SPEC_ZONES_100M
        x_min, x_max = -100.0, 100.0
        y_min, y_max = -100.0, 100.0
        input_path = args.input if args.input else "data/semantic_kitti/dataset/sequences/00/velodyne/000000.bin"
    else:
        zones = SAMPLE_ZONES
        x_min, x_max = -3.0, 3.0
        y_min, y_max = -3.0, 3.0
        input_path = args.input if args.input else "outputs/filtered_sample_cloud.ply"

    if Path(input_path).suffix == ".bin":
        raw = np.fromfile(input_path, dtype=np.float32).reshape(-1, 4)
        points = raw[:, :3]
    else:
        points = load_points(input_path)

    print(f"Loaded points: {len(points):,} from {input_path}")
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

