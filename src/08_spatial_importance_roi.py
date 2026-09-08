from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def save_map(
    layer: np.ndarray,
    title: str,
    output_path: Path,
    cmap: str,
    label: str,
    vmin: float | None = None,
    vmax: float | None = None
) -> None:
    masked = np.ma.masked_invalid(layer)
    plt.figure(figsize=(8, 7))
    image = plt.imshow(
        masked,
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax
    )
    plt.colorbar(image, label=label)
    plt.title(title)
    plt.xlabel("X grid cell")
    plt.ylabel("Y grid cell")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved: {output_path}")

def create_distance_priority(
    grid_rows: int, grid_cols: int, resolution_m: float
) -> np.ndarray:
    """Higher score for cells closer to the map centre/sensor."""
    x = (np.arange(grid_cols) + 0.5) * resolution_m
    y = (np.arange(grid_rows) + 0.5) * resolution_m
    
    x = x - np.mean(x)
    y = y - np.mean(y)
    
    xx, yy = np.meshgrid(x, y)
    distance = np.sqrt(xx**2 + yy**2)
    max_distance = np.max(distance)
    
    if max_distance < 1e-8:
        return np.ones((grid_rows, grid_cols), dtype=np.float32)
        
    return (1.0 - distance / max_distance).astype(np.float32)

def main():
    phase2_path = Path("outputs/phase2/map_layers.npz")
    phase3_path = Path("outputs/phase3/terrain_layers.npz")
    
    output_dir = Path("outputs/phase5")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not phase2_path.exists():
        raise FileNotFoundError(
            "Missing outputs/phase2/map_layers.npz. Run Phase 2 first."
        )
    if not phase3_path.exists():
        raise FileNotFoundError(
            "Missing outputs/phase3/terrain_layers.npz. Run Phase 3 first."
        )
        
    phase2 = np.load(phase2_path)
    phase3 = np.load(phase3_path)
    
    point_count = phase2["point_count"].astype(np.float32)
    occupied = phase2["occupied"].astype(bool)
    
    slope_degrees = phase3["slope_degrees"].astype(np.float32)
    roughness = phase3["roughness"].astype(np.float32)
    step_height = phase3["step_height"].astype(np.float32)
    traversability = phase3["traversability"].astype(np.float32)
    
    grid_rows, grid_cols = occupied.shape
    
    # Must match src/05_create_2point5d_map.py.
    grid_resolution_m = 0.02
    
    # Initial terrain thresholds from Phase 3.
    max_safe_slope_deg = 15.0
    max_safe_roughness_m = 0.05
    max_safe_step_m = 0.15
    
    distance_priority = create_distance_priority(
        grid_rows, grid_cols, grid_resolution_m
    )
    
    slope_risk = np.clip(slope_degrees / max_safe_slope_deg, 0.0, 1.0)
    roughness_risk = np.clip(
        roughness / max_safe_roughness_m, 0.0, 1.0
    )
    step_risk = np.clip(step_height / max_safe_step_m, 0.0, 1.0)
    
    terrain_risk = (
        0.40 * slope_risk +
        0.30 * roughness_risk +
        0.30 * step_risk
    ).astype(np.float32)
    terrain_risk[~occupied] = np.nan
    
    # Low point density means lower geometric confidence.
    # Cells with zero points are unknown; cells with many points are confident.
    density_confidence = np.clip(
        np.log1p(point_count) / np.log(8.0), 0.0, 1.0
    ).astype(np.float32)
    uncertainty = 1.0 - density_confidence
    uncertainty[~occupied] = 1.0
    
    # Spatial Importance Score:
    # Distance = 25%
    # Terrain risk = 50%
    # Uncertainty = 25%
    #
    # In later stages, add:
    # planned path, object class, object motion, collision probability.
    importance_score = (
        0.25 * distance_priority +
        0.50 * np.nan_to_num(terrain_risk, nan=0.0) +
        0.25 * uncertainty
    ).astype(np.float32)
    
    # Unknown/unobserved cells remain visible as uncertain, but they
    # do not falsely become "safe."
    importance_score[~occupied] = np.nan
    
    # ROI classes:
    # 0 = background
    # 1 = important
    # 2 = critical
    # 3 = unknown
    roi_class = np.full(occupied.shape, 3, dtype=np.uint8)
    
    roi_class[occupied & (importance_score < 0.40)] = 0
    roi_class[
        occupied & (importance_score >= 0.40) & (importance_score < 0.70)
    ] = 1
    roi_class[occupied & (importance_score >= 0.70)] = 2
    
    # Demo resolution recommendation for the small Open3D sample cloud:
    # background = 10 cm
    # important = 5 cm
    # critical = 2 cm
    # unknown = 5 cm (requires caution / more observations)
    recommended_resolution = np.full(
        occupied.shape, np.nan, dtype=np.float32
    )
    
    recommended_resolution[roi_class == 0] = 0.10
    recommended_resolution[roi_class == 1] = 0.05
    recommended_resolution[roi_class == 2] = 0.02
    recommended_resolution[roi_class == 3] = 0.05
    
    print("FOVEAX Phase 5 — Spatial Importance & ROI Manager")
    print(f"Grid shape: {grid_rows} rows × {grid_cols} cols")
    print(f"Grid resolution: {grid_resolution_m:.3f} m")
    
    save_map(
        distance_priority,
        "FOVEAX ROI Manager — Distance Priority",
        output_dir / "distance_priority.png",
        "viridis",
        "Priority (1 = closest)"
    )
    
    save_map(
        terrain_risk,
        "FOVEAX ROI Manager — Terrain Risk",
        output_dir / "terrain_risk.png",
        "inferno",
        "Terrain risk (0 = low, 1 = high)",
        vmin=0,
        vmax=1
    )
    
    save_map(
        uncertainty,
        "FOVEAX ROI Manager — Observation Uncertainty",
        output_dir / "uncertainty_map.png",
        "cividis",
        "Uncertainty (0 = confident, 1 = uncertain)",
        vmin=0,
        vmax=1
    )
    
    save_map(
        importance_score,
        "FOVEAX ROI Manager — Spatial Importance Score",
        output_dir / "importance_score.png",
        "plasma",
        "Importance (0 = low, 1 = critical)",
        vmin=0,
        vmax=1
    )
    
    roi_colormap = plt.matplotlib.colors.ListedColormap(
        ["#3498db", "#f1c40f", "#e74c3c", "#7f8c8d"]
    )
    plt.figure(figsize=(8, 7))
    image = plt.imshow(
        roi_class,
        origin="lower",
        cmap=roi_colormap,
        interpolation="nearest",
        vmin=0,
        vmax=3
    )
    colorbar = plt.colorbar(image, ticks=[0, 1, 2, 3])
    colorbar.ax.set_yticklabels(
        ["Background", "Important", "Critical", "Unknown"]
    )
    plt.title("FOVEAX ROI Manager — Region Classes")
    plt.xlabel("X grid cell")
    plt.ylabel("Y grid cell")
    plt.tight_layout()
    plt.savefig(output_dir / "roi_classes.png", dpi=180)
    plt.close()
    print(f"Saved: {output_dir / 'roi_classes.png'}")
    
    resolution_colormap = plt.matplotlib.colors.ListedColormap(
        ["#e74c3c", "#f1c40f", "#3498db"]
    )
    resolution_display = np.full(occupied.shape, np.nan, dtype=np.float32)
    resolution_display[recommended_resolution == 0.02] = 0
    resolution_display[recommended_resolution == 0.05] = 1
    resolution_display[recommended_resolution == 0.10] = 2
    
    plt.figure(figsize=(8, 7))
    image = plt.imshow(
        np.ma.masked_invalid(resolution_display),
        origin="lower",
        cmap=resolution_colormap,
        interpolation="nearest",
        vmin=0,
        vmax=2
    )
    colorbar = plt.colorbar(image, ticks=[0, 1, 2])
    colorbar.ax.set_yticklabels(["2 cm", "5 cm", "10 cm"])
    plt.title("FOVEAX ROI Manager — Recommended Local Resolution")
    plt.xlabel("X grid cell")
    plt.ylabel("Y grid cell")
    plt.tight_layout()
    plt.savefig(
        output_dir / "recommended_resolution.png", dpi=180
    )
    plt.close()
    print(f"Saved: {output_dir / 'recommended_resolution.png'}")
    
    background_count = int(np.sum(roi_class == 0))
    important_count = int(np.sum(roi_class == 1))
    critical_count = int(np.sum(roi_class == 2))
    unknown_count = int(np.sum(roi_class == 3))
    
    valid_importance = importance_score[occupied]
    
    metrics_text = (
        "FOVEAX Phase 5 — Spatial Importance & ROI Manager\n"
        "\n"
        "Importance formula:\n"
        "0.25 × distance priority\n"
        "+ 0.50 × terrain risk\n"
        "+ 0.25 × observation uncertainty\n"
        "\n"
        f"Grid size: {grid_rows} rows × {grid_cols} cols\n"
        f"Occupied cells: {int(np.sum(occupied))}\n"
        f"Background cells: {background_count}\n"
        f"Important cells: {important_count}\n"
        f"Critical cells: {critical_count}\n"
        f"Unknown cells: {unknown_count}\n"
        f"Mean importance: {np.nanmean(valid_importance):.3f}\n"
        f"Maximum importance: {np.nanmax(valid_importance):.3f}\n"
        "\n"
        "Demo resolution policy:\n"
        "Critical = 2 cm\n"
        "Important = 5 cm\n"
        "Background = 10 cm\n"
        "Unknown = 5 cm + caution\n"
    )
    metrics_path = output_dir / "phase5_metrics.txt"
    metrics_path.write_text(metrics_text, encoding="utf-8")
    
    print("\n" + metrics_text)
    print(f"Saved: {metrics_path}")
    
    np.savez_compressed(
        output_dir / "roi_layers.npz",
        distance_priority=distance_priority,
        terrain_risk=terrain_risk,
        uncertainty=uncertainty,
        importance_score=importance_score,
        roi_class=roi_class,
        recommended_resolution=recommended_resolution,
        occupied=occupied
    )
    print(f"Saved: {output_dir / 'roi_layers.npz'}")

if __name__ == "__main__":
    main()
