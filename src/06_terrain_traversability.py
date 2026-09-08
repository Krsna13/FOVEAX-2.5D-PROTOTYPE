from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import generic_filter


def nan_std(values: np.ndarray) -> float:
    """Local standard deviation ignoring NaN, requires >=3 valid values."""
    valid = values[~np.isnan(values)]
    return float(np.std(valid)) if len(valid) >= 3 else np.nan


def nan_max_difference(values: np.ndarray) -> float:
    """Max absolute difference from the center cell to any valid neighbour."""
    center = values[len(values) // 2]
    if np.isnan(center):
        return np.nan
    valid = values[~np.isnan(values)]
    if len(valid) < 2:
        return np.nan
    return float(np.max(np.abs(valid - center)))


def save_map(
    layer: np.ndarray,
    title: str,
    output_path: Path,
    cmap: str,
    label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    masked = np.ma.masked_invalid(layer)
    plt.figure(figsize=(8, 7))
    image = plt.imshow(
        masked,
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(image, label=label)
    plt.title(title)
    plt.xlabel("X grid cell")
    plt.ylabel("Y grid cell")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    input_path = Path("outputs/phase2/map_layers.npz")
    output_dir = Path("outputs/phase3")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Phase 2 map file not found: {input_path}. "
            "Run src/05_create_2point5d_map.py first."
        )

    maps = np.load(input_path)
    z_mean = maps["z_mean"].astype(np.float32)
    point_count = maps["point_count"].astype(np.float32)
    occupied = maps["occupied"].astype(bool)

    print("Loaded Phase 2 2.5D map.")
    print(f"Grid shape: {z_mean.shape} (rows, cols)")
    print(f"Occupied cells: {int(np.sum(occupied)):,}")

    # Must match the resolution used in src/05_create_2point5d_map.py
    grid_resolution_m = 0.02

    # Unknown cells remain NaN throughout terrain analysis.
    elevation = z_mean.copy()
    elevation[~occupied] = np.nan

    if np.all(np.isnan(elevation)):
        raise ValueError("Elevation map has no occupied cells.")

    # Fill NaNs temporarily for gradient computation.
    # Acceptable for the Phase 3 prototype; will be replaced by proper
    # ground interpolation / Bayesian elevation fusion later.
    elevation_filled = elevation.copy()
    fill_value = np.nanmedian(elevation_filled)
    elevation_filled[np.isnan(elevation_filled)] = fill_value

    gradient_y, gradient_x = np.gradient(
        elevation_filled,
        grid_resolution_m,
        grid_resolution_m,
    )

    slope_degrees = np.degrees(
        np.arctan(np.sqrt(gradient_x**2 + gradient_y**2))
    ).astype(np.float32)
    slope_degrees[~occupied] = np.nan

    roughness = generic_filter(
        elevation,
        nan_std,
        size=5,
        mode="constant",
        cval=np.nan,
    ).astype(np.float32)
    roughness[~occupied] = np.nan

    step_height = generic_filter(
        elevation,
        nan_max_difference,
        size=3,
        mode="constant",
        cval=np.nan,
    ).astype(np.float32)
    step_height[~occupied] = np.nan

    # Initial prototype thresholds — tune for the target vehicle later.
    max_safe_slope_deg = 15.0
    max_safe_roughness_m = 0.05
    max_safe_step_m = 0.15

    slope_risk = np.clip(slope_degrees / max_safe_slope_deg, 0.0, 1.0)
    roughness_risk = np.clip(roughness / max_safe_roughness_m, 0.0, 1.0)
    step_risk = np.clip(step_height / max_safe_step_m, 0.0, 1.0)

    total_risk = 0.40 * slope_risk + 0.30 * roughness_risk + 0.30 * step_risk
    traversability = (1.0 - total_risk).astype(np.float32)
    traversability[~occupied] = np.nan

    # Class codes: 0 = safe, 1 = caution, 2 = blocked, 3 = unknown
    terrain_class = np.full(traversability.shape, 3, dtype=np.uint8)
    terrain_class[(traversability >= 0.70) & occupied] = 0
    terrain_class[
        (traversability >= 0.40) & (traversability < 0.70) & occupied
    ] = 1
    terrain_class[(traversability < 0.40) & occupied] = 2

    # --- Save layer images ---
    save_map(
        slope_degrees,
        "FOVEAX Terrain Map — Slope",
        output_dir / "slope_degrees.png",
        "turbo",
        "Slope (degrees)",
        vmin=0,
        vmax=max_safe_slope_deg * 2,
    )

    save_map(
        roughness,
        "FOVEAX Terrain Map — Roughness",
        output_dir / "roughness.png",
        "magma",
        "Local elevation std dev (m)",
        vmin=0,
        vmax=max_safe_roughness_m * 2,
    )

    save_map(
        step_height,
        "FOVEAX Terrain Map — Step Height",
        output_dir / "step_height.png",
        "inferno",
        "Max local height difference (m)",
        vmin=0,
        vmax=max_safe_step_m * 2,
    )

    save_map(
        traversability,
        "FOVEAX Terrain Map — Traversability Score",
        output_dir / "traversability_score.png",
        "RdYlGn",
        "Traversability (0 = blocked, 1 = safe)",
        vmin=0,
        vmax=1,
    )

    class_colormap = plt.matplotlib.colors.ListedColormap(
        ["#2ecc71", "#f1c40f", "#e74c3c", "#6c757d"]
    )
    class_labels = ["Safe", "Caution", "Blocked", "Unknown"]

    plt.figure(figsize=(8, 7))
    image = plt.imshow(
        terrain_class,
        origin="lower",
        cmap=class_colormap,
        interpolation="nearest",
        vmin=0,
        vmax=3,
    )
    cbar = plt.colorbar(image, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(class_labels)
    plt.title("FOVEAX Terrain Map — Traversability Classes")
    plt.xlabel("X grid cell")
    plt.ylabel("Y grid cell")
    plt.tight_layout()
    plt.savefig(output_dir / "traversability_classes.png", dpi=180)
    plt.close()
    print(f"Saved: {output_dir / 'traversability_classes.png'}")

    # --- Save terrain layers ---
    np.savez_compressed(
        output_dir / "terrain_layers.npz",
        slope_degrees=slope_degrees,
        roughness=roughness,
        step_height=step_height,
        traversability=traversability,
        terrain_class=terrain_class,
        occupied=occupied,
        z_mean=z_mean,
        point_count=point_count,
    )
    print(f"Saved: {output_dir / 'terrain_layers.npz'}")

    # --- Write metrics summary ---
    valid = occupied
    n_valid = int(np.sum(valid))
    n_total = valid.size

    def stats(arr: np.ndarray) -> dict:
        v = arr[valid]
        return {
            "mean": float(np.nanmean(v)),
            "median": float(np.nanmedian(v)),
            "min": float(np.nanmin(v)),
            "max": float(np.nanmax(v)),
            "p10": float(np.nanpercentile(v, 10)),
            "p90": float(np.nanpercentile(v, 90)),
        }

    s_slope = stats(slope_degrees)
    s_rough = stats(roughness)
    s_step = stats(step_height)
    s_trav = stats(traversability)

    n_safe = int(np.sum(terrain_class == 0))
    n_caution = int(np.sum(terrain_class == 1))
    n_blocked = int(np.sum(terrain_class == 2))
    n_unknown = int(np.sum(terrain_class == 3))

    lines = [
        "FOVEAX Phase 3 — Terrain Analysis Metrics",
        "=" * 48,
        "",
        f"Grid: {n_total} cells ({z_mean.shape[0]} x {z_mean.shape[1]})",
        f"Resolution: {grid_resolution_m:.3f} m/cell",
        f"Occupied cells: {n_valid:,} ({100 * n_valid / n_total:.2f}%)",
        "",
        "--- Slope (degrees) ---",
        f"  Mean: {s_slope['mean']:.3f}",
        f"  Median: {s_slope['median']:.3f}",
        f"  Min / Max: {s_slope['min']:.3f} / {s_slope['max']:.3f}",
        f"  P10 / P90: {s_slope['p10']:.3f} / {s_slope['p90']:.3f}",
        f"  Max safe threshold: {max_safe_slope_deg:.1f} deg",
        "",
        "--- Roughness (m, local std dev) ---",
        f"  Mean: {s_rough['mean']:.4f}",
        f"  Median: {s_rough['median']:.4f}",
        f"  Min / Max: {s_rough['min']:.4f} / {s_rough['max']:.4f}",
        f"  P10 / P90: {s_rough['p10']:.4f} / {s_rough['p90']:.4f}",
        f"  Max safe threshold: {max_safe_roughness_m:.3f} m",
        "",
        "--- Step Height (m, max neighbour diff) ---",
        f"  Mean: {s_step['mean']:.4f}",
        f"  Median: {s_step['median']:.4f}",
        f"  Min / Max: {s_step['min']:.4f} / {s_step['max']:.4f}",
        f"  P10 / P90: {s_step['p10']:.4f} / {s_step['p90']:.4f}",
        f"  Max safe threshold: {max_safe_step_m:.3f} m",
        "",
        "--- Traversability Score ---",
        f"  Mean: {s_trav['mean']:.4f}",
        f"  Median: {s_trav['median']:.4f}",
        f"  Min / Max: {s_trav['min']:.4f} / {s_trav['max']:.4f}",
        f"  P10 / P90: {s_trav['p10']:.4f} / {s_trav['p90']:.4f}",
        "",
        "--- Class Breakdown ---",
        f"  Safe (0):     {n_safe:>6} ({100 * n_safe / n_valid:5.2f}%)  #2ecc71",
        f"  Caution (1):  {n_caution:>6} ({100 * n_caution / n_valid:5.2f}%)  #f1c40f",
        f"  Blocked (2):  {n_blocked:>6} ({100 * n_blocked / n_valid:5.2f}%)  #e74c3c",
        f"  Unknown (3):  {n_unknown:>6} ({(n_unknown / n_total) * 100:5.2f}% of grid)  #6c757d",
        "",
        "--- Weights ---",
        f"  Slope:      0.40",
        f"  Roughness:  0.30",
        f"  Step:       0.30",
        "",
        "Note: These thresholds and weights are prototype defaults.",
        "Tune them for the target vehicle before field use.",
    ]

    metrics_path = output_dir / "phase3_metrics.txt"
    metrics_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {metrics_path}")

    print("\n--- Class breakdown ---")
    print(f"  Safe:     {n_safe:>6} ({100 * n_safe / n_valid:5.2f}%)")
    print(f"  Caution:  {n_caution:>6} ({100 * n_caution / n_valid:5.2f}%)")
    print(f"  Blocked:  {n_blocked:>6} ({100 * n_blocked / n_valid:5.2f}%)")
    print(f"  Unknown:  {n_unknown:>6} ({(n_unknown / n_total) * 100:5.2f}% of grid)")


if __name__ == "__main__":
    main()
