"""FOVEAX Phase 9 Dashboard Snapshot Exporter.

Generates high-resolution publication-quality visualization snapshots of the
dual 2D grid panels (Elevation, Traversability, ROI) and tracked object
overlays for both SemanticKITTI and RELLIS-3D.

Saves:
    - outputs/phase9/dashboard_semantickitti_overview.png
    - outputs/phase9/dashboard_rellis3d_overview.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dashboard.data_streamer import DataStreamerThread
from src.perception.object_detector import MockObjectDetector


def render_dashboard_snapshot(
    source: str,
    sequence: str,
    frame_idx: int,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Render a comprehensive dashboard snapshot."""
    streamer = DataStreamerThread(
        source=source,
        sequence=sequence,
        num_frames=frame_idx + 1,
        rate_hz=10.0,
    )
    streamer._setup_sources()

    state = streamer.process_frame(frame_idx, timestamp_s=0.1 * frame_idx)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#1e1e2e")
    fig.suptitle(
        f"FOVEAX Real-Time Perception Dashboard -- {source.upper()} [{sequence} Frame {frame_idx:04d}] {title_suffix}",
        fontsize=16,
        fontweight="bold",
        color="#cdd6f4",
        y=0.98,
    )

    extent = streamer.grid_extent  # x_min, x_max, y_min, y_max

    # 1. Elevation Map
    ax0 = axes[0]
    ax0.set_facecolor("#11111b")
    elev = state.grid_maps.get("elevation")
    if elev is not None and not np.all(np.isnan(elev)):
        im0 = ax0.imshow(
            elev,
            extent=[extent[0], extent[1], extent[2], extent[3]],
            cmap="turbo",
            origin="upper",
        )
        cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
        cbar0.set_label("Elevation Z (m)", color="#cdd6f4")
        cbar0.ax.yaxis.set_tick_params(color="#cdd6f4")
        plt.setp(plt.getp(cbar0.ax.axes, "yticklabels"), color="#cdd6f4")
    ax0.set_title("2.5D Elevation Map (Turbo)", color="#89b4fa", fontsize=12, fontweight="bold")
    ax0.set_xlabel("X (Forward, m)", color="#cdd6f4")
    ax0.set_ylabel("Y (Lateral, m)", color="#cdd6f4")
    ax0.tick_params(colors="#a6adc8")

    # 2. Traversability Map (Canonical convention: 1.0 safe = Green, 0.0 blocked = Red)
    ax1 = axes[1]
    ax1.set_facecolor("#11111b")
    trav = state.grid_maps.get("traversability")
    if trav is not None and not np.all(np.isnan(trav)):
        im1 = ax1.imshow(
            trav,
            extent=[extent[0], extent[1], extent[2], extent[3]],
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            origin="upper",
        )
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label("Traversability Score (1.0=Safe, 0.0=Blocked)", color="#cdd6f4")
        cbar1.ax.yaxis.set_tick_params(color="#cdd6f4")
        plt.setp(plt.getp(cbar1.ax.axes, "yticklabels"), color="#cdd6f4")

    ax1.set_title("Terrain Traversability (RdYlGn)", color="#a6e3a1", fontsize=12, fontweight="bold")
    ax1.set_xlabel("X (Forward, m)", color="#cdd6f4")
    ax1.tick_params(colors="#a6adc8")

    # 3. ROI & Tracked Objects Map
    ax2 = axes[2]
    ax2.set_facecolor("#11111b")
    roi = state.grid_maps.get("roi")
    if roi is not None and not np.all(np.isnan(roi)):
        im2 = ax2.imshow(
            roi,
            extent=[extent[0], extent[1], extent[2], extent[3]],
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            origin="upper",
        )
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label("ROI Importance Weight", color="#cdd6f4")
        cbar2.ax.yaxis.set_tick_params(color="#cdd6f4")
        plt.setp(plt.getp(cbar2.ax.axes, "yticklabels"), color="#cdd6f4")

    # Overlay tracked object positions
    for t in state.tracks:
        tx, ty = t.state[0], t.state[1]
        color = "#f38ba8" if t.dynamic else "#fab387"
        ax2.plot(tx, ty, "o", color=color, markersize=8)
        ax2.text(
            tx + 0.8,
            ty + 0.8,
            f"ID:{t.track_id}\n{np.linalg.norm(t.state[3:5]):.1f}m/s",
            color=color,
            fontsize=8,
            fontweight="bold",
        )

    ax2.set_title("Dynamic ROI & Kalman Tracks", color="#f38ba8", fontsize=12, fontweight="bold")
    ax2.set_xlabel("X (Forward, m)", color="#cdd6f4")
    ax2.tick_params(colors="#a6adc8")

    # Telemetry banner
    stats_text = (
        f"Points: {len(state.points):,d}  |  Tracks: {len(state.tracks)}  |  "
        f"FPS: {state.metrics.fps:.1f}  |  Latency: {state.metrics.latency_ms:.1f}ms  |  "
        f"CPU: {state.metrics.cpu_percent:.1f}%  |  RAM: {state.metrics.ram_used_gb:.1f}/{state.metrics.ram_total_gb:.1f} GB"
    )
    fig.text(
        0.5,
        0.02,
        stats_text,
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#fab387",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#313244", edgecolor="#45475a"),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.94])
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"[SUCCESS] Dashboard snapshot saved: {output_path}")


def main():
    out_dir = _PROJECT_ROOT / "outputs/phase9"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. SemanticKITTI snapshot
    render_dashboard_snapshot(
        source="semantickitti",
        sequence="00",
        frame_idx=0,
        output_path=out_dir / "dashboard_semantickitti_overview.png",
        title_suffix="(Velodyne HDL-64E)",
    )

    # 2. RELLIS-3D snapshot
    render_dashboard_snapshot(
        source="rellis3d",
        sequence="00000",
        frame_idx=0,
        output_path=out_dir / "dashboard_rellis3d_overview.png",
        title_suffix="(Ouster OS1-64 Trail)",
    )


if __name__ == "__main__":
    main()
