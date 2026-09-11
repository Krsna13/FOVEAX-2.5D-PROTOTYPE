"""FOVEAX Phase 8A — 3D object detection and multi-object tracking.

Demonstrates the model-agnostic detection and tracking pipeline built in
Phase 8A.  Supports:

    1. Synthetic sample mode (--source sample): a synthetic point cloud with
       a moving obstacle cluster for tracking pipeline validation.
    2. SemanticKITTI mode (--source semantickitti): reads .bin frames if
       available on disk.

    ⚠️  The mock geometric detector is NOT an AI model.  It performs
    deterministic Euclidean clustering of points above a ground threshold and
    assigns class_name="unknown_obstacle".  Use it only to validate the tracking
    pipeline, not for safety claims.

Usage
-----
    python src/11_object_detection_tracking.py --source sample --detector mock --frames 10
    python src/11_object_detection_tracking.py --source semantickitti --sequence 00 --start-frame 000000 --frames 5
"""

from __future__ import annotations

import json
import sys
import time
import argparse
import textwrap
from pathlib import Path
from typing import Any

# --- Resolve project root so the import works from any cwd ---------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure robust UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.perception.object_detector import (  # noqa: E402
    Detection3D,
    GroundTruthBoxProvider,
    MockObjectDetector,
    ObjectDetector,
    OpenPCDetDetector,
    validate_points_xyzi,
)
from src.perception.grid_overlay import generate_dynamic_object_overlay  # noqa: E402
from src.tracking.multi_object_tracker import (  # noqa: E402
    MultiObjectTracker,
    TrackState,
    _TRACKER_DEFAULTS,
)



# ============================================================================
# Data loading
# ============================================================================

def load_open3d_sample_cloud() -> np.ndarray:
    """Load a sample point cloud via Open3D and return Nx4 float32.

    Uses the EaglePointCloud sample if available; falls back to generating
    a synthetic knot-like cloud.

    Returns
    -------
    np.ndarray, shape (N, 4), dtype float32
        Columns: [x, y, z, intensity].
    """
    try:
        import open3d as o3d
        cloud = o3d.data.EaglePointCloud()
        pcd = o3d.io.read_point_cloud(cloud.path)
        arr = np.asarray(pcd.points, dtype=np.float64)
    except Exception:
        # Generate a synthetic torus knot cloud as a fallback.
        n = 5000
        rng = np.random.default_rng(42)
        t = np.linspace(0, 4 * np.pi, n)
        R = 3.0
        r = 1.0
        x = (R + r * np.cos(2 * t)) * np.cos(t)
        y = (R + r * np.cos(2 * t)) * np.sin(t)
        z = r * np.sin(2 * t) + rng.normal(0, 0.05, n)
        arr = np.column_stack((x, y, z))
        arr = arr - arr.mean(axis=0)

    arr = arr - arr.mean(axis=0)
    arr = arr * 2.0
    n = arr.shape[0]
    points = np.column_stack(
        (arr, np.full(n, 1.0, dtype=np.float64))
    ).astype(np.float32)
    return points


def load_semantickitti_frame(
    sequence_dir: Path,
    frame_id: str,
) -> np.ndarray:
    """Load a SemanticKITTI Velodyne .bin scan.

    Parameters
    ----------
    sequence_dir : Path
        Path to the sequence directory containing a ``velodyne/`` subfolder.
    frame_id : str
        Frame ID, e.g. "000000".

    Returns
    -------
    np.ndarray, shape (N, 4), dtype float32
        Columns: [x, y, z, intensity].
    """
    velodyne_dir = sequence_dir / "velodyne"
    point_path = velodyne_dir / f"{frame_id}.bin"
    if not point_path.exists():
        raise FileNotFoundError(f"Velodyne scan not found: {point_path}")
    raw = np.fromfile(point_path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(
            f"Point file size ({raw.size}) is not divisible by 4."
        )
    return raw.reshape(-1, 4)


def repeated_cloud_with_translation(
    base: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Return a copy of *base* translated by *translation*."""
    out = base.copy()
    out[:, 0] += float(translation[0])
    out[:, 1] += float(translation[1])
    out[:, 2] += float(translation[2])
    return out


# ============================================================================
# Synthetic sample generation
# ============================================================================

def generate_synthetic_sample_cloud(
    n_static: int = 2000,
    n_moving: int = 150,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic point cloud with a moving obstacle cluster.

    Parameters
    ----------
    n_static : int
        Number of static background points.
    n_moving : int
        Number of points in the moving obstacle cluster.
    seed : int
        RNG seed for deterministic generation.

    Returns
    -------
    static_points : np.ndarray, shape (n_static, 4), float32
        Static background points [x, y, z, intensity].
    moving_points : np.ndarray, shape (n_moving, 4), float32
        Moving obstacle points at t=0, before any translation.
    """
    rng = np.random.default_rng(seed)

    # Static background: spread over a 20x20m area, BELOW ground.
    static_x = rng.uniform(-10, 10, n_static)
    static_y = rng.uniform(-10, 10, n_static)
    static_z = rng.uniform(-0.5, 0.1, n_static)  # mostly below ground
    static_i = rng.uniform(0.1, 0.8, n_static)
    static_points = np.column_stack([
        static_x, static_y, static_z, static_i
    ]).astype(np.float32)

    # Moving obstacle: a compact cluster at the origin, ABOVE ground.
    # Spread enough to survive voxel downsampling (voxel_size=0.2m) and
    # remain connected under BFS with max_gap=0.75m.
    # With std=0.9 the cluster spans ~3m which gives ~15x15 voxels
    # before filtering, ensuring >100 points survive voxel downsampling.
    moving_x = rng.normal(0, 0.9, n_moving)
    moving_y = rng.normal(0, 0.9, n_moving)
    moving_z = rng.normal(1.0, 0.2, n_moving)
    moving_i = rng.uniform(0.5, 1.0, n_moving)
    moving_points = np.column_stack([
        moving_x, moving_y, moving_z, moving_i
    ]).astype(np.float32)

    return static_points, moving_points


# ============================================================================
# BEV rendering
# ============================================================================

def render_bev_image(
    points: np.ndarray,
    detections: list[Detection3D],
    tracks: list[TrackState],
    frame_idx: int,
    output_path: Path,
    source: str,
    detector_label: str = "MOCK GEOMETRIC BASELINE — NOT AI DETECTION",
) -> None:
    """Render a BEV (top-down) image and save to *output_path*.

    Layout:
        - LiDAR points in grey
        - Detection boxes in orange
        - Tracked boxes in blue
        - Arrows for velocity direction
        - Label each box: Track ID, class, confidence, speed

    Parameters
    ----------
    points : np.ndarray, shape (N, 4), float32
    detections : list[Detection3D]
    tracks : list[TrackState]
    frame_idx : int
    output_path : Path
    source : str
        Source name for title.
    detector_label : str
        Provenance label for plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # LiDAR points in grey.
    ax.scatter(
        points[:, 0], points[:, 1],
        c="lightgrey", s=0.3, alpha=0.5, marker=".",
        label="LiDAR points"
    )

    # Detection boxes in orange.
    for d in detections:
        cx, cy, cz = d.center_xyz
        l, w, h = d.size_lwh
        yaw = d.yaw_rad
        # Draw rectangle.
        theta = yaw
        corners = np.array([
            [-l/2, -w/2],
            [ l/2, -w/2],
            [ l/2,  w/2],
            [-l/2,  w/2],
            [-l/2, -w/2],
        ])
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ])
        rotated = corners @ rot.T + np.array([cx, cy])
        ax.plot(rotated[:, 0], rotated[:, 1], "orange", linewidth=1.5, alpha=0.8)
        ax.fill(rotated[:, 0], rotated[:, 1], "orange", alpha=0.15)

    # Tracked boxes in blue with velocity arrows.
    for t in tracks:
        cx, cy, cz = t.state[0], t.state[1], t.state[2]
        l, w, h = t.size_lwh
        yaw = t.yaw_rad
        speed = np.linalg.norm(t.state[3:6])
        vx, vy, vz = t.state[3], t.state[4], t.state[5]

        # Rectangle.
        theta = yaw
        corners = np.array([
            [-l/2, -w/2],
            [ l/2, -w/2],
            [ l/2,  w/2],
            [-l/2,  w/2],
            [-l/2, -w/2],
        ])
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ])
        rotated = corners @ rot.T + np.array([cx, cy])
        color = "blue" if t.dynamic else "purple"
        ax.plot(rotated[:, 0], rotated[:, 1], color, linewidth=2.0, alpha=0.9)
        ax.fill(rotated[:, 0], rotated[:, 1], color, alpha=0.20)

        # Velocity arrow.
        if speed > 0.01:
            ax.annotate(
                "",
                xy=(cx + vx * 0.3, cy + vy * 0.3),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2.0),
            )

        # Label: Track ID, class, confidence, speed.
        label = (
            f"ID {t.track_id} | {t.class_name}\n"
            f"conf {t.confidence:.2f} | speed {speed:.2f} m/s"
        )
        ax.text(
            cx, cy + 0.4,
            label,
            fontsize=7,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"FOVEAX Phase 8 — BEV | {source} | frame {frame_idx:04d}\n[{detector_label}]")

    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ============================================================================
# JSONL output helpers
# ============================================================================

def detection_to_dict(d: Detection3D, frame_idx: int, timestamp_s: float) -> dict:
    """Convert a Detection3D to a JSON-serialisable dict."""
    return {
        "frame_idx": frame_idx,
        "timestamp_s": round(timestamp_s, 6),
        "class_id": d.class_id,
        "class_name": d.class_name,
        "confidence": round(float(d.confidence), 6),
        "center_xyz": [round(float(x), 4) for x in d.center_xyz],
        "size_lwh": [round(float(x), 4) for x in d.size_lwh],
        "yaw_rad": round(float(d.yaw_rad), 6),
        "source": d.source,
        "metadata": d.metadata,
    }


def track_to_dict(t: TrackState, frame_idx: int, timestamp_s: float) -> dict:
    """Convert a TrackState to a JSON-serialisable dict."""
    speed = float(np.linalg.norm(t.state[3:6]))
    return {
        "frame_idx": frame_idx,
        "timestamp_s": round(timestamp_s, 6),
        "track_id": int(t.track_id),
        "class_name": str(t.class_name),
        "confidence": round(float(t.confidence), 6),
        "state": [round(float(x), 4) for x in t.state],
        "covariance_diag": [round(float(x), 4) for x in np.diag(t.covariance)],
        "size_lwh": [round(float(x), 4) for x in t.size_lwh],
        "yaw_rad": round(float(t.yaw_rad), 6),
        "speed_mps": round(speed, 4),
        "velocity_xyz": [round(float(x), 4) for x in t.state[3:6]],
        "age_frames": int(t.age_frames),
        "hits": int(t.hits),
        "missed_frames": int(t.missed_frames),
        "dynamic": bool(t.dynamic),
        "source": str(t.source),
    }


# ============================================================================
# Metrics
# ============================================================================

def write_metrics(
    source: str,
    n_frames: int,
    avg_input_points: float,
    avg_detections: float,
    active_tracks_final: int,
    dynamic_tracks_final: int,
    avg_detection_time: float,
    avg_tracking_time: float,
    avg_total_time: float,
    output_dir: Path,
) -> str:
    """Write phase8_metrics.txt and return the text."""
    lines = [
        "=" * 60,
        f"FOVEAX Phase 8A — Object Detection & Tracking Metrics",
        f"Source: {source}",
        "=" * 60,
        "",
        f"Total frames            : {n_frames}",
        f"Average input points/frame: {avg_input_points:,.1f}",
        f"Average mock detections/frame: {avg_detections:.2f}",
        f"Active tracks (final frame): {active_tracks_final}",
        f"Dynamic tracks (final frame): {dynamic_tracks_final}",
        "",
        f"Average detection time  : {avg_detection_time * 1000:.3f} ms",
        f"Average tracking time    : {avg_tracking_time * 1000:.3f} ms",
        f"Average total time/frame : {avg_total_time * 1000:.3f} ms",
        "",
        "NOTE: MockObjectDetector is a deterministic GEOMETRIC BASELINE,",
        "NOT an AI object detector.  Do not use these results for",
        "safety claims.  Phase 8B will integrate PointPillars or",
        "CenterPoint via OpenPCDet for real AI detections.",
        "",
        f"Output directory: {output_dir}",
    ]
    text = "\n".join(lines)
    (output_dir / "phase8_metrics.txt").write_text(text, encoding="utf-8")
    return text


def log_runtime_benchmark_to_report(
    detector_name: str,
    device_name: str,
    detection_times: list[float],
    report_path: Path = Path("outputs/phase8/model_environment_report.txt"),
) -> str:
    """Log measured per-frame inference latency and throughput (Task K).

    Appends or updates the runtime benchmark section in
    outputs/phase8/model_environment_report.txt.
    Does not claim real-time capability; reports measured numbers only.

    Parameters
    ----------
    detector_name : str
        Name of the detector class / architecture.
    device_name : str
        Execution device (e.g. CPU or CUDA).
    detection_times : list[float]
        List of per-frame detection times in seconds.
    report_path : Path
        Path to environment report.

    Returns
    -------
    str
        Summary benchmark text.
    """
    if not detection_times:
        return ""

    times_arr = np.array(detection_times, dtype=np.float64) * 1000.0  # to ms
    mean_ms = float(np.mean(times_arr))
    min_ms = float(np.min(times_arr))
    max_ms = float(np.max(times_arr))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    benchmark_text = (
        "=== Runtime Benchmark Measurements ===\n"
        f"Detector Architecture: {detector_name}\n"
        f"Execution Device     : {device_name}\n"
        f"Frames Evaluated     : {len(detection_times)}\n"
        f"Inference Latency    : mean={mean_ms:.2f} ms | min={min_ms:.2f} ms | max={max_ms:.2f} ms\n"
        f"Throughput           : {fps:.2f} FPS\n"
        "NOTE: Measured benchmark numbers only. No real-time capability claims made.\n"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    current_content = report_path.read_text(encoding="utf-8") if report_path.exists() else ""

    import datetime
    run_separator = f"\n{'='*70}\nRun timestamp: {datetime.datetime.now().isoformat()}\n{'='*70}\n"
    new_content = current_content.rstrip() + "\n" + run_separator + benchmark_text

    report_path.write_text(new_content, encoding="utf-8")
    return benchmark_text


def render_summary_image(
    frames_dir: Path,
    output_path: Path,
    source: str,
    n_frames: int,
) -> None:
    """Create a grid summary of all BEV frames."""
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        return

    n_cols = min(5, len(frame_files))
    n_rows = (len(frame_files) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, ax in enumerate(axes.flat):
        if idx < len(frame_files):
            img = plt.imread(frame_files[idx])
            ax.imshow(img)
            ax.set_title(f"Frame {idx:04d}", fontsize=9)
        else:
            ax.axis("off")
        ax.axis("off")

    fig.suptitle(
        f"FOVEAX Phase 8A — Summary | {source}\n[MOCK GEOMETRIC BASELINE — NOT AI DETECTION]",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ============================================================================
# Sample mode runner
# ============================================================================

def run_sample_mode(
    detector: ObjectDetector | None,
    tracker: MultiObjectTracker,
    n_frames: int,
    dt: float,
    ground_threshold: float,
    cluster_distance: float,
    min_cluster_points: int,
    track_gate: float,
    dynamic_speed_threshold: float,
    output_dir: Path,
    save_overlay: bool = True,
) -> None:
    """Run synthetic sample mode with a moving obstacle.

    Creates a synthetic point cloud with a moving obstacle cluster that
    translates across frames.  Saves BEV images, JSONL outputs, and metrics.

    ⚠️  This is synthetic test data only.  The sample cloud movement is NOT
    real detection performance.
    """
    print("=" * 70)
    print("FOVEAX Phase 8A — Sample Mode (Synthetic Test Only)")
    print("=" * 70)
    print()
    print("[WARNING] SYNTHETIC TEST ONLY - NOT REAL DETECTION PERFORMANCE")
    print()

    # Generate synthetic data.
    static_pts, moving_pts = generate_synthetic_sample_cloud(
        n_static=2000, n_moving=150
    )
    print(f"Generated synthetic cloud: {len(static_pts)} static + {len(moving_pts)} moving points")

    # Create output directories.
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    detections_jsonl = output_dir / "detections.jsonl"
    tracks_jsonl = output_dir / "tracks.jsonl"

    # Use provided detector or build mock
    if detector is not None:
        det = detector
    else:
        det = MockObjectDetector(
            ground_threshold_m=ground_threshold,
            cluster_min_points=min_cluster_points,
            cluster_max_dist_m=cluster_distance,
        )

    if isinstance(det, MockObjectDetector):
        detector_label = "MOCK GEOMETRIC BASELINE — NOT AI DETECTION"
    else:
        detector_label = f"{type(det).__name__} (AI Detection)"

    # Rebuild tracker with requested parameters.
    trk = MultiObjectTracker(
        gating_threshold_m=track_gate,
        max_missed_frames=_TRACKER_DEFAULTS["max_missed_frames"],
        min_hits_for_dynamic=_TRACKER_DEFAULTS["min_hits_for_dynamic"],
        dynamic_speed_threshold_mps=dynamic_speed_threshold,
        measure_noise_std=_TRACKER_DEFAULTS["measure_noise_std"],
        process_noise_std=_TRACKER_DEFAULTS["process_noise_std"],
    )

    # Clear previous output files.
    if detections_jsonl.exists():
        detections_jsonl.unlink()
    if tracks_jsonl.exists():
        tracks_jsonl.unlink()

    total_detection_time = 0.0
    total_tracking_time = 0.0
    total_points = 0
    total_detections = 0
    detection_times: list[float] = []

    translation_step = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    current_time = 0.0

    for frame_idx in range(n_frames):
        # Build frame: static + translated moving points.
        offset = translation_step * frame_idx
        frame_moving = repeated_cloud_with_translation(moving_pts, offset)
        points = np.concatenate([static_pts, frame_moving], axis=0)

        total_points += len(points)

        # 1. Detect.
        t0 = time.perf_counter()
        detections = det.detect(points, timestamp_s=current_time)
        t1 = time.perf_counter()
        detection_time = t1 - t0
        total_detection_time += detection_time
        detection_times.append(detection_time)

        # 2. Track.
        t2 = time.perf_counter()
        tracks = trk.update(detections, timestamp_s=current_time)
        t3 = time.perf_counter()
        tracking_time = t3 - t2
        total_tracking_time += tracking_time

        total_detections += len(detections)

        # 3. Save BEV image.
        frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
        render_bev_image(
            points, detections, tracks, frame_idx, frame_path, source="sample", detector_label=detector_label
        )

        # 3b. Task J: Dynamic-object overlay onto Phase 3 & 5 grids
        if save_overlay:
            overlay_dir = output_dir / "dynamic_object_overlay"
            dynamic_targets = [t for t in tracks if t.dynamic]
            if not dynamic_targets:
                dynamic_targets = detections
            generate_dynamic_object_overlay(
                points_xyzi=points,
                dynamic_objects=dynamic_targets,
                output_dir=overlay_dir,
                frame_idx=frame_idx,
            )

        # 4. Write JSONL.
        with open(detections_jsonl, "a", encoding="utf-8") as f:
            for d in detections:
                f.write(json.dumps(detection_to_dict(d, frame_idx, current_time)) + "\n")

        with open(tracks_jsonl, "a", encoding="utf-8") as f:
            for t in tracks:
                f.write(json.dumps(track_to_dict(t, frame_idx, current_time)) + "\n")

        # Console output.
        print(f"--- Frame {frame_idx:04d}  time={current_time:.2f}s ---")
        print(f"  Points      : {len(points)}")
        print(f"  Detections  : {len(detections)}")
        for d in detections:
            c = d.center_xyz
            print(f"    {d.class_name!r} conf={d.confidence:.3f} center=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")
        print(f"  Tracks      : {len(tracks)}")
        for t in tracks:
            speed = np.linalg.norm(t.state[3:6])
            dyn = "DYNAMIC" if t.dynamic else "static"
            print(f"    ID {t.track_id} | {t.class_name} | conf {t.confidence:.2f} | {dyn} | speed {speed:.2f} m/s")
        print()

        current_time += dt

    # Write metrics.
    avg_pts = total_points / n_frames
    avg_dets = total_detections / n_frames
    active_final = len(trk.tracks)
    dynamic_final = sum(1 for t in trk.tracks if t.dynamic)
    avg_det_t = total_detection_time / n_frames
    avg_trk_t = total_tracking_time / n_frames
    avg_total_t = avg_det_t + avg_trk_t

    metrics_text = write_metrics(
        source="sample",
        n_frames=n_frames,
        avg_input_points=avg_pts,
        avg_detections=avg_dets,
        active_tracks_final=active_final,
        dynamic_tracks_final=dynamic_final,
        avg_detection_time=avg_det_t,
        avg_tracking_time=avg_trk_t,
        avg_total_time=avg_total_t,
        output_dir=output_dir,
    )
    print(metrics_text)
    print()

    # Task K: Runtime benchmark measurement logging
    device_name = str(getattr(det, "device", "CPU"))
    det_name = type(det).__name__
    benchmark_summary = log_runtime_benchmark_to_report(
        detector_name=det_name,
        device_name=device_name,
        detection_times=detection_times,
    )
    if benchmark_summary:
        print(benchmark_summary)

    # Render summary image.
    summary_path = output_dir / "phase8_summary.png"
    render_summary_image(frames_dir, summary_path, "sample", n_frames)
    print(f"Saved summary: {summary_path}")
    print(f"Saved BEV frames: {frames_dir}/")
    print(f"Saved detections: {detections_jsonl}")
    print(f"Saved tracks    : {tracks_jsonl}")
    if save_overlay:
        print(f"Saved dynamic overlays: {output_dir / 'dynamic_object_overlay'}/")



# ============================================================================
# SemanticKITTI runner
# ============================================================================

def run_semantickitti(
    detector: ObjectDetector,
    tracker: MultiObjectTracker,
    sequence: str,
    start_frame: str,
    n_frames: int,
    dt: float,
    output_dir: Path,
    save_overlay: bool = True,
) -> None:
    """Run detection + tracking on a sequence of SemanticKITTI frames.

    Saves BEV images, JSONL outputs, and metrics.
    """
    base_dir = Path("data/semantic_kitti/dataset/sequences") / sequence
    if not base_dir.exists():
        raise FileNotFoundError(
            f"SemanticKITTI sequence directory not found: {base_dir}\n"
            "Download and extract SemanticKITTI first."
        )

    try:
        start_idx = int(start_frame)
    except ValueError:
        raise ValueError(f"start_frame must be an integer string, got {start_frame!r}.")

    print("=" * 70)
    print(f"FOVEAX Phase 8A — SemanticKITTI Tracking ({sequence})")
    print("=" * 70)
    print(f"Sequence dir : {base_dir}")
    print(f"Start frame  : {start_frame}")
    print(f"Frame count  : {n_frames}")
    print(f"Frame delta  : {dt}s")
    print()

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    detections_jsonl = output_dir / "detections.jsonl"
    tracks_jsonl = output_dir / "tracks.jsonl"

    if detections_jsonl.exists():
        detections_jsonl.unlink()
    if tracks_jsonl.exists():
        tracks_jsonl.unlink()

    total_detection_time = 0.0
    total_tracking_time = 0.0
    total_points = 0
    total_detections = 0
    detection_times: list[float] = []

    if isinstance(detector, MockObjectDetector):
        detector_label = "MOCK GEOMETRIC BASELINE — NOT AI DETECTION"
    else:
        detector_label = f"{type(detector).__name__} (AI Detection)"

    current_time = 0.0

    for i in range(n_frames):
        frame_id = f"{start_idx + i:06d}"
        print(f"--- Frame {i} / {frame_id}  time={current_time:.2f}s ---")

        try:
            points = load_semantickitti_frame(base_dir, frame_id)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            current_time += dt
            continue

        total_points += len(points)

        t0 = time.perf_counter()
        detections = detector.detect(points, timestamp_s=current_time)
        t1 = time.perf_counter()
        detection_time = t1 - t0
        total_detection_time += detection_time
        detection_times.append(detection_time)

        t2 = time.perf_counter()
        tracks = tracker.update(detections, timestamp_s=current_time)
        t3 = time.perf_counter()
        total_tracking_time += t3 - t2

        total_detections += len(detections)

        # BEV image.
        frame_path = frames_dir / f"frame_{i:04d}.png"
        render_bev_image(
            points, detections, tracks, i, frame_path, f"semantickitti/{sequence}", detector_label=detector_label
        )

        # Task J: Dynamic-object overlay onto Phase 3 & 5 grids
        if save_overlay:
            overlay_dir = output_dir / "dynamic_object_overlay"
            dynamic_targets = [t for t in tracks if t.dynamic]
            if not dynamic_targets:
                dynamic_targets = detections
            generate_dynamic_object_overlay(
                points_xyzi=points,
                dynamic_objects=dynamic_targets,
                output_dir=overlay_dir,
                frame_idx=i,
            )

        # JSONL.
        with open(detections_jsonl, "a", encoding="utf-8") as f:
            for d in detections:
                f.write(json.dumps(detection_to_dict(d, i, current_time)) + "\n")
        with open(tracks_jsonl, "a", encoding="utf-8") as f:
            for t in tracks:
                f.write(json.dumps(track_to_dict(t, i, current_time)) + "\n")

        print(f"  Points      : {len(points)}")
        print(f"  Detections  : {len(detections)}")
        print(f"  Tracks      : {len(tracks)}")
        for t in tracks:
            speed = np.linalg.norm(t.state[3:6])
            dyn = "DYNAMIC" if t.dynamic else "static"
            print(f"    ID {t.track_id} | {t.class_name} | conf {t.confidence:.2f} | {dyn} | speed {speed:.2f} m/s")
        print()

        current_time += dt

    # Metrics.
    avg_pts = total_points / max(1, n_frames)
    avg_dets = total_detections / max(1, n_frames)
    active_final = len(tracker.tracks)
    dynamic_final = sum(1 for t in tracker.tracks if t.dynamic)
    avg_det_t = total_detection_time / max(1, n_frames)
    avg_trk_t = total_tracking_time / max(1, n_frames)

    metrics_text = write_metrics(
        source=f"semantickitti/{sequence}",
        n_frames=n_frames,
        avg_input_points=avg_pts,
        avg_detections=avg_dets,
        active_tracks_final=active_final,
        dynamic_tracks_final=dynamic_final,
        avg_detection_time=avg_det_t,
        avg_tracking_time=avg_trk_t,
        avg_total_time=avg_det_t + avg_trk_t,
        output_dir=output_dir,
    )
    print(metrics_text)
    print()

    # Task K: Runtime benchmark measurement logging
    device_name = str(getattr(detector, "device", "CPU"))
    det_name = type(detector).__name__
    benchmark_summary = log_runtime_benchmark_to_report(
        detector_name=det_name,
        device_name=device_name,
        detection_times=detection_times,
    )
    if benchmark_summary:
        print(benchmark_summary)

    summary_path = output_dir / "phase8_summary.png"
    render_summary_image(frames_dir, summary_path, f"semantickitti/{sequence}", n_frames)
    print(f"Saved summary: {summary_path}")
    if save_overlay:
        print(f"Saved dynamic overlays: {output_dir / 'dynamic_object_overlay'}/")



# ============================================================================
# CLI
# ============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FOVEAX Phase 8A — 3D object detection & tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python src/11_object_detection_tracking.py --source sample --detector mock --frames 10
                  Run synthetic sample mode with 10 frames.

              python src/11_object_detection_tracking.py --source semantickitti \\
                  --sequence 00 --start-frame 000000 --frames 5
                  Run on SemanticKITTI frames (requires dataset on disk).

              python src/11_object_detection_tracking.py --source sample --detector mock \\
                  --frames 10 --ground-threshold 0.25 --cluster-distance 0.75 \\
                  --min-cluster-points 15 --track-gate 2.0 --dynamic-speed-threshold 0.25
                  Run sample mode with custom detector/tracker parameters.
        """),
    )

    parser.add_argument(
        "--source",
        choices=["sample", "semantickitti"],
        default="sample",
        help=(
            "Data source (default: sample). "
            "'sample' = synthetic point cloud with moving obstacle. "
            "'semantickitti' = SemanticKITTI .bin frames (requires dataset)."
        ),
    )
    parser.add_argument(
        "--detector",
        choices=["mock", "pointpillars"],
        default="mock",
        help="Detector type (default: mock). mock geometric detector or real pointpillars.",
    )
    parser.add_argument(
        "--openpcdet-repo",
        type=str,
        default="external/OpenPCDet",
        help="Path to OpenPCDet repository (default: external/OpenPCDet).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to OpenPCDet pretrained checkpoint (.pth). Required for pointpillars.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to OpenPCDet PointPillars config (.yaml). Required for pointpillars.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on (auto, cpu, cuda). Default: auto.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.20,
        help="Score threshold for pointpillars detection (default: 0.20).",
    )
    parser.add_argument(
        "--duplicate-iou-threshold",
        type=float,
        default=0.70,
        help="IoU threshold for flagging duplicate 3D detection boxes (default: 0.70).",
    )
    parser.add_argument(
        "--save-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save dynamic-object overlay onto Phase 3/5 grids (default: True).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="Number of frames to process (default: 10).",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="00",
        help="SemanticKITTI sequence ID (default: 00).",
    )
    parser.add_argument(
        "--start-frame",
        type=str,
        default="000000",
        help="First SemanticKITTI frame ID (default: 000000).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Time delta between frames in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--ground-threshold",
        type=float,
        default=0.25,
        help="Ground threshold for the mock detector (default: 0.25 m).",
    )
    parser.add_argument(
        "--cluster-distance",
        type=float,
        default=0.75,
        help="Maximum cluster distance for the mock detector (default: 0.75 m).",
    )
    parser.add_argument(
        "--min-cluster-points",
        type=int,
        default=15,
        help="Minimum cluster points for the mock detector (default: 15).",
    )
    parser.add_argument(
        "--track-gate",
        type=float,
        default=2.0,
        help="Track-detection gating threshold in metres (default: 2.0).",
    )
    parser.add_argument(
        "--dynamic-speed-threshold",
        type=float,
        default=0.25,
        help="Speed threshold (m/s) for marking a track dynamic (default: 0.25).",
    )

    return parser.parse_args(argv)


def build_detector(args: argparse.Namespace) -> ObjectDetector:
    """Build the requested detector."""
    if args.detector == "mock":
        print("Building MockObjectDetector (deterministic geometric clusterer).")
        print("  [WARNING] This is NOT an AI detector. Do not use for safety claims.")
        return MockObjectDetector(
            ground_threshold_m=args.ground_threshold,
            cluster_min_points=args.min_cluster_points,
            cluster_max_dist_m=args.cluster_distance,
        )
    elif args.detector == "pointpillars":
        if not args.checkpoint or not args.config:
            raise ValueError("--checkpoint and --config are required for pointpillars detector.")
        from src.perception.openpcdet_predictor import OpenPCDetPointPillarsDetector
        print("Building OpenPCDetPointPillarsDetector.")
        return OpenPCDetPointPillarsDetector(
            repo_path=Path(args.openpcdet_repo),
            checkpoint_path=Path(args.checkpoint),
            config_path=Path(args.config),
            device=args.device,
            score_threshold=args.score_threshold,
            duplicate_iou_threshold=args.duplicate_iou_threshold,
        )
    else:
        raise ValueError(f"Unknown detector: {args.detector}")


def build_tracker(args: argparse.Namespace) -> MultiObjectTracker:
    """Build the tracker."""
    return MultiObjectTracker(
        gating_threshold_m=args.track_gate,
        max_missed_frames=_TRACKER_DEFAULTS["max_missed_frames"],
        min_hits_for_dynamic=_TRACKER_DEFAULTS["min_hits_for_dynamic"],
        dynamic_speed_threshold_mps=args.dynamic_speed_threshold,
        measure_noise_std=_TRACKER_DEFAULTS["measure_noise_std"],
        process_noise_std=_TRACKER_DEFAULTS["process_noise_std"],
    )


def main(argv: list[str] | None = None) -> None:
    """Run the Phase 8A detection and tracking pipeline."""
    args = parse_args(argv)

    print("=" * 70)
    print("FOVEAX Phase 8A/8B — 3D Object Detection and Multi-Object Tracking")
    print("=" * 70)
    print()

    # Validate --source and --frames.
    if args.frames < 1:
        raise ValueError("--frames must be >= 1.")

    # Build output directory.
    output_dir = Path("outputs/phase8") / args.source / args.detector
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.detector == "pointpillars":
        prov_text = (
            f"detector source: pointpillars_openpcdet_pretrained\n"
            f"model architecture: PointPillars\n"
            f"checkpoint path: {args.checkpoint}\n"
            f"config path: {args.config}\n"
            f"device: {args.device}\n"
            f"score threshold: {args.score_threshold}\n"
            f"duplicate iou threshold: {args.duplicate_iou_threshold}\n"
            "dataset compatibility statement: A pretrained model is only valid for the dataset it was trained on.\n"
            "WARNING: outputs are model predictions, not ground truth.\n"
        )
    else:
        prov_text = (
            "detector source: mock_geometric_clusterer\n"
            "outputs are geometric clusters, not AI predictions.\n"
        )
    (output_dir / "model_provenance.txt").write_text(prov_text)

    # Handle run modes
    if args.source == "sample":
        detector = build_detector(args)
        tracker = build_tracker(args)
        run_sample_mode(
            detector=detector,
            tracker=tracker,
            n_frames=args.frames,
            dt=args.dt,
            ground_threshold=args.ground_threshold,
            cluster_distance=args.cluster_distance,
            min_cluster_points=args.min_cluster_points,
            track_gate=args.track_gate,
            dynamic_speed_threshold=args.dynamic_speed_threshold,
            output_dir=output_dir,
            save_overlay=args.save_overlay,
        )
    elif args.source == "semantickitti":
        detector = build_detector(args)
        tracker = build_tracker(args)
        run_semantickitti(
            detector=detector,
            tracker=tracker,
            sequence=args.sequence,
            start_frame=args.start_frame,
            n_frames=args.frames,
            dt=args.dt,
            output_dir=output_dir,
            save_overlay=args.save_overlay,
        )
    else:
        raise ValueError(f"Unknown source/detector combination: {args.source}/{args.detector}")



if __name__ == "__main__":
    main()
