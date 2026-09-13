import math
import time
import numpy as np
import psutil
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

import os
from pathlib import Path

from typing import Any
from PyQt5.QtCore import QThread, pyqtSignal

from src.dashboard.dashboard_state import FrameState, HardwareMetrics, assert_coordinate_frame_consistency
from src.tracking.multi_object_tracker import MultiObjectTracker
from src.perception.object_detector import ObjectDetector, MockObjectDetector
import importlib
_trk = importlib.import_module('src.11_object_detection_tracking')
generate_synthetic_sample_cloud = _trk.generate_synthetic_sample_cloud
repeated_cloud_with_translation = _trk.repeated_cloud_with_translation

try:
    from src.perception.rellis3d_loader import load_rellis3d_frame
    HAS_RELLIS_LOADER = True
except ImportError:
    HAS_RELLIS_LOADER = False

class DataStreamerThread(QThread):
    state_ready = pyqtSignal(FrameState)
    
    def __init__(
        self,
        source: str = "sample",
        detector: ObjectDetector = None,
        num_frames: int = 100,
        sequence: str = "00",
        data_root: str | Path | None = None,
        rate_hz: float = 10.0,
        semantic_predictor: Any = None,
        predictor_mode: str = "ground_truth",
    ):
        super().__init__()
        self.source = source.lower()
        self.detector = detector if detector else MockObjectDetector()
        self.semantic_predictor = semantic_predictor
        # "ground_truth" | "mock" | "salsanext" -- matches
        # src/dashboard/export_web_dashboard_data.py's --predictor choices.
        # Only used to decide what the live accuracy comparison's
        # "prediction" side is; semantic_predictor itself is what actually
        # produces predictions when this is not "ground_truth".
        self.predictor_mode = predictor_mode
        self.tracker = MultiObjectTracker()
        self.num_frames = num_frames
        self.sequence = sequence
        self.data_root = Path(data_root) if data_root else None
        self.rate_hz = rate_hz
        self.running = True
        
        # Grid parameters for 2D maps
        self.grid_res = 0.5
        self.grid_extent = (-20.0, 20.0, -20.0, 20.0) # x_min, x_max, y_min, y_max
        self.x_bins = int((self.grid_extent[1] - self.grid_extent[0]) / self.grid_res)
        self.y_bins = int((self.grid_extent[3] - self.grid_extent[2]) / self.grid_res)
        
        global HAS_PYNVML
        
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
            except pynvml.NVMLError:
                HAS_PYNVML = False

    def _attach_terrain_features(self, detections: list, points: np.ndarray) -> None:
        """Compute real geometric features per detection and stash them in
        `det.metadata`, so the tracker (and, from there, the display layer)
        can surface them without recomputing or guessing.

        Every feature here is derived from this detection's own real member
        points (selected from the real frame point cloud, including the
        ground points MockObjectDetector itself discards before clustering
        -- needed here to estimate real local ground height) via
        src/perception/terrain_features.py. A feature is simply absent from
        metadata when there isn't enough real data to support it (e.g. a
        one-point detection can't support a plane fit) -- never a
        placeholder value.
        """
        from src.perception.terrain_features import (
            ground_clearance_m,
            is_overhang,
            is_probable_rock_heuristic,
            local_ground_z_percentile,
            points_in_oriented_box,
            slope_angle_deg,
        )

        ground_like = {"DRIVABLE_GROUND", "ROUGH_TERRAIN"}

        for d in detections:
            member_mask = points_in_oriented_box(points, d.center_xyz, d.size_lwh)
            member_xyz = points[member_mask, :3]
            metadata: dict = {}

            if d.class_name in ground_like:
                slope = slope_angle_deg(member_xyz)
                if slope is not None:
                    metadata["slope_deg"] = slope

            ground_z = local_ground_z_percentile(points, d.center_xyz[:2], radius_m=1.5)
            if ground_z is not None:
                bbox_bottom_z = float(d.center_xyz[2] - d.size_lwh[2] / 2.0)
                clearance = ground_clearance_m(bbox_bottom_z, ground_z)
                metadata["ground_clearance_m"] = clearance
                metadata["is_overhang"] = is_overhang(clearance, d.class_name)

            volume_m3 = float(d.size_lwh[0] * d.size_lwh[1] * d.size_lwh[2])
            metadata["is_probable_rock"] = is_probable_rock_heuristic(
                volume_m3, int(member_mask.sum()), d.class_name
            )

            d.metadata.update(metadata)

    def _generate_maps(self, points: np.ndarray, tracks: list) -> dict:
        """Generate simple 2D grid maps for dashboard visualization."""
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        x_min, x_max, y_min, y_max = self.grid_extent

        # Filter points within bounds
        mask = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
        x, y, z = x[mask], y[mask], z[mask]

        if len(x) == 0:
            empty = np.full((self.y_bins, self.x_bins), np.nan, dtype=np.float32)
            empty_overhead = np.full((self.y_bins, self.x_bins), -1, dtype=np.int8)
            self._last_extras = {
                "road_width_m": None,
                "elevation_profile": [],
                "elevation_hazards_m": [],
                "hazard_clusters": [],
                "mean_uncertainty": None,
                "occupied_fraction": None,
                "centerline_profile": [],
                "centerline_markers": [],
            }
            return {
                "elevation": empty, "traversability": empty, "roi": empty,
                "uncertainty": empty, "overhead": empty_overhead,
            }

        col = np.floor((x - x_min) / self.grid_res).astype(np.int32)
        row = np.floor((y - y_min) / self.grid_res).astype(np.int32)

        # Need to flip Y axis for image rendering (0,0 is top-left in image, bottom-left in world)
        # However, matplotlib imshow handles origin='lower'. For PyQt QImage, (0,0) is top-left.
        # Let's just create standard arrays and let UI handle it.
        flat_idx = row * self.x_bins + col

        total_cells = self.y_bins * self.x_bins
        z_max_arr = np.full(total_cells, -np.inf, dtype=np.float32)
        np.maximum.at(z_max_arr, flat_idx, z)
        z_max_arr[z_max_arr == -np.inf] = np.nan

        elevation = z_max_arr.reshape(self.y_bins, self.x_bins)

        # Traversability mock: High if elevation variance is high
        z_min_arr = np.full(total_cells, np.inf, dtype=np.float32)
        np.minimum.at(z_min_arr, flat_idx, z)
        z_min_arr[z_min_arr == np.inf] = np.nan
        z_min_arr = z_min_arr.reshape(self.y_bins, self.x_bins)

        height_range = elevation - z_min_arr
        # Traversability: 1.0 = safe (flat), 0.0 = blocked (rough/obstacle),
        # matching canonical convention from 06_terrain_traversability.py
        traversability = 1.0 - np.clip(height_range / 1.0, 0.0, 1.0)

        # Point count per cell (real occupancy density) -- used below for
        # the real Phase 5 (src/08_spatial_importance_roi.py) uncertainty
        # formula, applied verbatim to this live local grid.
        point_count = np.bincount(flat_idx, minlength=total_cells).astype(np.float32)
        occupied = point_count > 0
        density_confidence = np.clip(np.log1p(point_count) / np.log(8.0), 0.0, 1.0)
        uncertainty_flat = 1.0 - density_confidence
        uncertainty_flat[~occupied] = 1.0
        uncertainty = uncertainty_flat.reshape(self.y_bins, self.x_bins)
        point_count_2d = point_count.reshape(self.y_bins, self.x_bins)
        occupied_2d = occupied.reshape(self.y_bins, self.x_bins)

        # ROI mock: higher around dynamic objects
        roi = np.zeros_like(elevation)
        for t in tracks:
            if t.dynamic:
                tx, ty = t.state[0], t.state[1]
                t_col = int((tx - x_min) / self.grid_res)
                t_row = int((ty - y_min) / self.grid_res)
                if 0 <= t_col < self.x_bins and 0 <= t_row < self.y_bins:
                    # Draw a blob
                    for i in range(-2, 3):
                        for j in range(-2, 3):
                            r, c = t_row + i, t_col + j
                            if 0 <= r < self.y_bins and 0 <= c < self.x_bins:
                                roi[r, c] = max(roi[r, c], 1.0 - 0.2*max(abs(i), abs(j)))

        self._last_extras = self._compute_real_extras(
            traversability, elevation, z_min_arr, point_count_2d, occupied_2d, uncertainty, tracks
        )

        # Overhead-clearance classification (EV-2): per-cell, from this
        # frame's real raw point z-values only -- distinguishes a genuine
        # vertical gap under a raised obstacle (overhead_obstacle) from a
        # column solid all the way to the ground (solid_obstacle). See
        # src/perception/overhead_detection.py for the algorithm.
        from src.perception.overhead_detection import compute_overhead_clearance_grid
        overhead = compute_overhead_clearance_grid(
            x, y, z, x_min, y_min, self.x_bins, self.y_bins, self.grid_res
        )

        # Flip vertically so row 0 is max Y (top of image)
        return {
            "elevation": np.flipud(elevation),
            "traversability": np.flipud(traversability),
            "roi": np.flipud(roi),
            "uncertainty": np.flipud(uncertainty),
            "overhead": np.flipud(overhead),
        }

    def _compute_real_extras(
        self,
        traversability: np.ndarray,
        elevation: np.ndarray,
        z_min_2d: np.ndarray,
        point_count_2d: np.ndarray,
        occupied_2d: np.ndarray,
        uncertainty_2d: np.ndarray,
        tracks: list,
    ) -> dict:
        """Compute road width, elevation profile, hazard clusters and the
        Phase 5 uncertainty summary from the real (pre-flip) local grid.

        All arrays are indexed [row=y, col=x] over self.grid_extent, not yet
        flipped for display -- kept that way here so real-world x/y math
        below stays simple.

        FORWARD_AXIS = +Y. This convention (road width / elevation
        profile sweep over rows at a fixed centerline column, i.e. Y is
        "ahead" and X is "lateral") must match
        src/dashboard/export_web_dashboard_data.py::compute_ego_terrain_status,
        which previously used the opposite (+X forward) and was corrected
        to agree with this file -- do not let the two drift apart again.
        """
        x_min, x_max, y_min, y_max = self.grid_extent

        # --- Road width: contiguous Safe (>=0.70) run of columns at the
        # ego's row (y nearest 0), containing the ego's column (x nearest 0).
        ego_row = int(np.clip(round((0.0 - y_min) / self.grid_res), 0, self.y_bins - 1))
        ego_col = int(np.clip(round((0.0 - x_min) / self.grid_res), 0, self.x_bins - 1))
        road_width_m = None
        row_trav = traversability[ego_row, :]
        safe_mask = np.nan_to_num(row_trav, nan=0.0) >= 0.70
        if safe_mask[ego_col]:
            left = ego_col
            while left - 1 >= 0 and safe_mask[left - 1]:
                left -= 1
            right = ego_col
            while right + 1 < self.x_bins and safe_mask[right + 1]:
                right += 1
            road_width_m = float((right - left + 1) * self.grid_res)

        # --- Elevation profile + hazard markers along the ego-forward
        # centerline (column nearest x=0), real z_max per row.
        centerline_col = ego_col
        elevation_profile = []
        elevation_hazards_m = []
        for r in range(self.y_bins):
            z = elevation[r, centerline_col]
            if not np.isnan(z):
                y_world = y_min + (r + 0.5) * self.grid_res
                elevation_profile.append((float(y_world), float(z)))
                t = traversability[r, centerline_col]
                if not np.isnan(t) and t < 0.40:
                    elevation_hazards_m.append(float(y_world))

        # --- Hazard clusters: connected components of Blocked (<0.40)
        # cells, same method as export_web_dashboard_data.find_hazard_clusters.
        hazard_clusters = []
        try:
            from scipy import ndimage

            from src.perception.terrain_features import classify_hazard_kind

            blocked = np.nan_to_num(traversability, nan=1.0) < 0.40
            labeled, n_clusters = ndimage.label(blocked)
            cell_area_m2 = self.grid_res ** 2
            for cluster_id in range(1, n_clusters + 1):
                cmask = labeled == cluster_id
                rows_idx, cols_idx = np.nonzero(cmask)
                if len(rows_idx) == 0:
                    continue
                min_trav = float(np.nanmin(traversability[cmask]))
                mean_density = float(point_count_2d[cmask].mean())
                confidence = float(np.clip(mean_density / 20.0, 0.0, 1.0))
                centroid_row = float(rows_idx.mean())
                centroid_col = float(cols_idx.mean())
                world_x = x_min + (centroid_col + 0.5) * self.grid_res
                world_y = y_min + (centroid_row + 0.5) * self.grid_res
                distance_m = float(np.hypot(world_x, world_y))
                severity = "Critical" if min_trav < 0.20 else "Caution"

                # Real Pothole/Bump classification: this cluster's own real
                # min/max elevation vs. a real local baseline taken from the
                # real occupied ring of cells immediately around it
                # (dilate minus the cluster itself) -- never the cluster's
                # own cells, or a dip would be compared against itself.
                dilated = ndimage.binary_dilation(cmask, iterations=2)
                ring = dilated & ~cmask & occupied_2d
                baseline_z = float(np.nanmedian(elevation[ring])) if np.any(ring) else None
                z_max_local = float(np.nanmax(elevation[cmask]))
                z_min_local = float(np.nanmin(z_min_2d[cmask]))
                kind, extent_m = classify_hazard_kind(z_max_local, z_min_local, baseline_z)

                hazard_clusters.append({
                    "cluster_id": cluster_id,
                    "x": round(world_x, 3),
                    "y": round(world_y, 3),
                    "distance_m": round(distance_m, 2),
                    "area_m2": round(len(rows_idx) * cell_area_m2, 2),
                    "min_traversability": round(min_trav, 3),
                    "severity": severity,
                    "confidence": round(confidence, 3),
                    "kind": kind,
                    "extent_m": round(extent_m, 3) if extent_m is not None else None,
                })
            hazard_clusters.sort(key=lambda c: c["distance_m"])
        except ImportError:
            pass

        # --- Uncertainty summary (Phase 5 formula, see _generate_maps).
        occupied_fraction = float(occupied_2d.mean())
        mean_uncertainty = float(np.mean(uncertainty_2d))

        # --- Real forward-corridor cross-section (src/perception/centerline_profile.py).
        # No existing ego heading/position tracking was found anywhere in
        # this pipeline (checked dashboard_state.py, this module, and the
        # tracker) -- every FrameState is a single ego-centered frame with
        # no odometry. ego_position=(0,0) is the pipeline's established
        # ego-origin convention; heading=+Y matches this module's own
        # pre-existing forward-axis convention above (elevation_profile,
        # road_width already treat y as the forward axis). Capped to this
        # local grid's own 20m forward extent, not the function's 100m
        # default, since this grid does not extend further.
        from src.perception.centerline_profile import (
            compute_centerline_profile, find_corridor_markers,
        )
        centerline = compute_centerline_profile(
            elevation, z_min_2d, point_count_2d, self.grid_extent, self.grid_res,
            ego_position=(0.0, 0.0), ego_heading_rad=math.pi / 2.0,
            width_m=2.0, max_range_m=y_max, bin_size_m=1.0,
        )
        centerline_profile = [
            {
                "distance_m": b.distance_m,
                "height_above_baseline_m": b.height_above_baseline_m,
                "depth_below_baseline_m": b.depth_below_baseline_m,
                "point_count": b.point_count,
            }
            for b in centerline.bins
        ]

        corridor_objects = list(hazard_clusters)
        for t in tracks:
            tx, ty = float(t.state[0]), float(t.state[1])
            corridor_objects.append({
                "x": tx, "y": ty,
                "distance_m": round(float(np.hypot(tx, ty)), 2),
                "label": t.class_name,
                "type": "track",
                "track_id": t.track_id,
            })
        for h in hazard_clusters:
            h.setdefault("label", h["severity"])
            h.setdefault("type", "hazard")
        centerline_markers = find_corridor_markers(
            corridor_objects, self.grid_extent,
            ego_position=(0.0, 0.0), ego_heading_rad=math.pi / 2.0,
            width_m=2.0, max_range_m=y_max,
        )

        return {
            "road_width_m": road_width_m,
            "elevation_profile": elevation_profile,
            "elevation_hazards_m": elevation_hazards_m,
            "hazard_clusters": hazard_clusters,
            "occupied_fraction": occupied_fraction,
            "mean_uncertainty": mean_uncertainty,
            "centerline_profile": centerline_profile,
            "centerline_markers": centerline_markers,
        }

    def _compute_live_accuracy(
        self, points: np.ndarray, gt_labels: np.ndarray, pred_labels: np.ndarray
    ) -> dict:
        """Real Near/Mid/Far/Overall distance-bucketed accuracy for the
        current frame, using the SAME src/10b_eval_distance_metrics.py
        ::compute_metrics() function as the already-validated offline
        evaluation -- imported and called directly, not reimplemented.

        Uses the identical bucket boundaries as that script's main():
        Near 0-15m, Mid 15-35m, Far 35-100m, Overall 0-100m (2D radial
        distance). compute_metrics() itself prints a full report to
        stdout per call (by design, for the offline CLI tool) -- that
        output is redirected away here since this runs every live frame,
        without touching the reused function itself.
        """
        import io
        import contextlib
        import importlib

        eval_module = importlib.import_module("src.10b_eval_distance_metrics")
        compute_metrics = eval_module.compute_metrics

        x = points[:, 0]
        y = points[:, 1]
        distance = np.sqrt(x ** 2 + y ** 2)

        mask_near = (distance >= 0.0) & (distance < 15.0)
        mask_mid = (distance >= 15.0) & (distance < 35.0)
        mask_far = (distance >= 35.0) & (distance <= 100.0)
        mask_overall = (distance >= 0.0) & (distance <= 100.0)

        with contextlib.redirect_stdout(io.StringIO()):
            near = compute_metrics(gt_labels, pred_labels, mask_near, "Near (0-15m)")
            mid = compute_metrics(gt_labels, pred_labels, mask_mid, "Mid (15-35m)")
            far = compute_metrics(gt_labels, pred_labels, mask_far, "Far (35-100m)")
            overall = compute_metrics(gt_labels, pred_labels, mask_overall, "Overall (0-100m)")

        return {"near": near, "mid": mid, "far": far, "overall": overall}

    def _get_hardware_metrics(self, fps: float, target_fps: float, latency: float) -> HardwareMetrics:
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        
        hm = HardwareMetrics(
            fps=fps,
            target_fps=target_fps,
            achieved_fps=fps,
            latency_ms=latency * 1000.0,
            cpu_percent=cpu_percent,
            ram_percent=mem.percent,
            ram_used_gb=mem.used / (1024**3),
            ram_total_gb=mem.total / (1024**3),
            gpu_available=HAS_PYNVML
        )
        
        if HAS_PYNVML:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                hm.gpu_percent = util.gpu
                hm.vram_used_gb = mem_info.used / (1024**3)
                hm.vram_total_gb = mem_info.total / (1024**3)
                hm.vram_percent = (mem_info.used / mem_info.total) * 100.0
            except:
                hm.gpu_available = False
                
        return hm

    def _setup_sources(self):
        """Prepare file list or synthetic generators based on self.source."""
        self._sample_static_pts = None
        self._sample_moving_pts = None
        self._sample_translation_step = np.array([0.5, 0.0, 0.0], dtype=np.float64)
        self._file_list = []

        if self.source == "sample":
            self._sample_static_pts, self._sample_moving_pts = generate_synthetic_sample_cloud(
                n_static=2000, n_moving=150
            )
        elif self.source == "semantickitti":
            kitti_root = self.data_root or Path("data/semantic_kitti/dataset/sequences")
            seq_dir = kitti_root / self.sequence / "velodyne"
            if seq_dir.exists():
                self._file_list = sorted(list(seq_dir.glob("*.bin")))
            if not self._file_list:
                print(f"[WARNING] No SemanticKITTI .bin files found in {seq_dir}. Falling back to sample generator.")
                self.source = "sample"
                self._sample_static_pts, self._sample_moving_pts = generate_synthetic_sample_cloud(
                    n_static=2000, n_moving=150
                )
        elif self.source == "rellis3d":
            rellis_root = self.data_root or Path("C:/dev/data/rellis3d")
            if not rellis_root.exists():
                rellis_root = Path("data/rellis3d")
            seq_dir = rellis_root / "Rellis-3D" / self.sequence / "os1_cloud_node_kitti_bin"
            if seq_dir.exists():
                self._file_list = sorted(list(seq_dir.glob("*.bin")))
            if not self._file_list:
                print(f"[WARNING] No RELLIS-3D .bin files found in {seq_dir}. Falling back to sample generator.")
                self.source = "sample"
                self._sample_static_pts, self._sample_moving_pts = generate_synthetic_sample_cloud(
                    n_static=2000, n_moving=150
                )

    def _get_frame_data(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Fetch (points, semantic_labels) for frame_idx from active source."""
        if self.source == "sample":
            offset = self._sample_translation_step * frame_idx
            frame_moving = repeated_cloud_with_translation(self._sample_moving_pts, offset)
            return np.concatenate([self._sample_static_pts, frame_moving], axis=0), None

        if not self._file_list:
            return np.zeros((0, 4), dtype=np.float32), None

        file_path = self._file_list[frame_idx % len(self._file_list)]

        if self.source == "rellis3d" and HAS_RELLIS_LOADER:
            rellis_root = self.data_root or Path("C:/dev/data/rellis3d")
            if not rellis_root.exists():
                rellis_root = Path("data/rellis3d")
            try:
                frame_id = file_path.stem
                rf = load_rellis3d_frame(rellis_root, self.sequence, frame_id)
                return rf.points, rf.foveax_class_ids
            except Exception:
                # Fallback to direct raw bin loading if labels missing
                raw = np.fromfile(file_path, dtype=np.float32)
                return raw.reshape(-1, 4), None

        # Standard SemanticKITTI or raw binary
        raw = np.fromfile(file_path, dtype=np.float32)
        if raw.size % 4 != 0:
            return np.zeros((0, 4), dtype=np.float32), None
        points = raw.reshape(-1, 4)

        semantic_labels = None
        label_file = file_path.parent.parent / "labels" / f"{file_path.stem}.label"
        if label_file.exists():
            try:
                from src.perception.semantic_predictor import (
                    extract_semantickitti_ids,
                    remap_semantickitti_to_foveax,
                )
                raw_labels = np.fromfile(label_file, dtype=np.uint32)
                sem_ids, _ = extract_semantickitti_ids(raw_labels)
                semantic_labels = remap_semantickitti_to_foveax(sem_ids)
            except Exception:
                semantic_labels = None

        return points, semantic_labels

    def _get_frame_points(self, frame_idx: int) -> np.ndarray:
        """Fetch points for frame_idx from active source."""
        points, _ = self._get_frame_data(frame_idx)
        return points

    def process_frame(self, frame_idx: int, timestamp_s: float, fps: float = 10.0) -> FrameState:
        """Process a single frame through detection, tracking, grid mapping, and metrics."""
        t_start = time.perf_counter()
        points, gt_labels = self._get_frame_data(frame_idx)
        # gt_labels is REAL ground truth ONLY (loaded from a .label file or
        # the RELLIS-3D loader) -- never a model prediction. Kept separate
        # from any prediction below so the two can be honestly compared.

        semantic_labels = gt_labels

        # If external semantic predictor is supplied and no real ground
        # truth exists for this frame, fall back to its prediction for
        # object-classification cross-referencing (existing behavior,
        # unchanged).
        if semantic_labels is None and getattr(self, "semantic_predictor", None) is not None and len(points) > 0:
            try:
                prediction = self.semantic_predictor.predict(points)
                semantic_labels = prediction.class_ids
            except Exception:
                pass

        # --- Real distance-bucketed accuracy (Task: live metrics) -------
        # Reuses src/10b_eval_distance_metrics.py::compute_metrics()
        # directly -- the exact same, already-validated function used by
        # the offline evaluation script, not a reimplementation. Only
        # computed when real ground truth exists for this frame; a
        # "ground_truth" predictor_mode compares gt against itself
        # (trivially ~100%, still real, not fabricated -- labeled as such
        # in the UI) while "mock"/"salsanext" compare a real independent
        # prediction against real ground truth.
        has_ground_truth = gt_labels is not None
        accuracy_metrics = None
        if has_ground_truth and len(points) > 0:
            if self.predictor_mode == "ground_truth" or self.semantic_predictor is None:
                pred_labels_for_accuracy = gt_labels
            else:
                try:
                    accuracy_prediction = self.semantic_predictor.predict(points)
                    pred_labels_for_accuracy = accuracy_prediction.class_ids
                except Exception:
                    pred_labels_for_accuracy = None

            if pred_labels_for_accuracy is not None:
                accuracy_metrics = self._compute_live_accuracy(points, gt_labels, pred_labels_for_accuracy)

        # Coordinate frame sanity check
        if getattr(self.detector, "_SOURCE", "") != "mock_geometric_clusterer" and len(points) > 0:
            x_max_cloud = np.max(points[:, 0])
            if x_max_cloud < 0 or x_max_cloud > 100:
                print(f"[WARNING] Point cloud X range out of expected KITTI coordinate bounds (x_max={x_max_cloud:.1f}).")

        # Detect & Track with semantic cross-referencing
        try:
            detections = self.detector.detect(points, timestamp_s=timestamp_s, semantic_labels=semantic_labels)
        except TypeError:
            detections = self.detector.detect(points, timestamp_s=timestamp_s)

        from src.perception.object_detector import classify_cluster_from_semantics
        for d in detections:
            if d.class_name == "unknown_obstacle":
                cid, cname = classify_cluster_from_semantics(
                    d.center_xyz, d.size_lwh, points, semantic_labels, fallback_class="unclassified"
                )
                d.class_id = cid
                d.class_name = cname

        self._attach_terrain_features(detections, points)

        tracks = self.tracker.update(detections, timestamp_s=timestamp_s)

        track_velocities = {
            t.track_id: (float(t.state[3]), float(t.state[4]))
            for t in tracks
        }

        # Grids
        grids = self._generate_maps(points, tracks)

        t_end = time.perf_counter()
        latency = t_end - t_start
        target_fps = self.rate_hz

        metrics = self._get_hardware_metrics(fps, target_fps, latency)

        extras = getattr(self, "_last_extras", {})
        state = FrameState(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            points=points,
            tracks=tracks,
            track_velocities=track_velocities,
            grid_maps=grids,
            grid_resolution_m=self.grid_res,
            grid_extent_m=self.grid_extent,
            metrics=metrics,
            road_width_m=extras.get("road_width_m"),
            elevation_profile=extras.get("elevation_profile", []),
            elevation_hazards_m=extras.get("elevation_hazards_m", []),
            hazard_clusters=extras.get("hazard_clusters", []),
            mean_uncertainty=extras.get("mean_uncertainty"),
            occupied_fraction=extras.get("occupied_fraction"),
            centerline_profile=extras.get("centerline_profile", []),
            centerline_markers=extras.get("centerline_markers", []),
            has_ground_truth=has_ground_truth,
            predictor_mode=self.predictor_mode,
            accuracy_metrics=accuracy_metrics,
        )
        state.coordinate_warnings = assert_coordinate_frame_consistency(state)
        for w in state.coordinate_warnings:
            print(f"[WARNING] {w}")

        self._log_session(state)
        return state

    def run(self):
        self._setup_sources()
        dt = 1.0 / self.rate_hz
        current_time = 0.0
        last_t = time.perf_counter()

        for frame_idx in range(self.num_frames):
            if not self.running:
                break

            t_start = time.perf_counter()
            fps = 1.0 / (t_start - last_t) if (t_start - last_t) > 0 else self.rate_hz
            last_t = t_start

            state = self.process_frame(frame_idx, current_time, fps)
            self.state_ready.emit(state)

            current_time += dt
            elapsed = time.perf_counter() - t_start
            sleep_t = max(0, dt - elapsed)
            time.sleep(sleep_t)
            
    def _log_session(self, state: FrameState):
        import json
        import os
        from pathlib import Path
        
        log_dir = Path("outputs/phase9/session")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "dashboard_session.jsonl"
        
        # Log rotation based on size (e.g. 10MB)
        if log_path.exists() and log_path.stat().st_size > 10 * 1024 * 1024:
            log_path.rename(log_dir / f"dashboard_session_{int(time.time())}.jsonl")
            
        record = {
            "schema_version": state.schema_version,
            "frame_idx": state.frame_idx,
            "timestamp_s": state.timestamp_s,
            "num_points": len(state.points),
            "num_tracks": len(state.tracks),
            "fps": state.metrics.fps,
            "target_fps": state.metrics.target_fps,
            "achieved_fps": state.metrics.achieved_fps,
            "latency_ms": state.metrics.latency_ms,
            "cpu_percent": state.metrics.cpu_percent
        }
        
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            
    def stop(self):
        self.running = False
        self.wait()
