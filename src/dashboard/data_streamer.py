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
    ):
        super().__init__()
        self.source = source.lower()
        self.detector = detector if detector else MockObjectDetector()
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
            return {"elevation": empty, "traversability": empty, "roi": empty}
        
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
        
        # Flip vertically so row 0 is max Y (top of image)
        return {
            "elevation": np.flipud(elevation),
            "traversability": np.flipud(traversability),
            "roi": np.flipud(roi)
        }

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

    def _get_frame_points(self, frame_idx: int) -> np.ndarray:
        """Fetch points for frame_idx from active source."""
        if self.source == "sample":
            offset = self._sample_translation_step * frame_idx
            frame_moving = repeated_cloud_with_translation(self._sample_moving_pts, offset)
            return np.concatenate([self._sample_static_pts, frame_moving], axis=0)

        if not self._file_list:
            return np.zeros((0, 4), dtype=np.float32)

        file_path = self._file_list[frame_idx % len(self._file_list)]

        if self.source == "rellis3d" and HAS_RELLIS_LOADER:
            rellis_root = self.data_root or Path("C:/dev/data/rellis3d")
            if not rellis_root.exists():
                rellis_root = Path("data/rellis3d")
            try:
                frame_id = file_path.stem
                rf = load_rellis3d_frame(rellis_root, self.sequence, frame_id)
                return rf.points
            except Exception:
                # Fallback to direct raw bin loading if labels missing
                raw = np.fromfile(file_path, dtype=np.float32)
                return raw.reshape(-1, 4)

        # Standard SemanticKITTI or raw binary
        raw = np.fromfile(file_path, dtype=np.float32)
        if raw.size % 4 != 0:
            return np.zeros((0, 4), dtype=np.float32)
        return raw.reshape(-1, 4)

    def process_frame(self, frame_idx: int, timestamp_s: float, fps: float = 10.0) -> FrameState:
        """Process a single frame through detection, tracking, grid mapping, and metrics."""
        t_start = time.perf_counter()
        points = self._get_frame_points(frame_idx)

        # Coordinate frame sanity check
        if getattr(self.detector, "_SOURCE", "") != "mock_geometric_clusterer" and len(points) > 0:
            x_max_cloud = np.max(points[:, 0])
            if x_max_cloud < 0 or x_max_cloud > 100:
                print(f"[WARNING] Point cloud X range out of expected KITTI coordinate bounds (x_max={x_max_cloud:.1f}).")

        # Detect & Track
        detections = self.detector.detect(points, timestamp_s=timestamp_s)
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
