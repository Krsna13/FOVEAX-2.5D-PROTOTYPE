import time
import numpy as np
import psutil
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

from PyQt5.QtCore import QThread, pyqtSignal

from src.dashboard.dashboard_state import FrameState, HardwareMetrics, assert_coordinate_frame_consistency
from src.tracking.multi_object_tracker import MultiObjectTracker
from src.perception.object_detector import ObjectDetector, MockObjectDetector
import importlib
_trk = importlib.import_module('src.11_object_detection_tracking')
generate_synthetic_sample_cloud = _trk.generate_synthetic_sample_cloud
repeated_cloud_with_translation = _trk.repeated_cloud_with_translation

class DataStreamerThread(QThread):
    state_ready = pyqtSignal(FrameState)
    
    def __init__(self, source: str = "sample", detector: ObjectDetector = None, num_frames: int = 100):
        super().__init__()
        self.source = source
        self.detector = detector if detector else MockObjectDetector()
        self.tracker = MultiObjectTracker()
        self.num_frames = num_frames
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

    def run(self):
        # Set up sample cloud
        static_pts, moving_pts = generate_synthetic_sample_cloud(n_static=2000, n_moving=150)
        translation_step = np.array([0.5, 0.0, 0.0], dtype=np.float64)
        
        current_time = 0.0
        dt = 0.1
        
        last_t = time.perf_counter()
        
        for frame_idx in range(self.num_frames):
            if not self.running:
                break
                
            t_start = time.perf_counter()
            
            # 1. Prepare data
            offset = translation_step * frame_idx
            frame_moving = repeated_cloud_with_translation(moving_pts, offset)
            points = np.concatenate([static_pts, frame_moving], axis=0)
            
            # 2. Detect & Track
            # Coordinate frame assertion for PointPillars (Task O)
            if self.detector._SOURCE != "mock_geometric_clusterer":
                # Typical KITTI range: x in [0, 70.4], y in [-40, 40], z in [-3, 1]
                # Just print a warning if completely out of bounds
                x_max_cloud = np.max(points[:, 0])
                if x_max_cloud < 0 or x_max_cloud > 100:
                    print(f"[WARNING] Point cloud X range out of expected KITTI coordinate bounds (x_max={x_max_cloud:.1f}). Ensure coordinate transformation is applied.")

            detections = self.detector.detect(points, timestamp_s=current_time)
            tracks = self.tracker.update(detections, timestamp_s=current_time)
            
            # Populate track velocities
            track_velocities = {}
            for t in tracks:
                track_velocities[t.track_id] = (float(t.state[3]), float(t.state[4]))
            
            # 3. Grids
            grids = self._generate_maps(points, tracks)

            t_end = time.perf_counter()
            latency = t_end - t_start
            fps = 1.0 / (t_end - last_t) if (t_end - last_t) > 0 else 0.0
            target_fps = 1.0 / dt
            last_t = t_end
            
            # 4. Metrics
            metrics = self._get_hardware_metrics(fps, target_fps, latency)
            
            # 5. Emit
            state = FrameState(
                frame_idx=frame_idx,
                timestamp_s=current_time,
                points=points,
                tracks=tracks,
                track_velocities=track_velocities,
                grid_maps=grids,
                grid_resolution_m=self.grid_res,
                grid_extent_m=self.grid_extent,
                metrics=metrics
            )

            # Cross-panel origin-consistency check (Task O). Non-blocking: logs
            # and surfaces warnings on the frame instead of crashing the thread.
            state.coordinate_warnings = assert_coordinate_frame_consistency(state)
            for w in state.coordinate_warnings:
                print(f"[WARNING] {w}")

            self.state_ready.emit(state)
            
            # 6. Log session data (Task Q)
            self._log_session(state)
            
            current_time += dt
            
            # Sleep to maintain roughly ~10Hz
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
