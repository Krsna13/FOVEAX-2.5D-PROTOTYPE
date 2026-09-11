"""Diagnostics and Telemetry module for Phase 10 ROS 2 Integration."""

import json
import os
import time
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class NodeDiagnostics:
    source_type: str = "live_sensor" # live_sensor, rosbag_replay, mock
    received_frames: int = 0
    processed_frames: int = 0
    dropped_frames_queue: int = 0
    dropped_frames_tf: int = 0
    total_invalid_points: int = 0
    processing_latency_ms_avg: float = 0.0
    end_to_end_latency_ms_avg: float = 0.0
    fps_received: float = 0.0
    fps_processed: float = 0.0

class DiagnosticsManager:
    def __init__(self, source_type: str = "live_sensor"):
        self.stats = NodeDiagnostics(source_type=source_type)
        
        # Setup logging paths
        self.log_dir = Path("outputs/phase10/ros2_session")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "ros2_session.jsonl"
        self.csv_path = self.log_dir / "ros2_metrics.csv"
        self.provenance_path = self.log_dir / "provenance.txt"
        
        # Write provenance
        with open(self.provenance_path, "w") as f:
            f.write(f"Session started at {time.time()}\n")
            f.write(f"Data source provenance: {source_type}\n")
            
        # Write CSV header if not exists
        if not self.csv_path.exists():
            with open(self.csv_path, "w") as f:
                f.write("timestamp,received,processed,dropped_queue,dropped_tf,fps_processed,latency_ms\n")
                
        self.last_report_time = time.time()
        self.frames_since_report = 0

    def rotate_logs_if_needed(self):
        """Rotate JSONL if > 50MB."""
        if self.jsonl_path.exists() and self.jsonl_path.stat().st_size > 50 * 1024 * 1024:
            self.jsonl_path.rename(self.log_dir / f"ros2_session_{int(time.time())}.jsonl")

    def log_frame(self, frame_metadata: dict, latency_ms: float, end_to_end_ms: float):
        """Log individual frame processing results."""
        self.stats.processed_frames += 1
        self.stats.total_invalid_points += frame_metadata.get("dropped_invalid_points", 0)
        
        # Moving averages
        alpha = 0.1
        if self.stats.processing_latency_ms_avg == 0.0:
            self.stats.processing_latency_ms_avg = latency_ms
            self.stats.end_to_end_latency_ms_avg = end_to_end_ms
        else:
            self.stats.processing_latency_ms_avg = (1 - alpha) * self.stats.processing_latency_ms_avg + alpha * latency_ms
            self.stats.end_to_end_latency_ms_avg = (1 - alpha) * self.stats.end_to_end_latency_ms_avg + alpha * end_to_end_ms
            
        self.frames_since_report += 1
        
        now = time.time()
        if now - self.last_report_time >= 1.0:
            self.stats.fps_processed = self.frames_since_report / (now - self.last_report_time)
            self.frames_since_report = 0
            self.last_report_time = now
            self._write_csv(now)

        self.rotate_logs_if_needed()
        record = {
            "timestamp": now,
            "latency_ms": latency_ms,
            "end_to_end_ms": end_to_end_ms,
            "metadata": frame_metadata,
            "stats": asdict(self.stats)
        }
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _write_csv(self, now: float):
        with open(self.csv_path, "a") as f:
            f.write(f"{now},{self.stats.received_frames},{self.stats.processed_frames},"
                    f"{self.stats.dropped_frames_queue},{self.stats.dropped_frames_tf},"
                    f"{self.stats.fps_processed:.1f},{self.stats.processing_latency_ms_avg:.1f}\n")

    def record_received(self):
        self.stats.received_frames += 1
        
    def record_dropped_queue(self):
        self.stats.dropped_frames_queue += 1
        
    def record_dropped_tf(self):
        self.stats.dropped_frames_tf += 1
