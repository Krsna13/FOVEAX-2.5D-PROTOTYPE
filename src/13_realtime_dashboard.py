import sys
import argparse
from pathlib import Path

# Fix sys.path for relative imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# torch must be imported before PyQt5 in this process: on Windows, PyQt5's
# bundled Qt DLLs conflict with torch's bundled CUDA DLLs (c10.dll) when
# PyQt5 loads first, causing an import-time crash (WinError 1114 / access
# violation). Only matters once a real (non-mock) torch-based detector is
# wired in below, but the ordering has to be set here regardless.
try:
    import torch  # noqa: F401
except ImportError:
    pass

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

from src.dashboard.qt_main_window import FoveaXDashboardWindow
from src.dashboard.open3d_viewer import FoveaX3DViewer
from src.dashboard.data_streamer import DataStreamerThread
from src.perception.object_detector import MockObjectDetector

def parse_args():
    parser = argparse.ArgumentParser(description="FOVEAX Phase 9 - Real-Time Dashboard")
    parser.add_argument(
        "--source", type=str, choices=["sample", "semantickitti", "rellis3d"],
        default="sample", help="Data source ('sample', 'semantickitti', 'rellis3d')"
    )
    parser.add_argument(
        "--detector", type=str, choices=["mock", "pointpillars"],
        default="mock", help="Detector type"
    )
    parser.add_argument(
        "--sequence", type=str, default=None,
        help="Sequence name (e.g. '00' for KITTI, '00000' for RELLIS-3D)"
    )
    parser.add_argument(
        "--frames", type=int, default=100,
        help="Number of frames to process"
    )
    parser.add_argument(
        "--rate-hz", type=float, default=10.0,
        help="Target streaming rate in Hz"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run streamer in headless mode without GUI/Open3D display (writes telemetry to outputs/phase9/session/)"
    )
    return parser.parse_args()

from multiprocessing import Process, Queue
import time

class DashboardApplication:
    def __init__(self, args):
        self.app = QApplication(sys.argv)
        
        # Initialize UI Components
        self.qt_window = FoveaXDashboardWindow()
        
        # Multiprocessing for Open3D Viewer
        self.o3d_queue = Queue(maxsize=10)
        from src.dashboard.open3d_viewer import run_open3d_process
        self.o3d_process = Process(target=run_open3d_process, args=(self.o3d_queue,))
        self.o3d_process.start()
        
        # Detector
        if args.detector == "mock":
            detector = MockObjectDetector()
        else:
            detector = MockObjectDetector()
            print("[WARNING] PointPillars requested but not fully implemented in CLI args, using mock.")
            
        # Sequence default
        seq = args.sequence
        if seq is None:
            seq = "00000" if args.source == "rellis3d" else "00"

        # Initialize Streamer
        self.streamer = DataStreamerThread(
            source=args.source,
            detector=detector,
            num_frames=args.frames,
            sequence=seq,
            rate_hz=args.rate_hz,
        )
        self.streamer.state_ready.connect(self.on_state_ready)
        
        # Frame desync management
        self.last_2d_render_time = 0.0
        self.target_2d_fps = 10.0 # Throttle 2D panel to 10 FPS
        
    def on_state_ready(self, state):
        """Called when DataStreamer emits a new FrameState."""
        
        # Send to 3D Viewer process (non-blocking, drop if full)
        try:
            if self.o3d_queue.full():
                self.o3d_queue.get_nowait() # drop oldest
            self.o3d_queue.put_nowait(state)
        except Exception:
            pass
            
        # Desync logic: Throttle 2D update if needed
        current_time = time.time()
        if (current_time - self.last_2d_render_time) >= (1.0 / self.target_2d_fps):
            self.qt_window.update_ui(state)
            self.last_2d_render_time = current_time
        
    def run(self):
        self.qt_window.show()
        self.streamer.start()
        
        # Start event loop
        exit_code = self.app.exec_()
        
        # Cleanup
        self.streamer.stop()
        self.o3d_queue.put("QUIT")
        self.o3d_process.join(timeout=2.0)
        if self.o3d_process.is_alive():
            self.o3d_process.terminate()
            
        sys.exit(exit_code)

def run_headless(args):
    """Run data streamer without PyQt5 GUI or Open3D windows."""
    print("=" * 60)
    print("FOVEAX Phase 9 - Headless Session Replay & Telemetry")
    print("=" * 60)
    print(f"Source:     {args.source}")
    seq = args.sequence or ("00000" if args.source == "rellis3d" else "00")
    print(f"Sequence:   {seq}")
    print(f"Frames:     {args.frames}")
    print(f"Rate:       {args.rate_hz} Hz")
    print("-" * 60)

    detector = MockObjectDetector()
    streamer = DataStreamerThread(
        source=args.source,
        detector=detector,
        num_frames=args.frames,
        sequence=seq,
        rate_hz=args.rate_hz,
    )
    streamer._setup_sources()

    dt = 1.0 / args.rate_hz
    start_time = time.time()

    for idx in range(args.frames):
        ts = idx * dt
        t0 = time.perf_counter()
        state = streamer.process_frame(idx, ts, fps=args.rate_hz)
        latency = (time.perf_counter() - t0) * 1000.0

        if (idx + 1) % 5 == 0 or idx == 0 or idx == args.frames - 1:
            print(
                f"[Frame {state.frame_idx + 1:3d}/{args.frames}] "
                f"Points: {len(state.points):6,d} | "
                f"Tracks: {len(state.tracks):2d} | "
                f"Latency: {latency:5.1f}ms | "
                f"CPU: {state.metrics.cpu_percent:4.1f}%"
            )

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Headless replay completed in {total_time:.2f}s ({args.frames / max(total_time, 1e-3):.1f} FPS)")
    print("Session telemetry saved to outputs/phase9/session/dashboard_session.jsonl")
    print("=" * 60)


def main():
    args = parse_args()
    if args.headless:
        run_headless(args)
    else:
        app = DashboardApplication(args)
        app.run()


if __name__ == "__main__":
    main()
