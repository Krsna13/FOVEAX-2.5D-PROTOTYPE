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
    parser.add_argument("--source", type=str, choices=["sample", "semantickitti"], default="sample", help="Data source")
    parser.add_argument("--detector", type=str, choices=["mock", "pointpillars"], default="mock", help="Detector type")
    # For pointpillars we would need more args, but sticking to basics for now as mock is default
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
            
        # Initialize Streamer
        self.streamer = DataStreamerThread(source=args.source, detector=detector)
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

def main():
    args = parse_args()
    app = DashboardApplication(args)
    app.run()

if __name__ == "__main__":
    main()
