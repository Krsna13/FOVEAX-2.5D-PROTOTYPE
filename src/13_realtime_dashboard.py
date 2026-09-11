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
    parser.add_argument(
        "--separate-windows", action="store_true",
        help="Force the original two-window layout (2D dashboard + a separate "
             "floating Open3D window) even if pywin32 is available. By default, "
             "when pywin32 is installed the Open3D view is embedded as a panel "
             "inside the main dashboard window."
    )
    return parser.parse_args()

from multiprocessing import Process, Queue
import time

class DashboardApplication:
    def __init__(self, args):
        self.app = QApplication(sys.argv)

        # --- Decide whether to attempt embedding the Open3D window ------
        # This is a windowing/visual change only: the Open3D render loop
        # still runs in its own process (self.o3d_process below), on its
        # own event loop, completely separate from the Qt event loop here
        # -- that split is unchanged and is not what this feature touches.
        from src.dashboard.win32_embed import WIN32_AVAILABLE
        self._attempt_embed = WIN32_AVAILABLE and not getattr(args, "separate_windows", False)
        self._embedded = False
        if WIN32_AVAILABLE and getattr(args, "separate_windows", False):
            print("[INFO] --separate-windows requested; using the original two-window layout.")
        elif not WIN32_AVAILABLE:
            print("[INFO] pywin32 (win32gui) is not installed -- falling back to the "
                  "original two-window layout. Install it with 'pip install pywin32' "
                  "to combine the 3D view into the main dashboard window.")

        # Initialize UI Components
        self.qt_window = FoveaXDashboardWindow(embed_3d=self._attempt_embed)

        # Multiprocessing for Open3D Viewer (unchanged: separate process,
        # separate render loop, communicated with only via self.o3d_queue).
        self.o3d_queue = Queue(maxsize=10)
        from src.dashboard.open3d_viewer import run_open3d_process
        self._ready_queue = Queue() if self._attempt_embed else None
        self.o3d_process = Process(
            target=run_open3d_process, args=(self.o3d_queue, self._ready_queue)
        )
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
        
    def _try_reparent(self):
        """Wait for the Open3D child process to report its window handle,
        then reparent it into self.qt_window.embed_widget.

        Called only when self._attempt_embed is True (win32gui available
        and not opted out). Any failure here -- timeout waiting for the
        handle, or an error during the actual reparent call -- is caught
        and logged; the Open3D window simply remains a normal, separate,
        free-floating top-level window in that case (the pre-existing
        behavior), and nothing else about the app's operation changes.
        """
        from src.dashboard.win32_embed import reparent_as_child

        # Poll rather than a single blocking get(timeout=...): this method
        # runs after self.qt_window.show() but before self.app.exec_(), so
        # nothing is pumping the Qt event loop yet -- a plain blocking wait
        # here would make the just-shown window appear frozen/unresponsive
        # for up to the full timeout. processEvents() between polls keeps
        # it painting/responsive while we wait for the child's handle.
        child_hwnd = None
        deadline = time.time() + 8.0
        while time.time() < deadline:
            try:
                child_hwnd = self._ready_queue.get_nowait()
                break
            except Exception:
                pass
            self.app.processEvents()
            time.sleep(0.05)

        if not child_hwnd:
            print("[INFO] Could not locate the Open3D window handle in time -- "
                  "showing it as a separate floating window instead (unchanged "
                  "prior behavior).")
            return

        try:
            embed_widget = self.qt_window.embed_widget
            # Force real native window handles to exist before SetParent.
            self.qt_window.winId()
            embed_widget.winId()
            parent_hwnd = int(embed_widget.winId())
            reparent_as_child(child_hwnd, parent_hwnd, embed_widget.width(), embed_widget.height())
            embed_widget.child_hwnd = child_hwnd
            self._embedded = True
            print("[INFO] Open3D 3D view embedded into the main dashboard window.")
        except Exception as e:
            print(f"[WARNING] Reparenting the Open3D window failed ({e!r}); "
                  "it will remain a separate floating window instead.")

    def run(self):
        self.qt_window.show()
        if self._attempt_embed:
            self._try_reparent()
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
