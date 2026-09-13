import sys
import argparse
import importlib
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
        default="rellis3d", help="Data source ('sample', 'semantickitti', 'rellis3d')"
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
        # Default: the curated ~2-minute real demo range. Sequence 00001
        # alone (real, unstitched) was confirmed to contain all three
        # terrain features -- real overhead obstacles from frame 0 onward,
        # a real depression (~3.5m into the corridor), and a real height
        # bump (~4.8m) -- see docs referenced in the task report. At the
        # real confirmed throughput (~14.7 FPS) and the --rate-hz default
        # below, 1700 real frames span ~121s (~2:01), comfortably inside
        # the sequence's 2319 available real frames -- no stitching or
        # looping needed.
        "--frames", type=int, default=1700,
        help="Number of frames to process"
    )
    parser.add_argument(
        "--rate-hz", type=float, default=14.0,
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
    parser.add_argument(
        "--screenshot-after", type=int, default=None,
        help="Save a QWidget.grab() of the main window to --screenshot-path "
             "after this many frames, then exit. Avoids relying on desktop "
             "screen capture (which can grab the wrong window when other "
             "windows overlap this one)."
    )
    parser.add_argument(
        "--screenshot-path", type=str, default="outputs/phase9/final_dashboard_confirmed.png",
    )
    parser.add_argument(
        "--capture-sequence", type=str, default=None,
        help="Advance through the entire real sequence saving a screenshot "
             "every --capture-every frames to this directory, then exit."
    )
    parser.add_argument(
        "--capture-every", type=int, default=10,
        help="Frame interval for --capture-sequence (default: every 10th real frame)."
    )
    parser.add_argument(
        "--predictor", type=str, choices=["ground_truth", "mock", "salsanext"],
        default="ground_truth",
        help="Semantic class source for the live accuracy comparison "
             "(default: 'ground_truth', matching export_web_dashboard_data.py's "
             "own default -- comparing ground truth against itself, trivially "
             "~100%%, clearly labeled as such). Use 'salsanext' to compare a "
             "real model prediction against real ground truth."
    )
    parser.add_argument("--salsanext-repo", type=str, default="external/SalsaNext")
    parser.add_argument("--checkpoint", type=str,
                         default="models/salsanext/pretrained/pretrained/SalsaNext")
    parser.add_argument("--config", type=str,
                         default="models/salsanext/pretrained/pretrained/arch_cfg.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fov-up", type=float, default=None)
    parser.add_argument("--fov-down", type=float, default=None)
    parser.add_argument("--rescale-intensity", action="store_true", default=False)
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

        # Real camera controls: the left panel's view/zoom/rotate/reset
        # buttons emit this signal, forwarded here as a command dict onto
        # the same o3d_queue (run_open3d_process distinguishes command
        # dicts from FrameState/QUIT items, see open3d_viewer.py). This
        # drives Open3D's actual ViewControl API, not a simulated camera.
        self.qt_window.camera_command_requested.connect(self._send_camera_command)

        # Detector
        if args.detector == "mock":
            detector = MockObjectDetector()
        else:
            detector = MockObjectDetector()
            print("[WARNING] PointPillars requested but not fully implemented in CLI args, using mock.")
            
        # Sequence default: "00001" for rellis3d -- the curated ~2-minute
        # real demo sequence (see --frames/--rate-hz defaults above).
        seq = args.sequence
        if seq is None:
            seq = "00001" if args.source == "rellis3d" else "00"

        # --- Playback controller ----------------------------------------
        # This dashboard replays a REAL, finite, recorded sequence of
        # LiDAR frames (RELLIS-3D/SemanticKITTI files, or the deterministic
        # synthetic sample generator) -- it is not a live sensor feed, so
        # frames are addressable by index and this can be a real pull-based
        # playback controller rather than a one-way autonomous stream.
        #
        # DataStreamerThread.process_frame(idx, ts, fps) is already a pure,
        # synchronous, fully-tested method that computes one real FrameState
        # for an arbitrary frame index (run_headless() already calls it this
        # way). Reusing it directly here -- driven by a single QTimer on the
        # Qt thread instead of DataStreamerThread's own background QThread
        # loop -- means every panel (3D view, 2D map, tracked objects,
        # hazard table, centerline profile) is updated from the exact same
        # FrameState object on every tick. The previous design ran two
        # independent, uncoordinated throttles (a fixed 10 FPS gate on the
        # 2D panels here, and a separate drop-oldest queue for the 3D view)
        # that could show two different frames on screen at once; this
        # design cannot desync because there is only one frame in flight.
        # Real semantic predictor for the live accuracy comparison
        # (Task: live metrics). Reuses
        # src/dashboard/export_web_dashboard_data.py::make_semantic_predictor
        # directly -- the exact same constructor already used and
        # validated by the web-export bridge -- rather than duplicating
        # its predictor-selection logic here.
        semantic_predictor = None
        if args.predictor != "ground_truth":
            export_module = importlib.import_module("src.dashboard.export_web_dashboard_data")
            semantic_predictor = export_module.make_semantic_predictor(args.predictor, args)

        self.streamer = DataStreamerThread(
            source=args.source,
            detector=detector,
            num_frames=args.frames,
            sequence=seq,
            rate_hz=args.rate_hz,
            semantic_predictor=semantic_predictor,
            predictor_mode=args.predictor,
        )
        self.streamer._setup_sources()

        # Cache of already-computed real FrameStates, keyed by frame index.
        # Stepping backward replays an already-computed real frame from
        # this cache -- it never re-runs the (stateful, Kalman-filtered)
        # tracker "backward in time", which would corrupt its real track
        # history. Stepping forward past the cache computes a new real
        # frame exactly once, advancing the tracker forward exactly as
        # normal playback always has.
        self._frame_cache: dict[int, object] = {}
        self._playback_idx = -1
        self._playing = True
        self._rate_hz = args.rate_hz

        self.qt_window.play_pause_requested.connect(self._toggle_play_pause)
        self.qt_window.step_forward_requested.connect(lambda: self._show_frame(self._playback_idx + 1))
        self.qt_window.step_backward_requested.connect(lambda: self._show_frame(self._playback_idx - 1))
        self.qt_window.rate_changed.connect(self._set_rate_hz)
        self.qt_window.restart_requested.connect(self._restart)
        self.qt_window.environment_requested.connect(self._switch_environment)
        self.qt_window.set_rate_hz(self._rate_hz)

        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._on_timer_tick)
        self._apply_timer_interval()

        # Throttle the informational, real point-cloud-registration (ICP)
        # ego-displacement estimate (src/perception/ego_motion_estimate.py)
        # -- it costs tens of ms per real frame pair, real but non-trivial
        # next to a 10 Hz (100 ms) frame budget, and it is purely
        # informational (never used to reposition any view), so it is
        # computed only every few real frames rather than every one.
        self._ego_motion_every_n_frames = 5
        self._last_ego_motion_m: float | None = None

        self._screenshot_after = getattr(args, "screenshot_after", None)
        self._screenshot_path = getattr(args, "screenshot_path", None)

    def _send_camera_command(self, cmd: str):
        """Forward a real camera command (from the dashboard's left panel)
        to the Open3D process via the shared o3d_queue."""
        try:
            self.o3d_queue.put_nowait({"cmd": cmd})
        except Exception:
            pass

    def _apply_timer_interval(self):
        interval_ms = max(1, int(round(1000.0 / self._rate_hz)))
        self.playback_timer.setInterval(interval_ms)

    def _set_rate_hz(self, hz: float):
        self._rate_hz = max(1.0, hz)
        self._apply_timer_interval()

    def _toggle_play_pause(self):
        self._playing = not self._playing
        self.qt_window.set_play_pause_state(self._playing)

    def _switch_environment(self, seq: str):
        """Switch between real RELLIS-3D recorded sequences."""
        if self.streamer.sequence == seq:
            return
            
        self.streamer.sequence = seq
        self.streamer._setup_sources()
        
        # Default view on switch: Traversability
        # so the difference (non-drivable cells) is immediately visible
        self.qt_window._on_map_type_clicked("traversability")
        self.qt_window.set_active_environment(seq)
        self._restart()

    def _restart(self):
        """Reset the curated real playback to frame 0 with completely
        fresh tracker state -- no track history, hazard clusters, or
        accuracy numbers carry over from the previous run.

        A fresh MultiObjectTracker() clears all track IDs back to 0 (its
        own __init__ resets _next_track_id and _tracks). The frame cache
        is also dropped entirely: cached FrameStates were computed against
        the OLD tracker instance, so keeping them would silently resurrect
        the old track history the moment playback revisited a cached
        frame index.
        """
        self.playback_timer.stop()
        from src.tracking.multi_object_tracker import MultiObjectTracker
        self.streamer.tracker = MultiObjectTracker()
        self._frame_cache.clear()
        self._playback_idx = -1
        self._last_ego_motion_m = None
        self._show_frame(0)
        self._playing = True
        self.qt_window.set_play_pause_state(True)
        self.playback_timer.start()

    def _on_timer_tick(self):
        if not self._playing:
            return
        if self._playback_idx + 1 >= self.streamer.num_frames:
            return  # real sequence exhausted; hold on the last real frame
        self._show_frame(self._playback_idx + 1)

    def _show_frame(self, idx: int):
        """Compute (or replay from cache) the real frame at *idx* and push
        the SAME FrameState to every panel: 3D view, 2D map, tracked
        objects, hazard table, and the centerline profile."""
        idx = max(0, min(idx, self.streamer.num_frames - 1))
        if idx == self._playback_idx and idx in self._frame_cache:
            return  # already showing this exact real frame

        went_backward = idx < self._playback_idx
        self._playback_idx = idx

        if idx in self._frame_cache:
            state = self._frame_cache[idx]
        else:
            ts = idx / self._rate_hz
            state = self.streamer.process_frame(idx, ts, fps=self._rate_hz)
            self._frame_cache[idx] = state

            # Informational-only real ego-displacement estimate, computed
            # against the immediately preceding REAL frame's real points
            # (never fabricated), throttled per the comment in __init__.
            if idx > 0 and idx % self._ego_motion_every_n_frames == 0 and (idx - 1) in self._frame_cache:
                try:
                    from src.perception.ego_motion_estimate import estimate_relative_displacement_m
                    prev_pts = self._frame_cache[idx - 1].points[:, :3]
                    curr_pts = state.points[:, :3]
                    self._last_ego_motion_m = estimate_relative_displacement_m(prev_pts, curr_pts)
                except Exception:
                    self._last_ego_motion_m = None

        # Push the identical FrameState to the 3D process and to every 2D
        # panel -- no independent throttling, so nothing can lag behind.
        try:
            if self.o3d_queue.full():
                self.o3d_queue.get_nowait()
            self.o3d_queue.put_nowait(state)
        except Exception:
            pass

        self.qt_window.update_ui(state)
        self.qt_window.set_frame_counter(idx + 1, self.streamer.num_frames)
        self.qt_window.set_ego_motion_estimate(self._last_ego_motion_m)

        if went_backward:
            print(f"[INFO] Stepped back to real frame {idx + 1}/{self.streamer.num_frames} "
                  "(replayed from cache, tracker not re-run).")

        self._maybe_take_screenshot(state)

    def _maybe_take_screenshot(self, state):
        if self._screenshot_after is None or self._playback_idx + 1 < self._screenshot_after:
            return
        self._screenshot_after = None  # only once
        from pathlib import Path
        out_path = Path(self._screenshot_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.qt_window.grab().save(str(out_path))
        print(f"[INFO] Screenshot saved to {out_path}")
        for i, tab_name in enumerate(["status", "terrain", "system"]):
            self.qt_window.tabs.setCurrentIndex(i)
            self.app.processEvents()
            tab_path = out_path.with_name(out_path.stem + f"_{tab_name}" + out_path.suffix)
            self.qt_window.grab().save(str(tab_path))
            print(f"[INFO] Screenshot saved to {tab_path}")

        self.qt_window.tabs.setCurrentIndex(0)
        self.qt_window._on_map_changed("overhead")
        self.qt_window.update_ui(state)  # force a re-render with the new layer selection
        self.app.processEvents()
        overhead_path = out_path.with_name(out_path.stem + "_overhead" + out_path.suffix)
        self.qt_window.grab().save(str(overhead_path))
        print(f"[INFO] Screenshot saved to {overhead_path}")
        self.qt_window.tabs.setCurrentIndex(0)
        self.qt_window._on_map_changed("elevation")
        self.app.processEvents()

        # Capture both PLAY and PAUSE button states for visual review
        # (Task: PAUSE/PLAY contrast fix).
        self.qt_window.set_play_pause_state(True)
        self.app.processEvents()
        play_path = out_path.with_name(out_path.stem + "_playing" + out_path.suffix)
        self.qt_window.grab().save(str(play_path))
        print(f"[INFO] Screenshot saved to {play_path}")

        self.qt_window.set_play_pause_state(False)
        self.app.processEvents()
        pause_path = out_path.with_name(out_path.stem + "_paused" + out_path.suffix)
        self.qt_window.grab().save(str(pause_path))
        print(f"[INFO] Screenshot saved to {pause_path}")

        self.app.quit()

    def _capture_frame_sequence(self, out_dir: str, every_n: int):
        """Advance through the real sequence, saving a screenshot every
        *every_n* real frames to *out_dir*, then quit. Used for the
        motion-playback verification sequence (--capture-sequence)."""
        from pathlib import Path
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        self.playback_timer.stop()
        self._playing = False

        for idx in range(self.streamer.num_frames):
            self._show_frame(idx)
            self.app.processEvents()
            if idx % every_n == 0 or idx == self.streamer.num_frames - 1:
                path = out_dir_path / f"frame_{idx:04d}.png"
                self.qt_window.grab().save(str(path))
                print(f"[INFO] Sequence screenshot saved: {path}")
        self.app.quit()

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
        # The Open3D child must spawn, import torch/open3d and create its GLFW
        # window before it can report a handle -- routinely well over 8s on a
        # loaded GPU. A short blocking wait here gave up early and left the
        # 3D view floating, so poll on a QTimer inside the running event loop
        # instead, for as long as the child itself keeps searching.
        self._reparent_deadline = time.time() + 60.0
        self._reparent_timer = QTimer()
        self._reparent_timer.setInterval(100)
        self._reparent_timer.timeout.connect(self._poll_reparent)
        self._reparent_timer.start()

    def _poll_reparent(self):
        try:
            child_hwnd = self._ready_queue.get_nowait()
        except Exception:
            if time.time() > self._reparent_deadline:
                self._reparent_timer.stop()
                print("[WARNING] Open3D window handle never arrived -- the 3D view "
                      "stays a separate floating window.", flush=True)
            return

        self._reparent_timer.stop()
        if not child_hwnd:
            print("[WARNING] Open3D child could not find its own window -- the 3D "
                  "view stays a separate floating window.", flush=True)
            return
        self._embed_child(child_hwnd)

    def _embed_child(self, child_hwnd: int):
        from src.dashboard.win32_embed import reparent_as_child

        try:
            embed_widget = self.qt_window.embed_widget
            # Force real native window handles to exist before SetParent.
            self.qt_window.winId()
            embed_widget.winId()
            parent_hwnd = int(embed_widget.winId())
            reparent_as_child(child_hwnd, parent_hwnd, embed_widget.width(), embed_widget.height())
            embed_widget.child_hwnd = child_hwnd
            self._embedded = True
            print("[INFO] Open3D 3D view embedded into the main dashboard window.", flush=True)
        except Exception as e:
            print(f"[WARNING] Reparenting the Open3D window failed ({e!r}); "
                  "it will remain a separate floating window instead.", flush=True)

    def run(self, capture_sequence: str | None = None, capture_every: int = 10):
        self.qt_window.show()
        if self._attempt_embed:
            self._try_reparent()

        self._show_frame(0)  # render the real first frame immediately

        if capture_sequence is not None:
            self._capture_frame_sequence(capture_sequence, capture_every)
        else:
            self.playback_timer.start()

        # Start event loop
        exit_code = self.app.exec_()

        # Cleanup
        self.playback_timer.stop()
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
    seq = args.sequence or ("00001" if args.source == "rellis3d" else "00")
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
        app.run(capture_sequence=args.capture_sequence, capture_every=args.capture_every)


if __name__ == "__main__":
    main()
