import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QGroupBox, QListWidget, QProgressBar,
    QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QFrame, QSizePolicy, QHeaderView, QSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from src.dashboard.dashboard_state import FrameState
from src.dashboard.win32_embed import resize_child

# Real resolution-zone spec (src/07_adaptive_2point5d_map.py SPEC_ZONES_100M),
# quoted directly for the System Details drawer -- not re-derived or guessed.
RESOLUTION_ZONES = [
    ("Near", "0-15 m", "5 cm"),
    ("Middle", "15-35 m", "20 cm"),
    ("Far", "35-100 m", "50 cm"),
]

# Canonical traversability convention (src/06_terrain_traversability.py).
TRAVERSABILITY_SAFE_MIN = 0.70
TRAVERSABILITY_CAUTION_MIN = 0.40

# Dashboard color tokens, matched to foveax_rough_dashboard.html's CSS
# custom properties (kept identical so the restyled Qt UI reads as the same
# visual system as the reference/web dashboards).
COLOR_BG = "#060B14"
COLOR_PANEL = "#0B1420"
COLOR_PANEL_ALT = "#0E1B2B"
COLOR_BORDER = "#16324A"
COLOR_TEXT = "#E2E8F0"
COLOR_TEXT_DIM = "#64748B"
COLOR_SAFE = "#34D399"
COLOR_CAUTION = "#FBBF24"
COLOR_CRITICAL = "#F43F5E"
COLOR_CYAN = "#22D3EE"
COLOR_DYNAMIC = "#A855F7"
COLOR_STATIC = "#FB923C"

# Shared typography scale, applied consistently across every widget added
# incrementally in separate tasks (confidence rings, hazard table,
# overhead legend, centerline profile, accuracy/efficiency/ego-motion
# cards) so they read as one coherent design rather than separately
# styled pieces:
#   NOTE      -- small dim explanatory/caveat text under a panel title.
#   BODY      -- default readable text (matches the base 11px in STYLESHEET).
#   EMPHASIS  -- a single inline number/word that should stand out a little
#                within its group (e.g. "Overall" accuracy).
#   HEADLINE  -- a card's one big real-time stat (e.g. road width).
#   HERO      -- the single most prominent status word on the whole
#                dashboard (ego terrain status) -- intentionally the
#                largest, there is exactly one of these per screen.
FONT_SIZE_NOTE = "9px"
FONT_SIZE_BODY = "11px"
FONT_SIZE_EMPHASIS = "12px"
FONT_SIZE_HEADLINE = "16px"
FONT_SIZE_HERO = "18px"

# Shared style string for small dim explanatory/caveat labels -- applied
# identically everywhere this kind of text appears (previously varied:
# some had this exact rule, some had no font-size at all).
NOTE_STYLE = f"color: {COLOR_TEXT_DIM}; font-size: {FONT_SIZE_NOTE};"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {COLOR_BG};
    color: {COLOR_TEXT};
    font-family: 'Segoe UI', Inter, sans-serif;
    font-size: 11px;
}}
QGroupBox {{
    background-color: {COLOR_PANEL};
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
    margin-top: 14px;
    font-weight: 600;
    color: {COLOR_CYAN};
    padding: 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}}
QLabel {{ color: {COLOR_TEXT}; }}
QPushButton {{
    background-color: {COLOR_PANEL_ALT};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    color: {COLOR_TEXT};
}}
QPushButton:hover {{ border: 1px solid {COLOR_CYAN}; }}
QPushButton:checked {{
    background-color: {COLOR_CYAN};
    color: #041019;
    font-weight: 600;
}}
QListWidget, QTableWidget, QComboBox {{
    background-color: {COLOR_PANEL_ALT};
    border: 1px solid {COLOR_BORDER};
    color: {COLOR_TEXT};
    gridline-color: {COLOR_BORDER};
}}
QHeaderView::section {{
    background-color: {COLOR_PANEL};
    color: {COLOR_TEXT_DIM};
    border: 1px solid {COLOR_BORDER};
    padding: 3px;
}}
QTabWidget::pane {{ border: 1px solid {COLOR_BORDER}; background: {COLOR_PANEL}; }}
QTabBar::tab {{
    background: {COLOR_PANEL_ALT};
    color: {COLOR_TEXT_DIM};
    padding: 6px 14px;
    border: 1px solid {COLOR_BORDER};
}}
QTabBar::tab:selected {{ color: {COLOR_CYAN}; background: {COLOR_PANEL}; font-weight: 600; }}
QProgressBar {{
    background-color: {COLOR_PANEL_ALT};
    border: 1px solid {COLOR_BORDER};
    border-radius: 3px;
    text-align: center;
    color: {COLOR_TEXT};
}}
QProgressBar::chunk {{ background-color: {COLOR_CYAN}; }}
"""


class Open3DEmbedWidget(QWidget):
    """Placeholder panel that hosts the reparented Open3D native window.

    This widget itself draws nothing -- it exists only to (a) provide a
    real native HWND for src/13_realtime_dashboard.py to reparent the
    Open3D child process's window into, and (b) forward Qt resize events
    to that reparented window via win32gui.MoveWindow, since a foreign
    HWND does not automatically follow its new Qt "parent" widget's size.

    `child_hwnd` is set externally once the reparenting handshake in
    DashboardApplication succeeds; until then (or if it never succeeds,
    see the fallback path there) this is just an empty black panel and the
    Open3D window shows separately, exactly as before this feature existed.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.child_hwnd = None
        # Forces Qt to back this widget with a real native window (HWND) up
        # front, rather than the "alien widget" optimization Qt normally
        # uses for plain child widgets on Windows -- SetParent needs a real
        # HWND to attach to.
        self.setAttribute(Qt.WA_NativeWindow, True)
        self.setStyleSheet("background-color: black;")
        self.setMinimumSize(300, 300)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.child_hwnd:
            resize_child(self.child_hwnd, self.width(), self.height())


class ElevationProfileWidget(QWidget):
    """Real elevation profile (z_max along the ego-forward centerline) with
    real hazard markers where traversability < 0.40 at the same column.

    No synthetic curve, no scripted pothole positions -- both the line and
    the markers are drawn directly from FrameState.elevation_profile /
    elevation_hazards_m, which src/dashboard/data_streamer.py computes from
    the real local grid each frame.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(110)
        self.profile: list[tuple[float, float]] = []
        self.hazards_m: list[float] = []

    def set_data(self, profile, hazards_m):
        self.profile = profile
        self.hazards_m = hazards_m
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COLOR_PANEL_ALT))
        w, h = self.width(), self.height()

        if not self.profile:
            painter.setPen(QColor(COLOR_TEXT_DIM))
            painter.drawText(self.rect(), Qt.AlignCenter, "No elevation data")
            return

        ys = [p[0] for p in self.profile]
        zs = [p[1] for p in self.profile]
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)
        y_span = max(y_max - y_min, 1e-6)
        z_span = max(z_max - z_min, 1e-6)

        def to_px(y_val, z_val):
            px = (y_val - y_min) / y_span * (w - 10) + 5
            py = h - 15 - (z_val - z_min) / z_span * (h - 25)
            return px, py

        painter.setPen(QPen(QColor(COLOR_CYAN), 1))
        prev = None
        for y_val, z_val in self.profile:
            px, py = to_px(y_val, z_val)
            if prev is not None:
                painter.drawLine(int(prev[0]), int(prev[1]), int(px), int(py))
            prev = (px, py)

        painter.setPen(QPen(QColor(COLOR_CRITICAL), 2))
        painter.setBrush(QColor(COLOR_CRITICAL))
        for hz_y in self.hazards_m:
            closest = min(self.profile, key=lambda p: abs(p[0] - hz_y))
            px, py = to_px(closest[0], closest[1])
            painter.drawEllipse(int(px) - 3, int(py) - 3, 6, 6)

        painter.setPen(QColor(COLOR_TEXT_DIM))
        painter.drawText(5, h - 3, f"{y_min:.0f}m")
        painter.drawText(w - 40, h - 3, f"{y_max:.0f}m")


# A bin needs at least this many real points backing it to be drawn at
# full confidence; sparser real bins are still drawn (never hidden), just
# with a lighter/dashed visual treatment so they aren't presented with
# the same confidence as well-observed bins.
LOW_CONFIDENCE_POINT_COUNT = 5


class CenterlineProfileChart(QWidget):
    """Real forward-corridor cross-section: real height above / depth
    below a real local ground baseline, from
    src/perception/centerline_profile.py, with real hazard/tracked-object
    markers at their real distances. A gap in the real data (no points in
    a bin) is drawn as a gap in the chart -- never interpolated across.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(5, 1.8), facecolor=COLOR_PANEL)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self._style_axes()

    def _style_axes(self):
        ax = self.ax
        ax.set_facecolor(COLOR_PANEL_ALT)
        for spine in ax.spines.values():
            spine.set_color(COLOR_BORDER)
        ax.tick_params(colors=COLOR_TEXT_DIM, labelsize=7)
        ax.xaxis.label.set_color(COLOR_TEXT_DIM)
        ax.yaxis.label.set_color(COLOR_TEXT_DIM)

    def set_data(self, profile_bins: list[dict], markers: list[dict]):
        ax = self.ax
        ax.clear()
        self._style_axes()
        ax.set_xlabel("Distance ahead (m)", fontsize=8)
        ax.set_ylabel("Height rel. baseline (m)", fontsize=8)
        ax.axhline(0.0, color=COLOR_TEXT_DIM, linewidth=0.8, linestyle="-")

        if not profile_bins:
            ax.text(0.5, 0.5, "No real corridor data for this frame", ha="center",
                     va="center", color=COLOR_TEXT_DIM, transform=ax.transAxes)
            self.canvas.draw_idle()
            return

        # Draw each real bin as its own bar -- gaps (point_count == 0)
        # are simply not drawn, never interpolated across.
        for b in profile_bins:
            d = b["distance_m"]
            n = b["point_count"]
            if n == 0:
                continue
            low_conf = n < LOW_CONFIDENCE_POINT_COUNT
            alpha = 0.35 if low_conf else 0.85
            hatch = "//" if low_conf else None

            h_above = b["height_above_baseline_m"]
            if h_above is not None and h_above > 0.0:
                ax.bar(d, h_above, width=0.9, bottom=0.0, color=COLOR_STATIC,
                       alpha=alpha, hatch=hatch, edgecolor=COLOR_BORDER, linewidth=0.3)

            d_below = b["depth_below_baseline_m"]
            if d_below is not None:
                ax.bar(d, d_below, width=0.9, bottom=0.0, color=COLOR_CRITICAL,
                       alpha=alpha, hatch=hatch, edgecolor=COLOR_BORDER, linewidth=0.3)

        # Real hazard/tracked-object markers, at the SAME distance_m
        # already shown elsewhere in the dashboard for that object.
        y_min, y_max = ax.get_ylim()
        for m in markers:
            d = m["distance_m"]
            label = m.get("label", m.get("severity", m.get("class_name", "?")))
            ax.axvline(d, color=COLOR_CYAN, linewidth=1.0, linestyle="--", alpha=0.8)
            ax.text(d, y_max, f" {label}\n {d:.1f}m", color=COLOR_CYAN, fontsize=6,
                    ha="left", va="top", rotation=0)

        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=COLOR_STATIC, label="Height above ground (bump)"),
            Patch(facecolor=COLOR_CRITICAL, label="Depth below ground (dip)"),
            Patch(facecolor=COLOR_TEXT_DIM, alpha=0.35, hatch="//", label="Low confidence (<5 pts)"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=6,
                  facecolor=COLOR_PANEL, edgecolor=COLOR_BORDER, labelcolor=COLOR_TEXT_DIM)

        self.figure.tight_layout()
        self.canvas.draw_idle()


class AlertCard(QFrame):
    """Real nearby object alert based on actual tracked positions and velocities."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"background-color: {COLOR_PANEL}; border: 1px solid {COLOR_CRITICAL}; border-radius: 6px;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)

        self.lbl_title = QLabel("")
        self.lbl_title.setStyleSheet(f"color: {COLOR_TEXT}; font-weight: 700; font-size: {FONT_SIZE_EMPHASIS};")

        self.lbl_body = QLabel("")
        self.lbl_body.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-weight: 600; font-size: {FONT_SIZE_BODY};")
        self.lbl_body.setWordWrap(True)

        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_body)
        self.setFixedWidth(320)
        self.setMinimumHeight(60)

    def set_data(self, track, dist: float, speed: float, direction: str, proximity: str):
        self.lbl_title.setText(f"{track.class_name} #{track.track_id}")
        conf_str = f"{track.confidence*100:.1f}%" if track.confidence > 0.0 else "N/A"
        text = (
            f"Distance: {dist:.1f} m\n"
            f"Speed: {speed:.1f} m/s\n"
            f"Direction: {direction}\n"
            f"{proximity}\n"
            f"Confidence: {conf_str}"
        )
        self.lbl_body.setText(text)


class FoveaXDashboardWindow(QMainWindow):
    # Emitted when a left-panel control requests a real Open3D camera
    # action ("zoom_in", "zoom_out", "rotate", "reset", "view_top",
    # "view_perspective", "view_side"). Connected externally (in
    # src/13_realtime_dashboard.py, which owns the o3d command queue) to
    # actually forward the command to the Open3D process.
    camera_command_requested = pyqtSignal(str)

    # Playback transport controls. Connected externally (in
    # src/13_realtime_dashboard.py, which owns the real frame-by-frame
    # replay loop and per-frame cache) -- this window only emits requests,
    # it never advances frames itself.
    play_pause_requested = pyqtSignal()
    step_forward_requested = pyqtSignal()
    step_backward_requested = pyqtSignal()
    rate_changed = pyqtSignal(float)
    restart_requested = pyqtSignal()
    
    # Environment switching
    environment_requested = pyqtSignal(str)

    def __init__(self, embed_3d: bool = False):
        super().__init__()
        self.setWindowTitle("FOVEAX Phase 9 - 2D & Metrics Dashboard")
        self.resize(1400, 900)
        self.setStyleSheet(STYLESHEET)

        # Default to the OVERHEAD layer so real overhead obstacles (and the
        # always-visible FORWARD CORRIDOR CROSS-SECTION panel below) make
        # real terrain diversity evident without an extra click.
        self.current_map_type = "overhead"
        self.embed_3d = embed_3d
        self.embed_widget = None
        self._map_type_buttons = {}
        self._init_ui()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------
    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        outer = QVBoxLayout(central_widget)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        outer.addWidget(self._build_header())
        outer.addWidget(self._build_metrics_bar())
        outer.addLayout(self._build_body(), stretch=1)
        outer.addWidget(self._build_centerline_panel())
        outer.addWidget(self._build_footer())

        self.threat_toast = self._build_threat_toast(central_widget)
        self.threat_toast.hide()

        self.alert_container = QWidget(central_widget)
        self.alert_layout = QVBoxLayout(self.alert_container)
        self.alert_layout.setContentsMargins(0, 0, 0, 0)
        self.alert_layout.setSpacing(6)
        # Positioned to avoid overlapping threat_toast at (250, 115) and drawer
        self.alert_container.move(620, 115)
        self.alert_container.hide()

        self.drawer = self._build_system_drawer(central_widget)
        self.drawer.hide()

    def _build_header(self):
        header = QFrame()
        header.setFixedHeight(46)
        header.setStyleSheet(f"background-color: {COLOR_PANEL}; border-bottom: 1px solid {COLOR_BORDER};")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(12, 4, 12, 4)

        title = QLabel("FOVEAX 2.5D LiDAR PERCEPTION")
        title.setStyleSheet(f"color: {COLOR_CYAN}; font-size: 14px; font-weight: 700;")
        layout.addWidget(title)
        
        self.env1_btn = QPushButton("ENVIRONMENT 1 (Seq 00001)")
        self.env1_btn.clicked.connect(lambda: self.environment_requested.emit("00001"))
        layout.addWidget(self.env1_btn)
        
        self.env2_btn = QPushButton("ENVIRONMENT 2 (Seq 00003)")
        self.env2_btn.clicked.connect(lambda: self.environment_requested.emit("00003"))
        layout.addWidget(self.env2_btn)
        self.set_active_environment("00001")
        
        layout.addStretch(1)

        # --- Playback transport controls -----------------------------
        # This dashboard replays a real, finite, recorded sequence of
        # LiDAR frames (not a live sensor feed) -- these controls step
        # through the real recorded frames, they never fabricate motion
        # between them (see step/rate wiring in 13_realtime_dashboard.py).
        self.step_back_btn = QPushButton("<< STEP")
        self.step_back_btn.clicked.connect(self.step_backward_requested.emit)
        layout.addWidget(self.step_back_btn)

        self.play_pause_btn = QPushButton("PAUSE")
        self.play_pause_btn.clicked.connect(self.play_pause_requested.emit)
        layout.addWidget(self.play_pause_btn)
        self._apply_play_pause_style(playing=True)

        self.step_fwd_btn = QPushButton("STEP >>")
        self.step_fwd_btn.clicked.connect(self.step_forward_requested.emit)
        layout.addWidget(self.step_fwd_btn)

        self.restart_btn = QPushButton("RESTART")
        self.restart_btn.setStyleSheet(f"QPushButton {{ border: 1px solid {COLOR_CYAN}; }}")
        self.restart_btn.clicked.connect(self.restart_requested.emit)
        layout.addWidget(self.restart_btn)

        rate_label = QLabel("Hz:")
        rate_label.setStyleSheet(f"color: {COLOR_TEXT_DIM};")
        layout.addWidget(rate_label)
        self.rate_spinbox = QSpinBox()
        self.rate_spinbox.setRange(1, 30)
        self.rate_spinbox.setValue(10)
        self.rate_spinbox.valueChanged.connect(lambda v: self.rate_changed.emit(float(v)))
        layout.addWidget(self.rate_spinbox)

        self.lbl_frame_counter = QLabel("Frame -- / --")
        self.lbl_frame_counter.setStyleSheet(f"color: {COLOR_TEXT}; font-weight: 600; padding: 0 8px;")
        layout.addWidget(self.lbl_frame_counter)

        layout.addStretch(1)

        self.lbl_fps_badge = QLabel("-- FPS")
        self.lbl_fps_badge.setStyleSheet(f"color: {COLOR_SAFE}; font-weight: 600;")
        layout.addWidget(self.lbl_fps_badge)

        drawer_btn = QPushButton("SYSTEM DETAILS")
        drawer_btn.clicked.connect(self._toggle_drawer)
        layout.addWidget(drawer_btn)
        return header

    def set_active_environment(self, seq: str):
        active_style = f"background-color: {COLOR_CYAN}; color: {COLOR_PANEL}; font-weight: 700; border: none; padding: 4px 8px; border-radius: 4px;"
        inactive_style = f"background-color: transparent; color: {COLOR_TEXT_DIM}; font-weight: 600; border: 1px solid {COLOR_BORDER}; padding: 4px 8px; border-radius: 4px;"
        self.env1_btn.setStyleSheet(active_style if seq == "00001" else inactive_style)
        self.env2_btn.setStyleSheet(active_style if seq == "00003" else inactive_style)

    def set_frame_counter(self, current: int, total: int):
        self.lbl_frame_counter.setText(f"Frame {current} / {total}")

    def set_play_pause_state(self, playing: bool):
        self._apply_play_pause_style(playing)

    def _apply_play_pause_style(self, playing: bool):
        # Explicit, self-contained inline style per state -- deliberately
        # NOT driven by the QPushButton:checked QSS pseudo-class, which
        # was found to sometimes not repaint after a programmatic
        # setChecked() call (a real, confirmed Qt style-repolish quirk),
        # leaving the button text low-contrast even though the logical
        # play/pause state itself was always correct. Setting a full
        # inline stylesheet directly guarantees readable text in both
        # states regardless of that repaint timing.
        self.play_pause_btn.setText("PAUSE" if playing else "PLAY")
        accent = COLOR_SAFE if playing else COLOR_CAUTION
        self.play_pause_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_PANEL_ALT};
                border: 1px solid {accent};
                border-radius: 4px;
                padding: 5px 8px;
                color: {COLOR_TEXT};
                font-weight: 600;
            }}
            QPushButton:hover {{ border: 1px solid {COLOR_CYAN}; }}
        """)

    def set_rate_hz(self, hz: float):
        self.rate_spinbox.blockSignals(True)
        self.rate_spinbox.setValue(int(round(hz)))
        self.rate_spinbox.blockSignals(False)

    def _populate_efficiency_stat(self, zones=None):
        """Compute (once, or when the zone config changes) the real Phase B
        memory-reduction stat via
        src/07_adaptive_2point5d_map.py::compute_memory_reduction_metrics()
        -- reused directly, never re-derived here. A static architectural
        property of the given zone config, not a per-frame measurement.
        """
        import importlib
        mod = importlib.import_module("src.07_adaptive_2point5d_map")
        metrics = mod.compute_memory_reduction_metrics(zones=zones)
        self.lbl_efficiency_cells.setText(
            f"Cell reduction vs. uniform 3D voxels: {metrics['cell_savings_vs_3d_pct']:.2f}%"
        )
        self.lbl_efficiency_bytes.setText(
            f"Byte savings vs. uniform 3D voxels: {metrics['byte_savings_vs_3d_pct']:.2f}%"
        )

    def set_accuracy_metrics(self, has_ground_truth: bool, predictor_mode: str, accuracy_metrics: dict | None):
        if predictor_mode == "ground_truth":
            self.lbl_accuracy_note.setText(
                "predictor=ground_truth: comparing real ground truth against "
                "itself (trivially ~100%). Use --predictor salsanext (or "
                "mock) to compare an independent real prediction."
            )
        else:
            self.lbl_accuracy_note.setText(f"predictor={predictor_mode}, compared against real ground truth.")

        if not has_ground_truth or accuracy_metrics is None:
            na_text = "N/A - no ground truth for this frame"
            self.lbl_accuracy_near.setText(f"Near (0-15m): {na_text}")
            self.lbl_accuracy_mid.setText(f"Mid (15-35m): {na_text}")
            self.lbl_accuracy_far.setText(f"Far (35-100m): {na_text}")
            self.lbl_accuracy_overall.setText(f"Overall (0-100m): {na_text}")
            return

        def fmt(bucket_name, result):
            if result is None:
                return f"{bucket_name}: N/A - no ground truth points in this bucket"
            return f"{bucket_name}: {result['accuracy']*100:.2f}% (n={result['n_points']:,})"

        self.lbl_accuracy_near.setText(fmt("Near (0-15m)", accuracy_metrics.get("near")))
        self.lbl_accuracy_mid.setText(fmt("Mid (15-35m)", accuracy_metrics.get("mid")))
        self.lbl_accuracy_far.setText(fmt("Far (35-100m)", accuracy_metrics.get("far")))
        self.lbl_accuracy_overall.setText(fmt("Overall (0-100m)", accuracy_metrics.get("overall")))

    def set_ego_motion_estimate(self, displacement_m: float | None):
        if displacement_m is None:
            self.lbl_ego_motion.setText("Est. relative displacement: N/A (first frame or estimate failed)")
        else:
            self.lbl_ego_motion.setText(f"Est. relative displacement: {displacement_m:.3f} m (since previous frame)")

    def _build_metrics_bar(self):
        bar = QFrame()
        bar.setFixedHeight(56)
        bar.setStyleSheet(f"background-color: {COLOR_PANEL}; border: 1px solid {COLOR_BORDER}; border-radius: 6px;")
        layout = QHBoxLayout(bar)

        def make_card(label_text):
            box = QVBoxLayout()
            val = QLabel("--")
            val.setStyleSheet(f"color: {COLOR_TEXT}; font-size: 15px; font-weight: 700;")
            cap = QLabel(label_text)
            cap.setStyleSheet(NOTE_STYLE)
            box.addWidget(val)
            box.addWidget(cap)
            layout.addLayout(box)
            return val

        self.lbl_m_fps = make_card("FPS")
        self.lbl_m_latency = make_card("LATENCY")
        self.lbl_m_points = make_card("POINTS/FRAME")
        self.lbl_m_cpu = make_card("CPU")
        self.lbl_m_ram = make_card("RAM")
        self.lbl_m_gpu = make_card("GPU")
        self.lbl_m_vram = make_card("VRAM")
        layout.addStretch(1)
        return bar

    def _build_body(self):
        body = QHBoxLayout()
        body.addWidget(self._build_left_panel(), stretch=2)
        body.addWidget(self._build_center_panel(), stretch=5 if self.embed_3d else 3)
        body.addWidget(self._build_right_panel(), stretch=2)
        return body

    def _build_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        cam_group = QGroupBox("VIEW / CAMERA (live Open3D)")
        cam_layout = QGridLayout()
        view_top = QPushButton("TOP")
        view_persp = QPushButton("PERSPECTIVE")
        view_side = QPushButton("SIDE")
        for i, (btn, cmd) in enumerate([
            (view_top, "view_top"), (view_persp, "view_perspective"), (view_side, "view_side")
        ]):
            btn.clicked.connect(lambda _, c=cmd: self.camera_command_requested.emit(c))
            cam_layout.addWidget(btn, 0, i)

        zoom_in = QPushButton("ZOOM +")
        zoom_out = QPushButton("ZOOM -")
        rotate_btn = QPushButton("ROTATE")
        reset_btn = QPushButton("RESET")
        for i, (btn, cmd) in enumerate([
            (zoom_in, "zoom_in"), (zoom_out, "zoom_out"),
            (rotate_btn, "rotate"), (reset_btn, "reset")
        ]):
            btn.clicked.connect(lambda _, c=cmd: self.camera_command_requested.emit(c))
            cam_layout.addWidget(btn, 1, i)
        cam_group.setLayout(cam_layout)
        layout.addWidget(cam_group)

        layers_group = QGroupBox("2D MAP LAYER")
        layers_layout = QVBoxLayout()
        for map_type in ["elevation", "traversability", "roi", "uncertainty", "overhead"]:
            btn = QPushButton(map_type.upper())
            btn.setCheckable(True)
            btn.setChecked(map_type == self.current_map_type)
            btn.clicked.connect(lambda _, mt=map_type: self._on_map_changed(mt))
            layers_layout.addWidget(btn)
            self._map_type_buttons[map_type] = btn
        layers_group.setLayout(layers_layout)
        layout.addWidget(layers_group)

        res_group = QGroupBox("ADAPTIVE RESOLUTION ZONES (spec)")
        res_layout = QVBoxLayout()
        for name, rng, res in RESOLUTION_ZONES:
            res_layout.addWidget(QLabel(f"{name}: {rng} -> {res}"))
        res_group.setLayout(res_layout)
        layout.addWidget(res_group)

        layout.addStretch(1)
        return panel

    def _build_center_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        if self.embed_3d:
            self.embed_widget = Open3DEmbedWidget()
            layout.addWidget(self.embed_widget, stretch=3)

        self.map_label = QLabel("Waiting for data...")
        self.map_label.setAlignment(Qt.AlignCenter)
        self.map_label.setMinimumSize(400, 300)
        self.map_label.setStyleSheet(f"background-color: black; color: {COLOR_TEXT_DIM}; border: 1px solid {COLOR_BORDER};")
        layout.addWidget(self.map_label, stretch=2)

        ego_relative_note = QLabel(
            "Ego-relative view: recentered on the vehicle each frame -- not an "
            "absolute-position world trajectory (this pipeline has no odometry)."
        )
        ego_relative_note.setStyleSheet(NOTE_STYLE + " font-style: italic;")
        ego_relative_note.setAlignment(Qt.AlignCenter)
        layout.addWidget(ego_relative_note)

        self.overhead_legend = QLabel(
            f"<span style='color:{COLOR_STATIC};'>&#9632;</span> Solid obstacle &nbsp;&nbsp;"
            f"<span style='color:{COLOR_CYAN};'>&#9645;</span> Overhead clearance (hatched, passable below) &nbsp;&nbsp;"
            f"<span style='color:#646464;'>&#9632;</span> Indeterminate (too few points)"
        )
        self.overhead_legend.setStyleSheet(NOTE_STYLE + " padding: 2px;")
        self.overhead_legend.hide()
        layout.addWidget(self.overhead_legend)
        return panel

    def _build_right_panel(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_status_tab(), "STATUS")
        self.tabs.addTab(self._build_terrain_tab(), "TERRAIN")
        self.tabs.addTab(self._build_system_tab(), "SYSTEM")
        return self.tabs

    def _build_status_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        status_group = QGroupBox("EGO TERRAIN STATUS")
        status_layout = QVBoxLayout()
        self.lbl_terrain_status = QLabel("UNKNOWN")
        self.lbl_terrain_status.setStyleSheet(f"font-size: {FONT_SIZE_HERO}; font-weight: 700; color: {COLOR_TEXT_DIM};")
        self.lbl_terrain_confidence = QLabel("Confidence: --")
        status_layout.addWidget(self.lbl_terrain_status)
        status_layout.addWidget(self.lbl_terrain_confidence)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        self.lbl_coord_warning = QLabel("")
        self.lbl_coord_warning.setStyleSheet(f"color: {COLOR_CAUTION}; font-weight: bold;")
        self.lbl_coord_warning.setWordWrap(True)
        self.lbl_coord_warning.hide()
        layout.addWidget(self.lbl_coord_warning)

        tracks_group = QGroupBox("TRACKED OBJECTS")
        tracks_layout = QVBoxLayout()
        self.list_tracks = QListWidget()
        tracks_layout.addWidget(self.list_tracks)
        tracks_group.setLayout(tracks_layout)
        layout.addWidget(tracks_group, stretch=1)

        threat_group = QGroupBox("NEAREST DYNAMIC OBJECT")
        threat_layout = QVBoxLayout()
        self.lbl_threat = QLabel("No dynamic objects tracked.")
        self.lbl_threat.setWordWrap(True)
        threat_layout.addWidget(self.lbl_threat)
        threat_group.setLayout(threat_layout)
        layout.addWidget(threat_group)

        return tab

    def _build_terrain_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        width_group = QGroupBox("ROAD WIDTH (Safe corridor, >=0.70)")
        width_layout = QVBoxLayout()
        self.lbl_road_width = QLabel("N/A")
        self.lbl_road_width.setStyleSheet(f"font-size: {FONT_SIZE_HEADLINE}; font-weight: 700; color: {COLOR_SAFE};")
        width_layout.addWidget(self.lbl_road_width)
        width_group.setLayout(width_layout)
        layout.addWidget(width_group)

        elev_group = QGroupBox("ELEVATION PROFILE (centerline, real hazards)")
        elev_layout = QVBoxLayout()
        self.elevation_widget = ElevationProfileWidget()
        elev_layout.addWidget(self.elevation_widget)
        elev_group.setLayout(elev_layout)
        layout.addWidget(elev_group)

        unc_group = QGroupBox("MAP UNCERTAINTY (Phase 5 formula)")
        unc_layout = QVBoxLayout()
        self.lbl_uncertainty = QLabel("Mean uncertainty: N/A")
        self.lbl_occupied = QLabel("Occupied grid fraction: N/A")
        unc_layout.addWidget(self.lbl_uncertainty)
        unc_layout.addWidget(self.lbl_occupied)
        unc_group.setLayout(unc_layout)
        layout.addWidget(unc_group)

        hazard_group = QGroupBox("HAZARD CLUSTERS (connected components)")
        hazard_layout = QVBoxLayout()
        self.hazard_table = QTableWidget(0, 4)
        self.hazard_table.setHorizontalHeaderLabels(["Dist (m)", "Min Trav.", "Severity", "Conf."])
        self.hazard_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.hazard_table.verticalHeader().setVisible(False)
        hazard_layout.addWidget(self.hazard_table)
        hazard_group.setLayout(hazard_layout)
        layout.addWidget(hazard_group, stretch=1)

        return tab

    def _build_system_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        stats_group = QGroupBox("HARDWARE")
        stats_layout = QVBoxLayout()
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_latency = QLabel("Latency: -- ms")
        self.lbl_cpu = QLabel("CPU: -- %")
        self.prog_cpu = QProgressBar()
        self.lbl_ram = QLabel("RAM: -- %")
        self.prog_ram = QProgressBar()
        self.lbl_gpu = QLabel("GPU: -- %")
        self.prog_gpu = QProgressBar()
        self.lbl_vram = QLabel("VRAM: -- %")
        self.prog_vram = QProgressBar()
        for w in [self.lbl_fps, self.lbl_latency, self.lbl_cpu, self.prog_cpu,
                  self.lbl_ram, self.prog_ram, self.lbl_gpu, self.prog_gpu,
                  self.lbl_vram, self.prog_vram]:
            stats_layout.addWidget(w)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        session_group = QGroupBox("SESSION")
        session_layout = QVBoxLayout()
        self.lbl_frame_idx = QLabel("Frame: --")
        session_layout.addWidget(self.lbl_frame_idx)
        session_group.setLayout(session_layout)
        layout.addWidget(session_group)

        accuracy_group = QGroupBox("ACCURACY (current frame, vs. real ground truth)")
        accuracy_layout = QVBoxLayout()
        self.lbl_accuracy_note = QLabel("")
        self.lbl_accuracy_note.setWordWrap(True)
        self.lbl_accuracy_note.setStyleSheet(NOTE_STYLE)
        accuracy_layout.addWidget(self.lbl_accuracy_note)
        self.lbl_accuracy_near = QLabel("Near (0-15m): --")
        self.lbl_accuracy_mid = QLabel("Mid (15-35m): --")
        self.lbl_accuracy_far = QLabel("Far (35-100m): --")
        self.lbl_accuracy_overall = QLabel("Overall (0-100m): --")
        self.lbl_accuracy_overall.setStyleSheet(f"font-size: {FONT_SIZE_HEADLINE}; font-weight: 700; color: {COLOR_SAFE};")
        for w in [self.lbl_accuracy_near, self.lbl_accuracy_mid,
                  self.lbl_accuracy_far, self.lbl_accuracy_overall]:
            accuracy_layout.addWidget(w)
        accuracy_group.setLayout(accuracy_layout)
        layout.addWidget(accuracy_group)

        efficiency_group = QGroupBox("EFFICIENCY (Phase B, static grid design property)")
        efficiency_layout = QVBoxLayout()
        efficiency_note = QLabel(
            "Fixed property of the current Near/Middle/Far resolution-zone "
            "config (src/07_adaptive_2point5d_map.py), not a live per-frame "
            "measurement -- recomputed only if that zone config changes."
        )
        efficiency_note.setWordWrap(True)
        efficiency_note.setStyleSheet(NOTE_STYLE)
        efficiency_layout.addWidget(efficiency_note)
        self.lbl_efficiency_cells = QLabel("Cell reduction vs. uniform 3D voxels: --")
        self.lbl_efficiency_bytes = QLabel("Byte savings vs. uniform 3D voxels: --")
        efficiency_layout.addWidget(self.lbl_efficiency_cells)
        efficiency_layout.addWidget(self.lbl_efficiency_bytes)
        efficiency_group.setLayout(efficiency_layout)
        layout.addWidget(efficiency_group)
        self._populate_efficiency_stat()

        motion_group = QGroupBox("EGO MOTION (informational only)")
        motion_layout = QVBoxLayout()
        motion_note = QLabel(
            "This pipeline has no odometry. The value below is a coarse, "
            "real point-cloud-registration (ICP) estimate of relative "
            "displacement between consecutive frames -- informational "
            "only. It does NOT reposition any view; every 2D/3D panel "
            "remains ego-relative (recentered each frame)."
        )
        motion_note.setWordWrap(True)
        motion_note.setStyleSheet(NOTE_STYLE)
        motion_layout.addWidget(motion_note)
        self.lbl_ego_motion = QLabel("Est. relative displacement: N/A")
        motion_layout.addWidget(self.lbl_ego_motion)
        motion_group.setLayout(motion_layout)
        layout.addWidget(motion_group)

        layout.addStretch(1)
        return tab

    def _build_centerline_panel(self):
        group = QGroupBox("FORWARD CORRIDOR CROSS-SECTION (real elevation, real hazards)")
        group.setFixedHeight(190)
        layout = QVBoxLayout(group)
        self.centerline_chart = CenterlineProfileChart()
        layout.addWidget(self.centerline_chart)
        return group

    def _build_footer(self):
        footer = QFrame()
        footer.setFixedHeight(22)
        footer.setStyleSheet(f"background-color: {COLOR_PANEL}; border-top: 1px solid {COLOR_BORDER};")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(10, 0, 10, 0)
        self.lbl_footer = QLabel("FOVEAX Phase 9 -- waiting for first frame")
        self.lbl_footer.setStyleSheet(NOTE_STYLE)
        layout.addWidget(self.lbl_footer)
        layout.addStretch(1)
        return footer

    def _build_threat_toast(self, parent):
        toast = QFrame(parent)
        toast.setStyleSheet(
            f"background-color: {COLOR_PANEL}; border: 1px solid {COLOR_CRITICAL}; border-radius: 6px;"
        )
        layout = QVBoxLayout(toast)
        layout.setContentsMargins(10, 8, 10, 8)
        self.lbl_toast_text = QLabel("")
        self.lbl_toast_text.setStyleSheet(f"color: {COLOR_TEXT}; font-weight: 600; font-size: {FONT_SIZE_BODY};")
        # Wrapping (rather than a fixed single-line box) is what fixed the
        # garbled/overlapping toast text: without it, a long message (real
        # class name + track id + distance + speed) simply overflowed the
        # frame's last-known size instead of reflowing to fit.
        self.lbl_toast_text.setWordWrap(True)
        layout.addWidget(self.lbl_toast_text)
        toast.setFixedWidth(340)
        toast.setMinimumHeight(50)
        toast.move(250, 115)
        return toast

    def _build_system_drawer(self, parent):
        drawer = QFrame(parent)
        drawer.setStyleSheet(
            f"background-color: {COLOR_PANEL}; border: 1px solid {COLOR_BORDER}; border-radius: 6px;"
        )
        layout = QVBoxLayout(drawer)

        title = QLabel("SYSTEM DETAILS")
        title.setStyleSheet(f"color: {COLOR_CYAN}; font-size: 13px; font-weight: 700;")
        layout.addWidget(title)

        layout.addWidget(QLabel(
            "Mode: Adaptive resolution (fixed Near/Middle/Far zone config,\n"
            "not a live-switchable runtime mode in this pipeline)."
        ))

        pipeline = QLabel(
            "PIPELINE:\n"
            "LiDAR point cloud\n"
            "  -> Object detection & tracking (src/perception, src/tracking)\n"
            "  -> 2.5D elevation / traversability / ROI / uncertainty grids\n"
            "     (src/dashboard/data_streamer.py, canonical Safe/Caution/\n"
            "     Blocked convention from src/06_terrain_traversability.py)\n"
            "  -> Open3D 3D view + Qt 2D dashboard (this window)"
        )
        pipeline.setWordWrap(True)
        layout.addWidget(pipeline)

        tags = QLabel("Tech stack: PyQt5, Open3D, NumPy, SciPy, PyTorch, pywin32 (win32gui)")
        tags.setWordWrap(True)
        tags.setStyleSheet(NOTE_STYLE)
        layout.addWidget(tags)

        close_btn = QPushButton("CLOSE")
        close_btn.clicked.connect(self._toggle_drawer)
        layout.addWidget(close_btn)

        drawer.setFixedWidth(320)
        return drawer

    def _toggle_drawer(self):
        if self.drawer.isVisible():
            self.drawer.hide()
        else:
            self.drawer.move(self.width() - self.drawer.width() - 20, 60)
            self.drawer.resize(self.drawer.width(), self.height() - 100)
            self.drawer.raise_()
            self.drawer.show()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.drawer.isVisible():
            self.drawer.move(self.width() - self.drawer.width() - 20, 60)
            self.drawer.resize(self.drawer.width(), self.height() - 100)
            
    def _update_alerts(self, alerts):
        """Update the real approaching object alert cards."""
        # Clear existing cards
        while self.alert_layout.count():
            child = self.alert_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        if not alerts:
            self.alert_container.hide()
            return
            
        for dist, track, speed, direction, prox in alerts:
            card = AlertCard(self.alert_container)
            card.set_data(track, dist, speed, direction, prox)
            self.alert_layout.addWidget(card)
            
        self.alert_container.adjustSize()
        self.alert_container.show()
        self.alert_container.raise_()

    # ------------------------------------------------------------------
    # Data wiring
    # ------------------------------------------------------------------
    def _on_map_changed(self, map_type: str):
        self.current_map_type = map_type
        for mt, btn in self._map_type_buttons.items():
            btn.setChecked(mt == map_type)
        self.overhead_legend.setVisible(map_type == "overhead")

    def update_ui(self, state: FrameState):
        """Update the dashboard UI from the FrameState."""
        # 1. Update Hardware Metrics
        m = state.metrics
        self.lbl_fps.setText(f"FPS: {m.fps:.1f}")
        self.lbl_latency.setText(f"Latency: {m.latency_ms:.1f} ms")
        self.lbl_fps_badge.setText(f"{m.fps:.1f} FPS")
        self.lbl_m_fps.setText(f"{m.fps:.1f}")
        self.lbl_m_latency.setText(f"{m.latency_ms:.0f} ms")
        self.lbl_m_points.setText(f"{len(state.points):,}")
        self.lbl_m_cpu.setText(f"{m.cpu_percent:.0f}%")
        self.lbl_m_ram.setText(f"{m.ram_percent:.0f}%")

        if state.coordinate_warnings:
            self.lbl_coord_warning.setText("\n".join(state.coordinate_warnings))
            self.lbl_coord_warning.show()
        else:
            self.lbl_coord_warning.clear()
            self.lbl_coord_warning.hide()

        self.lbl_cpu.setText(f"CPU: {m.cpu_percent:.1f}%")
        self.prog_cpu.setValue(int(m.cpu_percent))

        self.lbl_ram.setText(f"RAM: {m.ram_percent:.1f}% ({m.ram_used_gb:.1f}/{m.ram_total_gb:.1f} GB)")
        self.prog_ram.setValue(int(m.ram_percent))

        if m.gpu_available:
            self.lbl_gpu.setText(f"GPU: {m.gpu_percent:.1f}%")
            self.prog_gpu.setValue(int(m.gpu_percent))
            self.lbl_vram.setText(f"VRAM: {m.vram_percent:.1f}% ({m.vram_used_gb:.1f}/{m.vram_total_gb:.1f} GB)")
            self.prog_vram.setValue(int(m.vram_percent))
            self.lbl_m_gpu.setText(f"{m.gpu_percent:.0f}%")
            self.lbl_m_vram.setText(f"{m.vram_percent:.0f}%")
        else:
            self.lbl_gpu.setText("GPU: N/A")
            self.prog_gpu.setValue(0)
            self.lbl_vram.setText("VRAM: N/A")
            self.prog_vram.setValue(0)
            self.lbl_m_gpu.setText("N/A")
            self.lbl_m_vram.setText("N/A")

        # 2. Update Tracked Objects List + nearest dynamic threat
        self.list_tracks.clear()
        nearest_dynamic = None
        nearest_dist = None
        for t in state.tracks:
            speed = np.linalg.norm(t.state[3:6])
            dyn_str = "Dynamic" if t.dynamic else "Static"
            item_text = f"ID: {t.track_id} | {t.class_name} | {dyn_str} | {speed:.1f} m/s"
            self.list_tracks.addItem(item_text)
            if t.dynamic:
                dist = float(np.hypot(t.state[0], t.state[1]))
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest_dynamic = t

        if nearest_dynamic is not None:
            speed = float(np.linalg.norm(nearest_dynamic.state[3:6]))
            self.lbl_threat.setText(
                f"ID {nearest_dynamic.track_id} ({nearest_dynamic.class_name})\n"
                f"Distance: {nearest_dist:.1f} m | Speed: {speed:.1f} m/s"
            )
            self.lbl_toast_text.setText(
                f"DYNAMIC OBJECT: {nearest_dynamic.class_name} #{nearest_dynamic.track_id} "
                f"at {nearest_dist:.1f} m, {speed:.1f} m/s"
            )
            self.threat_toast.adjustSize()
            self.threat_toast.show()
            self.threat_toast.raise_()
        else:
            self.lbl_threat.setText("No dynamic objects tracked.")
            self.threat_toast.hide()

        # Update real alert cards for approaching objects
        alerts = []
        for t in state.tracks:
            dist = float(np.hypot(t.state[0], t.state[1]))
            if dist < 15.0 and t.dynamic:
                speed = float(np.linalg.norm(t.state[3:6]))
                x, y = float(t.state[0]), float(t.state[1])
                vx, vy = float(t.state[3]), float(t.state[4])
                
                # relative radial velocity = (x*vx + y*vy)/dist
                # Negative means the object is moving towards the ego (0,0)
                radial_v = (x*vx + y*vy) / dist if dist > 0.1 else 0.0
                
                # Classify direction using real vector math
                if radial_v < -0.2:
                    direction = "Approaching"
                elif radial_v > 0.2:
                    direction = "Receding"
                else:
                    direction = "Crossing"
                    
                if direction == "Approaching":
                    if dist < 10.0:
                        prox = "Proximity: Close (<10m)"
                    elif dist < 25.0:
                        prox = "Proximity: Near (10-25m)"
                    else:
                        prox = "Proximity: Far (>25m)"
                    alerts.append((dist, t, speed, direction, prox))
                    
        alerts.sort(key=lambda x: x[0])
        self._update_alerts(alerts[:3])

        # 3. Update 2D Map Image
        if self.current_map_type == "overhead":
            self._render_overhead_to_label(
                state.grid_maps.get("traversability"), state.grid_maps.get("overhead")
            )
        elif self.current_map_type in state.grid_maps:
            grid = state.grid_maps[self.current_map_type]
            self._render_grid_to_label(grid, self.current_map_type)
        else:
            self.map_label.setText(f"Map '{self.current_map_type}' not available.")

        # 4. Ego terrain status (from the real live traversability grid)
        self._update_terrain_status(state)

        # 5. Road width / elevation profile / uncertainty / hazards
        if state.road_width_m is not None:
            self.lbl_road_width.setText(f"{state.road_width_m:.2f} m")
        else:
            self.lbl_road_width.setText("N/A (no safe cell under ego)")

        self.elevation_widget.set_data(state.elevation_profile, state.elevation_hazards_m)
        self.centerline_chart.set_data(state.centerline_profile, state.centerline_markers)

        if state.mean_uncertainty is not None:
            self.lbl_uncertainty.setText(f"Mean uncertainty: {state.mean_uncertainty*100:.1f}%")
        else:
            self.lbl_uncertainty.setText("Mean uncertainty: N/A")
        if state.occupied_fraction is not None:
            self.lbl_occupied.setText(f"Occupied grid fraction: {state.occupied_fraction*100:.1f}%")
        else:
            self.lbl_occupied.setText("Occupied grid fraction: N/A")

        self._update_hazard_table(state.hazard_clusters)

        self.lbl_frame_idx.setText(f"Frame: {state.frame_idx} | t={state.timestamp_s:.2f}s")
        self.set_accuracy_metrics(state.has_ground_truth, state.predictor_mode, state.accuracy_metrics)
        self.lbl_footer.setText(
            f"Frame {state.frame_idx} | {len(state.points):,} pts | "
            f"{len(state.tracks)} tracks | {len(state.hazard_clusters)} hazard clusters"
        )

    def _update_terrain_status(self, state: FrameState):
        trav = state.grid_maps.get("traversability")
        if trav is None:
            self.lbl_terrain_status.setText("UNKNOWN")
            self.lbl_terrain_status.setStyleSheet(f"font-size: {FONT_SIZE_HERO}; font-weight: 700; color: {COLOR_TEXT_DIM};")
            self.lbl_terrain_confidence.setText("Confidence: N/A")
            return

        valid = ~np.isnan(trav)
        if not np.any(valid):
            self.lbl_terrain_status.setText("UNKNOWN")
            self.lbl_terrain_status.setStyleSheet(f"font-size: {FONT_SIZE_HERO}; font-weight: 700; color: {COLOR_TEXT_DIM};")
            self.lbl_terrain_confidence.setText("Confidence: N/A")
            return

        avg_trav = float(np.nanmean(trav[valid]))
        if avg_trav >= TRAVERSABILITY_SAFE_MIN:
            text, color = "DRIVABLE", COLOR_SAFE
        elif avg_trav >= TRAVERSABILITY_CAUTION_MIN:
            text, color = "CAUTION", COLOR_CAUTION
        else:
            text, color = "NON-DRIVABLE", COLOR_CRITICAL

        self.lbl_terrain_status.setText(text)
        self.lbl_terrain_status.setStyleSheet(f"font-size: {FONT_SIZE_HERO}; font-weight: 700; color: {color};")
        self.lbl_terrain_confidence.setText(f"Avg. traversability: {avg_trav:.2f}")

    def _update_hazard_table(self, clusters: list[dict]):
        self.hazard_table.setRowCount(len(clusters))
        for row, c in enumerate(clusters):
            self.hazard_table.setItem(row, 0, QTableWidgetItem(f"{c['distance_m']:.2f}"))
            self.hazard_table.setItem(row, 1, QTableWidgetItem(f"{c['min_traversability']:.2f}"))
            sev_item = QTableWidgetItem(c["severity"])
            if c["severity"] == "Critical":
                sev_item.setForeground(QColor(COLOR_CRITICAL))
            else:
                sev_item.setForeground(QColor(COLOR_CAUTION))
            self.hazard_table.setItem(row, 2, sev_item)
            self.hazard_table.setItem(row, 3, QTableWidgetItem(f"{c['confidence']*100:.0f}%"))

    def _render_overhead_to_label(self, trav_grid, overhead_grid):
        """Render the OVERHEAD layer: real terrain coloring for ordinary
        ground cells, a solid fill for solid_obstacle cells, a cross-hatch
        overlay for overhead_obstacle cells (real vertical clearance
        underneath, from src/perception/overhead_detection.py), and a
        muted fill for indeterminate (too-sparse) cells.
        """
        from src.perception.overhead_detection import (
            EMPTY, GROUND, SOLID_OBSTACLE, OVERHEAD_OBSTACLE, INDETERMINATE,
        )

        if overhead_grid is None:
            self.map_label.setText("Overhead clearance map not available.")
            return

        h, w = overhead_grid.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        ground_mask = overhead_grid == GROUND
        if trav_grid is not None:
            valid_trav = ~np.isnan(trav_grid)
            norm = np.zeros_like(trav_grid, dtype=np.float32)
            norm[valid_trav] = np.clip(trav_grid[valid_trav], 0.0, 1.0)
            cmap = mpl.colormaps["RdYlGn"]
            ground_rgba = (cmap(norm) * 255).astype(np.uint8)
            fillable_ground = ground_mask & valid_trav
            rgba[fillable_ground] = ground_rgba[fillable_ground]

        solid_mask = overhead_grid == SOLID_OBSTACLE
        rgba[solid_mask] = [251, 146, 60, 255]  # COLOR_STATIC

        overhead_mask = overhead_grid == OVERHEAD_OBSTACLE
        rgba[overhead_mask] = [34, 211, 238, 255]  # COLOR_CYAN base fill, hatched below

        indet_mask = overhead_grid == INDETERMINATE
        rgba[indet_mask] = [100, 100, 100, 255]

        rgba[overhead_grid == EMPTY] = [0, 0, 0, 0]

        # Cross-hatch pattern over overhead_obstacle cells only, using the
        # same diagonal-line technique as the lethal-traversability hatch
        # in _render_grid_to_label below.
        if np.any(overhead_mask):
            try:
                import cv2
                hatched = rgba.copy()
                hatch_color = (10, 10, 10, 255)
                spacing = 4
                for i in range(0, h + w, spacing):
                    cv2.line(hatched, (0, i), (i, 0), hatch_color, 1)
                rgba = np.where(overhead_mask[:, :, None], hatched, rgba)
            except ImportError:
                pass

        bytes_per_line = 4 * w
        q_img = QImage(rgba.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.map_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.map_label.setPixmap(scaled_pixmap)

    def _render_grid_to_label(self, grid: np.ndarray, map_type: str):
        # Handle nans
        valid_mask = ~np.isnan(grid)
        if not np.any(valid_mask):
            self.map_label.setText("Grid is empty")
            return

        # Normalize for colormap
        g_min, g_max = np.nanmin(grid), np.nanmax(grid)
        if g_max > g_min:
            norm_grid = (grid - g_min) / (g_max - g_min)
        else:
            norm_grid = np.zeros_like(grid)

        norm_grid[~valid_mask] = 0.0 # background

        # Apply colormap
        if map_type == "elevation":
            cmap = mpl.colormaps["turbo"]
        elif map_type == "traversability":
            cmap = mpl.colormaps["RdYlGn"] # Green = high = safe, Red = low = blocked, matches canonical convention
        elif map_type == "roi":
            cmap = mpl.colormaps["magma"]
        elif map_type == "uncertainty":
            cmap = mpl.colormaps["cividis"]
        else:
            cmap = mpl.colormaps["viridis"]

        # rgba image
        rgba = cmap(norm_grid)
        # set transparent for nan
        rgba[~valid_mask, 3] = 0.0

        # convert to uint8
        rgba_8 = (rgba * 255).astype(np.uint8)

        # Add hatch pattern for traversability lethal cells (Accessibility - Task R)
        if map_type == "traversability":
            try:
                import cv2
                # Blocked/lethal is traversability score < 0.40, matching canonical convention
                # (06_terrain_traversability.py, grid_overlay.py): 1.0 = safe, 0.0 = blocked
                lethal_mask = (grid < 0.40) & valid_mask
                if np.any(lethal_mask):
                    # Create a hatch pattern
                    hatch_color = (0, 0, 0, 255) # Black hatch
                    thickness = 1
                    spacing = 5

                    for i in range(0, rgba_8.shape[0] + rgba_8.shape[1], spacing):
                        # Draw diagonal lines on a mask
                        cv2.line(rgba_8, (0, i), (i, 0), hatch_color, thickness)

                    # Now clear the hatch where it's not lethal
                    # Best way: only apply hatch to lethal mask.
                    # Reconstruct:
                    rgba_8_no_hatch = (rgba * 255).astype(np.uint8)
                    rgba_8[~lethal_mask] = rgba_8_no_hatch[~lethal_mask]
            except ImportError:
                pass

        h, w, c = rgba_8.shape
        bytes_per_line = c * w
        q_img = QImage(rgba_8.data, w, h, bytes_per_line, QImage.Format_RGBA8888)

        # Scale to fit label while keeping aspect ratio
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.map_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.map_label.setPixmap(scaled_pixmap)
