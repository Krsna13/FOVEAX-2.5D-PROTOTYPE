import numpy as np
import matplotlib as mpl
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QComboBox, QGroupBox, QListWidget, QProgressBar
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from src.dashboard.dashboard_state import FrameState

class FoveaXDashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FOVEAX Phase 9 - 2D & Metrics Dashboard")
        self.resize(1000, 800)
        
        self.current_map_type = "elevation"
        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # --- LEFT PANEL: 2D View ---
        left_panel = QVBoxLayout()
        
        self.map_selector = QComboBox()
        self.map_selector.addItems(["elevation", "traversability", "roi"])
        self.map_selector.currentTextChanged.connect(self._on_map_changed)
        left_panel.addWidget(self.map_selector)
        
        self.map_label = QLabel("Waiting for data...")
        self.map_label.setAlignment(Qt.AlignCenter)
        self.map_label.setMinimumSize(600, 600)
        self.map_label.setStyleSheet("background-color: black; color: white;")
        left_panel.addWidget(self.map_label)
        
        main_layout.addLayout(left_panel, stretch=2)
        
        # --- RIGHT PANEL: Metrics & Stats ---
        right_panel = QVBoxLayout()
        
        # System Stats Group
        stats_group = QGroupBox("System & Performance")
        stats_layout = QVBoxLayout()
        
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_latency = QLabel("Latency: -- ms")

        self.lbl_coord_warning = QLabel("")
        self.lbl_coord_warning.setStyleSheet("color: orange; font-weight: bold;")
        self.lbl_coord_warning.setWordWrap(True)
        self.lbl_coord_warning.hide()
        
        self.lbl_cpu = QLabel("CPU: -- %")
        self.prog_cpu = QProgressBar()
        
        self.lbl_ram = QLabel("RAM: -- %")
        self.prog_ram = QProgressBar()
        
        self.lbl_gpu = QLabel("GPU: -- %")
        self.prog_gpu = QProgressBar()
        
        self.lbl_vram = QLabel("VRAM: -- %")
        self.prog_vram = QProgressBar()
        
        stats_layout.addWidget(self.lbl_fps)
        stats_layout.addWidget(self.lbl_latency)
        stats_layout.addWidget(self.lbl_coord_warning)
        stats_layout.addSpacing(10)
        stats_layout.addWidget(self.lbl_cpu)
        stats_layout.addWidget(self.prog_cpu)
        stats_layout.addWidget(self.lbl_ram)
        stats_layout.addWidget(self.prog_ram)
        stats_layout.addWidget(self.lbl_gpu)
        stats_layout.addWidget(self.prog_gpu)
        stats_layout.addWidget(self.lbl_vram)
        stats_layout.addWidget(self.prog_vram)
        
        stats_group.setLayout(stats_layout)
        right_panel.addWidget(stats_group)
        
        # Tracked Objects Group
        tracks_group = QGroupBox("Tracked Objects")
        tracks_layout = QVBoxLayout()
        self.list_tracks = QListWidget()
        tracks_layout.addWidget(self.list_tracks)
        tracks_group.setLayout(tracks_layout)
        
        right_panel.addWidget(tracks_group, stretch=1)
        
        main_layout.addLayout(right_panel, stretch=1)

    def _on_map_changed(self, text):
        self.current_map_type = text

    def update_ui(self, state: FrameState):
        """Update the dashboard UI from the FrameState."""
        # 1. Update Hardware Metrics
        m = state.metrics
        self.lbl_fps.setText(f"FPS: {m.fps:.1f}")
        self.lbl_latency.setText(f"Latency: {m.latency_ms:.1f} ms")

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
        else:
            self.lbl_gpu.setText("GPU: N/A")
            self.prog_gpu.setValue(0)
            self.lbl_vram.setText("VRAM: N/A")
            self.prog_vram.setValue(0)
            
        # 2. Update Tracked Objects List
        self.list_tracks.clear()
        for t in state.tracks:
            speed = np.linalg.norm(t.state[3:6])
            dyn_str = "Dynamic" if t.dynamic else "Static"
            item_text = f"ID: {t.track_id} | {t.class_name} | {dyn_str} | {speed:.1f} m/s"
            self.list_tracks.addItem(item_text)
            
        # 3. Update 2D Map Image
        if self.current_map_type in state.grid_maps:
            grid = state.grid_maps[self.current_map_type]
            self._render_grid_to_label(grid, self.current_map_type)
        else:
            self.map_label.setText(f"Map '{self.current_map_type}' not available.")

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
