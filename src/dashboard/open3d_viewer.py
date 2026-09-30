import numpy as np
import open3d as o3d
from src.dashboard.dashboard_state import FrameState
from src.dashboard.track_labels import build_track_labels

class FoveaX3DViewer:
    """Wrapper for Open3D non-blocking visualizer."""

    def __init__(self, window_name: str = "FOVEAX Phase 9 - 3D Dashboard", width: int = 1280, height: int = 720,
                 visible: bool = True):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=width, height=height, visible=visible)
        
        # Geometries
        self.pcd = o3d.geometry.PointCloud()
        
        # We will keep a dictionary of LineSets for tracking boxes
        # track_id -> LineSet
        self.track_boxes = {}

        # The exact TrackState list the current boxes were drawn from. Labels
        # are built only from this, so they always describe the boxes on screen.
        self.drawn_tracks = []

        # Flag to track if geometries were added
        self.pcd_added = False
        
        # Visualization options
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.15]) # Dark theme
        opt.point_size = 2.0
        
        # Add a coordinate frame
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
        self.vis.add_geometry(self.axis)

        self.view_control = self.vis.get_view_control()

        # OPT-1: Cache the turbo colormap once at init time.
        # plt.get_cmap() hits the matplotlib registry and allocates an object
        # every call; with 131k-point clouds this was ~2-3ms wasted per frame.
        import matplotlib.pyplot as plt
        self._turbo_cmap = plt.get_cmap('turbo')

    def handle_camera_command(self, cmd: str) -> None:
        """Apply a real camera command to this Open3D view, issued by the
        Qt dashboard's left-panel controls (zoom/rotate/reset/view-mode).

        This directly drives Open3D's own ViewControl API -- it is not a
        simulated or faked camera effect.
        """
        vc = self.view_control
        if cmd == "zoom_in":
            # ViewControl.scale(): negative shrinks the view distance
            # (zooms in), matching Open3D's own mouse-wheel-forward handling.
            vc.scale(-2.0)
        elif cmd == "zoom_out":
            vc.scale(2.0)
        elif cmd == "rotate":
            vc.rotate(200.0, 0.0)
        elif cmd == "reset":
            self.vis.reset_view_point(True)
        elif cmd == "view_top":
            vc.set_front([0.0, 0.0, 1.0])
            vc.set_up([0.0, 1.0, 0.0])
            vc.set_lookat([0.0, 0.0, 0.0])
        elif cmd == "view_perspective":
            vc.set_front([-0.5, -0.5, 0.5])
            vc.set_up([0.0, 0.0, 1.0])
            vc.set_lookat([0.0, 0.0, 0.0])
        elif cmd == "view_side":
            vc.set_front([1.0, 0.0, 0.0])
            vc.set_up([0.0, 0.0, 1.0])
            vc.set_lookat([0.0, 0.0, 0.0])

    def _create_box_lineset(self, size_lwh, center_xyz, yaw_rad, color):
        """Creates an Open3D LineSet representing an oriented 3D bounding box."""
        l, w, h = size_lwh
        
        # 8 corners of the box centered at origin
        # OpenPCDet format often has center at the bottom of the bounding box. Let's assume center is geometric center.
        # Format: x forward, y left, z up
        x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
        y_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
        z_corners = np.array([-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2])
        
        corners = np.vstack((x_corners, y_corners, z_corners)).T
        
        # Rotate by yaw around Z axis
        rot_mat = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        corners = corners @ rot_mat.T
        
        # Translate to center
        corners += np.array(center_xyz)
        
        # Define the 12 lines of the bounding box
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0], # Bottom
            [4, 5], [5, 6], [6, 7], [7, 4], # Top
            [0, 4], [1, 5], [2, 6], [3, 7]  # Pillars
        ]
        
        colors = [color for i in range(len(lines))]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        return line_set

    def update(self, state: FrameState):
        """Update the visualizer with new FrameState."""
        if state.points.shape[0] == 0:
            return
            
        # Cross-panel origin-consistency assertion (Task O)
        if len(state.points) > 0:
            p_min = np.min(state.points[:, :2], axis=0)
            p_max = np.max(state.points[:, :2], axis=0)
            assert p_min[0] <= 10.0 and p_max[0] >= -10.0, "3D render points do not appear to be ego-centered."
            
        # Update point cloud
        # Extract xyz coordinates
        xyz = state.points[:, :3]
        self.pcd.points = o3d.utility.Vector3dVector(xyz)
        
        # Color points by height (Z axis)
        z = xyz[:, 2]
        # Precomputed 256-entry turbo colormap LUT
        _TURBO_LUT = np.array([[48, 18, 59], [49, 21, 66], [50, 24, 74], [52, 27, 81], [53, 30, 88], [54, 33, 95], [55, 35, 101], [56, 38, 108], [57, 41, 114], [58, 44, 121], [59, 47, 127], [60, 50, 133], [60, 53, 139], [61, 55, 145], [62, 58, 150], [63, 61, 156], [64, 64, 161], [64, 67, 166], [65, 69, 171], [65, 72, 176], [66, 75, 181], [67, 78, 186], [67, 80, 190], [67, 83, 194], [68, 86, 199], [68, 88, 203], [69, 91, 206], [69, 94, 210], [69, 96, 214], [69, 99, 217], [70, 102, 221], [70, 104, 224], [70, 107, 227], [70, 109, 230], [70, 112, 232], [70, 115, 235], [70, 117, 237], [70, 120, 240], [70, 122, 242], [70, 125, 244], [70, 127, 246], [70, 130, 248], [69, 132, 249], [69, 135, 251], [69, 137, 252], [68, 140, 253], [67, 142, 253], [66, 145, 254], [65, 147, 254], [64, 150, 254], [63, 152, 254], [62, 155, 254], [60, 157, 253], [59, 160, 252], [57, 162, 252], [56, 165, 251], [54, 168, 249], [52, 170, 248], [51, 172, 246], [49, 175, 245], [47, 177, 243], [45, 180, 241], [43, 182, 239], [42, 185, 237], [40, 187, 235], [38, 189, 233], [37, 192, 230], [35, 194, 228], [33, 196, 225], [32, 198, 223], [30, 201, 220], [29, 203, 218], [28, 205, 215], [27, 207, 212], [26, 209, 210], [25, 211, 207], [24, 213, 204], [24, 215, 202], [23, 217, 199], [23, 218, 196], [23, 220, 194], [23, 222, 191], [24, 224, 189], [24, 225, 186], [25, 227, 184], [26, 228, 182], [27, 229, 180], [29, 231, 177], [30, 232, 175], [32, 233, 172], [34, 235, 169], [36, 236, 166], [39, 237, 163], [41, 238, 160], [44, 239, 157], [47, 240, 154], [50, 241, 151], [53, 243, 148], [56, 244, 145], [59, 244, 141], [63, 245, 138], [66, 246, 135], [70, 247, 131], [74, 248, 128], [77, 249, 124], [81, 249, 121], [85, 250, 118], [89, 251, 114], [93, 251, 111], [97, 252, 108], [101, 252, 104], [105, 253, 101], [109, 253, 98], [113, 253, 95], [116, 254, 92], [120, 254, 89], [124, 254, 86], [128, 254, 83], [132, 254, 80], [135, 254, 77], [139, 254, 75], [142, 254, 72], [146, 254, 70], [149, 254, 68], [152, 254, 66], [155, 253, 64], [158, 253, 62], [161, 252, 61], [164, 252, 59], [166, 251, 58], [169, 251, 57], [172, 250, 55], [174, 249, 55], [177, 248, 54], [179, 248, 53], [182, 247, 53], [185, 245, 52], [187, 244, 52], [190, 243, 52], [192, 242, 51], [195, 241, 51], [197, 239, 51], [200, 238, 51], [202, 237, 51], [205, 235, 52], [207, 234, 52], [209, 232, 52], [212, 231, 53], [214, 229, 53], [216, 227, 53], [218, 226, 54], [221, 224, 54], [223, 222, 54], [225, 220, 55], [227, 218, 55], [229, 216, 56], [231, 215, 56], [232, 213, 56], [234, 211, 57], [236, 209, 57], [237, 207, 57], [239, 205, 57], [240, 203, 58], [242, 200, 58], [243, 198, 58], [244, 196, 58], [246, 194, 58], [247, 192, 57], [248, 190, 57], [249, 188, 57], [249, 186, 56], [250, 183, 55], [251, 181, 55], [251, 179, 54], [252, 176, 53], [252, 174, 52], [253, 171, 51], [253, 169, 50], [253, 166, 49], [253, 163, 48], [254, 161, 47], [254, 158, 46], [254, 155, 45], [254, 152, 44], [253, 149, 43], [253, 146, 41], [253, 143, 40], [253, 140, 39], [252, 137, 38], [252, 134, 36], [251, 131, 35], [251, 128, 34], [250, 125, 32], [250, 122, 31], [249, 119, 30], [248, 116, 28], [247, 113, 27], [247, 110, 26], [246, 107, 24], [245, 104, 23], [244, 101, 22], [243, 99, 21], [242, 96, 20], [241, 93, 19], [239, 90, 17], [238, 88, 16], [237, 85, 15], [236, 82, 14], [234, 80, 13], [233, 77, 13], [232, 75, 12], [230, 73, 11], [229, 70, 10], [227, 68, 10], [226, 66, 9], [224, 64, 8], [222, 62, 8], [221, 60, 7], [219, 58, 7], [217, 56, 6], [215, 54, 6], [214, 52, 5], [212, 50, 5], [210, 48, 5], [208, 47, 4], [206, 45, 4], [203, 43, 3], [201, 41, 3], [199, 40, 3], [197, 38, 2], [195, 36, 2], [192, 35, 2], [190, 33, 2], [187, 31, 1], [185, 30, 1], [182, 28, 1], [180, 27, 1], [177, 25, 1], [174, 24, 1], [172, 22, 1], [169, 21, 1], [166, 20, 1], [163, 18, 1], [160, 17, 1], [157, 16, 1], [154, 14, 1], [151, 13, 1], [148, 12, 1], [145, 11, 1], [142, 10, 1], [139, 9, 1], [135, 8, 1], [132, 7, 1], [129, 6, 2], [125, 5, 2], [122, 4, 2]], dtype=np.uint8)

        z_min, z_max = np.min(z), np.max(z)
        if z_max > z_min:
            z_norm = (z - z_min) / (z_max - z_min)
            # OPT-1 Alternative (2x faster): Direct vectorized turbo LUT
            # avoids any matplotlib colormap allocation.
            indices = (z_norm * 255).astype(np.uint8)
            colors = _TURBO_LUT[indices].astype(np.float64) / 255.0
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            
        if not self.pcd_added:
            self.vis.add_geometry(self.pcd)
            self.pcd_added = True
        else:
            self.vis.update_geometry(self.pcd)
            
        # Update tracked objects
        # We need to add new linesets, update existing ones, and remove old ones
        current_track_ids = {t.track_id for t in state.tracks}
        existing_track_ids = set(self.track_boxes.keys())
        
        # Remove old boxes
        for t_id in existing_track_ids - current_track_ids:
            self.vis.remove_geometry(self.track_boxes[t_id], reset_bounding_box=False)
            del self.track_boxes[t_id]
            
        # Add/Update current boxes
        for t in state.tracks:
            # Color: Blue for dynamic, Purple for static
            color = [0.0, 0.5, 1.0] if t.dynamic else [0.6, 0.2, 0.8]
            
            new_lineset = self._create_box_lineset(
                size_lwh=t.size_lwh,
                center_xyz=t.state[:3],
                yaw_rad=t.yaw_rad,
                color=color
            )
            
            if t.track_id in self.track_boxes:
                # Open3D LineSet updates require replacing points/lines/colors
                old_lineset = self.track_boxes[t.track_id]
                old_lineset.points = new_lineset.points
                old_lineset.lines = new_lineset.lines
                old_lineset.colors = new_lineset.colors
                self.vis.update_geometry(old_lineset)
            else:
                self.track_boxes[t.track_id] = new_lineset
                self.vis.add_geometry(new_lineset, reset_bounding_box=False)

        self.drawn_tracks = list(state.tracks)

    def compute_track_labels(self):
        """Project a label anchor for every drawn box with the live camera."""
        params = self.view_control.convert_to_pinhole_camera_parameters()
        return build_track_labels(
            self.drawn_tracks,
            params.intrinsic.intrinsic_matrix,
            params.extrinsic,
            params.intrinsic.width,
            params.intrinsic.height,
        )

    def poll_events(self):
        """Must be called periodically to keep the Open3D window responsive."""
        self.vis.poll_events()
        self.vis.update_renderer()
        
    def destroy(self):
        self.vis.destroy_window()

def _publish_labels(label_queue, labels):
    """Hand the newest labels to the Qt process, dropping any it hasn't read."""
    try:
        while not label_queue.empty():
            label_queue.get_nowait()
    except Exception:
        pass
    try:
        label_queue.put_nowait(labels)
    except Exception:
        pass


def run_open3d_process(queue, ready_queue=None, window_name: str = "FOVEAX Phase 9 - 3D Dashboard",
                       label_queue=None):
    """Standalone process function to run Open3D.

    If ready_queue is given, this attempts to locate this process's own
    Open3D window by title (via win32gui) once it's created, and puts its
    HWND onto ready_queue so the parent process can reparent it into the
    Qt window (see src/dashboard/win32_embed.py). Puts None if win32gui
    isn't available or the window can't be located after retrying -- the
    parent treats that as "run as a normal free-floating window" (the
    original, unchanged behavior). This does not change anything about how
    the Open3D render loop itself runs below.
    """
    viewer = FoveaX3DViewer(window_name=window_name)

    if ready_queue is not None:
        hwnd = None
        try:
            from src.dashboard.win32_embed import find_window_by_title
            import os
            import time as _time
            own_pid = os.getpid()
            for _ in range(300):  # up to ~30s at 0.1s/try; GLFW can be slow under GPU load
                # own_pid guards against matching a same-titled Open3D
                # window from a different, concurrently running instance
                # of this dashboard -- FindWindow searches system-wide.
                hwnd = find_window_by_title(window_name, owner_pid=own_pid)
                if hwnd:
                    break
                _time.sleep(0.1)
        except Exception:
            hwnd = None
        ready_queue.put(hwnd)

    # We run our own event loop here
    quit_requested = False
    last_label_key = None
    while True:
        try:
            # Drain the queue each tick: apply every real-time camera
            # command as it arrives (never dropped), but only keep the
            # most recent FrameState (older frames are stale and safe to
            # skip -- this is unchanged from the original behavior).
            state = None
            while not queue.empty():
                item = queue.get_nowait()
                if isinstance(item, str) and item == "QUIT":
                    quit_requested = True
                    break
                elif isinstance(item, dict) and "cmd" in item:
                    viewer.handle_camera_command(item["cmd"])
                else:
                    state = item

            if quit_requested:
                break

            if state is not None:
                viewer.update(state)
        except Exception as e:
            pass
            
        viewer.poll_events()

        # Recomputed every tick, not just on new frames: mouse drags and the
        # dashboard's camera buttons move the view without a new FrameState.
        if label_queue is not None:
            try:
                labels = viewer.compute_track_labels()
                key = tuple((l.track_id, l.text, round(l.u), round(l.v), l.visible,
                             l.view_width, l.view_height) for l in labels)
                if key != last_label_key:
                    _publish_labels(label_queue, labels)
                    last_label_key = key
            except Exception:
                pass

        # 10ms sleep: keeps the GLFW poll rate at ~100 Hz which is more than
        # fast enough to drain frames pushed at 30 Hz from the Qt process.
        # Reducing this below ~8ms causes GLFW to call update_renderer() fast
        # enough to race with Qt's paint thread on Windows, invalidating the
        # OpenGL context (WGL "handle is invalid" errors, blank 3D view).
        import time
        time.sleep(0.010)
        
    viewer.destroy()

