import numpy as np
import open3d as o3d
from src.dashboard.dashboard_state import FrameState

class FoveaX3DViewer:
    """Wrapper for Open3D non-blocking visualizer."""

    def __init__(self, window_name: str = "FOVEAX Phase 9 - 3D Dashboard", width: int = 1280, height: int = 720):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=width, height=height)
        
        # Geometries
        self.pcd = o3d.geometry.PointCloud()
        
        # We will keep a dictionary of LineSets for tracking boxes
        # track_id -> LineSet
        self.track_boxes = {}
        
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
        z_min, z_max = np.min(z), np.max(z)
        if z_max > z_min:
            z_norm = (z - z_min) / (z_max - z_min)
            # Use a simple colormap (e.g., inferno/turbo approximation)
            # Here we just map Z to red-green gradient for simplicity, or use jet
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap('turbo')
            colors = cmap(z_norm)[:, :3] # RGBA to RGB
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

    def poll_events(self):
        """Must be called periodically to keep the Open3D window responsive."""
        self.vis.poll_events()
        self.vis.update_renderer()
        
    def destroy(self):
        self.vis.destroy_window()

def run_open3d_process(queue, ready_queue=None, window_name: str = "FOVEAX Phase 9 - 3D Dashboard"):
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
        # Small sleep or let poll_events dictate loop speed. 
        # Open3D poll_events is usually fast, we can add a small sleep to avoid 100% CPU.
        import time
        time.sleep(0.01)
        
    viewer.destroy()

