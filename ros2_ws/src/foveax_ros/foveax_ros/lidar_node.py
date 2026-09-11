"""FOVEAX ROS 2 LiDAR Node.

Connects the FOVEAX Phase 8 pipeline to a live ROS 2 LiDAR stream.
Operates on a Single-threaded Executor where the ROS callback pushes raw messages
to a bounded thread-safe queue. A separate worker thread processes the FOVEAX pipeline.
"""

import sys
import os
import json
import time
import threading
import queue
from pathlib import Path

# Fix sys.path for FOVEAX imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.integrations.ros2_pointcloud_adapter import pointcloud2_to_numpy, ROS2_AVAILABLE
from src.integrations.ros2_tf_adapter import TF2Adapter
from foveax_ros.diagnostics import DiagnosticsManager

from src.perception.object_detector import MockObjectDetector
from src.tracking.multi_object_tracker import MultiObjectTracker
from src.dashboard.data_streamer import DataStreamerThread

if not ROS2_AVAILABLE:
    print("[ERROR] ROS 2 is not installed or not sourced in this environment.")
    print("This node requires rclpy to run. See docs/phase10_ros2_integration.md")
    sys.exit(1)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String


class FoveaxLidarNode(Node):
    def __init__(self):
        super().__init__('foveax_lidar_node')
        
        # 1. Check ROS_DISTRO (Task S)
        target_distro = "jazzy" # or humble
        current_distro = os.environ.get("ROS_DISTRO", "unknown").lower()
        if current_distro not in ["jazzy", "humble"]:
            self.get_logger().warn(f"Developed/Tested against Jazzy/Humble. Current ROS_DISTRO is {current_distro}.")
            
        # 2. Parameters
        self.declare_parameter('input_topic', '/lidar/points')
        self.declare_parameter('output_frame', 'base_link')
        self.declare_parameter('detector', 'mock')
        self.declare_parameter('source_type', 'live_sensor') # Task T
        self.declare_parameter('queue_size', 5)
        self.declare_parameter('max_points', 200000)
        
        self.input_topic = self.get_parameter('input_topic').value
        self.output_frame = self.get_parameter('output_frame').value
        self.detector_type = self.get_parameter('detector').value
        self.source_type = self.get_parameter('source_type').value
        self.queue_size = self.get_parameter('queue_size').value
        self.max_points = self.get_parameter('max_points').value
        
        self.get_logger().info(f"Initializing FOVEAX Lidar Node (Source: {self.source_type})")
        
        # 3. Adapters & Pipeline
        self.tf_adapter = TF2Adapter(self, buffer_duration_sec=10.0)
        self.diagnostics = DiagnosticsManager(source_type=self.source_type)
        
        if self.detector_type == 'mock':
            self.detector = MockObjectDetector()
        else:
            self.get_logger().warn("PointPillars detector requested but not fully integrated. Falling back to Mock.")
            self.detector = MockObjectDetector()
            
        self.tracker = MultiObjectTracker()
        
        # For map generation (reusing data_streamer logic)
        self.streamer_logic = DataStreamerThread(source="ros2", detector=self.detector)
        
        # 4. Publishers
        self.pub_detections = self.create_publisher(String, '/foveax/detections', 10)
        self.pub_tracks = self.create_publisher(String, '/foveax/tracks', 10)
        self.pub_maps = self.create_publisher(String, '/foveax/maps_metadata', 10)
        
        # 5. Worker Thread & Queue (Task O)
        self.msg_queue = queue.Queue(maxsize=self.queue_size)
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_running = True
        self.worker_thread.start()
        
        # 6. Subscriber with sensor_data QoS (Task O)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.sub_points = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.points_callback,
            qos_profile
        )

    def points_callback(self, msg: PointCloud2):
        """ROS callback. Runs on the executor thread. ONLY enqueues data."""
        self.diagnostics.record_received()
        
        try:
            self.msg_queue.put_nowait(msg)
        except queue.Full:
            self.diagnostics.record_dropped_queue()

    def _worker_loop(self):
        """Runs the FOVEAX pipeline completely outside the ROS event loop."""
        while self.worker_running:
            try:
                # Block until we get a message
                msg = self.msg_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            t_start = time.perf_counter()
            msg_arrival_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            # 1. Convert PC2 -> Numpy (Task P, U applied in adapter)
            try:
                points_xyzi, meta = pointcloud2_to_numpy(msg, max_points=self.max_points)
            except Exception as e:
                self.get_logger().error(f"PointCloud conversion failed: {e}")
                continue
                
            # 2. Transform to output_frame (Task Q)
            T = self.tf_adapter.get_transform_matrix(
                source_frame=msg.header.frame_id,
                target_frame=self.output_frame,
                timestamp_s=meta["timestamp"],
                timeout_s=0.1
            )
            
            if T is None and self.output_frame != msg.header.frame_id:
                self.diagnostics.record_dropped_tf()
                continue
                
            points_transformed = self.tf_adapter.transform_points(points_xyzi, T)
            
            # 3. Run Pipeline (Detect -> Track -> Map)
            detections = self.detector.detect(points_transformed, timestamp_s=meta["timestamp"])
            tracks = self.tracker.update(detections, timestamp_s=meta["timestamp"])
            grids = self.streamer_logic._generate_maps(points_transformed, tracks)
            
            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000.0
            
            # Calculate end-to-end latency if time synchronization permits
            current_time = time.time()
            end_to_end_ms = (current_time - msg_arrival_time) * 1000.0
            
            self.diagnostics.log_frame(meta, latency_ms, end_to_end_ms)
            
            # 4. Publish Results (Task R)
            self._publish_results(meta["timestamp"], detections, tracks, grids)

    def _publish_results(self, timestamp: float, detections: list, tracks: list, grids: dict):
        """Publish JSON strings to avoid custom message compilation overheads."""
        # Detections
        det_data = []
        for d in detections:
            det_data.append({
                "class_name": d.class_name,
                "confidence": d.confidence,
                "center": d.center_xyz.tolist(),
                "size": d.size_lwh.tolist()
            })
        
        msg_det = String(data=json.dumps({"timestamp": timestamp, "detections": det_data}))
        self.pub_detections.publish(msg_det)
        
        # Tracks
        track_data = []
        for t in tracks:
            track_data.append({
                "id": t.track_id,
                "class_name": t.class_name,
                "dynamic": t.dynamic,
                "state": t.state.tolist()
            })
            
        msg_track = String(data=json.dumps({"timestamp": timestamp, "tracks": track_data}))
        self.pub_tracks.publish(msg_track)
        
        # Map Metadata
        # Sending full grids over std_msgs/String is slow, so we only send metadata for now
        map_meta = String(data=json.dumps({
            "timestamp": timestamp,
            "resolution": self.streamer_logic.grid_res,
            "extent": self.streamer_logic.grid_extent
        }))
        self.pub_maps.publish(map_meta)

    def destroy_node(self):
        self.worker_running = False
        self.worker_thread.join(timeout=2.0)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = FoveaxLidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
