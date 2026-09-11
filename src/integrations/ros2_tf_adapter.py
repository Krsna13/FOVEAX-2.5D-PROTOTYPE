"""Adapter for TF2 Coordinate Transformations.

This module provides a robust interface to tf2_ros, converting PointCloud2 arrays
to target coordinate frames (e.g. base_link or map).
"""

from __future__ import annotations

import numpy as np

try:
    import rclpy
    from rclpy.time import Time
    from rclpy.duration import Duration
    from tf2_ros import Buffer, TransformListener, TransformException
    from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

class TF2Adapter:
    def __init__(self, node_handle=None, buffer_duration_sec: float = 10.0):
        """Initialize the TF2 adapter.
        
        Args:
            node_handle: rclpy Node object.
            buffer_duration_sec: Duration for the tf2 buffer (Task Q).
        """
        self.node = node_handle
        self.tf_buffer = None
        self.tf_listener = None
        
        if ROS2_AVAILABLE and self.node is not None:
            # 10s buffer as required by Task Q
            self.tf_buffer = Buffer(cache_time=Duration(seconds=buffer_duration_sec))
            self.tf_listener = TransformListener(self.tf_buffer, self.node)

    def get_transform_matrix(
        self, 
        source_frame: str, 
        target_frame: str, 
        timestamp_s: float,
        timeout_s: float = 0.1
    ) -> np.ndarray | None:
        """Lookup and return the 4x4 homogenous transform matrix.
        
        Args:
            source_frame: Original frame of the data (e.g. lidar_frame).
            target_frame: Desired frame (e.g. base_link or map).
            timestamp_s: Timestamp in seconds.
            timeout_s: Maximum time to block waiting for extrapolation (Task Q).
            
        Returns:
            4x4 numpy array representing the transform, or None if failed.
        """
        if not ROS2_AVAILABLE or self.tf_buffer is None:
            return None
            
        if source_frame == target_frame:
            return np.eye(4, dtype=np.float32)

        try:
            sec = int(timestamp_s)
            nanosec = int((timestamp_s - sec) * 1e9)
            time_obj = Time(seconds=sec, nanoseconds=nanosec)
            timeout_obj = rclpy.duration.Duration(seconds=timeout_s)
            
            # This handles both past (dropped) and future (extrapolation wait) exceptions (Task Q)
            # If the transform is not available after the timeout, it raises an exception.
            t = self.tf_buffer.lookup_transform(
                target_frame, 
                source_frame, 
                time_obj, 
                timeout=timeout_obj
            )
            
            # Convert Transform to 4x4 matrix
            trans = t.transform.translation
            rot = t.transform.rotation
            
            # Quaternion to Rotation Matrix
            q0, q1, q2, q3 = rot.w, rot.x, rot.y, rot.z
            R = np.array([
                [1 - 2*q2**2 - 2*q3**2,     2*q1*q2 - 2*q0*q3,     2*q1*q3 + 2*q0*q2],
                [    2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2,     2*q2*q3 - 2*q0*q1],
                [    2*q1*q3 - 2*q0*q2,     2*q2*q3 + 2*q0*q1, 1 - 2*q1**2 - 2*q2**2]
            ])
            
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            T[0, 3] = trans.x
            T[1, 3] = trans.y
            T[2, 3] = trans.z
            
            return T
            
        except ExtrapolationException as e:
            if self.node:
                self.node.get_logger().warn(f"TF2 Extrapolation Exception: {e}")
            return None
        except (LookupException, ConnectivityException, TransformException) as e:
            if self.node:
                self.node.get_logger().warn(f"TF2 Exception: {e}")
            return None

    def transform_points(self, points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """Apply a 4x4 homogenous transform to an (N, 4) point cloud array."""
        if transform_matrix is None:
            return points
            
        # points are [x, y, z, intensity]
        xyz = points[:, :3]
        
        # Homogenous coordinates
        xyz_h = np.ones((len(xyz), 4), dtype=np.float32)
        xyz_h[:, :3] = xyz
        
        # Apply transform
        transformed_xyz_h = xyz_h @ transform_matrix.T
        
        # Reconstruct output
        out = points.copy()
        out[:, :3] = transformed_xyz_h[:, :3]
        return out
