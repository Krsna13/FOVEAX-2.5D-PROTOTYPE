"""Adapter for converting ROS 2 PointCloud2 to NumPy arrays and vice-versa.

This module provides functionality to translate between `sensor_msgs/msg/PointCloud2`
and the (N, 4) float32 NumPy arrays expected by the FOVEAX pipeline.
"""

from __future__ import annotations

import sys
import struct
import numpy as np

# We try to import sensor_msgs_py, but if ROS 2 is not installed, we fallback
# to standard struct unpacking to allow unit testing.
try:
    from sensor_msgs.msg import PointCloud2, PointField
    from sensor_msgs_py import point_cloud2 as pc2
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    
    class PointField:
        INT8 = 1
        UINT8 = 2
        INT16 = 3
        UINT16 = 4
        INT32 = 5
        UINT32 = 6
        FLOAT32 = 7
        FLOAT64 = 8

def _get_struct_fmt(is_bigendian: bool, fields: list, field_names: list[str]) -> str:
    """Generate struct format string for unpacking."""
    fmt = '>' if is_bigendian else '<'
    offset = 0
    
    # We want to extract specific fields in order
    type_mappings = {
        PointField.INT8: 'b',
        PointField.UINT8: 'B',
        PointField.INT16: 'h',
        PointField.UINT16: 'H',
        PointField.INT32: 'i',
        PointField.UINT32: 'I',
        PointField.FLOAT32: 'f',
        PointField.FLOAT64: 'd',
    }
    
    for field_name in field_names:
        field = next((f for f in fields if f.name == field_name), None)
        if field is None:
            continue
            
        # Add padding if needed
        if field.offset > offset:
            fmt += f'{field.offset - offset}x'
            
        fmt += type_mappings.get(field.datatype, 'B')
        offset = field.offset + 4 # approximation, real size depends on datatype
        
    return fmt

def voxel_downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    """Downsample point cloud deterministically using voxel grid if max_points is exceeded."""
    if len(points) <= max_points:
        return points
        
    # Find a voxel size that gets us close to max_points.
    # Start small and increase. This is a deterministic approximation.
    # In a real system, we'd use a more sophisticated approach, but this fulfills Task P.
    voxel_size = 0.1
    for _ in range(10):
        # Quantize points to voxel grid
        voxel_indices = np.floor(points[:, :3] / voxel_size).astype(np.int32)
        
        # Use lexsort to group by voxel indices
        order = np.lexsort((voxel_indices[:, 2], voxel_indices[:, 1], voxel_indices[:, 0]))
        voxel_indices = voxel_indices[order]
        points_sorted = points[order]
        
        # Find unique voxels (keep first point in each voxel)
        diff = np.any(voxel_indices[1:] != voxel_indices[:-1], axis=1)
        unique_mask = np.concatenate(([True], diff))
        
        downsampled = points_sorted[unique_mask]
        if len(downsampled) <= max_points:
            return downsampled
            
        voxel_size *= 1.5
        
    # If still too many, just truncate the downsampled result deterministically
    return downsampled[:max_points]

def pointcloud2_to_numpy(msg, fields=("x", "y", "z", "intensity"), max_points: int | None = None) -> tuple[np.ndarray, dict]:
    """Convert sensor_msgs/msg/PointCloud2 to (N, 4) float32 array.
    
    Returns:
        points: np.ndarray shape (N, 4)
        metadata: dict
    """
    field_names = [f.name for f in msg.fields]
    
    if "x" not in field_names or "y" not in field_names or "z" not in field_names:
        raise ValueError(f"PointCloud2 must contain 'x', 'y', and 'z' fields. Found: {field_names}")
        
    intensity_available = "intensity" in field_names
    
    # We will use numpy to parse the buffer efficiently
    # This correctly handles byte offsets and endianness
    dtype_list = []
    
    # Define mapping from PointField types to numpy types
    _np_types = {
        PointField.INT8: np.int8,
        PointField.UINT8: np.uint8,
        PointField.INT16: np.int16,
        PointField.UINT16: np.uint16,
        PointField.INT32: np.int32,
        PointField.UINT32: np.uint32,
        PointField.FLOAT32: np.float32,
        PointField.FLOAT64: np.float64,
    }

    endian = '>' if msg.is_bigendian else '<'
    
    # Sort fields by offset
    sorted_fields = sorted(msg.fields, key=lambda f: f.offset)
    
    for f in sorted_fields:
        if f.name in fields:
            np_type = np.dtype(_np_types.get(f.datatype, np.uint8))
            dtype_list.append((f.name, f"{endian}{np_type.char}"))
        else:
            # Add dummy field for padding if needed to maintain structured array alignment
            # In practice, if we just read the exact structured type it works if we define all fields,
            # or we can read the raw buffer and construct the structured array carefully.
            pass
            
    # To properly handle arbitrary PointCloud2 formats, we define the full structured type:
    full_dtype = []
    current_offset = 0
    dummy_idx = 0
    for f in sorted_fields:
        if f.offset > current_offset:
            full_dtype.append((f'dummy_{dummy_idx}', f'V{f.offset - current_offset}'))
            dummy_idx += 1
        
        np_type = np.dtype(_np_types.get(f.datatype, np.uint8))
        full_dtype.append((f.name, f"{endian}{np_type.char}"))
        current_offset = f.offset + np_type.itemsize
        
    if msg.point_step > current_offset:
        full_dtype.append((f'dummy_{dummy_idx}', f'V{msg.point_step - current_offset}'))

    # Parse buffer
    cloud_data = np.frombuffer(msg.data, dtype=full_dtype)
    
    # Extract xyz
    x = cloud_data['x'].astype(np.float32)
    y = cloud_data['y'].astype(np.float32)
    z = cloud_data['z'].astype(np.float32)
    
    if intensity_available:
        intensity = cloud_data['intensity'].astype(np.float32)
    else:
        # Task U: Explicitly default to 0.0
        intensity = np.zeros_like(x, dtype=np.float32)
        
    points = np.column_stack((x, y, z, intensity))
    
    initial_count = len(points)
    
    # Remove NaN and Inf
    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    
    dropped_invalid = initial_count - len(points)
    
    # Limit points deterministically if configured (Task P)
    if max_points is not None and len(points) > max_points:
        points = voxel_downsample(points, max_points)
        
    metadata = {
        "frame_id": msg.header.frame_id,
        "timestamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        "point_count": len(points),
        "intensity_available": intensity_available,
        "dropped_invalid_points": dropped_invalid
    }
    
    return points, metadata

def numpy_to_pointcloud2(points: np.ndarray, frame_id: str, timestamp: float) -> 'PointCloud2':
    """Convert (N, 4) numpy array to sensor_msgs/msg/PointCloud2."""
    if not ROS2_AVAILABLE:
        raise ImportError("ROS 2 is not available in this environment.")
        
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp.sec = int(timestamp)
    msg.header.stamp.nanosec = int((timestamp - msg.header.stamp.sec) * 1e9)
    
    msg.height = 1
    msg.width = len(points)
    
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
    ]
    
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    
    msg.data = points.astype(np.float32).tobytes()
    return msg
