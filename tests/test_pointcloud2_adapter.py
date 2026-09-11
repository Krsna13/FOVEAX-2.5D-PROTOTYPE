import pytest
import numpy as np

# Mock classes to simulate PointCloud2 without ROS 2 installed
class MockPointField:
    INT8 = 1
    UINT8 = 2
    INT16 = 3
    UINT16 = 4
    INT32 = 5
    UINT32 = 6
    FLOAT32 = 7
    FLOAT64 = 8

    def __init__(self, name, offset, datatype, count=1):
        self.name = name
        self.offset = offset
        self.datatype = datatype
        self.count = count

class MockHeader:
    def __init__(self):
        self.frame_id = "lidar_frame"
        self.stamp = self.Stamp()
        
    class Stamp:
        def __init__(self):
            self.sec = 12345
            self.nanosec = 678900000

class MockPointCloud2:
    def __init__(self):
        self.header = MockHeader()
        self.fields = []
        self.is_bigendian = False
        self.point_step = 0
        self.row_step = 0
        self.height = 1
        self.width = 0
        self.is_dense = True
        self.data = b''

from src.integrations.ros2_pointcloud_adapter import pointcloud2_to_numpy, voxel_downsample

def create_mock_pc2(points: np.ndarray, include_intensity=True):
    msg = MockPointCloud2()
    msg.width = len(points)
    
    msg.fields = [
        MockPointField('x', 0, MockPointField.FLOAT32),
        MockPointField('y', 4, MockPointField.FLOAT32),
        MockPointField('z', 8, MockPointField.FLOAT32),
    ]
    
    if include_intensity:
        msg.fields.append(MockPointField('intensity', 12, MockPointField.FLOAT32))
        msg.point_step = 16
    else:
        msg.point_step = 12
        
    msg.row_step = msg.point_step * msg.width
    
    if include_intensity:
        # points should be Nx4
        msg.data = points.astype(np.float32).tobytes()
    else:
        # points should be Nx3
        msg.data = points[:, :3].astype(np.float32).tobytes()
        
    return msg

def test_pointcloud2_to_numpy_with_intensity():
    pts = np.array([
        [1.0, 2.0, 3.0, 0.5],
        [4.0, 5.0, 6.0, 0.8]
    ], dtype=np.float32)
    msg = create_mock_pc2(pts, include_intensity=True)
    
    out, meta = pointcloud2_to_numpy(msg)
    
    np.testing.assert_array_equal(out, pts)
    assert meta["intensity_available"] == True
    assert meta["point_count"] == 2
    assert meta["dropped_invalid_points"] == 0

def test_pointcloud2_to_numpy_without_intensity():
    pts = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], dtype=np.float32)
    msg = create_mock_pc2(pts, include_intensity=False)
    
    out, meta = pointcloud2_to_numpy(msg)
    
    # Should default intensity to 0.0
    expected = np.array([
        [1.0, 2.0, 3.0, 0.0],
        [4.0, 5.0, 6.0, 0.0]
    ], dtype=np.float32)
    
    np.testing.assert_array_equal(out, expected)
    assert meta["intensity_available"] == False

def test_pointcloud2_to_numpy_nan_removal():
    pts = np.array([
        [1.0, 2.0, 3.0, 0.5],
        [np.nan, 5.0, 6.0, 0.8],
        [7.0, np.inf, 9.0, 0.1]
    ], dtype=np.float32)
    msg = create_mock_pc2(pts, include_intensity=True)
    
    out, meta = pointcloud2_to_numpy(msg)
    
    expected = np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32)
    
    np.testing.assert_array_equal(out, expected)
    assert meta["point_count"] == 1
    assert meta["dropped_invalid_points"] == 2

def test_voxel_downsample():
    # Create 10,000 points
    np.random.seed(42)
    pts = np.random.rand(10000, 4).astype(np.float32)
    
    downsampled = voxel_downsample(pts, max_points=1000)
    
    assert len(downsampled) <= 1000
    assert len(downsampled) > 0
    # ensure it didn't just truncate without thinking (hard to assert robustly but length check is fine)
