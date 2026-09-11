import pytest
import numpy as np

from src.integrations.ros2_tf_adapter import TF2Adapter

def test_transform_points_none():
    adapter = TF2Adapter()
    
    pts = np.array([
        [1.0, 2.0, 3.0, 0.5],
        [4.0, 5.0, 6.0, 0.8]
    ], dtype=np.float32)
    
    # Passing None should return the original points untouched
    out = adapter.transform_points(pts, None)
    
    np.testing.assert_array_equal(out, pts)

def test_transform_points_translation():
    adapter = TF2Adapter()
    
    pts = np.array([
        [1.0, 2.0, 3.0, 0.5],
        [4.0, 5.0, 6.0, 0.8]
    ], dtype=np.float32)
    
    # 4x4 matrix with pure translation (+10x, -5y, +2z)
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = 10.0
    T[1, 3] = -5.0
    T[2, 3] = 2.0
    
    out = adapter.transform_points(pts, T)
    
    expected = np.array([
        [11.0, -3.0, 5.0, 0.5],
        [14.0, 0.0, 8.0, 0.8]
    ], dtype=np.float32)
    
    np.testing.assert_array_almost_equal(out, expected)

def test_transform_points_rotation_translation():
    adapter = TF2Adapter()
    
    pts = np.array([
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.8]
    ], dtype=np.float32)
    
    # 90 degrees around Z axis, then translate +10x
    T = np.eye(4, dtype=np.float32)
    T[0, 0] = 0.0
    T[0, 1] = -1.0
    T[1, 0] = 1.0
    T[1, 1] = 0.0
    T[0, 3] = 10.0
    
    out = adapter.transform_points(pts, T)
    
    # point 1: (1, 0) rotated 90deg -> (0, 1) translated -> (10, 1)
    # point 2: (0, 1) rotated 90deg -> (-1, 0) translated -> (9, 0)
    expected = np.array([
        [10.0, 1.0, 0.0, 0.5],
        [9.0, 0.0, 0.0, 0.8]
    ], dtype=np.float32)
    
    np.testing.assert_array_almost_equal(out, expected)
