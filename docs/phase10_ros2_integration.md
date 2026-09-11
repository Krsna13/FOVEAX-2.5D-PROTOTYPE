# Phase 10: ROS 2 Live Sensor Integration

This document covers the implementation details for Phase 10, connecting the FOVEAX perception and tracking pipeline to a live ROS 2 stream.

## Architecture & Threading Model

To ensure deterministic performance and avoid dropping sensor frames due to heavy perception computation, the FOVEAX Lidar Node uses a decoupled threading model:
1. **ROS Executor Thread (Single-Threaded)**: Subscribes to `/lidar/points` using a `sensor_data` QoS profile. The callback does strictly $O(1)$ work by enqueuing the raw `sensor_msgs/PointCloud2` onto a thread-safe queue.
2. **FOVEAX Worker Thread**: Dequeues messages, converts them to NumPy (handling NaN removal and default intensities), transforms them via `tf2_ros`, and runs the heavy detection/tracking pipeline. 

## Node Lifecycle

`foveax_lidar_node` is implemented as a plain `rclpy.Node`, not a `LifecycleNode`, for this phase. Lifecycle management (configure/activate/deactivate/cleanup states) is deferred to a future hardening pass ahead of Jetson/embedded deployment in Phase 11, since the added complexity is not needed for local development and testing.

## TF2 Integration and Buffering

We use `tf2_ros` to transform points from the sensor frame to a stable tracking frame (e.g., `base_link`). 
- A **10-second `tf2_ros.Buffer`** is explicitly configured. 
- The node handles `ExtrapolationException` by waiting gracefully with a short timeout (`0.1s`), preventing crashes on transient TF drops or delays. If a transform fails completely, the frame is logged as dropped in the diagnostics.

## QoS Profiles

The subscriber utilizes the `sensor_data` QoS profile:
- **Reliability**: BEST_EFFORT
- **Durability**: VOLATILE
- **Depth**: 5

This is critical because LiDAR drivers almost universally publish `PointCloud2` topics using `sensor_data` QoS. A mismatch (e.g., subscribing with `reliable` QoS) would result in a silent failure to connect.

## Output Formatting

To prevent performance bottlenecks associated with Python serialization of custom ROS 2 messages, the node serializes its detection, tracking, and map metadata outputs into compact JSON strings and publishes them over standard `std_msgs/msg/String` topics:
- `/foveax/detections`
- `/foveax/tracks`
- `/foveax/maps_metadata`

## Telemetry and Diagnostics

A comprehensive diagnostics manager logs node performance directly to `outputs/phase10/ros2_session/ros2_session.jsonl`. It tracks:
- Frame provenance (`live_sensor`, `mock`, etc.)
- Queue drops and TF lookup failures.
- In-pipeline processing latency vs end-to-end latency.
- NaN point drops and downsampling metrics.

Logs are automatically rotated when they exceed 50MB.

## Building and Running

Ensure you have a ROS 2 Jazzy or Humble environment sourced.

```bash
cd ros2_ws
colcon build --packages-select foveax_ros
source install/setup.bash

# Run using the launch file
ros2 launch foveax_ros foveax.launch.py source_type:=live_sensor input_topic:=/lidar/points
```
