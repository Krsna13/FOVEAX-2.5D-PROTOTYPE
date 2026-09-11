import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    input_topic = LaunchConfiguration('input_topic')
    output_frame = LaunchConfiguration('output_frame')
    detector = LaunchConfiguration('detector')
    source_type = LaunchConfiguration('source_type')
    queue_size = LaunchConfiguration('queue_size')
    max_points = LaunchConfiguration('max_points')

    declare_input_topic = DeclareLaunchArgument(
        'input_topic', default_value='/lidar/points',
        description='Topic to subscribe for PointCloud2 data')
        
    declare_output_frame = DeclareLaunchArgument(
        'output_frame', default_value='base_link',
        description='Target TF2 frame for pointcloud transformation')
        
    declare_detector = DeclareLaunchArgument(
        'detector', default_value='mock',
        description='Detector type (mock or pointpillars)')
        
    declare_source_type = DeclareLaunchArgument(
        'source_type', default_value='live_sensor',
        description='Data provenance type (live_sensor, rosbag_replay, mock)')
        
    declare_queue_size = DeclareLaunchArgument(
        'queue_size', default_value='5',
        description='Worker queue depth for dropping frames gracefully')
        
    declare_max_points = DeclareLaunchArgument(
        'max_points', default_value='200000',
        description='Max points before voxel downsampling kicks in')

    foveax_node = Node(
        package='foveax_ros',
        executable='foveax_lidar_node',
        name='foveax_lidar_node',
        output='screen',
        parameters=[{
            'input_topic': input_topic,
            'output_frame': output_frame,
            'detector': detector,
            'source_type': source_type,
            'queue_size': queue_size,
            'max_points': max_points
        }]
    )

    return LaunchDescription([
        declare_input_topic,
        declare_output_frame,
        declare_detector,
        declare_source_type,
        declare_queue_size,
        declare_max_points,
        foveax_node
    ])
