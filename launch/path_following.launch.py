from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        # RViz2 with preconfigured settings
        DeclareLaunchArgument(
            'rviz_config',
            default_value=PathJoinSubstitution([
                FindPackageShare('waypoint_follower'),
                'rviz',
                'path_following.rviz'
            ]),
            description='Full path to RViz config file'
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            output='screen'
        ),

        # Odometry conversion node
        Node(
            package='waypoint_follower',
            executable='odometry_conversion',
            name='odometry_conversion',
            output='screen'
        ),

        # Path follower node
        Node(
            package='waypoint_follower',
            executable='path_follower',
            name='path_follower',
            output='screen'
        )
    ])
