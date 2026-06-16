"""Launch the live KITTI segmentation player and (optionally) RViz2.

Examples:
    ros2 launch kitti_seg_sim sim.launch.py
    ros2 launch kitti_seg_sim sim.launch.py color_source:=gt rate_hz:=15.0
    ros2 launch kitti_seg_sim sim.launch.py start_frame:=3633 rviz:=false
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("kitti_seg_sim"), "config", "kitti_seg.rviz"])

    args = [
        DeclareLaunchArgument("color_source", default_value="pred",
                              description="pred (run model) | gt (labels)"),
        DeclareLaunchArgument("rate_hz", default_value="10.0"),
        DeclareLaunchArgument("start_frame", default_value="0"),
        DeclareLaunchArgument("end_frame", default_value="-1"),
        DeclareLaunchArgument("loop", default_value="true"),
        DeclareLaunchArgument("data_root", default_value=""),
        DeclareLaunchArgument("rviz", default_value="true",
                              description="also start RViz2"),
    ]

    # only override data_root if the user passed one (else node default applies)
    params = {
        "color_source": LaunchConfiguration("color_source"),
        "rate_hz": ParameterValue(LaunchConfiguration("rate_hz"), value_type=float),
        "start_frame": ParameterValue(LaunchConfiguration("start_frame"), value_type=int),
        "end_frame": ParameterValue(LaunchConfiguration("end_frame"), value_type=int),
        "loop": ParameterValue(LaunchConfiguration("loop"), value_type=bool),
    }

    player = Node(
        package="kitti_seg_sim",
        executable="player_node",
        name="kitti_seg_player",
        output="screen",
        parameters=[params],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        condition=IfCondition(LaunchConfiguration("rviz")),
        output="log",
    )

    return LaunchDescription(args + [player, rviz])
