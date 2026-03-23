"""
bio_nav_data.launch.py

Author      : Chandan Sheikder
Email       : chandan@bit.edu.cn
Phone       : +8618222390506
Affiliation : Beijing Institute of Technology (BIT)
Date        : 2026-03-23

Description:
    Module for Bio Nav Data.Launch
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="bio_nav_data_ros",
                executable="data_generation_node",
                name="bio_nav_data_node",
                output="screen",
                parameters=[
                    {
                        "publish_frequency_hz": 1.0,
                        "frame_id": "map",
                        "auto_generate_on_startup": True,
                        "save_results_to_disk": False,
                        "generate_plots": False,
                    }
                ],
            ),
        ]
    )
