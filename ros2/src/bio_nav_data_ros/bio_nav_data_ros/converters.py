"""Helper utilities for converting dataframes to ROS2 message types."""

from __future__ import annotations

import json
from typing import Dict

import pandas as pd
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.time import Time
from std_msgs.msg import String


def _dataframe_to_path(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    frame_id: str,
    stamp: Time,
) -> Path:
    """Convert the provided dataframe columns to a ``nav_msgs/Path`` message."""
    path = Path()
    path.header.frame_id = frame_id
    path.header.stamp = stamp.to_msg()

    for _, row in df.iterrows():
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = stamp.to_msg()
        pose.pose.position.x = float(row[x_col])
        pose.pose.position.y = float(row[y_col])
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        path.poses.append(pose)

    return path


def build_trajectory_paths(
    df: pd.DataFrame,
    frame_id: str,
    stamp: Time,
) -> Dict[str, Path]:
    """Build path messages for all available trajectories."""
    return {
        'ground_truth': _dataframe_to_path(df, 'ground_truth_x', 'ground_truth_y', frame_id, stamp),
        'our_framework': _dataframe_to_path(df, 'our_framework_x', 'our_framework_y', frame_id, stamp),
        'traditional_slam': _dataframe_to_path(df, 'traditional_slam_x', 'traditional_slam_y', frame_id, stamp),
    }


def dataframe_to_json(df: pd.DataFrame) -> String:
    """Convert a dataframe to a JSON string ROS message."""
    msg = String()
    records = df.to_dict(orient='records')
    msg.data = json.dumps(records, default=float)
    return msg


def dict_to_json(data: Dict) -> String:
    """Convert a dictionary to a JSON string ROS message."""
    msg = String()
    msg.data = json.dumps(data, default=float)
    return msg
