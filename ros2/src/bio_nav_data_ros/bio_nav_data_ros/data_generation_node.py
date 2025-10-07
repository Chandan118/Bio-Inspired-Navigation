"""ROS2 node that wraps the BioNavData pipeline and publishes results."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from nav_msgs.msg import Path as PathMsg

# Ensure the core package is importable when running from the source tree
try:  # pragma: no cover - fallback for local runs
    from bio_nav_data.pipeline import BioNavDataGenerator
    from bio_nav_data.utils.config import Config
except ImportError:  # pragma: no cover - fallback for local runs
    repo_root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo_root))
    from bio_nav_data.pipeline import BioNavDataGenerator
    from bio_nav_data.utils.config import Config

from .converters import build_trajectory_paths, dataframe_to_json, dict_to_json


class BioNavDataRosNode(Node):
    """Expose the data-generation pipeline as a ROS2 node."""

    def __init__(self) -> None:
        super().__init__('bio_nav_data_node')

        # Parameters for controlling behaviour
        self.declare_parameter('publish_frequency_hz', 1.0)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('auto_generate_on_startup', True)
        self.declare_parameter('output_directory', '')
        self.declare_parameter('save_results_to_disk', False)
        self.declare_parameter('generate_plots', False)

        # Data-generation parameters
        self.declare_parameter('trajectory.n_points', 101)
        self.declare_parameter('trajectory.distance_m', 100.0)
        self.declare_parameter('trajectory.correction_interval', 20)
        self.declare_parameter('trajectory.random_seed', 42)
        self.declare_parameter('energy.n_trials', 100)
        self.declare_parameter('energy.random_seed', 42)

        self._publisher_frame_id = self.get_parameter('frame_id').value

        config = self._build_config()
        self.generator = BioNavDataGenerator(config)

        self.latest_data: Optional[Dict[str, Any]] = None
        self.latest_stats: Optional[Dict[str, Any]] = None

        # ROS interfaces
        self.trajectory_publishers = {
            'ground_truth': self.create_publisher(PathMsg, 'bio_nav_data/trajectory/path/ground_truth', 10),
            'our_framework': self.create_publisher(PathMsg, 'bio_nav_data/trajectory/path/our_framework', 10),
            'traditional_slam': self.create_publisher(PathMsg, 'bio_nav_data/trajectory/path/traditional_slam', 10),
        }
        self.trajectory_json_pub = self.create_publisher(String, 'bio_nav_data/trajectory/json', 10)
        self.energy_json_pub = self.create_publisher(String, 'bio_nav_data/energy/json', 10)
        self.performance_json_pub = self.create_publisher(String, 'bio_nav_data/performance/json', 10)
        self.analytics_json_pub = self.create_publisher(String, 'bio_nav_data/analytics/json', 10)

        self.trigger_service = self.create_service(
            Trigger,
            'bio_nav_data/generate',
            self._handle_generate_request,
        )

        publish_frequency = float(self.get_parameter('publish_frequency_hz').value)
        self.publish_timer = None
        if publish_frequency > 0:
            self.publish_timer = self.create_timer(
                1.0 / publish_frequency,
                self._publish_latest_data,
            )

        if bool(self.get_parameter('auto_generate_on_startup').value):
            self._run_pipeline(triggered=False)

        self.get_logger().info('bio_nav_data_ros node initialised')

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _build_config(self) -> Config:
        """Initialise configuration using declared ROS parameters."""
        output_dir = self.get_parameter('output_directory').get_parameter_value().string_value
        config = Config(output_dir) if output_dir else Config()

        trajectory_updates = {
            'n_points': int(self.get_parameter('trajectory.n_points').value),
            'distance_m': float(self.get_parameter('trajectory.distance_m').value),
            'correction_interval': int(self.get_parameter('trajectory.correction_interval').value),
            'random_seed': int(self.get_parameter('trajectory.random_seed').value),
        }
        config.update_trajectory_params(**trajectory_updates)

        energy_updates = {
            'n_trials': int(self.get_parameter('energy.n_trials').value),
            'random_seed': int(self.get_parameter('energy.random_seed').value),
        }
        config.update_energy_params(**energy_updates)

        return config

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def _run_pipeline(self, *, triggered: bool) -> bool:
        """Execute the pipeline and cache results for publishing."""
        try:
            data = self.generator.generate_all_data()
            self.latest_data = data
            self.latest_stats = self._build_analytics_payload(data)

            if bool(self.get_parameter('save_results_to_disk').value):
                self.generator.save_all_data(data)
            if bool(self.get_parameter('generate_plots').value):
                self.generator.generate_all_plots(data)

            # Publish immediately so subscribers get fresh data
            self._publish_latest_data()

            status = 'triggered' if triggered else 'startup'
            self.get_logger().info('Pipeline run (%s) completed successfully', status)
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            self.get_logger().error('Pipeline execution failed: %s', exc)
            return False

    def _build_analytics_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build a combined analytics dictionary for downstream consumers."""
        trajectory_stats = self.generator.trajectory_generator.get_statistics(data['trajectory'])
        energy_stats = self.generator.energy_generator.get_statistics(data['energy'])
        energy_improvements = self.generator.energy_generator.calculate_improvements(data['energy'])
        energy_quality = self.generator.energy_generator.validate_data_quality(data['energy'])
        config_validation = self.generator.config.validate_config()

        return {
            'trajectory_stats': trajectory_stats,
            'energy_stats': energy_stats,
            'energy_improvements_percent': energy_improvements,
            'energy_quality_checks': energy_quality,
            'config_validation': config_validation,
        }

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _handle_generate_request(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        success = self._run_pipeline(triggered=True)
        response.success = success
        response.message = 'Data generated successfully' if success else 'Pipeline execution failed (see logs).'
        return response

    def _publish_latest_data(self) -> None:
        if not self.latest_data:
            return

        stamp = self.get_clock().now()
        paths = build_trajectory_paths(
            self.latest_data['trajectory'],
            self._publisher_frame_id,
            stamp,
        )
        for key, publisher in self.trajectory_publishers.items():
            publisher.publish(paths[key])

        self.trajectory_json_pub.publish(dataframe_to_json(self.latest_data['trajectory']))
        self.energy_json_pub.publish(dataframe_to_json(self.latest_data['energy']))
        self.performance_json_pub.publish(dataframe_to_json(self.latest_data['performance']))

        if self.latest_stats:
            self.analytics_json_pub.publish(dict_to_json(self.latest_stats))

    # ------------------------------------------------------------------
    # Lifecycle utilities
    # ------------------------------------------------------------------
    def destroy_node(self) -> bool:
        if self.publish_timer:
            self.publish_timer.cancel()
        return super().destroy_node()


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = BioNavDataRosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover - user stop
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


__all__ = ['BioNavDataRosNode', 'main']
