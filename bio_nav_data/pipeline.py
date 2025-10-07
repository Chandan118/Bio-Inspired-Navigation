"""
Core pipeline orchestration for bio-inspired navigation data generation.

This module exposes the high-level ``BioNavDataGenerator`` class that coordinates
data generation, persistence, and visualization. It is reused by both the CLI
entry point and the ROS2 integration.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, Optional

from .data_generators.energy import EnergyDataGenerator
from .data_generators.performance import (
    create_performance_summary,
    export_performance_report,
)
from .data_generators.trajectory import TrajectoryDataGenerator
from .utils.config import Config
from .utils.logger import get_logger, log_execution_time
from .visualizers.plots import (
    create_combined_figure,
    plot_energy_consumption,
    plot_performance_comparison,
    plot_trajectory,
    setup_plotting_style,
)


class BioNavDataGenerator:
    """
    Main application class for generating bio-inspired navigation data.

    This class orchestrates the entire data generation and visualization process,
    providing a clean interface for generating research data and figures.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the data generator.

        Args:
            config: Configuration object (optional)
        """
        self.config = config or Config()
        self.logger = get_logger("BioNavDataGenerator")

        # Initialize data generators
        self.trajectory_generator = TrajectoryDataGenerator(
            **self.config.get_trajectory_params()
        )
        self.energy_generator = EnergyDataGenerator(
            **self.config.get_energy_params()
        )

        # Setup plotting style
        setup_plotting_style()

        self.logger.info("BioNavDataGenerator initialized successfully")

    @log_execution_time
    def generate_all_data(self) -> Dict[str, Any]:
        """
        Generate all research data.

        Returns:
            dict: Dictionary containing all generated dataframes
        """
        self.logger.info("Starting data generation process...")

        try:
            # Generate trajectory data
            self.logger.info("Generating trajectory data...")
            trajectory_df = self.trajectory_generator.generate()

            # Generate energy consumption data
            self.logger.info("Generating energy consumption data...")
            energy_df = self.energy_generator.generate()

            # Generate performance summary
            self.logger.info("Generating performance summary...")
            performance_df = create_performance_summary()

            # Validate data quality
            self._validate_generated_data(
                trajectory_df, energy_df, performance_df
            )

            self.logger.info("All data generated successfully")

            return {
                "trajectory": trajectory_df,
                "energy": energy_df,
                "performance": performance_df,
            }

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Error generating data: %s", exc)
            raise

    @log_execution_time
    def save_all_data(self, data: Dict[str, Any]) -> None:
        """
        Save all generated data to files.

        Args:
            data: Dictionary containing dataframes
        """
        self.logger.info("Saving all data to files...")

        try:
            # Save trajectory data
            trajectory_path = self.config.get_file_path("trajectory_data")
            data["trajectory"].to_csv(trajectory_path, index=False)
            self.logger.info("Saved trajectory data to %s", trajectory_path)

            # Save energy data
            energy_path = self.config.get_file_path("energy_data")
            data["energy"].to_csv(energy_path, index=False)
            self.logger.info("Saved energy data to %s", energy_path)

            # Save performance data
            performance_path = self.config.get_file_path("performance_data")
            export_performance_report(
                data["performance"], str(performance_path)
            )
            self.logger.info("Saved performance data to %s", performance_path)

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Error saving data: %s", exc)
            raise

    @log_execution_time
    def generate_all_plots(self, data: Dict[str, Any]) -> None:
        """
        Generate all research plots.

        Args:
            data: Dictionary containing dataframes
        """
        self.logger.info("Generating all plots...")

        try:
            # Generate trajectory plot
            trajectory_plot_path = self.config.get_file_path("trajectory_plot")
            plot_trajectory(
                data["trajectory"], save_path=str(trajectory_plot_path)
            )
            self.logger.info(
                "Generated trajectory plot: %s", trajectory_plot_path
            )

            # Generate energy consumption plot
            energy_plot_path = self.config.get_file_path("energy_plot")
            plot_energy_consumption(
                data["energy"], save_path=str(energy_plot_path)
            )
            self.logger.info("Generated energy plot: %s", energy_plot_path)

            # Generate performance comparison plot
            performance_plot_path = self.config.get_file_path(
                "performance_plot"
            )
            plot_performance_comparison(
                data["performance"], save_path=str(performance_plot_path)
            )
            self.logger.info(
                "Generated performance plot: %s", performance_plot_path
            )

            # Generate combined figure
            combined_plot_path = self.config.get_file_path("combined_plot")
            create_combined_figure(
                data["trajectory"],
                data["energy"],
                save_path=str(combined_plot_path),
            )
            self.logger.info(
                "Generated combined plot: %s", combined_plot_path
            )

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Error generating plots: %s", exc)
            raise

    def run_complete_pipeline(self) -> None:
        """Run the complete data generation and visualization pipeline."""
        try:
            self.logger.info("Running complete data generation pipeline...")

            # Generate all data
            data = self.generate_all_data()

            # Save all data
            self.save_all_data(data)

            # Generate all plots
            self.generate_all_plots(data)

            # Print summary
            self._print_summary(data)

            self.logger.info("Pipeline completed successfully!")

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Pipeline failed: %s", exc)
            self.logger.error(traceback.format_exc())
            raise

    def _validate_generated_data(
        self, trajectory_df, energy_df, performance_df
    ) -> None:
        """Validate generated dataframes before downstream processing."""
        if trajectory_df.empty:
            raise ValueError("Trajectory dataframe is empty")
        if energy_df.empty:
            raise ValueError("Energy dataframe is empty")
        if performance_df.empty:
            raise ValueError("Performance dataframe is empty")

    def _print_summary(self, data: Dict[str, Any]) -> None:
        """
        Print a summary of the generated data and results.

        Args:
            data: Dictionary containing generated data
        """
        print("\n" + "=" * 60)
        print("BIO-INSPIRED NAVIGATION DATA GENERATION SUMMARY")
        print("=" * 60)

        # Data summary
        print("\n📊 DATA SUMMARY:")
        print(f"  • Trajectory data: {len(data['trajectory'])} points")
        print(
            "  • Energy data: "
            f"{len(data['energy'])} trials across "
            f"{data['energy']['Method'].nunique()} methods"
        )
        print(
            f"  • Performance metrics: {len(data['performance'])} metrics"
        )

        # Energy improvements
        energy_improvements = self.energy_generator.calculate_improvements(
            data["energy"]
        )
        print("\n⚡ ENERGY IMPROVEMENTS:")
        for baseline, improvement in energy_improvements.items():
            label = baseline.replace("vs_", "vs ").replace("_", " ")
            print(f"  • {label}: {improvement:.1f}% reduction")

        # File locations
        print("\n📁 OUTPUT FILES:")
        paths = self.config.get_all_paths()
        for file_type, path in paths.items():
            if path.exists():
                print(f"  • {file_type}: {path}")

        print("\n" + "=" * 60)


__all__ = ["BioNavDataGenerator"]
