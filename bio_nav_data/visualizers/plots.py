"""
Visualization Module for Bio-Inspired Navigation Research

This module provides high-quality plotting functions for generating publication-ready
figures from the bio-inspired navigation research data.
"""

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Set global plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


class PlotConfig:
    """Configuration class for plot styling and settings."""

    # Figure sizes
    FIGURE_SIZES = {
        "small": (8, 6),
        "medium": (10, 8),
        "large": (12, 10),
        "wide": (14, 6),
    }

    # Color schemes
    COLORS = {
        "ground_truth": "#000000",
        "our_framework": "#1f77b4",
        "traditional_slam": "#d62728",
        "baseline_1": "#ff7f0e",
        "baseline_2": "#2ca02c",
        "other_baseline": "#9467bd",
    }

    # Line styles
    LINE_STYLES = {"ground_truth": "--", "our_framework": "-", "traditional_slam": "-."}

    # Font settings
    FONT_SIZE = {"title": 16, "axis_label": 12, "tick_label": 10, "legend": 11}


def plot_trajectory(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
    show_grid: bool = True,
    show_legend: bool = True,
) -> plt.Figure:
    """
    Create a high-quality trajectory plot (Figure 13).

    Args:
        df: Trajectory dataframe with coordinate columns
        save_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        dpi: DPI for saved figure
        show_grid: Whether to show grid
        show_legend: Whether to show legend

    Returns:
        plt.Figure: The created figure object
    """
    logger.info("Creating trajectory plot...")

    try:
        fig, ax = plt.subplots(figsize=figsize)

        # Plot trajectories
        ax.plot(
            df["ground_truth_x"],
            df["ground_truth_y"],
            color=PlotConfig.COLORS["ground_truth"],
            linestyle=PlotConfig.LINE_STYLES["ground_truth"],
            linewidth=2.5,
            label="Ground Truth",
            alpha=0.9,
        )

        ax.plot(
            df["traditional_slam_x"],
            df["traditional_slam_y"],
            color=PlotConfig.COLORS["traditional_slam"],
            linestyle=PlotConfig.LINE_STYLES["traditional_slam"],
            linewidth=2,
            label="Traditional SLAM (Drift)",
            alpha=0.8,
        )

        ax.plot(
            df["our_framework_x"],
            df["our_framework_y"],
            color=PlotConfig.COLORS["our_framework"],
            linestyle=PlotConfig.LINE_STYLES["our_framework"],
            linewidth=2.5,
            label="Our Framework (Corrected)",
            alpha=0.9,
        )

        # Customize plot
        ax.set_title(
            "Figure 13: Simulated Localization Accuracy",
            fontsize=PlotConfig.FONT_SIZE["title"],
            fontweight="bold",
        )
        ax.set_xlabel("X Coordinate (m)", fontsize=PlotConfig.FONT_SIZE["axis_label"])
        ax.set_ylabel("Y Coordinate (m)", fontsize=PlotConfig.FONT_SIZE["axis_label"])

        # Set aspect ratio and limits
        ax.set_aspect("equal", "box")

        # Add grid if requested
        if show_grid:
            ax.grid(True, alpha=0.3)

        # Add legend if requested
        if show_legend:
            ax.legend(
                fontsize=PlotConfig.FONT_SIZE["legend"],
                loc="upper right",
                framealpha=0.9,
            )

        # Adjust layout
        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(
                save_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            logger.info(f"Trajectory plot saved to {save_path}")

        return fig

    except Exception as e:
        logger.error(f"Error creating trajectory plot: {e}")
        raise


def plot_energy_consumption(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
    show_outliers: bool = True,
    show_statistics: bool = True,
) -> plt.Figure:
    """
    Create a high-quality energy consumption box plot (Figure 15B).

    Args:
        df: Energy consumption dataframe
        save_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        dpi: DPI for saved figure
        show_outliers: Whether to show outliers
        show_statistics: Whether to show statistical annotations

    Returns:
        plt.Figure: The created figure object
    """
    logger.info("Creating energy consumption plot...")

    try:
        fig, ax = plt.subplots(figsize=figsize)

        # Create box plot
        sns.boxplot(
            data=df,
            x="Method",
            y="Energy_Consumption_Watts",
            ax=ax,
            showfliers=show_outliers,
        )

        # Customize plot
        ax.set_title(
            "Figure 15B: Simulated Energy Consumption",
            fontsize=PlotConfig.FONT_SIZE["title"],
            fontweight="bold",
        )
        ax.set_xlabel("Navigation Method", fontsize=PlotConfig.FONT_SIZE["axis_label"])
        ax.set_ylabel(
            "Energy Consumption (Watts)", fontsize=PlotConfig.FONT_SIZE["axis_label"]
        )

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=15, ha="right")

        # Add statistical annotations if requested
        if show_statistics:
            _add_energy_statistics(ax, df)

        # Adjust layout
        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(
                save_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            logger.info(f"Energy consumption plot saved to {save_path}")

        return fig

    except Exception as e:
        logger.error(f"Error creating energy consumption plot: {e}")
        raise


def _add_energy_statistics(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Add statistical annotations to the energy consumption plot."""
    try:
        # Calculate statistics for each method
        stats = (
            df.groupby("Method")["Energy_Consumption_Watts"]
            .agg(["mean", "std"])
            .round(1)
        )

        # Add mean values as text annotations
        for i, method in enumerate(df["Method"].unique()):
            mean_val = stats.loc[method, "mean"]
            std_val = stats.loc[method, "std"]

            # Position text above the box
            ax.text(
                i,
                ax.get_ylim()[1] * 0.95,
                f"μ={mean_val:.1f}\nσ={std_val:.1f}",
                ha="center",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        # Add improvement annotation
        our_method_mean = stats.loc["Our Method", "mean"]
        baseline1_mean = stats.loc["Baseline SLAM 1", "mean"]
        improvement = ((baseline1_mean - our_method_mean) / baseline1_mean) * 100

        ax.text(
            0.5,
            ax.get_ylim()[1] * 0.85,
            f"~{improvement:.0f}% Energy Reduction\nvs. Baseline SLAM 1",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
        )

    except Exception as e:
        logger.warning(f"Could not add statistics annotations: {e}")


def plot_performance_comparison(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
) -> plt.Figure:
    """
    Create a performance comparison bar chart.

    Args:
        df: Performance summary dataframe
        save_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        dpi: DPI for saved figure

    Returns:
        plt.Figure: The created figure object
    """
    logger.info("Creating performance comparison plot...")

    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Absolute values comparison
        x = np.arange(len(df))
        width = 0.35

        ax1.bar(
            x - width / 2,
            df["Proposed_System_Value"],
            width,
            label="Proposed System",
            color=PlotConfig.COLORS["our_framework"],
        )
        ax1.bar(
            x + width / 2,
            df["Baseline_ORB_SLAM3_Value"],
            width,
            label="Baseline ORB-SLAM3",
            color=PlotConfig.COLORS["traditional_slam"],
        )

        ax1.set_xlabel("Performance Metrics")
        ax1.set_ylabel("Value")
        ax1.set_title("Performance Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(df["Metric"], rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Improvement percentages
        colors = ["green" if x > 0 else "red" for x in df["Improvement_Percentage"]]
        ax2.barh(df["Metric"], df["Improvement_Percentage"], color=colors, alpha=0.7)
        ax2.set_xlabel("Improvement (%)")
        ax2.set_title("Improvement Percentages")
        ax2.axvline(x=0, color="black", linestyle="-", alpha=0.5)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(
                save_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            logger.info(f"Performance comparison plot saved to {save_path}")

        return fig

    except Exception as e:
        logger.error(f"Error creating performance comparison plot: {e}")
        raise


def create_combined_figure(
    trajectory_df: pd.DataFrame,
    energy_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8),
    dpi: int = 300,
) -> plt.Figure:
    """
    Create a combined figure with both trajectory and energy plots.

    Args:
        trajectory_df: Trajectory dataframe
        energy_df: Energy consumption dataframe
        save_path: Path to save the plot (optional)
        figsize: Figure size as (width, height)
        dpi: DPI for saved figure

    Returns:
        plt.Figure: The created figure object
    """
    logger.info("Creating combined figure...")

    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Trajectory
        ax1.plot(
            trajectory_df["ground_truth_x"],
            trajectory_df["ground_truth_y"],
            color=PlotConfig.COLORS["ground_truth"],
            linestyle=PlotConfig.LINE_STYLES["ground_truth"],
            linewidth=2,
            label="Ground Truth",
        )

        ax1.plot(
            trajectory_df["traditional_slam_x"],
            trajectory_df["traditional_slam_y"],
            color=PlotConfig.COLORS["traditional_slam"],
            linestyle=PlotConfig.LINE_STYLES["traditional_slam"],
            linewidth=2,
            label="Traditional SLAM",
        )

        ax1.plot(
            trajectory_df["our_framework_x"],
            trajectory_df["our_framework_y"],
            color=PlotConfig.COLORS["our_framework"],
            linestyle=PlotConfig.LINE_STYLES["our_framework"],
            linewidth=2,
            label="Our Framework",
        )

        ax1.set_title("Localization Accuracy")
        ax1.set_xlabel("X Coordinate (m)")
        ax1.set_ylabel("Y Coordinate (m)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect("equal", "box")

        # Plot 2: Energy Consumption
        sns.boxplot(data=energy_df, x="Method", y="Energy_Consumption_Watts", ax=ax2)
        ax2.set_title("Energy Consumption")
        ax2.set_xlabel("Navigation Method")
        ax2.set_ylabel("Energy Consumption (Watts)")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha="right")

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(
                save_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            logger.info(f"Combined figure saved to {save_path}")

        return fig

    except Exception as e:
        logger.error(f"Error creating combined figure: {e}")
        raise


def setup_plotting_style() -> None:
    """Set up consistent plotting style across all figures."""
    plt.rcParams.update(
        {
            "figure.figsize": (10, 8),
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
        }
    )

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    logger.info("Plotting style configured")
