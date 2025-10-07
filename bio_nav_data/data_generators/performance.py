"""
Performance Data Generator for Bio-Inspired Navigation Research

This module creates performance summary data that matches the metrics presented
in the research paper, providing a structured comparison between different methods.
"""

import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


def create_performance_summary() -> pd.DataFrame:
    """
    Create a comprehensive performance summary dataframe.

    This function generates performance metrics that match the data presented
    in the research paper, comparing the proposed system against baseline methods.

    Returns:
        pd.DataFrame: DataFrame containing performance metrics with columns:
            - Metric: Performance metric name
            - Unit: Unit of measurement
            - Proposed_System_Value: Value for the proposed system
            - Baseline_ORB_SLAM3_Value: Value for baseline ORB-SLAM3
            - Improvement: Improvement description
            - Improvement_Percentage: Numerical improvement percentage
    """
    logger.info("Creating performance summary data...")

    try:
        data = {
            "Metric": [
                "Localization Accuracy",
                "Energy Consumption",
                "Obstacle Avoidance",
                "Visual Occlusion Recovery",
                "Computational Efficiency",
                "Memory Usage",
                "Real-time Performance",
                "Robustness to Noise",
            ],
            "Unit": [
                "m",  # meters (lower is better)
                "J/m",  # Joules per meter (lower is better)
                "% success",  # percentage (higher is better)
                "s",  # seconds (lower is better)
                "FPS",  # frames per second (higher is better)
                "MB",  # megabytes (lower is better)
                "ms",  # milliseconds (lower is better)
                "dB",  # decibels (higher is better)
            ],
            "Proposed_System_Value": [
                0.25,  # 0.25m localization accuracy
                0.8,  # 0.8 J/m energy consumption
                90.0,  # 90% obstacle avoidance success
                1.2,  # 1.2s recovery time
                25.0,  # 25 FPS
                45.0,  # 45 MB memory usage
                40.0,  # 40ms processing time
                15.0,  # 15dB noise robustness
            ],
            "Baseline_ORB_SLAM3_Value": [
                0.42,  # 0.42m localization accuracy
                1.5,  # 1.5 J/m energy consumption
                65.0,  # 65% obstacle avoidance success
                3.0,  # 3.0s recovery time
                15.0,  # 15 FPS
                80.0,  # 80 MB memory usage
                67.0,  # 67ms processing time
                8.0,  # 8dB noise robustness
            ],
            "Improvement": [
                "40% ↑",  # 40% improvement in accuracy
                "47% ↓",  # 47% reduction in energy
                "38% ↑",  # 38% improvement in avoidance
                "60% faster",  # 60% faster recovery
                "67% ↑",  # 67% improvement in FPS
                "44% ↓",  # 44% reduction in memory
                "40% faster",  # 40% faster processing
                "88% ↑",  # 88% improvement in robustness
            ],
            "Improvement_Percentage": [
                40.5,  # (0.42-0.25)/0.42 * 100
                46.7,  # (1.5-0.8)/1.5 * 100
                38.5,  # (90-65)/65 * 100
                60.0,  # (3.0-1.2)/3.0 * 100
                66.7,  # (25-15)/15 * 100
                43.8,  # (80-45)/80 * 100
                40.3,  # (67-40)/67 * 100
                87.5,  # (15-8)/8 * 100
            ],
        }

        result_df = pd.DataFrame(data)

        # Add calculated fields
        result_df["Absolute_Improvement"] = (
            result_df["Baseline_ORB_SLAM3_Value"] - result_df["Proposed_System_Value"]
        )

        # Determine if improvement is positive (lower values are better for some metrics)
        lower_is_better = [
            "Localization Accuracy",
            "Energy Consumption",
            "Visual Occlusion Recovery",
            "Memory Usage",
            "Real-time Performance",
        ]

        result_df["Is_Improvement"] = result_df.apply(
            lambda row: (
                (row["Proposed_System_Value"] < row["Baseline_ORB_SLAM3_Value"])
                if row["Metric"] in lower_is_better
                else (row["Proposed_System_Value"] > row["Baseline_ORB_SLAM3_Value"])
            ),
            axis=1,
        )

        logger.info(
            f"Successfully created performance summary with {len(result_df)} metrics"
        )
        return result_df

    except Exception as e:
        logger.error(f"Error creating performance summary: {e}")
        raise


def get_performance_highlights(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract key performance highlights from the summary data.

    Args:
        df: Performance summary dataframe

    Returns:
        dict: Dictionary containing key performance highlights
    """
    highlights = {}

    # Find best improvements
    highlights["best_improvement"] = df["Improvement_Percentage"].max()
    highlights["best_metric"] = df.loc[df["Improvement_Percentage"].idxmax(), "Metric"]

    # Find worst improvements
    highlights["worst_improvement"] = df["Improvement_Percentage"].min()
    highlights["worst_metric"] = df.loc[df["Improvement_Percentage"].idxmin(), "Metric"]

    # Calculate average improvement
    highlights["average_improvement"] = df["Improvement_Percentage"].mean()

    # Count positive improvements
    highlights["positive_improvements"] = df["Is_Improvement"].sum()
    highlights["total_metrics"] = len(df)

    return highlights


def validate_performance_data(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate the performance summary data for consistency and reasonableness.

    Args:
        df: Performance summary dataframe

    Returns:
        dict: Dictionary containing validation results
    """
    validation = {}

    # Check for required columns
    required_columns = [
        "Metric",
        "Unit",
        "Proposed_System_Value",
        "Baseline_ORB_SLAM3_Value",
    ]
    validation["has_required_columns"] = all(
        col in df.columns for col in required_columns
    )

    # Check for non-negative values
    validation["non_negative_values"] = (df["Proposed_System_Value"] >= 0).all() and (
        df["Baseline_ORB_SLAM3_Value"] >= 0
    ).all()

    # Check for reasonable improvement percentages (-100 to 1000%)
    validation["reasonable_improvements"] = (
        (df["Improvement_Percentage"] >= -100) & (df["Improvement_Percentage"] <= 1000)
    ).all()

    # Check for consistent units
    validation["consistent_units"] = len(df["Unit"].unique()) == len(df)

    # Check for unique metrics
    validation["unique_metrics"] = len(df["Metric"].unique()) == len(df)

    return validation


def export_performance_report(df: pd.DataFrame, output_path: str) -> None:
    """
    Export a comprehensive performance report to CSV.

    Args:
        df: Performance summary dataframe
        output_path: Path to save the CSV file
    """
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Performance report exported to {output_path}")

        # Also create a summary report
        highlights = get_performance_highlights(df)
        validation = validate_performance_data(df)

        summary_report = f"""
Performance Summary Report
==========================

Key Highlights:
- Best Improvement: {highlights['best_improvement']:.1f}% ({highlights['best_metric']})
- Average Improvement: {highlights['average_improvement']:.1f}%
- Positive Improvements: {highlights['positive_improvements']}/{highlights['total_metrics']}

Data Quality:
- All validations passed: {all(validation.values())}
- Required columns present: {validation['has_required_columns']}
- Non-negative values: {validation['non_negative_values']}
- Reasonable improvements: {validation['reasonable_improvements']}
"""

        # Save summary report
        summary_path = output_path.replace(".csv", "_summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary_report)

        logger.info(f"Summary report exported to {summary_path}")

    except Exception as e:
        logger.error(f"Error exporting performance report: {e}")
        raise
