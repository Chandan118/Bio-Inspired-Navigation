#!/usr/bin/env python3
"""Pytest suite for the Bio-Inspired Navigation Data toolkit."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

from bio_nav_data import BioNavDataGenerator, Config
from bio_nav_data.data_generators import EnergyDataGenerator, TrajectoryDataGenerator
from bio_nav_data.visualizers.plots import plot_energy_consumption, plot_trajectory

try:  # Optional import for type checking without hard dependency
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover - should not occur with requirements installed
    Figure = type("FigureStub", (), {})  # type: ignore[misc, assignment]


@pytest.fixture()
def temp_config(tmp_path: Path) -> Config:
    """Provide a configuration rooted in a temporary directory."""
    return Config(base_path=str(tmp_path))


def test_imports() -> None:
    """Key classes and functions should be importable."""
    assert BioNavDataGenerator is not None
    assert Config is not None
    assert TrajectoryDataGenerator is not None
    assert EnergyDataGenerator is not None
    assert plot_trajectory is not None
    assert plot_energy_consumption is not None


def test_configuration_validation(temp_config: Config) -> None:
    """Configuration updates and validation should succeed."""
    temp_config.update_trajectory_params(n_points=200)
    temp_config.update_energy_params(n_trials=150)

    validation = temp_config.validate_config()
    assert all(validation.values()), f"Configuration validation failed: {validation}"


def test_data_generation_shapes() -> None:
    """Trajectory/Energy generators should honour requested sizes."""
    traj_gen = TrajectoryDataGenerator(n_points=50)
    trajectory_df = traj_gen.generate()
    assert len(trajectory_df) == 50

    energy_gen = EnergyDataGenerator(n_trials=20)
    energy_df = energy_gen.generate()
    assert len(energy_df) == 4 * 20  # four methods defined in generator


def test_visualization_returns_figures() -> None:
    """Plot helpers must return matplotlib Figure objects."""
    traj_df = TrajectoryDataGenerator(n_points=20).generate()
    energy_df = EnergyDataGenerator(n_trials=10).generate()

    fig_traj = plot_trajectory(traj_df)
    fig_energy = plot_energy_consumption(energy_df)

    assert isinstance(fig_traj, Figure)
    assert isinstance(fig_energy, Figure)


@pytest.mark.parametrize("cli_args", [
    [],
    ["--config-only"],
    ["--validate"],
])
def test_main_cli(tmp_path: Path, cli_args: List[str]) -> None:
    """Exercise the CLI entry point with common flags."""
    import sys
    from subprocess import run

    cmd = [
        sys.executable,
        str(Path.cwd() / "main.py"),
        *cli_args,
        "--output-dir",
        str(tmp_path),
    ]

    result = run(cmd, check=False)
    assert result.returncode == 0, f"CLI exited with {result.returncode}"


def test_pipeline_end_to_end(temp_config: Config, tmp_path: Path) -> None:
    """Full pipeline should emit expected artifacts."""
    generator = BioNavDataGenerator(temp_config)
    data = generator.generate_all_data()

    assert {"trajectory", "energy", "performance"}.issubset(data.keys())

    generator.save_all_data(data)
    generator.generate_all_plots(data)

    outputs = list((tmp_path / "output").glob("*.csv"))
    plots = list((tmp_path / "plots").glob("*.png"))

    assert outputs, "Expected CSV outputs were not created"
    assert plots, "Expected PNG plots were not created"
