import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# --- Plotting Functions (Modified to show plots) ---

def plot_trajectory(df: pd.DataFrame, save_path: str):
    """Saves a plot of the trajectory data and displays it."""
    print(f"Creating trajectory plot at '{save_path}'...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(df['ground_truth_x'], df['ground_truth_y'], 'k--', label='Ground Truth', linewidth=2)
    ax.plot(df['traditional_slam_x'], df['traditional_slam_y'], 'r-.', label='Traditional SLAM (Drift)', alpha=0.8)
    ax.plot(df['our_framework_x'], df['our_framework_y'], 'b-', label='Our Framework (Corrected)', linewidth=2.5)
    
    ax.set_title('Figure 13: Simulated Localization Accuracy', fontsize=16)
    ax.set_xlabel('X Coordinate (m)', fontsize=12)
    ax.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show() # <-- ADD THIS LINE TO DISPLAY THE PLOT
    plt.close(fig)

def plot_energy_consumption(df: pd.DataFrame, save_path: str):
    """Saves a box plot of the energy consumption data and displays it."""
    print(f"Creating energy consumption plot at '{save_path}'...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.boxplot(data=df, x='Method', y='Energy_Consumption_Watts', ax=ax)
    
    try:
        mean_our_method = df[df['Method'] == 'Our Method']['Energy_Consumption_Watts'].mean()
        mean_baseline1 = df[df['Method'] == 'Baseline SLAM 1']['Energy_Consumption_Watts'].mean()
        if mean_baseline1 > 0:
            reduction = 100 * (1 - mean_our_method / mean_baseline1)
            ax.text(0.5, 0.9, f'~{reduction:.0f}% Energy Reduction vs. Baseline 1',
                    horizontalalignment='center',
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    except (KeyError, ZeroDivisionError) as e:
        print(f"Could not calculate annotation for energy plot: {e}")

    ax.set_title('Figure 15B: Simulated Energy Consumption per Task', fontsize=16)
    ax.set_xlabel('Navigation Method', fontsize=12)
    ax.set_ylabel('Energy Consumption (Watts)', fontsize=12)
    plt.xticks(rotation=15)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show() # <-- ADD THIS LINE TO DISPLAY THE PLOT
    plt.close(fig)

# --- (The rest of the script remains the same) ---

def create_trajectory_data() -> pd.DataFrame:
    t = np.linspace(0, 2 * np.pi, 150)
    gt_x, gt_y = 5 * np.cos(t), 5 * np.sin(t)
    noise = np.random.randn(150) * 0.1
    drift_x, drift_y = np.linspace(0, 1.5, 150), np.linspace(0, -1, 150)
    slam_x, slam_y = gt_x + noise + drift_x, gt_y + noise + drift_y
    our_noise = np.random.randn(150) * 0.05
    our_x, our_y = gt_x + our_noise, gt_y + our_noise
    return pd.DataFrame({
        'ground_truth_x': gt_x, 'ground_truth_y': gt_y,
        'traditional_slam_x': slam_x, 'traditional_slam_y': slam_y,
        'our_framework_x': our_x, 'our_framework_y': our_y,
    })

def create_energy_data() -> pd.DataFrame:
    np.random.seed(42)
    methods = (['Baseline SLAM 1'] * 50) + (['Baseline SLAM 2'] * 50) + (['Our Method'] * 50)
    energy_vals = np.concatenate([
        np.random.normal(loc=12.5, scale=1.0, size=50),
        np.random.normal(loc=13.0, scale=1.2, size=50),
        np.random.normal(loc=9.0, scale=0.8, size=50)
    ])
    return pd.DataFrame({'Method': methods, 'Energy_Consumption_Watts': energy_vals})
    
if __name__ == "__main__":
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    trajectory_df = create_trajectory_data()
    trajectory_save_path = os.path.join(output_dir, "figure_13_trajectory.png")
    plot_trajectory(trajectory_df, trajectory_save_path)

    energy_df = create_energy_data()
    energy_save_path = os.path.join(output_dir, "figure_15b_energy.png")
    plot_energy_consumption(energy_df, energy_save_path)
    
    print("\nAll plots have been successfully generated, displayed, and saved in the 'plots' directory.")