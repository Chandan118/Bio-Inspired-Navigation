# ==============================================================================
#  COMPLETE, SELF-CONTAINED SCRIPT - PASTE THIS ENTIRE BLOCK IN ONE CELL
# ==============================================================================

# --- PREREQUISITES: Make sure required libraries are installed ---
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    print("✅ All required libraries are found.")
except ImportError:
    print("Installing required libraries: pandas, numpy, matplotlib, seaborn...")
    import pip
    pip.main(['install', 'pandas', 'numpy', 'matplotlib', 'seaborn'])
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    print("✅ Libraries installed successfully.")


# --------------------------------------------------------------------------
# --- DATA GENERATOR: TRAJECTORY (from data_generator/trajectory.py) ---
# --------------------------------------------------------------------------
class TrajectoryDataGenerator:
    """Generates sophisticated trajectory data matching Figure 13."""
    def __init__(self, n_points=101, distance_m=100, correction_interval=20):
        self.n_points = n_points
        self.distance_m = distance_m
        self.correction_interval = correction_interval
        np.random.seed(42)

    def generate(self):
        """Generates and returns the trajectory dataframe."""
        print("Generating complex trajectory data...")
        distances = np.linspace(0, self.distance_m, self.n_points)
        ground_truth = pd.DataFrame({'x': distances, 'y': np.zeros(self.n_points)})
        x_drift = np.cumsum(np.random.normal(0, 0.1, self.n_points))
        y_drift = np.cumsum(np.random.normal(0, 0.3, self.n_points))
        traditional_slam = pd.DataFrame({'x': ground_truth['x'] + x_drift, 'y': ground_truth['y'] + y_drift})
        our_framework_x, our_framework_y = np.zeros(self.n_points), np.zeros(self.n_points)
        for i in range(1, self.n_points):
            if i % self.correction_interval == 0:
                our_framework_x[i] = ground_truth['x'][i] + np.random.normal(0, 0.05)
                our_framework_y[i] = ground_truth['y'][i] + np.random.normal(0, 0.2)
            else:
                our_framework_x[i] = our_framework_x[i-1] + (ground_truth['x'][i] - ground_truth['x'][i-1]) + np.random.normal(0, 0.08)
                our_framework_y[i] = our_framework_y[i-1] + np.random.normal(0, 0.1)
        our_framework = pd.DataFrame({'x': our_framework_x, 'y': our_framework_y})
        return pd.DataFrame({
            'distance_m': distances, 'ground_truth_x': ground_truth['x'], 'ground_truth_y': ground_truth['y'],
            'our_framework_x': our_framework['x'], 'our_framework_y': our_framework['y'],
            'traditional_slam_x': traditional_slam['x'], 'traditional_slam_y': traditional_slam['y']
        })

# --------------------------------------------------------------------------
# --- DATA GENERATOR: ENERGY (from data_generator/energy.py) ---
# --------------------------------------------------------------------------
class EnergyDataGenerator:
    """Generates simulated energy consumption data for Figure 15B."""
    def __init__(self, n_trials=100):
        self.n_trials = n_trials
        np.random.seed(42)

    def generate(self):
        """Generates and returns the energy consumption dataframe."""
        print("Generating energy consumption data...")
        our_method_energy = np.random.normal(loc=30, scale=4, size=self.n_trials)
        baseline1_energy = np.random.normal(loc=50, scale=6, size=self.n_trials)
        baseline2_energy = np.random.normal(loc=55, scale=7.5, size=self.n_trials)
        other_baseline_energy = np.random.normal(loc=48, scale=5, size=self.n_trials)
        data = {
            'Method': ['Our Method'] * self.n_trials + ['Baseline SLAM 1'] * self.n_trials +
                      ['Baseline SLAM 2'] * self.n_trials + ['Other Baseline'] * self.n_trials,
            'Energy_Consumption_Watts': np.concatenate([our_method_energy, baseline1_energy, baseline2_energy, other_baseline_energy]),
            'Trial': list(range(1, self.n_trials + 1)) * 4
        }
        return pd.DataFrame(data)

# --------------------------------------------------------------------------
# --- DATA GENERATOR: PERFORMANCE (from data_generator/performance.py) ---
# --------------------------------------------------------------------------
def create_performance_summary():
    """Creates a dataframe summarizing performance metrics from Table S3."""
    print("Generating performance metrics summary...")
    data = {
        'Metric': ['Localization Accuracy', 'Energy Consumption', 'Obstacle Avoidance', 'Visual Occlusion Recovery'],
        'Unit': ['m', 'J/m', '% success', 's'],
        'Proposed_System_Value': [0.25, 0.8, 90.0, 1.2],
        'Baseline_ORB_SLAM3_Value': [0.42, 1.5, 65.0, 3.0],
        'Improvement': ['40% ↑', '47% ↓', '38% ↑', '60% faster']
    }
    return pd.DataFrame(data)

# --------------------------------------------------------------------------
# --- VISUALIZER: PLOTS (from visualizer/plots.py) ---
# --------------------------------------------------------------------------
def plot_trajectory(df: pd.DataFrame, save_path: str):
    """Saves a plot of the trajectory data (Figure 13)."""
    print(f"Creating plot -> {save_path}")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(df['ground_truth_x'], df['ground_truth_y'], 'k--', label='Ground Truth', linewidth=2)
    ax.plot(df['traditional_slam_x'], df['traditional_slam_y'], 'r-.', label='Traditional SLAM (Drift)', alpha=0.8)
    ax.plot(df['our_framework_x'], df['our_framework_y'], 'b-', label='Our Framework (Corrected)', linewidth=2.5)
    ax.set_title('Figure 13: Simulated Localization Accuracy', fontsize=16)
    ax.set_xlabel('X Coordinate (m)'); ax.set_ylabel('Y Coordinate (m)'); ax.legend(); ax.set_aspect('equal', 'box')
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

def plot_energy_consumption(df: pd.DataFrame, save_path: str):
    """Saves a box plot of the energy consumption data (Figure 15B)."""
    print(f"Creating plot -> {save_path}")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8)); sns.boxplot(data=df, x='Method', y='Energy_Consumption_Watts', ax=ax)
    ax.set_title('Figure 15B: Simulated Energy Consumption', fontsize=16)
    ax.set_xlabel('Navigation Method'); ax.set_ylabel('Energy Consumption (Watts)'); plt.xticks(rotation=15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

# --------------------------------------------------------------------------
# --- MAIN EXECUTION: Run everything ---
# --------------------------------------------------------------------------
print("\n--- STARTING DATA AND FIGURE GENERATION ---\n")
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
print(f"Files will be saved in the '{output_dir}/' directory.")

# File paths
path_trajectory_csv = os.path.join(output_dir, 'S1_localization_accuracy.csv')
path_energy_csv = os.path.join(output_dir, 'S1_energy_consumption.csv')
path_performance_csv = os.path.join(output_dir, 'S1_swarm_performance_summary.csv')
path_trajectory_plot = os.path.join(output_dir, 'Figure13_Localization_Accuracy.png')
path_energy_plot = os.path.join(output_dir, 'Figure15B_Energy_Consumption.png')

# 1. Generate all data
traj_gen = TrajectoryDataGenerator(); traj_df = traj_gen.generate(); traj_df.to_csv(path_trajectory_csv, index=False)
print(f"Saved: {path_trajectory_csv}")
energy_gen = EnergyDataGenerator(); energy_df = energy_gen.generate(); energy_df.to_csv(path_energy_csv, index=False)
print(f"Saved: {path_energy_csv}")
perf_df = create_performance_summary(); perf_df.to_csv(path_performance_csv, index=False)
print(f"Saved: {path_performance_csv}\n")

# 2. Generate all plots
print("--- GENERATING PLOTS ---")
plot_trajectory(traj_df, path_trajectory_plot)
plot_energy_consumption(energy_df, path_energy_plot)

print("\n--- PROCESS COMPLETE. Check the 'output' folder. ---")