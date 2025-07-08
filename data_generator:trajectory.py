import pandas as pd
import numpy as np

class TrajectoryDataGenerator:
    """
    Generates sophisticated trajectory data matching Figure 13.
    This model includes periodic error correction for the proposed framework.
    """
    def __init__(self, n_points=101, distance_m=100, correction_interval=20):
        self.n_points = n_points
        self.distance_m = distance_m
        self.correction_interval = correction_interval
        np.random.seed(42) # Ensure reproducibility

    def generate(self):
        """ Returns the trajectory dataframe."""
        print(" Complex trajectory data for Figure 13...")
        
        # Ground truth path
        distances = np.linspace(0, self.distance_m, self.n_points)
        ground_truth = pd.DataFrame({'x': distances, 'y': np.zeros(self.n_points)})

        # --- Simulate Traditional SLAM with accumulating drift ---
        x_drift = np.cumsum(np.random.normal(0, 0.1, self.n_points))
        y_drift = np.cumsum(np.random.normal(0, 0.3, self.n_points))
        traditional_slam = pd.DataFrame({
            'x': ground_truth['x'] + x_drift,
            'y': ground_truth['y'] + y_drift
        })

        # --- Simulate Our Framework with periodic error correction ---
        our_framework_x = np.zeros(self.n_points)
        our_framework_y = np.zeros(self.n_points)
        for i in range(1, self.n_points):
            if i % self.correction_interval == 0:
                our_framework_x[i] = ground_truth['x'][i] + np.random.normal(0, 0.05)
                our_framework_y[i] = ground_truth['y'][i] + np.random.normal(0, 0.2)
            else:
                our_framework_x[i] = our_framework_x[i-1] + (ground_truth['x'][i] - ground_truth['x'][i-1]) + np.random.normal(0, 0.08)
                our_framework_y[i] = our_framework_y[i-1] + np.random.normal(0, 0.1)
        
        our_framework = pd.DataFrame({'x': our_framework_x, 'y': our_framework_y})

        final_df = pd.DataFrame({
            'distance_m': distances,
            'ground_truth_x': ground_truth['x'],
            'ground_truth_y': ground_truth['y'],
            'our_framework_x': our_framework['x'],
            'our_framework_y': our_framework['y'],
            'traditional_slam_x': traditional_slam['x'],
            'traditional_slam_y': traditional_slam['y'],
        })
        
        return final_df

# ===================================================================
# ADD THIS BLOCK TO ACTUALLY RUN THE CODE AND SEE THE OUTPUT
# ===================================================================
if __name__ == '__main__':
    # 1. Create an instance of the data generator
    generator = TrajectoryDataGenerator()
    
    # 2. Call the generate method to get the data
    trajectory_data = generator.generate()
    
    # 3. Print the first few rows of the generated data to the console
    print("\n--- Generated Data (First 5 Rows) ---")
    print(trajectory_data.head())
    
    # 4. (Optional) Save the data to a CSV file
    output_filename = "trajectory_data.csv"
    trajectory_data.to_csv(output_filename, index=False)
    print(f"\nSuccessfully saved data to {output_filename}")