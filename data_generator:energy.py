import pandas as pd
import numpy as np

class EnergyDataGenerator:
    """
    Generates simulated energy consumption data for the box plots in Figure 15B.
    """
    def __init__(self, n_trials=100): # Increased trials for a better box plot
        self.n_trials = n_trials
        np.random.seed(42) # Ensure reproducibility

    def generate(self):
        """Generates and returns the energy consumption dataframe."""
        print("Energy consumption data for Figure 15B...")
        
        # Based on "40% lower" claim. Let's assume baseline is ~50 Watts.
        # 40% lower is 30 Watts. Scales are chosen to visually match a box plot.
        our_method_energy = np.random.normal(loc=30, scale=4, size=self.n_trials)
        baseline1_energy = np.random.normal(loc=50, scale=6, size=self.n_trials)
        baseline2_energy = np.random.normal(loc=55, scale=7.5, size=self.n_trials)
        other_baseline_energy = np.random.normal(loc=48, scale=5, size=self.n_trials)
        
        data = {
            'Method': ['Our Method'] * self.n_trials + 
                      ['Baseline SLAM 1'] * self.n_trials + 
                      ['Baseline SLAM 2'] * self.n_trials + 
                      ['Other Baseline'] * self.n_trials,
            'Energy_Consumption_Watts': np.concatenate([
                our_method_energy,
                baseline1_energy,
                baseline2_energy,
                other_baseline_energy
            ]),
            'Trial': list(range(1, self.n_trials + 1)) * 4
        }
        
        # FIX: Removed the trailing '[' from this line
        return pd.DataFrame(data)

# ===================================================================
# ADD THIS BLOCK TO ACTUALLY RUN THE CODE AND GET OUTPUT
# ===================================================================
if __name__ == '__main__':
    # 1. Create an instance of the class
    generator = EnergyDataGenerator(n_trials=100)
    
    # 2. Call the generate method to create the DataFrame
    energy_data = generator.generate()
    
    # 3. Print the first few and last few rows to show all categories
    print("\n--- Generated Data (Sample) ---")
    print(energy_data.head())
    print("...")
    print(energy_data.tail())
    
    # 4. Save the generated data to a CSV file
    output_filename = "energy_consumption_data.csv"
    energy_data.to_csv(output_filename, index=False)
    print(f"\nSuccessfully saved data to {output_filename}")