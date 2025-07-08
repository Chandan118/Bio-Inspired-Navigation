import pandas as pd

def create_performance_summary():
    """
    Creates a dataframe summarizing the performance metrics from Table S3.
    This is a direct transcription of the table into a clean, machine-readable format.
    """
    print("Performance metrics summary for Table S3...")
    
    data = {
        'Metric': [
            'Localization Accuracy',
            'Energy Consumption',
            'Obstacle Avoidance',
            'Visual Occlusion Recovery'
        ],
        'Unit': ['m', 'J/m', '% success', 's'],
        'Proposed_System_Value': [0.25, 0.8, 90.0, 1.2],
        'Baseline_ORB_SLAM3_Value': [0.42, 1.5, 65.0, 3.0],
        'Improvement': ['40% ↑', '47% ↓', '38% ↑', '60% faster']
    }
    
    # FIX: Removed the extra text from the end of this line
    return pd.DataFrame(data)

# ===================================================================
# ADD THIS BLOCK TO ACTUALLY RUN THE CODE AND SEE THE OUTPUT
# ===================================================================
if __name__ == '__main__':
    # 1. Call the function to create the DataFrame
    summary_table = create_performance_summary()
    
    # 2. Print the entire resulting DataFrame to the console
    print("\n--- Summary Table ---")
    print(summary_table.to_string()) # .to_string() prints it nicely without truncation
    
    # 3. (Optional) Save the data to a CSV file
    output_filename = "performance_summary.csv"
    summary_table.to_csv(output_filename, index=False)
    print(f"\nSuccessfully saved data to {output_filename}")