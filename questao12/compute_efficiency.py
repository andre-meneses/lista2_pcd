import pandas as pd

# Load the CSV file
file_path = 'results.csv'
df = pd.read_csv(file_path)

# Calculate the average elapsed time for each combination of processors and size
average_time_df = df.groupby(['Processors', 'Size'])['Elapsed Time'].mean().reset_index()

# Extract the serial time (when Processors == 1)
serial_time = average_time_df[average_time_df['Processors'] == 1].set_index('Size')['Elapsed Time']

# Calculate efficiency
def compute_efficiency(row):
    if row['Processors'] == 1:
        return 1  # Efficiency is 1 for serial execution
    serial_time_for_size = serial_time.loc[row['Size']]
    efficiency = serial_time_for_size / (row['Processors'] * row['Elapsed Time'])
    return efficiency

# Apply the compute_efficiency function to each row
average_time_df['Efficiency'] = average_time_df.apply(compute_efficiency, axis=1)

# Pivot the DataFrame
pivot_df = average_time_df.pivot(index='Size', columns='Processors', values='Efficiency')

# Save the results to a new CSV file
output_file_path = 'results_pivot_with_efficiency_avg.csv'
pivot_df.to_csv(output_file_path)

print(f"Efficiency calculations saved to {output_file_path}")

