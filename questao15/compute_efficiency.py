import pandas as pd

# Load the CSV file
file_path = 'timings_2.csv'
df = pd.read_csv(file_path)

# Calculate the average elapsed time for each combination of cores and n
average_time_df = df.groupby(['cores', 'n'])['time'].mean().reset_index()

# Extract the serial time (when cores == 1)
serial_time = average_time_df[average_time_df['cores'] == 1].set_index('n')['time']

# Calculate efficiency
def compute_efficiency(row):
    if row['cores'] == 1:
        return 1  # Efficiency is 1 for serial execution
    serial_time_for_size = serial_time.loc[row['n']]
    efficiency = serial_time_for_size / (row['cores'] * row['time'])
    return efficiency

# Apply the compute_efficiency function to each row
average_time_df['Efficiency'] = average_time_df.apply(compute_efficiency, axis=1)

# Pivot the DataFrame
pivot_df = average_time_df.pivot(index='n', columns='cores', values='Efficiency')

# Save the results to a new CSV file
output_file_path = 'efficiency_timings_2.csv'
pivot_df.to_csv(output_file_path)

print(f"Efficiency calculations saved to {output_file_path}")

