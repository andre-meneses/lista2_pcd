import pandas as pd

# Load the CSV file
file_path = 'timings_2.csv'
df = pd.read_csv(file_path)

# Calculate the average elapsed time for each combination of cores and n
average_time_df = df.groupby(['cores', 'n'])['time'].mean().reset_index()

# Extract the serial time (when cores == 1)
serial_time = average_time_df[average_time_df['cores'] == 1].set_index('n')['time']

# Calculate speedup
def compute_speedup(row):
    if row['cores'] == 1:
        return 1  # Speedup is 1 for serial execution
    serial_time_for_size = serial_time.loc[row['n']]
    speedup = serial_time_for_size / row['time']
    return speedup

# Apply the compute_speedup function to each row
average_time_df['Speedup'] = average_time_df.apply(compute_speedup, axis=1)

# Pivot the DataFrame
pivot_df = average_time_df.pivot(index='n', columns='cores', values='Speedup')

# Save the results to a new CSV file
output_file_path = 'speedup_timings_2.csv'
pivot_df.to_csv(output_file_path)

print(f"Speedup calculations saved to {output_file_path}")

