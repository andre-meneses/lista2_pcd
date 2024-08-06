import pandas as pd

# Load the CSV file
file_path = 'results.csv'
df = pd.read_csv(file_path)

# Extract the serial time (when Processors == 1)
serial_time = df[df['Processors'] == 1].set_index(['Rows', 'Columns'])['Elapsed Time']

# Calculate efficiency
def compute_efficiency(row):
    if row['Processors'] == 1:
        return 1  # Efficiency is 1 for serial execution
    serial_time_for_size = serial_time.loc[(row['Rows'], row['Columns'])]
    efficiency = serial_time_for_size / (row['Processors'] * row['Elapsed Time'])
    return efficiency

# Apply the compute_efficiency function to each row
df['Efficiency'] = df.apply(compute_efficiency, axis=1)

# Pivot the DataFrame
df['Size'] = df['Rows'].astype(str) + ' x ' + df['Columns'].astype(str)
pivot_df = df.pivot(index='Size', columns='Processors', values='Efficiency')

# Save the results to a new CSV file
output_file_path = 'results_with_efficiency.csv'
pivot_df.to_csv(output_file_path)

print(f"Efficiency calculations saved to {output_file_path}")

