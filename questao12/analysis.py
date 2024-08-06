import pandas as pd

def compute_statistics(file_path):
    # Load the data from CSV
    data = pd.read_csv(file_path)

    # Group the data by 'Processors' and 'Size' to calculate statistics for each configuration
    grouped_data = data.groupby(['Processors', 'Size'])

    # Calculate mean, median, and minimum for each group
    stats = grouped_data['Elapsed Time'].agg(['mean', 'median', 'min']).reset_index()

    # Rename columns for clarity
    stats.columns = ['Processors', 'Size', 'Average Time', 'Median Time', 'Minimum Time']

    return stats

# Usage example
if __name__ == "__main__":
    file_path = 'results.csv'  # Update with your actual CSV file path
    result_stats = compute_statistics(file_path)
    print(result_stats)
    # Optionally, save the results back to a new CSV
    result_stats.to_csv('statistics_results.csv', index=False)

