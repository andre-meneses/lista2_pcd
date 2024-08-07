import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def estimate_coefficients(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Extract data for calculations
    n = data['Size']
    p = data['Processors']
    t_parallel = data['Elapsed Time']

    # Prepare the data for regression
    X = np.column_stack((n / p, np.log2(p)))
    y = t_parallel

    # Perform linear regression
    model = LinearRegression()
    model.fit(X, y)

    # Extract coefficients
    a, b = model.coef_[0], model.coef_[1]
    return a, b

# Usage example
file_path = 'results.csv'  # Replace with your actual file path
a, b = estimate_coefficients(file_path)
print("Coefficient a:", a)
print("Coefficient b:", b)

