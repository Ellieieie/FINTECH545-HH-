import pandas as pd
import numpy as np

data = pd.read_csv('/Users/ellieieie_/Desktop/DailyReturn.csv')

# Function to calculate the exponentially weighted covariance matrix
def exponentially_weighted_covariance(data, lambda_):
    returns = data.values
    n = returns.shape[0]  # Number of observations
    cov_matrix = np.zeros((returns.shape[1], returns.shape[1]))

    # Initialize the covariance matrix with the first observation
    cov_matrix = np.cov(returns, rowvar=False)

    # Calculate the exponentially weighted covariance matrix
    for t in range(1, n):
        cov_matrix = lambda_ * cov_matrix + (1 - lambda_) * np.outer(returns[t] - returns[t-1], returns[t] - returns[t-1])

    return cov_matrix

# Example usage
lambda_ = 0.9
cov_matrix = exponentially_weighted_covariance(data, lambda_)
print("Exponentially Weighted Covariance Matrix:\n", cov_matrix)


import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas.core.window.ewm import EWM

# Example using pandas ewm
def weighted_covariance(data, lambda_):
    ewm = data.ewm(span=(2/(1 - lambda_) - 1)).mean()
    return ewm.cov()

# Calculate using pandas
data = pd.read_csv('DailyReturn.csv')
cov_matrix_package = weighted_covariance(data, lambda_)
print("Package-based Covariance Matrix:\n", cov_matrix_package)
