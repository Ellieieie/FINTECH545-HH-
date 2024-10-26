import numpy as np
from scipy.stats import skew, kurtosis

# Initial parameters
P0 = 100
sigma = 0.1
np.random.seed(1234)  # Set a random seed for reproducibility
simR = np.random.normal(0, sigma, 1000000)

# Brownian Motion
P1 = P0 + simR
print(f"Expect (μ,σ,skew,kurt)=({P0},{sigma},0,0)")
print(f"({np.mean(P1)},{np.std(P1)},{skew(P1)},{kurtosis(P1)})")

# Arithmetic Returns
P1 = P0 * (1 + simR)
print(f"Expect (μ,σ,skew,kurt)=({P0},{sigma * P0},0,0)")
print(f"({np.mean(P1)},{np.std(P1)},{skew(P1)},{kurtosis(P1)})")

# Geometric Brownian Motion
P1 = P0 * np.exp(simR)
expected_mean = np.exp(np.log(P0) + (sigma**2) / 2)
expected_std = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2 * np.log(P0) + sigma**2))
expected_skew = (np.exp(sigma**2) + 2) * np.sqrt(np.exp(sigma**2) - 1)
expected_kurt = np.exp(4 * sigma**2) + 2 * np.exp(3 * sigma**2) + 3 * np.exp(2 * sigma**2) - 6

print("Expect (μ,σ,skew,kurt)=(")
print(f"        {expected_mean},")
print(f"        {expected_std},")
print(f"        {expected_skew},")
print(f"        {expected_kurt})")
print(f"Simulated (μ,σ,skew,kurt)=(")
print(f"        {np.mean(P1)},")
print(f"        {np.std(P1)},")
print(f"        {skew(P1)},")
print(f"        {kurtosis(P1)})")
