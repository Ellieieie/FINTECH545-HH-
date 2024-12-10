import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load the data
file_path = 'data.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Calculate the expected returns and covariance matrix
expected_returns = data.mean().values
cov_matrix = data.cov().values
risk_free_rate = 0.0475  # 4.75% risk-free rate

# Function to calculate portfolio return
def portfolio_return(weights, returns):
    return np.dot(weights, returns)

# Function to calculate portfolio standard deviation
def portfolio_std(weights, covariance_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

# Function to calculate negative Sharpe Ratio (for minimization)
def neg_sharpe_ratio(weights, returns, covariance_matrix, rf_rate):
    port_return = portfolio_return(weights, returns)
    port_std = portfolio_std(weights, covariance_matrix)
    return -(port_return - rf_rate) / port_std

# Function to calculate Expected Shortfall (ES) for a given portfolio
def expected_shortfall(weights, returns, alpha=0.025):
    portfolio_returns = np.dot(data.values, weights)
    var = np.percentile(portfolio_returns, 100 * alpha)
    es = portfolio_returns[portfolio_returns < var].mean()
    return abs(es)

# Function to calculate the new risk-adjusted return metric RR_p
def neg_risk_adjusted_return(weights, returns, covariance_matrix, rf_rate, alpha=0.025):
    port_return = portfolio_return(weights, returns)
    es = expected_shortfall(weights, returns, alpha)
    return -(port_return - rf_rate) / es

# Constraints: weights sum to 1
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

# Bounds: no short can be less than -100%
bounds = [(-1, 1) for _ in range(len(expected_returns))]

# Initial guess for weights
initial_weights = np.ones(len(expected_returns)) / len(expected_returns)

# Optimize for maximum Sharpe Ratio
result_sharpe = minimize(neg_sharpe_ratio, initial_weights, args=(expected_returns, cov_matrix, risk_free_rate),
                         method='SLSQP', bounds=bounds, constraints=constraints)

# Extract optimal weights and maximum Sharpe Ratio
optimal_weights_sharpe = result_sharpe.x
max_sharpe_ratio = -result_sharpe.fun

# Optimize for maximum RR_p
result_rrp = minimize(neg_risk_adjusted_return, initial_weights, args=(expected_returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)

# Extract optimal weights and maximum RR_p
optimal_weights_rrp = result_rrp.x
max_rrp = -result_rrp.fun

# Combine weights for visualization
weights_df = pd.DataFrame({
    "Assets": ["A1", "A2", "A3"],
    "Max Sharpe Ratio Weights": optimal_weights_sharpe,
    "Max RR_p Weights": optimal_weights_rrp
})

# Output results to console
print("=== Results Summary ===\n")

# 1. Max Sharpe Ratio Portfolio
print("1. Max Sharpe Ratio Portfolio:")
print(f"  Optimal Weights: {optimal_weights_sharpe}")
print(f"  Maximum Sharpe Ratio: {max_sharpe_ratio}\n")

# 2. Max RR_p Portfolio
print("2. Max RR_p Portfolio:")
print(f"  Optimal Weights: {optimal_weights_rrp}")
print(f"  Maximum RR_p: {max_rrp}\n")

# Plot weight distributions for comparison
plt.figure(figsize=(10, 6))
x = np.arange(len(weights_df["Assets"]))

plt.bar(x - 0.2, weights_df["Max Sharpe Ratio Weights"], width=0.4, label="Max Sharpe Ratio")
plt.bar(x + 0.2, weights_df["Max RR_p Weights"], width=0.4, label="Max RR_p")

plt.xticks(x, weights_df["Assets"])
plt.title("Comparison of Portfolio Weights")
plt.ylabel("Weights")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Create a table to compare key metrics
comparison_table = pd.DataFrame({
    "Portfolio Type": ["Max Sharpe Ratio", "Max RR_p"],
    "Metric Value": [max_sharpe_ratio, max_rrp],
    "A1 Weight": [optimal_weights_sharpe[0], optimal_weights_rrp[0]],
    "A2 Weight": [optimal_weights_sharpe[1], optimal_weights_rrp[1]],
    "A3 Weight": [optimal_weights_sharpe[2], optimal_weights_rrp[2]]
})

print("Comparison Table:")
print(comparison_table)
