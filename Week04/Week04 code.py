import pandas as pd
import numpy as np
from scipy import stats
import scipy.stats as stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg

#Question 1

file_path = '/Users/ellieieie_/Desktop/DailyPrices.csv'
prices = pd.read_csv(file_path)

sigma = 0.1
P_t_minus_1 = prices.iloc[0, 1]  
n_simulations = 100000

r_t = np.random.normal(0, sigma, n_simulations)

P_t_arithmetic = P_t_minus_1 * (1 + r_t) 
P_t_log = P_t_minus_1 * np.exp(r_t)       
P_t_brownian = P_t_minus_1 + sigma * r_t 
mean_arithmetic = np.mean(P_t_arithmetic)
std_arithmetic = np.std(P_t_arithmetic)

mean_log = np.mean(P_t_log)
std_log = np.std(P_t_log)

mean_brownian = np.mean(P_t_brownian)
std_brownian = np.std(P_t_brownian)

expected_mean_arithmetic = P_t_minus_1
expected_std_arithmetic = P_t_minus_1 * sigma

expected_mean_log = P_t_minus_1 * np.exp(0.5 * sigma**2)
expected_std_log = P_t_minus_1 * np.sqrt(np.exp(sigma**2) - 1)

expected_mean_brownian = P_t_minus_1
expected_std_brownian = sigma

print("Empirical Mean and Standard Deviation of Simulated Prices:")
print(f"1. Arithmetic Return: Mean = {mean_arithmetic:.2f}, Std Dev = {std_arithmetic:.2f}")
print(f"2. Log Return: Mean = {mean_log:.2f}, Std Dev = {std_log:.2f}")
print(f"3. Classical Brownian Motion: Mean = {mean_brownian:.2f}, Std Dev = {std_brownian:.2f}")

print("\nTheoretical Expected Value and Standard Deviation:")
print(f"1. Arithmetic Return: Mean = {expected_mean_arithmetic:.2f}, Std Dev = {expected_std_arithmetic:.2f}")
print(f"2. Log Return: Mean = {expected_mean_log:.2f}, Std Dev = {expected_std_log:.2f}")
print(f"3. Classical Brownian Motion: Mean = {expected_mean_brownian:.2f}, Std Dev = {expected_std_brownian:.2f}")

#Problem 2

prices = pd.read_csv('/Users/ellieieie_/Desktop/DailyPrices.csv')
prices["Date"] = pd.to_datetime(prices["Date"])
prices.set_index("Date", inplace=True)
prices = prices.asfreq('B')  
prices.ffill(inplace=True)


def return_calculate(prices, method="arithmetic"):
    if method == "arithmetic":
        returns = prices.pct_change().dropna()
    elif method == "log":
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("Unsupported method. Choose 'arithmetic' or 'log'.")
    return returns

returns = return_calculate(prices, method="arithmetic")

holdings = {"GOOGL": 10, "NVDA": 75, "TSLA": 25, "META": 1}

nm = set(prices.columns).intersection(holdings.keys())
current_prices = prices.iloc[-1][list(nm)]
returns = returns[list(nm)]

returns["META"] = returns["META"] - returns["META"].mean()

portfolio_value = sum(current_prices[stock] * quantity for stock, quantity in holdings.items())
portfolio_returns = returns.dot(pd.Series(holdings))

mean_arithmetic = portfolio_returns.mean()
std_arithmetic = portfolio_returns.std()
print(f"\n2a. Arithmetic Return: Mean = {mean_arithmetic:.2f}, Std Dev = {std_arithmetic:.2f}")

confidence_level = 0.05
# 1. VaR using a normal distribution
var_normal = stats.norm.ppf(1 - confidence_level, mean_arithmetic, std_arithmetic)
# 2. VaR using a normal distribution with an exponentially weighted variance
lambda_ = 0.94
weights = np.array([(1 - lambda_) * lambda_ ** i for i in range(len(portfolio_returns))])[::-1]
weighted_mean = np.average(portfolio_returns, weights=weights)
weighted_var = np.average((portfolio_returns - weighted_mean) ** 2, weights=weights)
var_ewma = stats.norm.ppf(1 - confidence_level, weighted_mean, np.sqrt(weighted_var))
# 3. VaR using a MLE fitted T-distribution
params = stats.t.fit(portfolio_returns)
var_t = stats.t.ppf(1 - confidence_level, *params)
# 4. VaR using a fitted AR(1) model
model = ARIMA(portfolio_returns.values, order=(1, 0, 0)).fit()
forecast = model.forecast(steps=1)
forecast_value = forecast[0]
var_ar1 = forecast_value + stats.norm.ppf(1 - confidence_level) * portfolio_returns.std()
# 5. VaR using Historical Simulation
var_historical = portfolio_returns.quantile(confidence_level)
# Compare the 5 VaR values
print("\nVaR Comparison at 95% Confidence Level:")
print(f"1. Normal Distribution VaR: {var_normal:.2f}")
print(f"2. EWMA Normal Distribution VaR: {var_ewma:.2f}")
print(f"3. MLE fitted T-distribution VaR: {var_t:.2f}")
print(f"4. AR(1) Model VaR: {var_ar1:.2f}")
print(f"5. Historical Simulation VaR: {var_historical:.2f}")

#Questioin 3

prices = pd.read_csv("/Users/ellieieie_/Desktop/DailyPrices.csv")
portfolio = pd.read_csv("/Users/ellieieie_/Desktop/portfolio.csv")

prices["Date"] = pd.to_datetime(prices["Date"])
prices.set_index("Date", inplace=True)

returns = prices.pct_change().dropna()

portfolio_a = portfolio[portfolio["Portfolio"] == "A"].set_index("Stock")["Holding"]
portfolio_b = portfolio[portfolio["Portfolio"] == "B"].set_index("Stock")["Holding"]
portfolio_c = portfolio[portfolio["Portfolio"] == "C"].set_index("Stock")["Holding"]

returns_a = returns[portfolio_a.index.intersection(returns.columns)]
returns_b = returns[portfolio_b.index.intersection(returns.columns)]
returns_c = returns[portfolio_c.index.intersection(returns.columns)]

# Calculate portfolio values in dollars
current_prices = prices.iloc[-1]

# Only consider common stocks between current prices and the portfolio holdings
common_stocks_a = portfolio_a.index.intersection(current_prices.index)
common_stocks_b = portfolio_b.index.intersection(current_prices.index)
common_stocks_c = portfolio_c.index.intersection(current_prices.index)

portfolio_value_a = (current_prices[common_stocks_a] * portfolio_a[common_stocks_a]).sum()
portfolio_value_b = (current_prices[common_stocks_b] * portfolio_b[common_stocks_b]).sum()
portfolio_value_c = (current_prices[common_stocks_c] * portfolio_c[common_stocks_c]).sum()


def calculate_ew_cov(returns, lambda_):
    return returns.ewm(span=(2 / (1 - lambda_)) - 1).cov(pairwise=True).iloc[-len(returns.columns):, -len(returns.columns):]

lambda_ = 0.97
ew_cov_a = calculate_ew_cov(returns_a, lambda_)
ew_cov_b = calculate_ew_cov(returns_b, lambda_)
ew_cov_c = calculate_ew_cov(returns_c, lambda_)


def calculate_portfolio_var(weights, cov_matrix, portfolio_value, confidence_level=0.03):
    common_index = weights.index.intersection(cov_matrix.index.get_level_values(1).unique())
    if len(common_index) == 0:
        raise ValueError("No matching stocks between weights and covariance matrix.")

    weights = weights.loc[common_index]
    relevant_cov_matrix = cov_matrix.loc[(slice(None), common_index), common_index].droplevel(0)

    portfolio_variance = weights.T @ relevant_cov_matrix.values @ weights
    portfolio_std_dev = np.sqrt(portfolio_variance)
    var_97 = stats.norm.ppf(1 - confidence_level) * portfolio_std_dev
    var_97_dollar = var_97 * portfolio_value
    return var_97_dollar

weights_a = portfolio_a / portfolio_a.sum()
weights_b = portfolio_b / portfolio_b.sum()
weights_c = portfolio_c / portfolio_c.sum()

try:
    var_a_97 = calculate_portfolio_var(weights_a, ew_cov_a, portfolio_value_a)
    var_b_97 = calculate_portfolio_var(weights_b, ew_cov_b, portfolio_value_b)
    var_c_97 = calculate_portfolio_var(weights_c, ew_cov_c, portfolio_value_c)
except ValueError as e:
    print(f"Error calculating VaR: {e}")


def calculate_historical_var(returns, weights, portfolio_value, confidence_level=0.03):
    common_stocks = returns.columns.intersection(weights.index)
    aligned_returns = returns[common_stocks]
    aligned_weights = weights[common_stocks]
    portfolio_returns = aligned_returns.dot(aligned_weights)
    var_hist = -np.percentile(portfolio_returns, 100 * confidence_level) * portfolio_value
    return var_hist

var_a_hist = calculate_historical_var(returns_a, weights_a, portfolio_value_a)
var_b_hist = calculate_historical_var(returns_b, weights_b, portfolio_value_b)
var_c_hist = calculate_historical_var(returns_c, weights_c, portfolio_value_c)

total_holdings = portfolio.groupby("Stock")["Holding"].sum()
common_stocks = total_holdings.index.intersection(returns.columns).intersection(current_prices.index)
weights_total = total_holdings[common_stocks] / total_holdings[common_stocks].sum()
aligned_returns_total = returns[common_stocks]
portfolio_value_total = (current_prices[common_stocks] * total_holdings[common_stocks]).sum()
total_portfolio_returns = aligned_returns_total.dot(weights_total)
var_total_hist = -np.percentile(total_portfolio_returns, 100 * 0.03) * portfolio_value_total

print("VaR of Portfolios at 97% Confidence Level (in $):")
if 'var_a_97' in locals():
    print(f"1. Portfolio A Exponentially Weighted VaR: ${var_a_97:.2f}")
if 'var_b_97' in locals():
    print(f"2. Portfolio B Exponentially Weighted VaR: ${var_b_97:.2f}")
if 'var_c_97' in locals():
    print(f"3. Portfolio C Exponentially Weighted VaR: ${var_c_97:.2f}")

print("\nHistorical VaR of Portfolios at 97% Confidence Level (in $):")
print(f"1. Portfolio A Historical VaR: ${var_a_hist:.2f}")
print(f"2. Portfolio B Historical VaR: ${var_b_hist:.2f}")
print(f"3. Portfolio C Historical VaR: ${var_c_hist:.2f}")
print(f"Total Historical VaR of All Portfolios Combined at 97% Confidence Level (in $): ${var_total_hist:.2f}")
