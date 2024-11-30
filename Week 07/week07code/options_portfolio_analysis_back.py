import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime

# Problem 1: Implement GBSM Greeks and Binomial Tree for American Options
def bs_greeks(S, X, T, r, b, sigma, option_type='call'):
    """
    Calculate Black-Scholes Greeks for both call and put options.
    """
    d1 = (np.log(S / X) + (b + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * norm.pdf(d1)
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * X * np.exp(-r * T) * norm.cdf(d2)) if option_type == 'call' else (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * X * np.exp(-r * T) * norm.cdf(-d2))
    rho = X * T * np.exp(-r * T) * norm.cdf(d2) if option_type == 'call' else -X * T * np.exp(-r * T) * norm.cdf(-d2)

    return delta, gamma, vega, theta, rho

# Finite difference for Greeks
def finite_difference_greeks(S, X, T, r, b, sigma, option_type='call', epsilon=0.01):
    """
    Calculate Greeks using finite difference approximation.
    """
    delta_plus = bs_price(S + epsilon, X, T, r, b, sigma, option_type)
    delta_minus = bs_price(S - epsilon, X, T, r, b, sigma, option_type)
    delta = (delta_plus - delta_minus) / (2 * epsilon)
    
    return delta

# Binomial Tree for American Options
def binomial_tree_american(S, X, T, r, sigma, N, option_type='call', dividend=0):
    """
    Binomial Tree valuation for American Options (both call and put).
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity
    ST = np.array([S * (u ** j) * (d ** (N - j)) for j in range(N + 1)])
    
    # Initialize option values at maturity
    if option_type == 'call':
        values = np.maximum(0, ST - X)
    else:
        values = np.maximum(0, X - ST)
    
    # Step back through the tree
    for i in range(N - 1, -1, -1):
        ST = ST[:-1] / u
        values = np.maximum(values[:-1] * p + values[1:] * (1 - p), (ST - X) if option_type == 'call' else (X - ST))
        # Adjust for dividends if applicable
        if dividend > 0 and i == int(N * 0.25):
            ST -= dividend
    
    return values[0]

# Problem 2: Simulate AAPL returns and calculate VaR/ES
def simulate_aapl_var_es(current_price, days, sigma, num_simulations=10000):
    """
    Simulate AAPL returns and calculate VaR and ES.
    """
    simulated_prices = current_price * np.exp(sigma * np.sqrt(days / 252) * np.random.randn(num_simulations))
    losses = current_price - simulated_prices
    
    var_95 = np.percentile(losses, 95)
    es_95 = losses[losses >= var_95].mean()
    
    return var_95, es_95

# Problem 3: Fama-French 4 Factor Model
def fama_french_four_factor(stock_returns, factors):
    """
    Fit a Fama-French 4 Factor Model to stock returns.
    """
    import statsmodels.api as sm
    X = sm.add_constant(factors)
    model = sm.OLS(stock_returns, X).fit()
    return model

# Example Usage
if __name__ == "__main__":
    # Problem 1: Greeks Calculation Example
    S = 151.03  # Current Stock Price
    X = 165  # Strike Price
    T = (datetime(2022, 4, 15) - datetime(2022, 3, 13)).days / 365  # Time to Maturity in Years
    r = 0.0425  # Risk Free Rate
    b = 0.0053  # Continuously Compounding Coupon
    sigma = 0.3  # Volatility
    
    delta, gamma, vega, theta, rho = bs_greeks(S, X, T, r, b, sigma, 'call')
    print(f"Delta: {delta}, Gamma: {gamma}, Vega: {vega}, Theta: {theta}, Rho: {rho}")
    print("Problem 1 Successfully")

    # Problem 2: Simulate AAPL returns
    current_price = 165
    sigma = 0.2
    var_95, es_95 = simulate_aapl_var_es(current_price, 10, sigma)
    print(f"VaR 95%: {var_95}, ES 95%: {es_95}")
    print("Problem 2 Successfully")

    # Problem 3: Fit Fama-French Model
    stock_returns = pd.read_csv("../../../../../Downloads/H6530/code/DailyPrices.csv")
    factors = pd.read_csv("../../../../../Downloads/H6530/code/F-F_Research_Data_Factors_daily.CSV")
    momentum = pd.read_csv("../../../../../Downloads/H6530/code/F-F_Momentum_Factor_daily.CSV")
    factors.columns = [col.strip() for col in factors.columns]  # Strip any whitespace from column names
    momentum.columns = [col.strip() for col in momentum.columns]
    
    # Convert 'Date' columns to datetime, ignoring any parsing errors
    factors['Date'] = pd.to_datetime(factors['Date'], format='%Y%m%d', errors='coerce')
    momentum['Date'] = pd.to_datetime(momentum['Date'], format='%Y%m%d', errors='coerce')
    
    # Drop rows with invalid dates
    factors = factors.dropna(subset=['Date'])
    momentum = momentum.dropna(subset=['Date'])
    
    # Merge factors and momentum data
    factors = pd.merge(factors, momentum[['Date', 'Mom']], on='Date', how='inner')
    
    # Align stock returns and factors based on dates
    stock_returns['Date'] = pd.to_datetime(stock_returns['Date'], errors='coerce')
    stock_returns = stock_returns.dropna(subset=['Date'])
    merged_data = pd.merge(stock_returns, factors, on='Date', how='inner')
    
    # Extract aligned stock returns and factor data
    aligned_stock_returns = merged_data.drop(columns=['Date', 'Mkt-RF', 'SMB', 'HML', 'Mom']).values
    aligned_factors = merged_data[['Mkt-RF', 'SMB', 'HML', 'Mom']].values
    
    # Standardize the factor data for better model performance
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    aligned_factors = scaler.fit_transform(aligned_factors)
    
    with open('../../../../../Downloads/H6530/code/Fama_French_Results.txt', 'w') as f:
        for i, stock in enumerate(merged_data.columns.difference(['Date', 'Mkt-RF', 'SMB', 'HML', 'Mom'])):
            f.write(f"Model for {stock}\n")
            stock_returns = merged_data[stock].values
            model = fama_french_four_factor(stock_returns, aligned_factors)
            f.write(model.summary().as_text() + '\n')
    print("Problem 3 Successfully, read  Fama_French_Results.txt")