import pandas as pd
import numpy as np
from scipy.stats import norm, t
from sklearn.decomposition import PCA
import threading
from scipy.stats import t


#Problem 2
prices = pd.read_csv('/Users/ellieieie_/Desktop/problem1.csv')

returns = prices['x'] - prices['x'].mean()


# 1. VaR using Normal distribution with exponentially weighted variance (lambda=0.97)
lambda_ = 0.97

def ew_variance(returns, lambda_):
    ewma = np.zeros_like(returns)
    ewma[0] = returns[0] ** 2
    for t in range(1, len(returns)):
        ewma[t] = lambda_ * ewma[t - 1] + (1 - lambda_) * returns[t] ** 2
    return ewma

s2 = ew_variance(returns.values, lambda_)
d2 = norm(0, np.sqrt(s2[-1])) 
VaR2 = -d2.ppf(0.03) 
ES2 = -d2.expect(lambda x: x, lb=-np.inf, ub=-VaR2) 

# 2. VaR using MLE fitted T distribution
def fit_t_distribution(data):
    params = t.fit(data)
    return params
mle_params = fit_t_distribution(returns)
m, s, nu = mle_params

d3 = t(df=nu, loc=m, scale=s)
VaR3 = -d3.ppf(0.05)

ES3 = -d3.expect(lambda x: x, lb=-np.inf, ub=-VaR3) 


# 3. VaR using Historical Simulation
VaR4 = -np.percentile(returns, 5)  
ES4 = -np.mean(returns[returns <= -VaR4])

print(f"VaR using Normal distribution (EWMA variance): {VaR2}")
print(f"Expected Shortfall using Normal distribution: {ES2}")

print(f"\nVaR using MLE-fitted T-distribution: {VaR3}")
print(f"Expected Shortfall using MLE-fitted T-distribution: {ES3}")

print(f"\nVaR using Historical Simulation: {VaR4}")
print(f"Expected Shortfall using Historical Simulation: {ES4}")

#Problem 3
portfolio = pd.read_csv('/Users/ellieieie_/Desktop/portfolio.csv')
returns = pd.read_csv('/Users/ellieieie_/Desktop/DailyPrices.csv') 
# Remove stocks from portfolio with no data
portfolio = portfolio[portfolio['Stock'].isin(returns.columns)]

# Function to compute VaR and ES for Generalized T-distribution
def compute_var_es_t(returns, alpha=0.05):
    params = t.fit(returns)
    df, loc, scale = params
    VaR = t.ppf(alpha, df, loc=loc, scale=scale)
    ES = t.expect(lambda x: x, args=(df,), loc=loc, scale=scale, lb=-np.inf, ub=VaR) / alpha
    return VaR, ES

# Function to compute VaR and ES for Normal distribution
def compute_var_es_normal(returns, alpha=0.05):
    mu, sigma = norm.fit(returns)
    VaR = norm.ppf(alpha, loc=mu, scale=sigma)
    ES = mu - sigma * norm.pdf(norm.ppf(alpha)) / alpha
    return VaR, ES

current_prices = returns.iloc[-1]
# Filter portfolios A, B, and C
portfolio_A = portfolio[portfolio['Portfolio'] == 'A']
portfolio_B = portfolio[portfolio['Portfolio'] == 'B']
portfolio_C = portfolio[portfolio['Portfolio'] == 'C']
# Calculate VaR and ES for Portfolios A and B (Generalized T-distribution)
VaR_A, ES_A = {}, {}
for stock in portfolio_A['Stock']:
    stock_returns = returns[stock].dropna()
    VaR_A[stock], ES_A[stock] = compute_var_es_t(stock_returns)

VaR_B, ES_B = {}, {}
for stock in portfolio_B['Stock']:
    stock_returns = returns[stock].dropna()
    VaR_B[stock], ES_B[stock] = compute_var_es_t(stock_returns)

# Calculate VaR and ES for Portfolio C (Normal distribution)
VaR_C, ES_C = {}, {}
for stock in portfolio_C['Stock']:
    stock_returns = returns[stock].dropna()
    VaR_C[stock], ES_C[stock] = compute_var_es_normal(stock_returns)

# Express VaR and ES as $ by multiplying by the stock's holdings (current price * number of shares)
portfolio_A['current_value'] = portfolio_A['Stock'].map(current_prices) * portfolio_A['Holding']
portfolio_B['current_value'] = portfolio_B['Stock'].map(current_prices) * portfolio_B['Holding']
portfolio_C['current_value'] = portfolio_C['Stock'].map(current_prices) * portfolio_C['Holding']

# Convert VaR and ES to dollar terms
portfolio_A['VaR_$'] = portfolio_A['Stock'].map(VaR_A) * portfolio_A['current_value']
portfolio_A['ES_$'] = portfolio_A['Stock'].map(ES_A) * portfolio_A['current_value']

portfolio_B['VaR_$'] = portfolio_B['Stock'].map(VaR_B) * portfolio_B['current_value']
portfolio_B['ES_$'] = portfolio_B['Stock'].map(ES_B) * portfolio_B['current_value']

portfolio_C['VaR_$'] = portfolio_C['Stock'].map(VaR_C) * portfolio_C['current_value']
portfolio_C['ES_$'] = portfolio_C['Stock'].map(ES_C) * portfolio_C['current_value']

# Calculate portfolio-level VaR and ES by summing the dollar values
portfolio_VaR = {
    'A': portfolio_A['VaR_$'].sum(),
    'B': portfolio_B['VaR_$'].sum(),
    'C': portfolio_C['VaR_$'].sum(),
}

portfolio_ES = {
    'A': portfolio_A['ES_$'].sum(),
    'B': portfolio_B['ES_$'].sum(),
    'C': portfolio_C['ES_$'].sum(),
}

# Calculate total VaR and ES by summing across all portfolios
total_VaR = portfolio_VaR['A'] + portfolio_VaR['B'] + portfolio_VaR['C']
total_ES = portfolio_ES['A'] + portfolio_ES['B'] + portfolio_ES['C']

print("\nPortfolio VaR (A) : $", portfolio_VaR['A'])
print("Portfolio VaR (B) : $", portfolio_VaR['B'])
print("Portfolio VaR (C) : $", portfolio_VaR['C'])
print("Total VaR in $:", total_VaR)

print("Portfolio ES (A) : $", portfolio_ES['A'])
print("Portfolio ES (B) : $", portfolio_ES['B'])
print("Portfolio ES (C) : $", portfolio_ES['C'])
print("Total ES : $", total_ES)