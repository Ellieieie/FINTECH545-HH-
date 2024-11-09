import pandas as pd
import numpy as np
from scipy.stats import norm, t
from sklearn.decomposition import PCA
import threading
from scipy.stats import t
from joblib import Parallel, delayed
from scipy.stats import spearmanr

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

returns = returns.set_index("Date").pct_change().dropna().reset_index()

rnames = returns.columns

currentPrice = prices.iloc[-1]
stocks = portfolio['Stock']

# Define stocks by portfolio
tStocks = portfolio[portfolio['Portfolio'].isin(["A", "B"])]['Stock']
nStocks = portfolio[portfolio['Portfolio'] == "C"]['Stock']

# Remove mean from returns
returns = returns.apply(lambda x: x - x.mean())

# Fit models (Generalized T for tStocks, Normal for nStocks)
fittedModels = {}
for s in tStocks:
    params = t.fit(returns[s].dropna())
    fittedModels[s] = {'type': 't', 'params': params}

for s in nStocks:
    mu, sigma = norm.fit(returns[s].dropna())
    fittedModels[s] = {'type': 'normal', 'params': (mu, sigma)}

# Create U matrix and calculate Spearman correlations
U = pd.DataFrame({s: t.cdf(returns[s], *fittedModels[s]['params']) if fittedModels[s]['type'] == 't' 
                  else norm.cdf(returns[s], *fittedModels[s]['params']) 
                  for s in stocks})
R = U.corr(method='spearman')

# Check if matrix R is Positive Semi-Definite
evals = eigvals(R)
if np.all(evals >= -1e-8):
    print("Matrix is PSD")
else:
    print("Matrix is not PSD")

# Simulation using PCA-based copula
NSim = 50000
pca = PCA()
simU = pd.DataFrame(norm.cdf(pca.fit_transform(np.random.normal(0, 1, (NSim, len(stocks))))) , columns=stocks)

# Evaluate simulated returns from fitted models
def evaluate_simulated_return(stock, model, simU):
    if model['type'] == 't':
        return t.ppf(simU[stock], *model['params'])
    else:
        mu, sigma = model['params']
        return norm.ppf(simU[stock], loc=mu, scale=sigma)

simulatedReturns = pd.DataFrame({s: evaluate_simulated_return(s, fittedModels[s], simU) for s in stocks})

# Portfolio Valuation and Risk Calculation
def calcPortfolioRisk(simulatedReturns, portfolio, currentPrice):
    nVals = len(portfolio) * NSim
    pnl = []
    
    for i, row in portfolio.iterrows():
        stock = row['Stock']
        holding = row['Holding']
        price = currentPrice[stock]
        
        currentValue = holding * price
        simValues = holding * price * (1.0 + simulatedReturns[stock])
        
        pnl.append(simValues - currentValue)

    values = pd.concat(pnl, axis=1).sum(axis=1)
    VaR95 = -np.percentile(values, 5)
    ES95 = -values[values <= -VaR95].mean()
    
    return VaR95, ES95

risk_A = calcPortfolioRisk(simulatedReturns[portfolio[portfolio['Portfolio'] == 'A']['Stock']], portfolio[portfolio['Portfolio'] == 'A'], currentPrice)
risk_B = calcPortfolioRisk(simulatedReturns[portfolio[portfolio['Portfolio'] == 'B']['Stock']], portfolio[portfolio['Portfolio'] == 'B'], currentPrice)
risk_C = calcPortfolioRisk(simulatedReturns[portfolio[portfolio['Portfolio'] == 'C']['Stock']], portfolio[portfolio['Portfolio'] == 'C'], currentPrice)

print(f"Portfolio A VaR95: ${risk_A[0]:.2f}, ES95: ${risk_A[1]:.2f}")
print(f"Portfolio B VaR95: ${risk_B[0]:.2f}, ES95: ${risk_B[1]:.2f}")
print(f"Portfolio C VaR95: ${risk_C[0]:.2f}, ES95: ${risk_C[1]:.2f}")

# Full Portfolio Metrics with Different Covariances
def ewCovar(returns, lambda_=0.97):
    weights = np.array([(1 - lambda_) * (lambda_ ** i) for i in range(len(returns))])
    weights = weights[::-1] / weights.sum()  # Reverse weights to match order
    return np.cov(returns, aweights=weights, rowvar=False)

covar = ewCovar(returns, 0.97)
simulatedReturns = pd.DataFrame(pca.fit_transform(np.random.multivariate_normal(np.zeros(len(stocks)), covar, NSim)), columns=stocks)
risk_n  = calcPortfolioRisk(simulatedReturns, portfolio, currentPrice)

# Rename VaR and ES columns
risk_df = pd.DataFrame({
    'Portfolio': ['A', 'B', 'C', 'Total'],
    'VaR95': [risk_A[0], risk_B[0], risk_C[0], risk_A[0] + risk_B[0] + risk_C[0]],
    'ES95': [risk_A[1], risk_B[1], risk_C[1], risk_A[1] + risk_B[1] + risk_C[1]],
    'Normal_VaR': [risk_n[0]] * 3 + [sum(risk_n[:3])],
    'Normal_ES': [risk_n[1]] * 3 + [sum(risk_n[1:3])]
})
print(risk_df)