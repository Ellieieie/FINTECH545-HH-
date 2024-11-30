import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import brentq
from options_portfolio_analysis.py.gbsm import gbsm
from library.RiskStats import aggRisk
from library.return_calculate import return_calculate
from statsmodels.tsa.arima.model import ARIMA

# Problem #1
current_price = 165
current_date = datetime.strptime("03/03/2023", "%m/%d/%Y")
rf = 0.0525
dy = 0.0053
days_year = 365

expiration_date = datetime.strptime("03/17/2023", "%m/%d/%Y")
ttm = (expiration_date - current_date).days / days_year

strike = 165
iv = np.arange(0.10, 0.81, 0.02)

# gbsm(call, underlying, strike, ttm, rf, b, ivol)
call_vals = [gbsm(True, current_price, strike, ttm, rf, rf - dy, v).value for v in iv]
put_vals = [gbsm(False, current_price, strike, ttm, rf, rf - dy, v).value for v in iv]

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(iv, call_vals, label="Call Values")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(iv, put_vals, label="Put Values", color='red')
plt.legend()

plt.show()

# Problem #2
current_date = datetime.strptime("10/30/2023", "%m/%d/%Y")
rf = 0.0525
dy = 0.0057
current_price = 170.15
options = pd.read_csv("Project/AAPL_Options.csv")

options['Expiration'] = pd.to_datetime(options['Expiration'], format="%m/%d/%Y")
n = len(options)

# Calculate TTM
options['ttm'] = (options['Expiration'] - current_date).dt.days / days_year

# Calculate Implied Volatility
def implied_vol(row):
    return brentq(lambda x: gbsm(row['Type'] == "Call", current_price, row['Strike'], row['ttm'], rf, rf - dy, x).value - row['Last Price'], 0.01, 5.0)

options['ivol'] = options.apply(implied_vol, axis=1)
options['gbsm'] = options.apply(lambda row: gbsm(row['Type'] == "Call", current_price, row['Strike'], row['ttm'], rf, rf - dy, row['ivol']).value, axis=1)

calls = options['Type'] == "Call"
puts = ~calls

plt.figure(figsize=(10, 6))
plt.plot(options.loc[calls, 'Strike'], options.loc[calls, 'ivol'], label="Call Implied Vol")
plt.plot(options.loc[puts, 'Strike'], options.loc[puts, 'ivol'], label="Put Implied Vol", color='red')
plt.axvline(x=current_price, label="Current Price", linestyle="--", color='purple')
plt.title("Implied Volatilities")
plt.legend()
plt.show()

# Problem #3
current_s = 170.15
returns = return_calculate(pd.read_csv("Project/DailyPrices.csv"), method="LOG", date_column="Date")["AAPL"]
returns = returns - returns.mean()
sd = returns.std()
current_dt = datetime(2023, 10, 30)

portfolio = pd.read_csv("Project/problem3.csv")

# Convert Expiration Date for Options to Date object
portfolio['ExpirationDate'] = portfolio.apply(lambda row: pd.to_datetime(row['ExpirationDate'], format="%m/%d/%Y") if row['Type'] == 'Option' else pd.NaT, axis=1)

# Calculate Implied Volatility for portfolio options
def portfolio_implied_vol(row):
    if row['Type'] == 'Option':
        return brentq(lambda x: gbsm(row['OptionType'] == "Call", current_s, row['Strike'], (row['ExpirationDate'] - current_dt).days / 365, rf, rf - dy, x).value - row['CurrentPrice'], 0.01, 5.0)
    return np.nan

portfolio['ImpVol'] = portfolio.apply(portfolio_implied_vol, axis=1)

# Simulate Returns
n_sim = 10000
fwd_t = 10

# Fit the AR(1) model
model = ARIMA(returns, order=(1, 0, 0))
fit = model.fit()
coef = fit.params

# Simulate AR(1) returns
def ar1_simulation(y, coef, innovations, ahead=1):
    m = coef['const']
    a1 = coef['ar.L1']
    s = fit.bse['sigma2']**0.5

    l = len(y)
    n = len(innovations) // ahead
    out = np.zeros((ahead, n))

    y_last = y[-1] - m
    for i in range(n):
        yl = y_last
        for j in range(ahead):
            next_val = a1 * yl + s * innovations[i * ahead + j]
            yl = next_val
            out[j, i] = next_val

    out += m
    return out

innovations = np.random.randn(fwd_t * n_sim)
ar_sim = ar1_simulation(returns.values, coef, innovations, ahead=fwd_t)

# Sum returns since these are log returns and convert to final prices
sim_returns = ar_sim.sum(axis=0)
sim_prices = current_s * np.exp(sim_returns)

# Cross join portfolio with iterations
iteration = np.arange(1, n_sim + 1)
values = pd.merge(portfolio, pd.DataFrame({'iteration': iteration}), how='cross')
n_vals = len(values)

# Set the forward TTM
values['fwd_ttm'] = values.apply(lambda row: (row['ExpirationDate'] - current_dt).days / 365 - fwd_t / days_year if row['Type'] == 'Option' else np.nan, axis=1)

# Calculate values of each position
simulated_value = []
current_value = []
pnl = []

for i, row in values.iterrows():
    sim_price = sim_prices[row['iteration'] - 1]
    current_val = row['Holding'] * row['CurrentPrice']
    current_value.append(current_val)
    
    if row['Type'] == 'Option':
        sim_val = row['Holding'] * gbsm(row['OptionType'] == "Call", sim_price, row['Strike'], row['fwd_ttm'], rf, rf - dy, row['ImpVol']).value
    elif row['Type'] == 'Stock':
        sim_val = row['Holding'] * sim_price
    else:
        sim_val = 0
    
    simulated_value.append(sim_val)
    pnl.append(sim_val - current_val)

values['simulatedValue'] = simulated_value
values['pnl'] = pnl
values['currentValue'] = current_value

# Calculate risk
risk = aggRisk(values, ['Portfolio'])
risk.to_csv("problem3_risk.csv", index=False)
