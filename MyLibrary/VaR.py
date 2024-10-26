import pandas as pd
import numpy as np
from scipy.stats import norm, t
from statsmodels.tsa.arima.model import ARIMA
from expo_weights import expW
from exp_w_cov import ewCovar


# Load the data
prices = pd.read_csv('/Users/ellieieie_/Desktop/problem1.csv')
# Calculate returns, assuming 'Date' is the date column
returns = prices.set_index("Date").pct_change().dropna()

# Extract the META returns series
meta = returns["META"]
# Normalize the returns
meta_centered = meta - meta.mean()
s = meta_centered.std()

# Normal VaR
d1 = norm(0, s)
VaR1 = -d1.ppf(0.05)

# Exponentially Weighted Normal VaR
def ewCovar(data, lambda_):
    weights = np.array([(1 - lambda_) * lambda_**i for i in range(len(data))][::-1])
    weights /= weights.sum()  # Normalize to sum to 1
    mean_adjusted = data - np.average(data, weights=weights)
    return np.cov(mean_adjusted, aweights=weights)

s2 = np.sqrt(ewCovar(meta_centered.values.reshape(-1, 1), 0.94)[0, 0])
d2 = norm(0, s2)
VaR2 = -d2.ppf(0.05)

# Fit T-distribution and T VaR
def fit_general_t(data):
    # Fit data to a T-distribution and return shape parameters
    df, loc, scale = t.fit(data)
    dist = t(df, loc, scale)
    return loc, scale, df, dist

m, s, nu, d3 = fit_general_t(meta_centered)
VaR3 = -d3.ppf(0.05)

# Historical VaR
VaR4 = -np.percentile(meta_centered, 5)

# AR(1) Model VaR
ar_model = ARIMA(meta_centered, order=(1, 0, 0))
ar_fit = ar_model.fit()
ar_sim = ar_fit.simulate(nsamples=1000000)
VaR5 = -np.percentile(ar_sim, 5)

# Current price of META
current_price = prices["META"].iloc[-1]

# Display results
print(f"Normal VaR  : ${current_price * VaR1} - {100 * VaR1}%")
print(f"EWMA Normal VaR: ${current_price * VaR2} - {100 * VaR2}%")
print(f"T Dist VaR  : ${current_price * VaR3} - {100 * VaR3}%")
print(f"AR(1) VaR   : ${current_price * VaR5} - {100 * VaR5}%")
print(f"Historic VaR: ${current_price * VaR4} - {100 * VaR4}%")