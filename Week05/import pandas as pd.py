import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

prices = pd.read_csv('/Users/ellieieie_/Desktop/DailyPrices.csv')

def return_calculate(prices, method="DISCRETE", date_column="Date"):
    if date_column not in prices.columns:
        raise ValueError(f"dateColumn: {date_column} not in DataFrame columns: {prices.columns.tolist()}")

    vars = [col for col in prices.columns if col != date_column]
    p = prices[vars].values
    n, m = p.shape
    p2 = np.empty((n - 1, m), dtype=np.float64)

    for i in range(n - 1):
        for j in range(m):
            p2[i, j] = p[i + 1, j] / p[i, j]

    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in ('LOG', 'DISCRETE')")

    dates = prices[date_column].iloc[1:].reset_index(drop=True)
    out = pd.DataFrame({date_column: dates})

    for i, var in enumerate(vars):
        out[var] = p2[:, i]

    return out

returns_discrete = return_calculate(prices, method="DISCRETE")
print(returns_discrete.head())

returns_meta = returns_discrete["META"]
returns_meta = returns_meta - returns_meta.mean()
confidence_level = 0.05

# 1. VaR using a normal distribution
mean_meta = returns_meta.mean()
std_meta = returns_meta.std()
var_normal = stats.norm.ppf(1 - confidence_level, mean_meta, std_meta)

# 2. VaR using a normal distribution with an exponentially weighted variance (λ = 0.94)
lambda_ = 0.94
weights = np.array([(1 - lambda_) * lambda_ ** i for i in range(len(returns_meta))])[::-1]
weighted_mean = np.average(returns_meta, weights=weights)
weighted_var = np.average((returns_meta - weighted_mean) ** 2, weights=weights)
var_ewma = stats.norm.ppf(1 - confidence_level, weighted_mean, np.sqrt(weighted_var))

# 3. VaR using a MLE fitted T-distribution
params = stats.t.fit(returns_meta)
var_t = stats.t.ppf(1 - confidence_level, *params)

# 4. VaR using a fitted AR(1) model
model = ARIMA(returns_meta.values, order=(1, 0, 0)).fit()
forecast = model.forecast(steps=1)
forecast_value = forecast[0]
var_ar1 = forecast_value + stats.norm.ppf(1 - confidence_level) * returns_meta.std()

# 5. VaR using Historical Simulation
var_historical = returns_meta.quantile(confidence_level)

print("\nVaR Comparison at 95% Confidence Level:")
print(f"1. Normal Distribution VaR: {var_normal:.4f}")
print(f"2. EWMA Normal Distribution VaR: {var_ewma:.4f}")
print(f"3. MLE fitted T-distribution VaR: {var_t:.4f}")
print(f"4. AR(1) Model VaR: {var_ar1:.4f}")
print(f"5. Historical Simulation VaR: {var_historical:.4f}")







