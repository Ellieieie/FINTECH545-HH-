import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_volatility(option_type, market_price, S, K, T, r):

    market_price = float(market_price)
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)


    if not all(np.isfinite([market_price, S, K, T, r])) or market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan 


    def objective_function(sigma):
        if option_type.lower() == 'call':
            return black_scholes_call(S, K, T, r, sigma) - market_price
        elif option_type.lower() == 'put':
            return black_scholes_put(S, K, T, r, sigma) - market_price
            from scipy.optimize import newton

        try:
            implied_vol = brentq(objective_function, 1e-6, 10, xtol=1e-6)
            return implied_vol
        except (ValueError, RuntimeError):
            return np.nan