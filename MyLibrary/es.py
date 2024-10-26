import numpy as np
from scipy.stats import norm, t


def expected_shortfall_t(returns, alpha=0.05):
 
    # Fit the T-distribution to the returns data
    df, loc, scale = t.fit(returns)
    
    # Calculate Value at Risk (VaR) at the alpha level
    VaR = t.ppf(alpha, df, loc=loc, scale=scale)
    
    # Expected Shortfall (ES) calculation using T-distribution
    ES_t = t.expect(lambda x: x, args=(df,), loc=loc, scale=scale, lb=-np.inf, ub=VaR) / alpha
    
    return ES_t

def expected_shortfall_normal(returns, alpha=0.05):

    # Fit the Normal distribution to the returns data
    mu, sigma = norm.fit(returns)
    
    # Calculate Expected Shortfall (ES) using Normal distribution formula
    VaR = norm.ppf(alpha, loc=mu, scale=sigma)
    ES_n = mu - sigma * norm.pdf(norm.ppf(alpha)) / alpha
    
    return ES_n