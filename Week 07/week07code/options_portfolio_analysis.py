import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Define GBSM (Generalized Black-Scholes Model) function
class GBSM:
    def __init__(self, is_call, underlying, strike, ttm, rf, b, ivol):
        self.is_call = is_call
        self.underlying = underlying
        self.strike = strike
        self.ttm = ttm
        self.rf = rf
        self.b = b
        self.ivol = ivol

    @property
    def value(self):
        d1 = (np.log(self.underlying / self.strike) + (self.b + (self.ivol ** 2) / 2) * self.ttm) / (self.ivol * np.sqrt(self.ttm))
        d2 = d1 - self.ivol * np.sqrt(self.ttm)
        if self.is_call:
            value = (self.underlying * np.exp((self.b - self.rf) * self.ttm) * norm.cdf(d1) -
                     self.strike * np.exp(-self.rf * self.ttm) * norm.cdf(d2))
        else:
            value = (self.strike * np.exp(-self.rf * self.ttm) * norm.cdf(-d2) -
                     self.underlying * np.exp((self.b - self.rf) * self.ttm) * norm.cdf(-d1))
        return value

def gbsm(is_call, underlying, strike, ttm, rf, b, ivol):
    return GBSM(is_call, underlying, strike, ttm, rf, b, ivol)

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
    """Fit a Fama-French 4 Factor Model to stock returns."""
    X = sm.add_constant(factors)  # Add intercept
    model = sm.OLS(stock_returns, X).fit()  # Ordinary Least Squares regression
    return model

# Example Usage
if __name__ == "__main__":
    # Problem 1: Option Pricing Example
    current_price = 165
    current_date = datetime.strptime("03/03/2023", "%m/%d/%Y")
    rf = 0.0525
    dy = 0.0053
    days_year = 365

    expiration_date = datetime.strptime("03/17/2023", "%m/%d/%Y")
    ttm = (expiration_date - current_date).days / days_year

    strike = 165
    iv = np.arange(0.10, 0.81, 0.02)

    # Calculate call and put option values using GBSM and store in a DataFrame
    records = []
    for v in iv:
        call_option = gbsm(True, current_price, strike, ttm, rf, rf - dy, v)
        put_option = gbsm(False, current_price, strike, ttm, rf, rf - dy, v)
        records.append({
            'Implied Volatility': v,
            'Call Value': call_option.value,
            'Put Value': put_option.value
        })

    # Create DataFrame from records
    df = pd.DataFrame(records)

    # Display the DataFrame
    print(df)

    # Plot results
    plt.plot(iv, df['Call Value'], label="Call Values")
    plt.plot(iv, df['Put Value'], label="Put Values")
    plt.xlabel("Implied Volatility")
    plt.ylabel("Option Value")
    plt.legend()
    plt.title("Option Prices vs Implied Volatility")

    # Save the plot as a PNG file
    plt.savefig("option_prices_vs_iv.png")

    # Show the plot
    plt.show()
    print("Problem 1 Successfully, option_prices_vs_iv.png")

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
    scaler = StandardScaler()
    aligned_factors = scaler.fit_transform(aligned_factors)
    
    stock_data_records = []
    total_weight = 1.0
    num_stocks = len(merged_data.columns.difference(['Date', 'Mkt-RF', 'SMB', 'HML', 'Mom']))
    equal_weight = total_weight / num_stocks
    for i, stock in enumerate(merged_data.columns.difference(['Date', 'Mkt-RF', 'SMB', 'HML', 'Mom'])):
        stock_returns = merged_data[stock].values
        model = fama_french_four_factor(stock_returns, aligned_factors)
        stock_data_records.append({
            'Stock': stock,
            'Weight': equal_weight,  # Assign equal weight to each stock
            'Er': model.params[1],  # Expected return coefficient for market factor (e.g., Mkt-RF)
            'UnconstWeight': model.params.sum()  # Sum of coefficients as an example of unconstrained weight
        })

    # Create DataFrame for stock data
    stock_data_df = pd.DataFrame(stock_data_records)

    # Display the DataFrame
    print(stock_data_df)

    # Save the stock data to CSV
    stock_data_df.to_csv('Fama_French_Stock_Data.csv', index=False)
    print("Problem 3 Successfully, see Fama_French_Stock_Data.csv for details.")
