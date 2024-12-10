import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt

def simulate_returns_and_calculate_risk(daily_prices, portfolios, S, r):

    daily_prices['Date'] = pd.to_datetime(daily_prices['Date'])
    daily_prices.set_index('Date', inplace=True)
    daily_prices.index = daily_prices.index.to_period('D')  
    
    log_returns_df = pd.DataFrame(index=daily_prices.index)

    for stock in daily_prices.columns:
        print(f"Processing stock: {stock}")
        log_returns_df[f'Log Return {stock}'] = np.log(daily_prices[stock]).diff()

    daily_prices = pd.concat([daily_prices, log_returns_df], axis=1)

    daily_prices = daily_prices.copy()


    for stock in log_returns_df.columns:  
        log_returns = daily_prices[stock].dropna()  
        log_returns -= log_returns.mean() 

        model = AutoReg(log_returns, lags=1).fit()
        forecast = model.predict(start=len(log_returns), end=len(log_returns) + 9)

        original_stock_name = stock.replace('Log Return ', '')
        current_price = daily_prices[original_stock_name].iloc[-1]
        forecasted_prices = [current_price]
        for ret in forecast:
            forecasted_prices.append(forecasted_prices[-1] * np.exp(ret))

        mean_price = np.mean(forecasted_prices)
        var = np.percentile(forecasted_prices, 5)
        es = np.mean([p for p in forecasted_prices if p <= var])

        print(f"{original_stock_name} - Mean Forecasted Price: {mean_price:.2f}")
        print(f"{original_stock_name} - VaR (5%): {var:.2f}")
        print(f"{original_stock_name} - Expected Shortfall: {es:.2f}")

        plt.figure(figsize=(10, 6))
        plt.plot(forecasted_prices, label=f"Forecasted Price of {original_stock_name}")
        plt.xlabel("Day Ahead")
        plt.ylabel("Price")
        plt.title(f"Forecasted {original_stock_name} Price Over 10 Days")
        plt.legend()
        plt.grid()

        plt.savefig(f"results/forecasted_prices_{original_stock_name}.png")
        plt.show()
        plt.close() 
        
        
        
        