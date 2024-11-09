import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils import black_scholes_call, black_scholes_put, implied_volatility
from plot_options import plot_option_prices_vs_volatility, plot_implied_vol_vs_strike
from simulation import simulate_returns_and_calculate_risk
from data_processing import load_data

def main():
 
    S, K, T, r, coupon = 165, 165, (datetime(2023, 3, 17) - datetime(2023, 3, 3)).days / 365, 5.25 / 100, 0.53 / 100
    aapl_options, daily_prices, portfolios = load_data()
    
    volatilities = np.linspace(0.1, 0.8, 100)
    plot_option_prices_vs_volatility(S, K, T, r, volatilities)
    

    aapl_options['Expiration'] = pd.to_datetime(aapl_options['Expiration'])

    aapl_options['Market Price'] = aapl_options['Last Price']
    current_date = pd.to_datetime('2023-10-30')
    aapl_options['Time to Maturity'] = (aapl_options['Expiration'] - current_date).dt.days / 365
    aapl_options = aapl_options[aapl_options['Time to Maturity'] > 0]  # 过滤掉过期的期权
    
    risk_free_rate = 0.0525
    current_price = 170.15
    q = 0.0057
 
    aapl_options = aapl_options.dropna(subset=['Market Price', 'Stock', 'Strike', 'Time to Maturity'])

    aapl_options['Implied Volatility'] = aapl_options.apply(
        lambda row: implied_volatility(
            row['Type'], 
            row['Market Price'], 
            current_price,
            row['Strike'], 
            row['Time to Maturity'],
            risk_free_rate
        ), 
        axis=1
    )


    if not aapl_options['Implied Volatility'].isnull().all():
        plt.figure(figsize=(12, 6))

        call_options = aapl_options[aapl_options['Type'] == 'Call']
        plt.plot(call_options['Strike'], call_options['Implied Volatility'], marker='o', label='Call Options', color='blue')

        put_options = aapl_options[aapl_options['Type'] == 'Put']
        plt.plot(put_options['Strike'], put_options['Implied Volatility'], marker='o', label='Put Options', color='red')

        plt.title('Implied Volatility vs Strike Price')
        plt.xlabel('Strike Price')
        plt.ylabel('Implied Volatility')
        plt.grid()
        plt.legend()

        plt.savefig('results/implied_volatility_vs_strike_price.png')
        plt.show()
    else:
        print("No valid implied volatility data to plot.")
 
    print(aapl_options[['Stock', 'Expiration', 'Type', 'Strike', 'Market Price', 'Time to Maturity', 'Implied Volatility']])

    simulate_returns_and_calculate_risk(daily_prices, portfolios, S, r)

if __name__ == "__main__":
    main()

