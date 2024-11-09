import os
import matplotlib.pyplot as plt
from utils import black_scholes_call, black_scholes_put

os.makedirs("results", exist_ok=True)

def plot_option_prices_vs_volatility(S, K, T, r, volatilities):
    call_prices = [black_scholes_call(S, K, T, r, sigma) for sigma in volatilities]
    put_prices = [black_scholes_put(S, K, T, r, sigma) for sigma in volatilities]
    
    plt.figure(figsize=(10, 6))
    plt.plot(volatilities, call_prices, label="Call Option Price")
    plt.plot(volatilities, put_prices, label="Put Option Price")
    plt.xlabel("Implied Volatility")
    plt.ylabel("Option Price")
    plt.title("Option Price vs. Implied Volatility")
    plt.legend()
    plt.savefig("results/call_put_volatility_plot.png")
    plt.show()

def plot_implied_vol_vs_strike(aapl_options):
    plt.figure(figsize=(10, 6))
    for option_type, group in aapl_options.groupby('Type'):
        plt.plot(group['Strike'], group['Implied Volatility'], label=f"{option_type.capitalize()} Options")
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")
    plt.title("Implied Volatility vs. Strike Price for AAPL Options")
    plt.legend()
    plt.savefig("results/implied_vol_vs_strike.png")
    plt.show()

