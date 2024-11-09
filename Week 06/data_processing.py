import pandas as pd

def load_data():
    aapl_options = pd.read_csv('data/AAPL_Options.csv')
    daily_prices = pd.read_csv('data/DailyPrices.csv')
    portfolios = pd.read_csv('data/problem3.csv')
    return aapl_options, daily_prices, portfolios

