[Overview]

This project contains code for options analysis and time series modeling, focusing on implied volatility, portfolio evaluation, and risk analysis. It includes the following problems:

Problem 1: Option Value Calculation and Analysis

Calculate the time to maturity for a call and a put option given the current stock price, current date, expiration date, risk-free rate, and continuously compounding coupon rate.

Plot the value of both the call and the put options for a range of implied volatilities (10% to 80%).

Discuss the graphs, focusing on how supply and demand impact implied volatility.

Problem 2: Implied Volatility Analysis for AAPL Options

Calculate the implied volatility for options using data from AAPL_Options.csv.

Plot implied volatility against the strike price for calls and puts, then discuss the shape of the graphs and the market dynamics that could create such patterns.

Problem 3: Portfolio Value Simulation and Risk Analysis

Use the portfolio data from problem3.csv to analyze portfolio values over a range of underlying values.

Plot the portfolio values and explain the shapes using put-call parity.

Use DailyPrices.csv to calculate the log returns of AAPL, demean the series, and fit an AR(1) model to simulate AAPL returns 10 days ahead. Apply these returns to the current AAPL price to calculate the Mean, VaR, and ES.

[Download]

Adjust the file paths in the scripts as needed to match the locations of your CSV files:

Problem 1: Requires problem1.csv for initial input values.

Problem 2: Requires AAPL_Options.csv containing data on options.

Problem 3: Requires problem3.csv for portfolio data and DailyPrices.csv for price data.

[Requirements]

Ensure you have the following Python packages installed:

pandas, numpy, scipy, matplotlib, statsmodels

You can install the required packages using pip:

pip install pandas numpy scipy matplotlib statsmodels

[Virtual Environment Check]

If the method above fails, follow these steps to ensure the virtual environment is correctly set up:

Install pandas

Check Python version:

python3 --version

Install pip if needed:

python3 -m ensurepip --upgrade

Install pandas:

python3 -m pip install pandas

Verify installation:

python3 -m pip show pandas

Install other required packages (scipy, numpy, etc.) similarly using pip.

[Explanation]

Problem 1: Option Value Calculation and Graphs

Time to Maturity: Calculate using calendar days, not trading days.

Option Values: Plot call and put values for implied volatilities between 10% and 80%.

Discussion: Explain how changes in supply and demand impact implied volatility.

Problem 2: Implied Volatility Analysis

Calculate Implied Volatility: Extract from option prices using iterative techniques.

Plot and Discuss: Describe the relationship between implied volatility and strike price.

Problem 3: Portfolio Evaluation and Simulation

Portfolio Value: Use problem3.csv data to calculate and plot portfolio values based on varying underlying prices.

Put-Call Parity: Use to discuss portfolio value behavior.

Log Returns and AR(1) Model: Simulate returns and calculate Mean, VaR, and ES.

[Notes]

Ensure your CSV files are correctly formatted.

Install and update all necessary Python libraries before running the scripts.

If any issues arise or assistance is needed, contact: hh315@duke.edu

