[Installation]
```
pip install pandas numpy scipy matplotlib
```

[Explanation]
Asset Return Data
Ensure that data.csv (containing asset return data) is available in the project directory. The script uses this file to calculate portfolio optimization metrics.

Portfolio Optimization
The script performs portfolio optimization using two risk-adjusted return metrics:

Max Sharpe Ratio Portfolio: Optimizes the portfolio for the highest return relative to its volatility.
Max RR_p Portfolio: Optimizes for a new risk-adjusted return metric (RR_p) that focuses on reducing extreme losses (tail risk) while still accounting for the return.
Functions Used for Optimization:
Portfolio Return: Calculates the expected return of a portfolio.
Portfolio Standard Deviation: Computes the risk (volatility) of a portfolio.
Negative Sharpe Ratio: Used for maximizing the Sharpe ratio through minimization.
Expected Shortfall (ES): Calculates the expected shortfall (tail risk) for a portfolio.
Negative Risk-Adjusted Return (RR_p): Optimizes the portfolio for the best RR_p, considering both return and tail risk.
Optimization Procedure:
Sharpe Ratio Optimization: Maximizes the Sharpe ratio by adjusting the portfolio weights.
RR_p Optimization: Maximizes the RR_p, which accounts for the expected shortfall (tail risk) of the portfolio.
Visualization and Comparison: The results, including the optimal weights for both portfolios, are displayed in a bar chart for comparison.
Portfolio Data
data.csv: CSV file containing the asset returns data used for portfolio optimization.
Output:
Optimal Weights: The script outputs the optimal portfolio weights for both the maximum Sharpe ratio and maximum RR_p portfolios.
Comparison Table: Displays the comparison of the two portfolios’ performance metrics and weights.

[Descriptions]
data.csv: CSV file containing the historical asset return data for portfolio optimization.
Optimal Weights for Sharpe Ratio Portfolio: Returns the portfolio weights that maximize the Sharpe ratio.
Optimal Weights for RR_p Portfolio: Returns the portfolio weights that maximize RR_p, focusing on minimizing tail risks.
Portfolio Visualization
The script generates a bar chart comparing the portfolio weights for the two optimization methods (Sharpe ratio and RR_p). This visual representation helps in understanding how each portfolio is composed based on the optimized weights.

[Conclusion]
This project compares two popular portfolio optimization strategies: maximizing the Sharpe ratio and maximizing RR_p. The Sharpe ratio is ideal for investors who are focusing on optimizing returns relative to volatility, while RR_p is more suitable for those who are concerned with minimizing extreme losses, especially during market turbulence. The resulting portfolio metrics and visualizations allow you to assess which optimization method best suits your investment goals.