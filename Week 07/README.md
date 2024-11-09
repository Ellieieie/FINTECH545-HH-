[Installation]
Install the necessary libraries:
"
pip install pandas numpy scipy scikit-learn
"

[Explaination]
1. Individual Asset VaR and ES Calculation
To calculate VaR and ES for an individual asset:

Ensure problem1.csv (containing asset returns) is in the project directory.
Run the individual asset VaR and ES script, which computes:
Normal VaR with EWMA Variance: VaR and ES based on normal distribution with exponentially weighted variance.
MLE-Fitted T-Distribution VaR: Fits a t-distribution to returns and calculates VaR and ES.
Historical Simulation VaR: Uses historical returns for VaR and ES calculations.

2. Portfolio VaR and ES Aggregation
For portfolio-level VaR and ES calculations:

Ensure portfolio.csv (holdings data) and asset return data files are in the project directory.
Functions provided:
compute_var_es_t: Calculates VaR and ES for a generalized t-distribution.
compute_var_es_normal: Calculates VaR and ES using a normal distribution.
Portfolio Filtering: Filters portfolios A, B, and C based on portfolio.csv.
Portfolio VaR and ES aggregation sums individual asset VaR and ES values, weighted by holdings, to deliver a comprehensive portfolio-level risk assessment.

[Descriptions]
problem1.csv: CSV file containing individual asset returns.
portfolio.csv: CSV file detailing portfolio holdings for VaR/ES calculations.
DailyPrices.csv and F-F_Research_Data_Factors_daily.CSV: Used in additional risk modeling (e.g., Fama-French model, not described here).
