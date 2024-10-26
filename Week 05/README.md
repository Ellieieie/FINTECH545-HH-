[Overview]
This tool computes VaR and ES for individual assets and portfolios using:

Normal distribution with exponentially weighted variance.
Maximum Likelihood Estimate (MLE) fitted T-distribution.
Historical simulation.
For portfolios, VaR and ES values are computed by aggregating stock-specific risk metrics, weighted by holdings.

[Features]
VaR and ES Calculation:
Supports Normal and T-distribution-based VaR and ES calculations.
Includes Historical Simulation for VaR and ES.
Portfolio Aggregation:
Computes portfolio-level VaR and ES for multiple portfolios, supporting custom weighting by holdings.

[Installation]
Required libraries:
pandas
numpy
scipy
sklearn
Install the required libraries via pip: " pip install pandas numpy scipy scikit-learn "
setup: " 
git clone <repository_url>
cd <repository_folder>
"

[Description]
#Problem 2 - Individual Asset VaR and ES
Returns Calculation: Calculates returns for the asset from the problem1.csv file.
Normal VaR with EWMA Variance:
Computes VaR and ES using a Normal distribution with exponentially weighted variance (lambda = 0.97).
MLE-Fitted T-Distribution VaR:
Fits a T-distribution to the returns using Maximum Likelihood Estimation (MLE) and calculates VaR and ES.
Historical Simulation VaR:
Computes VaR and ES using historical returns.

#Problem 3 
Portfolio Data Loading:
Loads portfolio holdings and asset return data from CSV files.
Function Definitions:
compute_var_es_t: Calculates VaR and ES for a Generalized T-distribution.
compute_var_es_normal: Calculates VaR and ES for a Normal distribution.
Portfolio Filtering:
Filters portfolios (A, B, C) based on the holdings data in portfolio.csv.
Portfolio VaR and ES Aggregation:
Computes the portfolio’s VaR and ES in dollar terms by summing across individual assets weighted by holdings.