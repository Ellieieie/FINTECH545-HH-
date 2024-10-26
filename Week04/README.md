[Overview]
This project involves the analysis of financial returns using various statistical methods and models. The tasks are divided into three main problems, each focusing on calculating and comparing the Value at Risk (VaR) of different portfolios using different methodologies.

This README provides instructions on how to set up and run the code, as well as an overview of the files included in the submission.

[Requirements]
Required Python libraries:
pandas
numpy
scipy
statsmodels
You can install the required packages using the following command: "pip install pandas numpy scipy statsmodels"

Adjust file paths for your local system:
The dataset "/Users/ellieieie_/Desktop/DailyReturn.csv" , "/Users/ellieieie_/Desktop/DailyPrices.csv" ,  "/Users/ellieieie_/Desktop/portfolio.csv" should contain the daily return data for various assets in a CSV format.
Replace the file path in the script if necessary to match your local directory structure.

[Descriptions]
Problem 1: Expected Value and Standard Deviation of Prices
The code calculates and compares the expected value and standard deviation of price at time t (P_t) using three types of returns: arithmetic, log, and classical Brownian motion.
It uses the DailyReturn.csv data to simulate returns and compute the results.
Outputs include empirical and theoretical mean and standard deviation comparisons for each return type.

Problem 2: Value at Risk (VaR) Calculation
Uses DailyPrices.csv to calculate arithmetic returns.
Implements a function (return_calculate()) that calculates returns based on user input (arithmetic or log).
Calculates VaR for a portfolio consisting of holdings in GOOGL, NVDA, TSLA, and META using five different methods:
Normal Distribution
Normal Distribution with Exponentially Weighted Variance (λ = 0.94)
MLE fitted T-distribution
AR(1) Model
Historical Simulation
Outputs include comparisons of VaR values for each method.

Problem 3: Portfolio VaR Calculation
Uses Portfolio.csv and DailyPrices.csv to analyze the holdings of three different portfolios (A, B, and C).
Calculates VaR for each portfolio as well as the total VaR of all holdings using an exponentially weighted covariance with λ = 0.97.
Outputs include individual VaR for each portfolio and the combined total.
Discusses the choice of model and its impact on the results.

[Notes]
Make sure that your CSV file (DailyReturn.csv) is correctly formatted and located in the path specified in the script.
The PCA analysis helps in reducing dimensionality by retaining the most significant principal components, which is crucial for speeding up simulations.
When correcting the covariance matrix to be positive semi-definite, Higham's method ensures that the matrix can be used in multivariate normal simulations without errors.
For any questions or issues, feel free to contact: hh315@duke.edu