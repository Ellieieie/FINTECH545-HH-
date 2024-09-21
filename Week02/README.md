[Overview]
This project contains code for statistical analysis and time series modeling. It includes the following problems:

Moment Calculation: Compute statistical moments (mean, variance, skewness, kurtosis) from a CSV file.
Regression Analysis: Perform Ordinary Least Squares (OLS) and Maximum Likelihood Estimation (MLE) for regression with normal and t-distributions.
Conditional Distribution: Analyze and visualize the conditional distribution of one variable given another.
Time Series Analysis: Fit AR and MA models to time series data and determine the best model.

[Download]
Adjust the file paths in the scripts as needed to match the locations of your CSV files: 

Problem 1: Moment Calculation
CSV File Required: /Users/Name/Location/problem1.csv should contain a single column of numerical data.
Problem 2: Regression Analysis
CSV File Required: /Users/Name/Locationp/problem2.csv should contain two columns: x and y.
Problem 2.3: Conditional Distribution
CSV File Required: /Users/Name/Location/problem2_x.csv should contain two columns: x1 and x2.

[Requirements]
Ensure you have the following Python packages installed:
pandas, numpy, scipy, statsmodels, matplotlib, csv

You can install the required packages using pip:
pip install pandas numpy scipy statsmodels matplotlib

[Virtual Environment Check] (if method above failed)

1. make sure you have downloaded pandas; otherwise, follow the instruction below:
    a. Check Python Version, copy code: 
        python3 --version
    b. Install pip:
         python3 -m ensurepip --upgrade
    c. Install pandas using pip: 
        python3 -m pip install pandas
    d. Verify pandas Installation:
        python3 -m pip show pandas
    e. Check Virtual Environment Activation:
        source /path_to_your_virtualenv/bin/activate
    f: Use Python Path Explicitly:
        /usr/local/bin/python3 -m pip install pandas
2. make sure you have downloaded scipy; otherwise, follow the instruction below:
    a. Install pip for Python3 (If ensurepip from 1(b) doesn't work, you can try installing pip manually):
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py python3 get-pip.py
    b. Install scipy: 
        pip install scipy
2. make sure you have downloaded numpy; otherwise, follow the instruction below:
    a. Use pip to install numpy:
       pip install numpy
            , Or if pip is not working:
                pip3 install numpy
    b. Check pip Installation:
        python3 -m ensurepip --upgrade
        python3 -m pip install --upgrade pip
    c. Verify Installation
        python3 -c "import numpy; print(numpy.__version__)"

[Explaination]
Problem 1: 
a. calculate_moments(data): Calculates mean, variance, skewness, and kurtosis.
b. read_data_from_csv(file_path): Reads numerical data from a CSV file.
c. mean, variance, skewness, kurtosis: Values are printed for both custom calculations and those using numpy and scipy.

Problem 2:
a. OLS: Performs regression using Ordinary Least Squares and calculates R-squared.
b. MLE: Fits a Maximum Likelihood Estimation model with both normal and t-distribution assumptions.
c. R-squared: Measures the proportion of variance explained by the models.
Problem 2.3:
a. Conditional Distribution: Uses multivariate normal distribution to analyze and visualize the conditional distribution.
Problem 3:
a. Time Series: Fits AR and MA models to the time series data and selects the best model based on AIC and BIC criteria.
b. ACF/PACF Plots: Assists in identifying the appropriate model order.

[Notes]
1. Make sure that your CSV files are formatted correctly as described.
2. Ensure that the necessary libraries are installed and updated.
If you encounter any issues or need further assistance, please contact: hh315@duke.edu
