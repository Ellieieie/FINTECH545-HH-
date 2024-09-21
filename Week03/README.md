[Overview]
This project performs statistical analysis, covariance matrix computations, PCA analysis, and multivariate normal simulations based on exponentially weighted covariance matrices. The project consists of three main parts:

1. Exponentially Weighted Covariance and PCA Analysis
Calculates the exponentially weighted covariance matrix using different λ values.
Performs Principal Component Analysis (PCA) on the covariance matrix to determine the cumulative variance explained by principal components.
2. Nearest Positive Semi-Definite (PSD) Matrix Correction and Multivariate Normal Simulation
Corrects a covariance matrix using Higham's method to ensure it is positive semi-definite (PSD).
Simulates multivariate normal data using the corrected covariance matrix.
Compares the covariance matrix derived from direct simulation with PCA-based simulations at different explained variance thresholds.
3. Variance Computation Using Pearson and Exponentially Weighted Methods
Generates covariance matrices using Pearson correlation and variance methods.
Ensures the covariance matrices are PSD and performs multivariate simulations for various scenarios.
Compares the Frobenius norms of the covariance matrices from different methods to analyze their accuracy.

[Requirements]
Ensure you have the following Python packages installed:

pandas
numpy
matplotlib
scikit-learn
You can install these packages using pip:
"pip install pandas numpy matplotlib scikit-learn"

[Setup]
Adjust file paths for your local system:
The dataset "/Users/ellieieie_/Desktop/DailyReturn.csv" should contain the daily return data for various assets in a CSV format.
Replace the file path in the script if necessary to match your local directory structure.

[Explanation]
Problem 1: Exponentially Weighted Covariance and PCA
1.Function: exponentially_weighted_covariance(data, lambda_)
Computes the exponentially weighted covariance matrix using a decay factor, λ.
Visualizes the resulting covariance matrix using a heatmap.

2. Function: plot_covariance_matrix(cov_matrix, title)
Plots the covariance matrix using matplotlib.

3. PCA Analysis:
Uses PCA to analyze the variance explained by principal components for different λ values.
Visualizes the cumulative variance explained.

Problem 2: Nearest PSD Matrix Correction and Multivariate Normal Simulation
1. Function: near_psd(a, epsilon)
Ensures a given covariance matrix is positive semi-definite by correcting any negative eigenvalues using Higham's method.

2. Function: simulate_multivariate_normal(cov_matrix, n_samples, pca=False, explained_variance=None)
Simulates multivariate normal data using either direct covariance sampling or PCA-based dimensionality reduction.

Problem 3: Pearson and Exponentially Weighted Variance Computation
1. Function: nearest_psd(matrix, epsilon)
Corrects covariance matrices to be positive semi-definite.
2. Function: simulate_multivariate_normal()
Simulates multivariate normal samples using different covariance matrices generated through Pearson and exponentially weighted variance methods.

[Notes]
Make sure that your CSV file (DailyReturn.csv) is correctly formatted and located in the path specified in the script.
The PCA analysis helps in reducing dimensionality by retaining the most significant principal components, which is crucial for speeding up simulations.
When correcting the covariance matrix to be positive semi-definite, Higham's method ensures that the matrix can be used in multivariate normal simulations without errors.
For any questions or issues, feel free to contact: hh315@duke.edu
