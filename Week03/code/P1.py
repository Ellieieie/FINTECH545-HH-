import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/ellieieie_/Desktop/DailyReturn.csv')

def calc_exponential_cov(series_x, series_y, decay_factor):
    mean_x = series_x.iloc[0]
    mean_y = series_y.iloc[0]
    exp_cov = 0.0

    for idx in range(len(series_x)):
        exp_cov = exp_cov * decay_factor + (1 - decay_factor) * (series_x.iloc[idx] - mean_x) * (series_y.iloc[idx] - mean_y)
        mean_x = mean_x * decay_factor + (1 - decay_factor) * series_x.iloc[idx]
        mean_y = mean_y * decay_factor + (1 - decay_factor) * series_y.iloc[idx]

    return exp_cov

def build_covariance_matrix(dataframe, decay_factor):
    num_cols = len(dataframe.columns)
    cov_mat = np.zeros((num_cols, num_cols))

    for i, col_x in enumerate(dataframe.columns):
        for j, col_y in enumerate(dataframe.columns):
            cov_mat[i, j] = calc_exponential_cov(dataframe[col_x], dataframe[col_y], decay_factor)

    return pd.DataFrame(cov_mat, columns=dataframe.columns, index=dataframe.columns)

lambda_values = [0.5, 0.8, 0.94]

plt.figure(figsize=(10, 6))
for lam in lambda_values:
    cov_mat = build_covariance_matrix(df, lam)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
    top_eigenvalues = eigenvalues[::-1][:10]
    total_var = np.sum(eigenvalues)
    explained_var_ratio = top_eigenvalues / total_var
    cumulative_explained_var = np.cumsum(explained_var_ratio)
    plt.plot(cumulative_explained_var, label=f'lambda = {lam}')

plt.title('Explained Variance')
plt.xlabel('Number of Top PCA Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True)
plt.legend()
plt.show()
