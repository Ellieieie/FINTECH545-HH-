import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = pd.read_csv('/Users/ellieieie_/Desktop/DailyReturn.csv')


def exponentially_weighted_covariance(data, lambda_):
    returns = data.values
    n, m = returns.shape
    
    ewma_means = np.zeros((n, m))
    ewma_cov_matrix = np.zeros((m, m))

    alpha = 1 - lambda_
 
    ewma_means[0, :] = returns[0, :]
    for t in range(1, n):
        ewma_means[t, :] = alpha * returns[t, :] + (1 - alpha) * ewma_means[t-1, :]
 
    for t in range(n):
        centered_returns = returns[t, :] - ewma_means[t, :]
        ewma_cov_matrix += np.outer(centered_returns, centered_returns)
    
    ewma_cov_matrix /= n
    return ewma_cov_matrix

# Example usage
lambda_ = 0.9
cov_matrix = exponentially_weighted_covariance(data, lambda_)
print("Exponentially Weighted Covariance Matrix:\n", cov_matrix)


def plot_covariance_matrix(cov_matrix, title):
    plt.figure(figsize=(10, 8))
    plt.imshow(cov_matrix, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Assets')
    plt.ylabel('Assets')
    plt.show()

plot_covariance_matrix(cov_matrix, 'Exponentially Weighted Covariance Matrix')

lambda_values = [0.1, 0.3, 0.5, 0.7, 0.9]
cumulative_variances = {}

for lambda_ in lambda_values:
    cov_matrix = exponentially_weighted_covariance(data, lambda_)
  
    pca = PCA()
    pca.fit(cov_matrix)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    cumulative_variances[lambda_] = cumulative_variance

plt.figure(figsize=(14, 8))
for lambda_, cumulative_variance in cumulative_variances.items():
    plt.plot(cumulative_variance, label=f'λ = {lambda_}')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained by Principal Components for Different λ Values')
plt.legend()
plt.grid(True)
plt.show()


#Problem 2
from numpy.linalg import cholesky, eigh
import time

n = 500
sigma = np.full((n, n), 0.9)  
np.fill_diagonal(sigma, 1.0) 
sigma[0, 1] = sigma[1, 0] = 0.7357  


def near_psd(a, epsilon=0.0):
    n = a.shape[0]
    out = np.copy(a)
    
    invSD = None
    if not np.allclose(np.diag(out), 1):
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD

    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    out = vecs @ np.diag(vals) @ vecs.T

    if invSD is not None:
        out = invSD @ out @ invSD
    
    return out

start_time = time.time()
near_psd_matrix = near_psd(sigma) 
higham_time = time.time() - start_time
print(f"Higham's method took {higham_time:.4f} seconds")

vals, vecs = np.linalg.eigh(near_psd_matrix)
vals_subset = np.sqrt(np.diag(vals[3:6]))  
vecs_subset = vecs[:, 3:6]  
B = np.dot(vecs_subset, vals_subset)  

m = 100_000  
try:
    random_data = np.random.randn(3, m)
    r = np.dot(B, random_data) 
    print("Matrix multiplication successful")
except ValueError as e:
    print(f"Matrix multiplication failed: {e}")

is_psd = np.all(np.linalg.eigvals(near_psd_matrix) >= 0)
print(f"Is the corrected matrix PSD? {is_psd}")

frobenius_norm_original = np.linalg.norm(sigma)
frobenius_norm_psd = np.linalg.norm(near_psd_matrix)
print(f"Frobenius Norm - Original: {frobenius_norm_original}, Near PSD: {frobenius_norm_psd}")


# Problem 3
import time
from numpy.linalg import eigh


data = pd.read_csv('/Users/ellieieie_/Desktop/DailyReturn.csv')
corr_matrix = data.corr()
var_vector = data.var()
lambda_ew = 0.97
ew_variance = data.ewm(alpha=1-lambda_ew).var().iloc[-1]

cov_matrix_pearson_var = corr_matrix * var_vector
cov_matrix_pearson_ew_var = corr_matrix * ew_variance
cov_matrix_ew_var = np.diag(ew_variance) @ corr_matrix @ np.diag(ew_variance)
cov_matrix_pearson_var_ew = np.diag(var_vector) @ corr_matrix @ np.diag(var_vector)
cov_matrices = [
    cov_matrix_pearson_var,
    cov_matrix_pearson_ew_var,
    cov_matrix_ew_var,
    cov_matrix_pearson_var_ew
]


def nearest_psd(matrix, epsilon=1e-8):
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, epsilon)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def simulate_multivariate_normal(cov_matrix, n_samples, pca=False, explained_variance=None):
    mean_vector = np.zeros(cov_matrix.shape[0])  
    psd_matrix = nearest_psd(cov_matrix)  
    
    if not pca:
        return np.random.multivariate_normal(mean_vector, psd_matrix, n_samples)
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(psd_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        if explained_variance is not None:
            total_variance = np.sum(eigenvalues)
            cumulative_variance = np.cumsum(eigenvalues) / total_variance
            num_components = np.argmax(cumulative_variance >= explained_variance) + 1
        else:
            num_components = psd_matrix.shape[0]

        selected_eigenvalues = np.diag(eigenvalues[:num_components])
        selected_eigenvectors = eigenvectors[:, :num_components]
        z = np.random.normal(size=(n_samples, num_components))
        return (z @ np.sqrt(selected_eigenvalues) @ selected_eigenvectors.T) + mean_vector

n_samples = 25000
results = {}

for cov_matrix in cov_matrices:
    direct_samples = simulate_multivariate_normal(cov_matrix, n_samples)
    results['Direct'] = np.cov(direct_samples, rowvar=False)
    print(f"Covariance matrix from direct sampling:\n{results['Direct']}")
    
    for explained_variance in [1.0, 0.75, 0.5]:
        pca_samples = simulate_multivariate_normal(cov_matrix, n_samples, pca=True, explained_variance=explained_variance)
        results[f'PCA {explained_variance*100}%'] = np.cov(pca_samples, rowvar=False)
        print(f"Covariance matrix from PCA with {explained_variance*100}% explained variance:\n{results[f'PCA {explained_variance*100}%']}")

frobenius_norms = {}
for key, simulated_cov in results.items():
    frobenius_norm = np.linalg.norm(simulated_cov - cov_matrix, 'fro')
    frobenius_norms[key] = frobenius_norm
<<<<<<< HEAD
    print(f"Frobenius Norm for {key}: {frobenius_norm}")
=======
    print(f"Frobenius Norm for {key}: {frobenius_norm}")
>>>>>>> 965aa1e250202eef2aeae259f2281b25ece2bf10
