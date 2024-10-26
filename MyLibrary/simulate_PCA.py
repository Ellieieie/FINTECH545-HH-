import numpy as np
import pandas as pd
import time
from scipy.stats import norm, t
from simulate_Normal import simulateNormal
from scipy.linalg import sqrtm
from exp_w_cov import ewCovar

def simulate_pca(a, nsim, pctExp=1, mean=None, seed=1234):
    n = a.shape[0]

    # If mean is not provided, set it to a zero vector
    if mean is None:
        mean = np.zeros(n)
    else:
        mean = np.array(mean)
        if mean.shape[0] != n:
            raise ValueError(f"Mean length ({mean.shape[0]}) does not match matrix size ({n})")

    # Eigenvalue decomposition
    vals, vecs = np.linalg.eigh(a)
    vals = np.real(vals)
    vecs = np.real(vecs)

    # Flip the order to have eigenvalues and eigenvectors sorted from highest to lowest
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    # Calculate the total variance and find eigenvalues that contribute to the requested pctExp
    total_variance = np.sum(vals)
    posv = np.where(vals >= 1e-8)[0]

    if pctExp < 1:
        cumulative_variance = 0.0
        nval = 0
        for i in posv:
            cumulative_variance += vals[i] / total_variance
            nval += 1
            if cumulative_variance >= pctExp:
                break
        posv = posv[:nval]

    vals = vals[posv]
    vecs = vecs[:, posv]

    # Construct the factor loading matrix
    B = vecs @ np.diag(np.sqrt(vals))

    # Set random seed and generate random normal variables
    np.random.seed(seed)
    r = np.random.normal(0.0, 1.0, (len(vals), nsim))

    # Apply factor loading matrix and transpose
    out = (B @ r).T

    # Add mean vector to each simulation
    out += mean

    return out


# Sample data for rets and nm (replace with your actual data)
rets = pd.DataFrame({
    'AAPL': np.random.normal(0, 0.01, 1000),
    'META': np.random.normal(0, 0.01, 1000),
    'GOOG': np.random.normal(0, 0.01, 1000)
})
nm = ['AAPL', 'META', 'GOOG']  # Selected stocks

# Define covariance matrices
ewma_cov = ewCovar(np.array(rets[nm]), 0.97)
pearson_cov = np.cov(rets[nm], rowvar=False)
pearson_std = np.sqrt(np.diag(pearson_cov))
pearson_cor = np.corrcoef(rets[nm], rowvar=False)

ewma_std = np.sqrt(np.diag(ewma_cov))
ewma_cor = np.diag(1 / ewma_std) @ ewma_cov @ np.diag(1 / ewma_std)

# Dictionary of covariance matrices for different matrix types
matrixLookup = {
    "EWMA": ewma_cov,
    "EWMA_COR_PEARSON_STD": np.diag(pearson_std) @ ewma_cor @ np.diag(pearson_std),
    "PEARSON": pearson_cov,
    "PEARSON_COR_EWMA_STD": np.diag(ewma_std) @ pearson_cor @ np.diag(ewma_std),
}

# Simulation types and storage for results
simType = ["Full", "PCA=1", "PCA=0.75", "PCA=0.5"]
matrixType = ["EWMA", "EWMA_COR_PEARSON_STD", "PEARSON", "PEARSON_COR_EWMA_STD"]

# Initialize result storage
matrix = []
simulation = []
runtimes = []
norms = []

# Perform simulations
for sim in simType:
    for mat in matrixType:
        matrix.append(mat)
        simulation.append(sim)
        c = matrixLookup[mat]

        if sim == "Full":
            start = time.time()
            for _ in range(20):
                s = simulateNormal(25000, c)
            elapsed = (time.time() - start) / 20
        elif sim == "PCA=1":
            start = time.time()
            for _ in range(20):
                s = simulate_pca(c, 25000, pctExp=1)
            elapsed = (time.time() - start) / 20
        elif sim == "PCA=0.75":
            start = time.time()
            for _ in range(20):
                s = simulate_pca(c, 25000, pctExp=0.75)
            elapsed = (time.time() - start) / 20
        else:  # PCA=0.5
            start = time.time()
            for _ in range(20):
                s = simulate_pca(c, 25000, pctExp=0.5)
            elapsed = (time.time() - start) / 20

        # Calculate covariance and error norms
        covar = np.cov(s, rowvar=False)
        runtime = elapsed
        norm = np.sum((covar - c) ** 2)

        # Store results
        runtimes.append(runtime)
        norms.append(norm)

# Compile results into a DataFrame
outTable = pd.DataFrame({
    "Matrix": matrix,
    "Simulation": simulation,
    "Runtime": runtimes,
    "Norm": norms
})

print(outTable)

