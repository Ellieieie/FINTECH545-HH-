import numpy as np
import time


def PCA_pctExplained(a):
    n = a.shape[0]

    # Get Eigenvalues and sort in descending order
    vals = np.linalg.eigvalsh(a)[::-1]  # Hermitian or symmetric matrix eigenvalues sorted in descending order

    # Total of Eigenvalues
    total = np.sum(vals)

    # Calculate cumulative explained variance
    out = np.empty(n)
    s = 0.0
    for i in range(n):
        s += vals[i]
        out[i] = s / total  # cumulative percentage of the total

    return out

# Function to make matrix nearly PSD with option for regularization epsilon
def near_psd(a, epsilon=0.0):
    n = a.shape[0]
    invSD = None
    out = a.copy()

    # Calculate the correlation matrix if we have a covariance matrix
    if np.count_nonzero(np.isclose(np.diag(out), 1.0)) != n:
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD

    # SVD and update the eigenvalues and scale
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = np.diag(1.0 / (np.sum(vecs ** 2 * vals, axis=1) ** 0.5))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T

    # Add back the variance
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = invSD @ out @ invSD

    return out


# Cholesky decomposition assuming PSD
def chol_psd(root, a):
    n = a.shape[0]
    root.fill(0.0)
    for j in range(n):
        s = 0.0
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])
        
        temp = a[j, j] - s
        if 0 >= temp >= -1e-8:
            temp = 0.0
        root[j, j] = np.sqrt(temp)

        if root[j, j] == 0.0:
            root[j, j+1:] = 0.0
        else:
            ir = 1.0 / root[j, j]
            for i in range(j+1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir

# Helper functions
def _getAplus(A):
    vals, vecs = np.linalg.eigh(A)
    vals = np.diag(np.maximum(vals, 0))
    return vecs @ vals @ vecs.T

def _getPS(A, W):
    W05 = np.sqrt(W)
    iW = np.linalg.inv(W05)
    return iW @ _getAplus(W05 @ A @ W05) @ iW

def _getPu(A, W):
    Aret = A.copy()
    np.fill_diagonal(Aret, 1.0)
    return Aret

def wgtNorm(A, W):
    W05 = np.sqrt(W)
    W05 = W05 @ A @ W05
    return np.sum(W05 ** 2)

# Higham's nearest PSD matrix
def higham_nearestPSD(pc, W=None, epsilon=1e-9, maxIter=100, tol=1e-9):
    n = pc.shape[0]
    if W is None:
        W = np.diag(np.ones(n))
    
    Yk = pc.copy()
    deltaS = np.zeros_like(pc)
    norml = np.finfo(np.float64).max
    i = 1

    while i <= maxIter:
        Rk = Yk - deltaS

        # PS Update
        Xk = _getPS(Rk, W)
        deltaS = Xk - Rk

        # Pu Update
        Yk = _getPu(Xk, W)

        # Norm calculation
        norm = wgtNorm(Yk - pc, W)

        # Smallest Eigenvalue check
        minEigVal = np.min(np.real(np.linalg.eigvals(Yk)))

        if abs(norm - norml) < tol and minEigVal > -epsilon:
            break

        norml = norm
        i += 1

    if i < maxIter:
        print(f"Converged in {i} iterations.")
    else:
        print(f"Convergence failed after {i - 1} iterations")

    return Yk
