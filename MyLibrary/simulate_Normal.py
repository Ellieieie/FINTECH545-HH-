import numpy as np


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


def simulateNormal(N, cov, mean=None, seed=1234):
    # Error checking
    n, m = cov.shape
    if n != m:
        raise ValueError(f"Covariance matrix is not square ({n}, {m})")

    # Output array for simulated data
    out = np.empty((n, N))

    # If mean is not provided, set it to a zero vector
    if mean is None:
        mean = np.zeros(n)
    else:
        mean = np.array(mean)
        if mean.shape[0] != n:
            raise ValueError(f"Mean length ({mean.shape[0]}) does not match covariance matrix size ({n})")

    # Cholesky decomposition to get the root of the covariance matrix
    l = np.zeros((n, n))
    chol_psd(l, cov) 

    # Set random seed and generate standard normal variables
    np.random.seed(seed)
    out = np.random.normal(0.0, 1.0, (n, N))

    # Apply the Cholesky root to the standard normals
    out = (l @ out).T

    # Add the mean to each simulation
    out += mean

    return out