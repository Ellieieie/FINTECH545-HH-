import numpy as np


# Function to calculate exponentially weighted covariance
def expW(m, lambd):
    w = np.empty(m)
    for i in range(m):
        w[i] = (1 - lambd) * lambd ** (m - i - 1)
    # Normalize weights to sum to 1
    w /= np.sum(w)
    return w

