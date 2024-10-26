import numpy as np
from expo_weights import expW


def ewCovar(x, lambd):
    m, n = x.shape

    # Calculate the weights
    w = expW(m, lambd)

    # Remove the weighted mean from the series and add the weights to the covariance calculation
    xm = np.sqrt(w)[:, np.newaxis] * (x - np.dot(w, x))

    # Calculate covariance
    covariance = np.dot(xm.T, xm)
    return covariance