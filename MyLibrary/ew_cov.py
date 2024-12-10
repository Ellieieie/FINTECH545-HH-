import numpy as np
from expo_weights import expW


def ewCovar(x, λ):
    m, n = x.shape
    w = expW(m, λ)
    weighted_mean = w @ x
    centered = x - weighted_mean
    sqrt_w = np.sqrt(w).reshape(-1, 1)
    xm = sqrt_w * centered
    cov = xm.T @ xm
    return cov