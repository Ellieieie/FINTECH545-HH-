import numpy as np


def expW(m, λ):
    w = np.zeros(m)
    for i in range(m):
        w[i] = (1 - λ) * (λ ** (m - i - 1))
    w = w / w.sum()
    return w
