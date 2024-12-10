import numpy as np
import pandas as pd
import time

def cholesky_psd(corr_data):
    corr_data = pd.DataFrame(corr_data)
    dim = len(corr_data)
    chol_mat = pd.DataFrame(0, index=range(dim), columns=range(dim))
    for i in range(dim):
        for j in range(i + 1):
            if i == j:
                chol_mat.iloc[i, j] = np.sqrt(corr_data.iloc[i, j] - np.sum(np.square(chol_mat.iloc[i, :j])))
            else:
                chol_mat.iloc[i, j] = (corr_data.iloc[i, j] - np.dot(chol_mat.iloc[i, :j], chol_mat.iloc[j, :j])) / chol_mat.iloc[j, j]
    return chol_mat.values

def higham_psd(mat, tolerance=1e-8, max_iter=100):
    adjusted = mat.copy()
    correction = np.zeros_like(mat)
    for _ in range(max_iter):
        residual = adjusted - correction
        sym_part = project_symmetric(residual)
        correction = sym_part - residual
        adjusted = project_psd(sym_part)
        if np.linalg.norm(adjusted - sym_part, ord='fro') < tolerance:
            break
    return adjusted

def project_symmetric(m):
    return (m + m.T) / 2

def project_psd(m):
    vals, vecs = np.linalg.eigh(m)
    vals[vals < 0] = 0
    return vecs @ np.diag(vals) @ vecs.T

def near_psd(mat, epsilon=0.0):
    dim = mat.shape[0]
    out_mat = np.copy(mat)
    inv_scale = None

    if not np.allclose(np.diag(out_mat), np.ones(dim)):
        inv_scale = np.diag(1.0 / np.sqrt(np.diag(out_mat)))
        out_mat = inv_scale @ out_mat @ inv_scale

    vals, vecs = np.linalg.eigh(out_mat)
    vals = np.maximum(vals, epsilon)
    scale_vec = 1.0 / np.sqrt(np.dot(vecs**2, vals))
    scale_mat = np.diag(scale_vec)
    l_mat = np.diag(np.sqrt(vals))
    b_mat = scale_mat @ vecs @ l_mat
    out_mat = b_mat @ b_mat.T

    if inv_scale is not None:
        inv_scale = np.diag(1.0 / np.diag(inv_scale))
        out_mat = inv_scale @ out_mat @ inv_scale

    return out_mat

dimension = 500
corr_mat = np.full((dimension, dimension), 0.9)
np.fill_diagonal(corr_mat, 1)
corr_mat[0, 1] = 0.7357
corr_mat[1, 0] = 0.7357

start = time.time()
near_corrected = near_psd(corr_mat)
elapsed = time.time() - start
vals = np.linalg.eigvals(near_corrected)
print(f'The matrix corrected by near_psd is PSD: {np.all(vals >= 0)}')
print(f'Runtime for near_psd: {elapsed:.6f} seconds')

start = time.time()
higham_corrected = higham_psd(corr_mat)
elapsed = time.time() - start
vals = np.linalg.eigvals(higham_corrected)
print(f'The matrix corrected by higham_psd is PSD: {np.all(vals >= 0)}')
print(f'Runtime for higham_psd: {elapsed:.6f} seconds')

dist_higham = np.linalg.norm(higham_corrected - corr_mat, ord='fro')
dist_near = np.linalg.norm(near_corrected - corr_mat, ord='fro')
print(f'Frobenius norm (higham_psd): {dist_higham:.6f}, Frobenius norm (near_psd): {dist_near:.6f}')
