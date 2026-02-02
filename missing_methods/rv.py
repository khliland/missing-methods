import numpy as np

from .pca_pls import pca


def rv(X1, X2, ncomp=None, center=True, tol=1e-06, maxiter=1000):
    """Compute the RV coefficient between two datasets using NA-safe PCA."""
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    if ncomp is None:
        ncomp = min([X1.shape[0] - 1, X1.shape[1], X2.shape[1]])
    out1 = pca(X1, ncomp, center=center, tol=tol, maxiter=maxiter)
    out2 = pca(X2, ncomp, center=center, tol=tol, maxiter=maxiter)

    Q1 = out1["scores"]
    Q2 = out2["scores"]
    C1 = Q1 @ Q1.T
    C2 = Q2 @ Q2.T
    numerator = np.trace(C1 @ C2)
    denom = np.sqrt(np.trace(C1 @ C1) * np.trace(C2 @ C2))
    return numerator / denom if denom > 0 else 0.0


def rv2(X1, X2, ncomp=None, center=True, tol=1e-06, maxiter=1000):
    """Compute the RV2 coefficient by zeroing the Gram diagonal."""
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    if ncomp is None:
        ncomp = min([X1.shape[0] - 1, X1.shape[1], X2.shape[1]])
    out1 = pca(X1, ncomp, center=center, tol=tol, maxiter=maxiter)
    out2 = pca(X2, ncomp, center=center, tol=tol, maxiter=maxiter)

    Q1 = out1["scores"]
    Q2 = out2["scores"]
    C1 = Q1 @ Q1.T
    C2 = Q2 @ Q2.T

    C1 = C1 - np.diag(np.diag(C1))
    C2 = C2 - np.diag(np.diag(C2))

    numerator = np.trace(C1 @ C2)
    denom = np.sqrt(np.trace(C1 @ C1) * np.trace(C2 @ C2))
    return numerator / denom if denom > 0 else 0.0


def rv_list(arrays, **rv_kwargs):
    """Build the symmetric RV matrix for a list of matrices."""
    n = len(arrays)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            val = rv(arrays[i], arrays[j], **rv_kwargs)
            mat[i, j] = val
            mat[j, i] = val
    return mat


def rv2_list(arrays, **rv_kwargs):
    """Build the symmetric RV2 matrix for a list of matrices."""
    n = len(arrays)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            val = rv2(arrays[i], arrays[j], **rv_kwargs)
            mat[i, j] = val
            mat[j, i] = val
    return mat
