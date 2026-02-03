import numpy as np

from .pca_pls import pca


def rv(X1, X2, ncomp=None, center=True, tol=1e-06, maxiter=1000):
    """Compute the RV coefficient between two datasets using NA-safe PCA.

    Args:
        X1: First data block (n_samples, n_features1).
        X2: Second data block (n_samples, n_features2).
        ncomp: Number of PCA components to use (defaults to min available dimensions).
        center: Whether to center both blocks before PCA.
        tol: Tolerance used when computing each PCA.
        maxiter: Maximum iterations for each PCA component.

    Returns:
        RV similarity in [0, 1], computed as the cosine of the Gram matrices.
    Note:
        The Gram inner products come from NIPALS outputs with MCAR-based scaling so RV stays comparable across missingness patterns.
    Example:
        >>> import numpy as np
        >>> from missing_methods import rv
        >>> X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9]])
        >>> Y = np.array([[2.4, 2.9], [0.6, 0.5], [2.1, 2.2]])
        >>> X[1, 0] = np.nan
        >>> Y[2, 0] = np.nan
        >>> float(rv(X, Y))
        0.7327...
    """
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
    """Compute the RV2 coefficient by zeroing the Gram diagonal.

    Args:
        X1: First data block with shape (n_samples, n_features1).
        X2: Second data block with shape (n_samples, n_features2).
        ncomp: Number of PCA components to use; defaults to min-dimension.
        center: Whether to center both matrices before PCA.
        tol: Convergence tolerance for the PCA routine.
        maxiter: Maximum iterations per component.

    Returns:
        RV2 similarity, computed after removing diagonal contributions from each Gram matrix.
    Note:
        The Gram matrices are built from scaled NIPALS scores assuming MCAR so diagonal-free similarities stay neutral.
    Example:
        >>> import numpy as np
        >>> from missing_methods import rv2
        >>> X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9]])
        >>> Y = np.array([[2.4, 2.9], [0.6, 0.5], [2.1, 2.2]])
        >>> X[1, 0] = np.nan
        >>> Y[2, 0] = np.nan
        >>> float(rv2(X, Y))
        0.6730...
    """
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
    """Build the symmetric RV matrix for a list of datasets.

    Args:
        arrays: Sequence of data blocks with matching row counts.
        **rv_kwargs: Passed through to `rv` for shared options.

    Returns:
        Symmetric RV similarity matrix of shape (n_blocks, n_blocks).
    Note:
        Each entry reuses the scaled RV helper, so the MCAR scaling assumption applies to the whole matrix.
    Example:
        >>> import numpy as np
        >>> from missing_methods import rv_list
        >>> X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9]])
        >>> Y = np.array([[2.4, 2.9], [0.6, 0.5], [2.1, 2.2]])
        >>> X[1, 0] = np.nan
        >>> Y[2, 0] = np.nan
        >>> arrays = [X, Y]
        >>> rv_list(arrays).shape
        (2, 2)
    """
    n = len(arrays)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            val = rv(arrays[i], arrays[j], **rv_kwargs)
            mat[i, j] = val
            mat[j, i] = val
    return mat


def rv2_list(arrays, **rv_kwargs):
    """Build the symmetric RV2 matrix for a list of datasets.

    Args:
        arrays: Sequence of data blocks with equal sample counts.
        **rv_kwargs: Passed through to `rv2`.

    Returns:
        Symmetric RV2 similarity matrix of shape (n_blocks, n_blocks).
    Note:
        Entries reuse the RV2 helper, preserving the MCAR-scaled inner-product geometry.
    Example:
        >>> import numpy as np
        >>> from missing_methods import rv2_list
        >>> X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9]])
        >>> Y = np.array([[2.4, 2.9], [0.6, 0.5], [2.1, 2.2]])
        >>> X[1, 0] = np.nan
        >>> Y[2, 0] = np.nan
        >>> arrays = [X, Y]
        >>> rv2_list(arrays).shape
        (2, 2)
    """
    n = len(arrays)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            val = rv2(arrays[i], arrays[j], **rv_kwargs)
            mat[i, j] = val
            mat[j, i] = val
    return mat
