import numpy as np

from .pca_pls import pls


def pairwise_rbf(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float,
    enforce_diag: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the RBF kernel between two sets while respecting NaNs."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n_samples_x, n_features = X.shape
    n_samples_y = Y.shape[0]
    mask_x = ~np.isnan(X)
    mask_y = ~np.isnan(Y)
    gamma = float(gamma)
    denom = n_features if n_features > 0 else 1
    K = np.zeros((n_samples_x, n_samples_y), dtype=float)
    for i in range(n_samples_x):
        xi = X[i]
        mi = mask_x[i]
        for j in range(n_samples_y):
            common = mi & mask_y[j]
            common_count = int(common.sum())
            if common_count == 0:
                continue
            diff = xi[common] - Y[j, common]
            scaled_sq = np.dot(diff, diff) * (denom / common_count)
            coverage = common_count / denom
            K[i, j] = np.exp(-gamma * scaled_sq) * coverage
    if enforce_diag and n_samples_x == n_samples_y and np.shares_memory(X, Y):
        np.fill_diagonal(K, 1.0)
    coverage_x = mask_x.mean(axis=1)
    return K, coverage_x


def _rbf_kernel(X: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    return pairwise_rbf(X, X, gamma, enforce_diag=True)


def _coverage_transform(coverage: np.ndarray) -> np.ndarray:
    """Helper for computing the coverage-weighted centering transform."""
    n = coverage.shape[0]
    safe_coverage = np.where(coverage > 0, coverage, 1.0)
    inv = np.reciprocal(safe_coverage)
    return np.eye(n, dtype=float) - np.outer(inv, coverage)


def _coverage_center(K: np.ndarray, coverage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the coverage-weighted centering transform and return the matrix."""
    C = _coverage_transform(coverage)
    return C @ K @ C, C


def kernel_pls(
    X,
    Y,
    ncomp,
    gamma=None,
    center=True,
    tol=1e-06,
    maxiter=1000,
):
    """Fit kernel PLS using an RBF Gram matrix that ignores NaNs.

    Args:
        X: Predictor matrix with shape (n_samples, n_features).
        Y: Response matrix with shape (n_samples, n_targets).
        ncomp: Number of latent components.
        gamma: Inverse kernel width; defaults to 1 / n_features when not provided.
        center: Whether to center the Y block before fitting.
        tol: Convergence tolerance for alternating updates.
        maxiter: Maximum iterations per component.

    Returns:
        Dictionary matching the `pls` return value with extra keys for the kernel.

    Note:
        The RBF kernel is computed only on shared observations. Kernel centering uses the coverage-weighted transforms so rows with different observed proportions stay comparable.
    Example:
        >>> import numpy as np
        >>> from missing_methods import kernel_pls
        >>> X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9]])
        >>> Y = np.array([[2.4], [0.6], [2.1]])
        >>> X[1, 0] = np.nan
        >>> result = kernel_pls(X, Y, ncomp=2)
        >>> result["scores"].shape
        (3, 2)
    """
    X = np.asarray(X, dtype=float)
    if gamma is None:
        gamma = 1.0 / max(1, X.shape[1])
    K, coverage = _rbf_kernel(X, gamma)
    centered_kernel, coverage_matrix = _coverage_center(K, coverage)

    Y = np.asarray(Y, dtype=float)
    if center:
        means_y = np.nanmean(Y, axis=0)
        means_y[np.isnan(means_y)] = 0.0
    else:
        means_y = np.zeros(Y.shape[1], dtype=float)
    Y_centered = Y - means_y

    result = pls(
        centered_kernel,
        Y_centered,
        ncomp,
        center=False,
        tol=tol,
        maxiter=maxiter,
    )
    result["means_y"] = means_y
    result["kernel"] = centered_kernel
    result["kernel_center"] = coverage_matrix
    result["gamma"] = gamma
    result["coverage"] = coverage
    return result
