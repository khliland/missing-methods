"""PCA-based imputation for matrices with missing values (MCAR-aware)."""

import numpy as np

from .nan_utils import _scaled_sumsq, _validate_sample_weight
from .pca_pls import _weighted_column_mean, pca


def _weighted_column_scale(X: np.ndarray, sample_weight) -> np.ndarray:
    means = _weighted_column_mean(X, sample_weight=sample_weight)
    residuals = X - means
    mask = ~np.isnan(X)
    weights = _validate_sample_weight(sample_weight, X.shape[0])
    weighted_counts = np.nansum(mask * weights[:, np.newaxis], axis=0)
    sumsq = _scaled_sumsq(residuals, axis=0, sample_weight=weights)
    denom = np.where(weighted_counts > 1, weighted_counts - 1, 1.0)
    variances = sumsq / denom
    scales = np.sqrt(variances)
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
    return scales


def _preprocess_for_imputation(
    X: np.ndarray,
    mode: str,
    sample_weight,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "standardize":
        means = _weighted_column_mean(X, sample_weight=sample_weight)
        scales = _weighted_column_scale(X, sample_weight=sample_weight)
    elif mode == "center":
        means = _weighted_column_mean(X, sample_weight=sample_weight)
        scales = np.ones(X.shape[1], dtype=float)
    elif mode == "none":
        means = np.zeros(X.shape[1], dtype=float)
        scales = np.ones(X.shape[1], dtype=float)
    else:
        raise ValueError("preprocessing must be one of: 'standardize', 'center', 'none'")
    transformed = (X - means) / scales
    return transformed, means, scales


def _impute_with_loadings(
    X: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    loadings: np.ndarray,
) -> np.ndarray:
    """Fill NaN entries by projecting each row onto stored PCA loadings.

    For each row with missing values, solves for the score vector using only
    the observed features (least squares), then reconstructs the missing cells.
    Rows with no observed features are filled with the column means.
    """
    X_filled = X.copy()
    mask = np.isnan(X)
    if not mask.any():
        return X_filled
    for i in range(X.shape[0]):
        missing_idx = mask[i]
        if not missing_idx.any():
            continue
        obs_idx = ~missing_idx
        if obs_idx.sum() == 0:
            X_filled[i, missing_idx] = means[missing_idx]
            continue
        P_obs = loadings[obs_idx, :]
        x_obs = (X[i, obs_idx] - means[obs_idx]) / scales[obs_idx]
        t = np.linalg.lstsq(P_obs, x_obs, rcond=None)[0]
        X_filled[i, missing_idx] = (
            loadings[missing_idx, :] @ t * scales[missing_idx] + means[missing_idx]
        )
    return X_filled


def pca_impute(
    X,
    *,
    ncomp=0.9,
    preprocessing="standardize",
    tol=1e-06,
    maxiter=1000,
    sample_weight=None,
):
    """Fill NaN entries in X using low-rank PCA reconstruction.

    Fits a MCAR-aware NIPALS PCA on the observed entries, then replaces each
    missing cell with its low-rank reconstruction. Observed cells are never
    altered. The returned dictionary contains the fitted parameters needed to
    impute new data via :func:`_impute_with_loadings`.

    Args:
        X: 2-D array with shape (n_samples, n_features), possibly containing NaNs.
        ncomp: Controls how many PCA components are used.

            - **float in (0, 1)** — minimum number of components whose cumulative
              explained variance reaches at least this fraction of the total
              variance.  Default is ``0.9`` (90 %).
            - **int ≥ 1** — exact number of components.

        preprocessing: One of ``"standardize"`` (default), ``"center"``, or
            ``"none"``. Applied before PCA and inverted on the imputed values.
        tol: Convergence tolerance for NIPALS.
        maxiter: Maximum NIPALS iterations per component.
        sample_weight: Optional row weights with shape (n_samples,).

    Returns:
        Dictionary with:

        - ``filled_X``: Full matrix with NaN cells replaced by the PCA
          reconstruction; observed values are unchanged.
        - ``means``: Column means used for centering.
        - ``scales``: Column scales used for scaling.
        - ``loadings``: PCA loadings with shape (n_features, ncomp), sufficient
          to impute new data with :func:`_impute_with_loadings`.
        - ``pca_result``: Full output dict from the internal :func:`pca` call.
        - ``ncomp``: Number of components actually used.
        - ``preprocessing``: Preprocessing mode that was applied.

    Example:
        >>> import numpy as np
        >>> from missing_methods.impute import pca_impute
        >>> X = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
        >>> result = pca_impute(X, ncomp=1)
        >>> result["filled_X"].shape
        (3, 3)
        >>> np.isnan(result["filled_X"]).any()
        False
        >>> result_auto = pca_impute(X)
        >>> result_auto["ncomp"] >= 1
        True
        >>> w = np.array([1.0, 0.5, 1.5])
        >>> weighted = pca_impute(X, ncomp=1, sample_weight=w)
        >>> weighted["filled_X"].shape
        (3, 3)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array")
    row_weights = _validate_sample_weight(sample_weight, X.shape[0])

    transformed, means, scales = _preprocess_for_imputation(
        X, preprocessing, sample_weight=row_weights
    )
    max_rank = min(max(X.shape[0] - 1, 1), X.shape[1])

    use_threshold = 0.0 < ncomp < 1.0
    if use_threshold:
        n_run = max_rank
    else:
        n_run = min(int(ncomp), max_rank)
        if n_run <= 0:
            raise ValueError("ncomp must be a positive integer or a float in (0, 1)")

    pca_result = pca(
        transformed,
        ncomp=n_run,
        center=False,
        tol=tol,
        maxiter=maxiter,
        sample_weight=row_weights,
    )

    if use_threshold:
        explained = pca_result["explained"]
        total = float(explained.sum())
        if total > 0:
            cumvar = np.cumsum(explained) / total
            hits = np.where(cumvar >= ncomp)[0]
            actual_ncomp = int(hits[0]) + 1 if hits.size > 0 else n_run
        else:
            actual_ncomp = 1
    else:
        actual_ncomp = n_run

    loadings = pca_result["loadings"][:, :actual_ncomp]
    scores = pca_result["scores"][:, :actual_ncomp]
    reconstructed = (scores @ loadings.T) * scales + means
    X_filled = np.where(np.isnan(X), reconstructed, X)

    return {
        "filled_X": X_filled,
        "means": means,
        "scales": scales,
        "loadings": loadings,
        "pca_result": pca_result,
        "ncomp": actual_ncomp,
        "preprocessing": preprocessing,
    }


__all__ = ["pca_impute"]
