import numpy as np

from .nan_utils import _normalize, _standardize, _validate_sample_weight


def _as_2d(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr[:, np.newaxis], True
    if arr.ndim != 2:
        raise ValueError("arr must be 1-D or 2-D")
    return arr, False


def _restore_shape(arr2d: np.ndarray, was_1d: bool) -> np.ndarray:
    if was_1d:
        return arr2d[:, 0]
    return arr2d


def _weighted_mle_variance(values: np.ndarray, weights: np.ndarray) -> float:
    w_sum = float(np.sum(weights))
    if w_sum <= 0.0:
        return np.nan
    mean = float(np.sum(weights * values) / w_sum)
    centered = values - mean
    return float(np.sum(weights * centered * centered) / w_sum)


def _optimize_lambda(objective, lower: float = -2.0, upper: float = 2.0) -> float:
    left = float(lower)
    right = float(upper)
    best_lambda = 1.0
    best_value = -np.inf
    for grid_size in (81, 81, 81):
        grid = np.linspace(left, right, num=grid_size)
        scores = np.array([objective(lmbda) for lmbda in grid], dtype=float)
        idx = int(np.nanargmax(scores))
        if np.isfinite(scores[idx]) and scores[idx] > best_value:
            best_value = float(scores[idx])
            best_lambda = float(grid[idx])
        if idx == 0:
            left, right = float(grid[0]), float(grid[1])
        elif idx == grid_size - 1:
            left, right = float(grid[-2]), float(grid[-1])
        else:
            left, right = float(grid[idx - 1]), float(grid[idx + 1])
    return best_lambda


def _yeo_johnson_apply(values: np.ndarray, lmbda: float) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    pos = values >= 0
    if np.isclose(lmbda, 0.0):
        out[pos] = np.log1p(values[pos])
    else:
        out[pos] = (np.power(values[pos] + 1.0, lmbda) - 1.0) / lmbda

    neg = ~pos
    if np.isclose(lmbda, 2.0):
        out[neg] = -np.log1p(-values[neg])
    else:
        out[neg] = -((np.power(1.0 - values[neg], 2.0 - lmbda) - 1.0) / (2.0 - lmbda))
    return out


def _box_cox_apply(values: np.ndarray, lmbda: float) -> np.ndarray:
    if np.any(values <= 0.0):
        raise ValueError("Box-Cox requires strictly positive observed values")
    if np.isclose(lmbda, 0.0):
        return np.log(values)
    return (np.power(values, lmbda) - 1.0) / lmbda


def _fit_yeo_johnson_lambda(values: np.ndarray, weights: np.ndarray) -> float:
    eps = np.finfo(float).tiny
    pos = values >= 0
    neg = ~pos
    log_jac_pos = np.log1p(values[pos])
    log_jac_neg = np.log1p(-values[neg])
    w_pos = weights[pos]
    w_neg = weights[neg]
    w_total = float(np.sum(weights))

    def objective(lmbda: float) -> float:
        transformed = _yeo_johnson_apply(values, lmbda)
        var = _weighted_mle_variance(transformed, weights)
        if not np.isfinite(var) or var <= 0.0:
            return -np.inf
        log_jacobian = float((lmbda - 1.0) * np.sum(w_pos * log_jac_pos) + (1.0 - lmbda) * np.sum(w_neg * log_jac_neg))
        return -0.5 * w_total * np.log(var + eps) + log_jacobian

    return _optimize_lambda(objective)


def _fit_box_cox_lambda(values: np.ndarray, weights: np.ndarray) -> float:
    eps = np.finfo(float).tiny
    if np.any(values <= 0.0):
        raise ValueError("Box-Cox requires strictly positive observed values")
    log_values = np.log(values)
    w_total = float(np.sum(weights))

    def objective(lmbda: float) -> float:
        transformed = _box_cox_apply(values, lmbda)
        var = _weighted_mle_variance(transformed, weights)
        if not np.isfinite(var) or var <= 0.0:
            return -np.inf
        log_jacobian = float((lmbda - 1.0) * np.sum(weights * log_values))
        return -0.5 * w_total * np.log(var + eps) + log_jacobian

    return _optimize_lambda(objective)


def normalize(arr: np.ndarray, sample_weight=None) -> np.ndarray:
    """Normalize each column of the input array to have unit norm, handling NaNs.

    Args:
        arr: 2-D array with numerical data, possibly containing NaNs.
        sample_weight: Optional row weights with shape (n_samples,).
    Returns:
        Normalized array with the same shape as input.

    Example:
        >>> import numpy as np
        >>> from missing_methods import normalize
        >>> X = np.array([[3.0, np.nan], [0.0, 4.0]])
        >>> normalized = normalize(X)
        >>> weighted = normalize(X, sample_weight=np.array([1.0, 2.0]))
        >>> normalized.shape
        (2, 2)
        >>> weighted.shape
        (2, 2)
        >>> float(np.nanmax(normalized))
        1.0
    """
    arr = np.asarray(arr, dtype=float)
    return _normalize(arr, sample_weight=sample_weight)


def standardize(arr: np.ndarray, sample_weight=None) -> np.ndarray:
    """Standardize each column of the input array to have zero mean and unit variance, handling NaNs.

    Args:
        arr: 2-D array with numerical data, possibly containing NaNs.
        sample_weight: Optional row weights with shape (n_samples,).
    Returns:
        Standardized array with the same shape as input.

    Example:
        >>> import numpy as np
        >>> from missing_methods import standardize
        >>> X = np.array([[1.0, 2.0], [np.nan, 4.0]])
        >>> baseline = standardize(X)
        >>> np.nanmean(baseline, axis=0)
        array([0., 0.])
        >>> w = np.array([1.0, 2.0])
        >>> standardized = standardize(X, sample_weight=w)
        >>> round(float(np.average(standardized[:, 1], weights=w)), 7)
        0.0
    """
    arr = np.asarray(arr, dtype=float)
    return _standardize(arr, sample_weight=sample_weight)


def yeo_johnson(
    arr: np.ndarray,
    lambdas: np.ndarray | None = None,
    sample_weight=None,
    return_lambdas: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Apply a NaN-aware Yeo-Johnson transform column-wise.

    If `lambdas` is omitted, one lambda per column is estimated by maximizing
    the weighted Gaussian log-likelihood over observed entries.

    Args:
        arr: 1-D or 2-D numerical array, possibly with NaNs.
        lambdas: Optional shape-(n_features,) lambdas to reuse.
        sample_weight: Optional row weights with shape (n_samples,).
        return_lambdas: If True, return `(transformed, lambdas_used)`.
    Returns:
        Transformed array (same shape as input), and optionally lambdas.
    """
    arr2d, was_1d = _as_2d(arr)
    weights = _validate_sample_weight(sample_weight, arr2d.shape[0])
    n_features = arr2d.shape[1]
    if lambdas is None:
        lambda_arr = np.ones(n_features, dtype=float)
        for col in range(n_features):
            x_col = arr2d[:, col]
            observed = ~np.isnan(x_col)
            if not np.any(observed):
                lambda_arr[col] = 1.0
                continue
            x_obs = x_col[observed]
            w_obs = weights[observed]
            lambda_arr[col] = _fit_yeo_johnson_lambda(x_obs, w_obs)
    else:
        lambda_arr = np.asarray(lambdas, dtype=float).reshape(-1)
        if lambda_arr.size != n_features:
            raise ValueError("lambdas must have shape (n_features,)")

    transformed = arr2d.copy()
    for col in range(n_features):
        x_col = transformed[:, col]
        observed = ~np.isnan(x_col)
        if not np.any(observed):
            continue
        x_col[observed] = _yeo_johnson_apply(x_col[observed], float(lambda_arr[col]))

    restored = _restore_shape(transformed, was_1d)
    if return_lambdas:
        return restored, lambda_arr
    return restored


def box_cox(
    arr: np.ndarray,
    lambdas: np.ndarray | None = None,
    sample_weight=None,
    return_lambdas: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Apply a NaN-aware Box-Cox transform column-wise.

    If `lambdas` is omitted, one lambda per column is estimated by maximizing
    the weighted Gaussian log-likelihood over observed entries.

    Args:
        arr: 1-D or 2-D numerical array, possibly with NaNs.
        lambdas: Optional shape-(n_features,) lambdas to reuse.
        sample_weight: Optional row weights with shape (n_samples,).
        return_lambdas: If True, return `(transformed, lambdas_used)`.
    Returns:
        Transformed array (same shape as input), and optionally lambdas.
    """
    arr2d, was_1d = _as_2d(arr)
    weights = _validate_sample_weight(sample_weight, arr2d.shape[0])
    n_features = arr2d.shape[1]
    if lambdas is None:
        lambda_arr = np.ones(n_features, dtype=float)
        for col in range(n_features):
            x_col = arr2d[:, col]
            observed = ~np.isnan(x_col)
            if not np.any(observed):
                lambda_arr[col] = 1.0
                continue
            x_obs = x_col[observed]
            w_obs = weights[observed]
            lambda_arr[col] = _fit_box_cox_lambda(x_obs, w_obs)
    else:
        lambda_arr = np.asarray(lambdas, dtype=float).reshape(-1)
        if lambda_arr.size != n_features:
            raise ValueError("lambdas must have shape (n_features,)")

    transformed = arr2d.copy()
    for col in range(n_features):
        x_col = transformed[:, col]
        observed = ~np.isnan(x_col)
        if not np.any(observed):
            continue
        x_col[observed] = _box_cox_apply(x_col[observed], float(lambda_arr[col]))

    restored = _restore_shape(transformed, was_1d)
    if return_lambdas:
        return restored, lambda_arr
    return restored


__all__ = ["normalize", "standardize", "yeo_johnson", "box_cox"]