import numpy as np


def _rescale_by_proportion(values, counts, total):
    counts_arr = np.asarray(counts, dtype=float)
    total_arr = np.asarray(total, dtype=float)
    factors = np.zeros_like(counts_arr, dtype=float)
    mask = counts_arr > 0
    if np.shape(total_arr) != np.shape(counts_arr):
        total_broadcast = np.broadcast_to(total_arr, counts_arr.shape)
    else:
        total_broadcast = total_arr
    factors[mask] = total_broadcast[mask] / counts_arr[mask]
    return values * factors


def _safe_matvec(mat: np.ndarray, vec: np.ndarray, scale: bool = True) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    vec = np.asarray(vec, dtype=float)
    if mat.shape[1] != vec.size:
        raise ValueError("mat and vec must align in the second dimension")
    # Multiply only where both mat values and vec entries are observed.
    mask = ~np.isnan(mat) & ~np.isnan(vec[np.newaxis, :])
    prod = np.where(mask, mat * vec[np.newaxis, :], 0.0)
    sums = np.nansum(prod, axis=1)
    if not scale:
        return sums
    counts = mask.sum(axis=1)
    return _rescale_by_proportion(sums, counts, mat.shape[1])


def _safe_crossprod(mat: np.ndarray, vec: np.ndarray, scale: bool = True) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    vec = np.asarray(vec, dtype=float)
    if mat.shape[0] != vec.size:
        raise ValueError("mat and vec must align in the first dimension")
    # Similarly guard cross-products by observed overlaps before scaling.
    mask = ~np.isnan(mat) & ~np.isnan(vec[:, np.newaxis])
    prod = np.where(mask, mat * vec[:, np.newaxis], 0.0)
    sums = np.nansum(prod, axis=0)
    if not scale:
        return sums
    counts = mask.sum(axis=0)
    return _rescale_by_proportion(sums, counts, mat.shape[0])


def _scaled_sumsq(arr: np.ndarray, axis=None, scale: bool = True) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    sq = arr * arr
    sums = np.nansum(sq, axis=axis)
    if not scale:
        return sums
    mask = ~np.isnan(arr)
    counts = mask.sum(axis=axis)
    total = arr.size if axis is None else arr.shape[axis]
    # Use the observed proportion to normalize each sum-of-squares.
    return _rescale_by_proportion(sums, counts, total)


def _normalize(vec: np.ndarray, scale: bool = True) -> np.ndarray:
    eps = np.finfo(float).eps
    arr = np.asarray(vec, dtype=float)
    if arr.ndim == 1:
        sumsq = _scaled_sumsq(arr, axis=None, scale=scale)
        norm = np.sqrt(sumsq)
        if norm <= eps:
            return arr
        return arr / norm
    norms = np.sqrt(_scaled_sumsq(arr, axis=0, scale=scale))
    safe_norms = np.where(norms > eps, norms, 1.0)
    return arr / safe_norms[np.newaxis, :]


def _standardize(arr: np.ndarray, scale: bool = True) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        mean = np.nanmean(arr)
        residual = arr - mean
        mask = ~np.isnan(arr)
        count = int(mask.sum())
        denom = count - 1 if count > 1 else 1
        base_sumsq = np.nansum(residual * residual)
        sums = _rescale_by_proportion(base_sumsq, count, arr.size) if scale else base_sumsq
        var = sums / denom
        std = np.sqrt(var) if var > 0 else 0.0
        if std == 0 or np.isnan(std):
            std = 1.0
        return residual / std
    means = np.nanmean(arr, axis=0, keepdims=True)
    residuals = arr - means
    # Compute variance per column using the scaled sums before standardizing.
    sumsq = _scaled_sumsq(residuals, axis=0, scale=scale)
    counts = (~np.isnan(arr)).sum(axis=0)
    denom = np.where(counts > 1, counts - 1, 1.0)
    variances = sumsq / denom
    stds = np.sqrt(variances)
    stds = np.where(np.isfinite(stds) & (stds != 0), stds, 1.0)
    return residuals / stds


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if not np.any(mask):
        return 0.0
    x = x[mask]
    y = y[mask]
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = np.sqrt(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered))
    return np.sum(x_centered * y_centered) / denom if denom > 0 else 0.0
