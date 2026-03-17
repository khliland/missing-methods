import numpy as np


def _validate_sample_weight(sample_weight, n_samples: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n_samples, dtype=float)
    weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    if weights.size != n_samples:
        raise ValueError("sample_weight must have shape (n_samples,)")
    if np.any(weights < 0):
        raise ValueError("sample_weight cannot contain negative values")
    if np.sum(weights) <= 0:
        raise ValueError("sample_weight must sum to a positive value")
    return weights


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


def _safe_crossprod(
    mat: np.ndarray,
    vec: np.ndarray,
    scale: bool = True,
    sample_weight=None,
) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    vec = np.asarray(vec, dtype=float)
    if mat.shape[0] != vec.size:
        raise ValueError("mat and vec must align in the first dimension")
    # Similarly guard cross-products by observed overlaps before scaling.
    mask = ~np.isnan(mat) & ~np.isnan(vec[:, np.newaxis])
    weights = _validate_sample_weight(sample_weight, mat.shape[0])
    prod = np.where(mask, mat * vec[:, np.newaxis], 0.0)
    prod = prod * weights[:, np.newaxis]
    sums = np.nansum(prod, axis=0)
    if not scale:
        return sums
    weighted_counts = np.nansum(mask * weights[:, np.newaxis], axis=0)
    total = np.sum(weights)
    return _rescale_by_proportion(sums, weighted_counts, total)


def _scaled_sumsq(arr: np.ndarray, axis=None, scale: bool = True, sample_weight=None) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    sq = arr * arr
    mask = ~np.isnan(arr)
    if sample_weight is None:
        sums = np.nansum(sq, axis=axis)
        if not scale:
            return sums
        counts = mask.sum(axis=axis)
        total = arr.size if axis is None else arr.shape[axis]
        # Use the observed proportion to normalize each sum-of-squares.
        return _rescale_by_proportion(sums, counts, total)

    if arr.ndim == 1:
        weights = _validate_sample_weight(sample_weight, arr.shape[0])
        weighted_sq = np.where(mask, sq * weights, 0.0)
        sums = np.nansum(weighted_sq)
        if not scale:
            return sums
        counts = np.nansum(mask * weights)
        total = np.sum(weights)
        return _rescale_by_proportion(sums, counts, total)

    if axis != 0:
        raise ValueError("sample_weight is currently supported for axis=None (1-D) or axis=0")

    weights = _validate_sample_weight(sample_weight, arr.shape[0])
    weighted_sq = np.where(mask, sq * weights[:, np.newaxis], 0.0)
    sums = np.nansum(weighted_sq, axis=0)
    if not scale:
        return sums
    counts = np.nansum(mask * weights[:, np.newaxis], axis=0)
    total = np.sum(weights)
    return _rescale_by_proportion(sums, counts, total)


def _normalize(vec: np.ndarray, scale: bool = True, sample_weight=None) -> np.ndarray:
    eps = np.finfo(float).eps
    arr = np.asarray(vec, dtype=float)
    if arr.ndim == 1:
        sumsq = _scaled_sumsq(arr, axis=None, scale=scale, sample_weight=sample_weight)
        norm = np.sqrt(sumsq)
        if norm <= eps:
            return arr
        return arr / norm
    norms = np.sqrt(_scaled_sumsq(arr, axis=0, scale=scale, sample_weight=sample_weight))
    safe_norms = np.where(norms > eps, norms, 1.0)
    return arr / safe_norms[np.newaxis, :]


def _standardize(arr: np.ndarray, scale: bool = True, sample_weight=None) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        if sample_weight is None:
            mean = np.nanmean(arr)
            count = int((~np.isnan(arr)).sum())
        else:
            weights = _validate_sample_weight(sample_weight, arr.shape[0])
            mask = ~np.isnan(arr)
            total_w = np.nansum(weights * mask)
            mean = np.nansum(np.where(mask, arr * weights, 0.0)) / total_w if total_w > 0 else 0.0
            count = total_w
        residual = arr - mean
        mask = ~np.isnan(arr)
        denom = count - 1 if count > 1 else 1
        if sample_weight is None:
            base_sumsq = np.nansum(residual * residual)
            sums = _rescale_by_proportion(base_sumsq, count, arr.size) if scale else base_sumsq
        else:
            base_sumsq = np.nansum(np.where(mask, residual * residual * weights, 0.0))
            total = np.sum(weights)
            sums = _rescale_by_proportion(base_sumsq, count, total) if scale else base_sumsq
        var = sums / denom
        std = np.sqrt(var) if var > 0 else 0.0
        if std == 0 or np.isnan(std):
            std = 1.0
        return residual / std
    if sample_weight is None:
        means = np.nanmean(arr, axis=0, keepdims=True)
        counts = (~np.isnan(arr)).sum(axis=0)
    else:
        weights = _validate_sample_weight(sample_weight, arr.shape[0])
        mask = ~np.isnan(arr)
        weighted_counts = np.nansum(mask * weights[:, np.newaxis], axis=0)
        weighted_sums = np.nansum(np.where(mask, arr * weights[:, np.newaxis], 0.0), axis=0)
        means = np.divide(
            weighted_sums,
            weighted_counts,
            out=np.zeros_like(weighted_sums, dtype=float),
            where=weighted_counts > 0,
        )[np.newaxis, :]
        counts = weighted_counts
    residuals = arr - means
    # Compute variance per column using the scaled sums before standardizing.
    sumsq = _scaled_sumsq(residuals, axis=0, scale=scale, sample_weight=sample_weight)
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
