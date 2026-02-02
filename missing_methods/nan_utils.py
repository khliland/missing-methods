import numpy as np


def _safe_matvec(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return np.nansum(mat * vec[np.newaxis, :], axis=1)


def _safe_crossprod(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return np.nansum(mat * vec[:, np.newaxis], axis=0)


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    eps = np.finfo(float).eps
    if arr.ndim == 1:
        norm = np.sqrt(np.nansum(arr * arr))
        return arr / norm if norm > eps else arr
    norms = np.sqrt(np.nansum(arr * arr, axis=0))
    norms_safe = np.where(norms > eps, norms, 1.0)
    return arr / norms_safe[np.newaxis, :]


def _standardize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    means = np.nanmean(arr, axis=0, keepdims=True)
    stds = np.nanstd(arr, axis=0, ddof=1, keepdims=True)
    stds = np.where(stds == 0, 1.0, stds)
    return (arr - means) / stds


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
