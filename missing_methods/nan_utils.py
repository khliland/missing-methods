import numpy as np


def _safe_matvec(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return np.nansum(mat * vec[np.newaxis, :], axis=1)


def _safe_crossprod(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return np.nansum(mat * vec[:, np.newaxis], axis=0)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.sqrt(np.nansum(vec * vec))
    return vec / norm if norm > np.finfo(float).eps else vec


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
