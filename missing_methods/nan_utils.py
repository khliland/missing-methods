import numpy as np


def _safe_matvec(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return np.nansum(mat * vec[np.newaxis, :], axis=1)


def _safe_crossprod(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return np.nansum(mat * vec[:, np.newaxis], axis=0)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.sqrt(np.nansum(vec * vec))
    return vec / norm if norm > np.finfo(float).eps else vec
