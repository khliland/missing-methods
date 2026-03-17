import numpy as np

from .nan_utils import _normalize, _standardize


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


__all__ = ["normalize", "standardize"]