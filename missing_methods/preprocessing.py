import numpy as np

from .nan_utils import _normalize, _standardize


def normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize each column of the input array to have unit norm, handling NaNs.

    Args:
        arr: 2-D array with numerical data, possibly containing NaNs.
    Returns:
        Normalized array with the same shape as input.

    Example:
        >>> import numpy as np
        >>> from missing_methods import normalize
        >>> X = np.array([[3.0, np.nan], [0.0, 4.0]])
        >>> normalized = normalize(X)
        >>> normalized.shape
        (2, 2)
        >>> float(np.nanmax(normalized))
        1.0
    """
    arr = np.asarray(arr, dtype=float)
    return _normalize(arr)


def standardize(arr: np.ndarray) -> np.ndarray:
    """Standardize each column of the input array to have zero mean and unit variance, handling NaNs.

    Args:
        arr: 2-D array with numerical data, possibly containing NaNs.
    Returns:
        Standardized array with the same shape as input.

    Example:
        >>> import numpy as np
        >>> from missing_methods import standardize
        >>> X = np.array([[1.0, 2.0], [np.nan, 4.0]])
        >>> standardized = standardize(X)
        >>> np.nanmean(standardized, axis=0)
        array([0., 0.])
    """
    arr = np.asarray(arr, dtype=float)
    return _standardize(arr)


__all__ = ["normalize", "standardize"]