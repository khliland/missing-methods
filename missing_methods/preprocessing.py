import numpy as np
from .nan_utils import _normalize, _standardize

def normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize each column of the input array to have unit norm, handling NaNs.

    Args:
        arr: 2-D array with numerical data, possibly containing NaNs.
    Returns:
        Normalized array with the same shape as input.
    """
    arr = np.asarray(arr, dtype=float)
    return _normalize(arr)

def standardize(arr: np.ndarray) -> np.ndarray:
    """Standardize each column of the input array to have zero mean and unit variance, handling NaNs.

    Args:
        arr: 2-D array with numerical data, possibly containing NaNs.
    Returns:
        Standardized array with the same shape as input.
    """
    arr = np.asarray(arr, dtype=float)
    return _standardize(arr)