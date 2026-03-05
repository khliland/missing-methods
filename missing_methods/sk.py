"""Scikit-learn compatible wrappers around the missing-methods functions."""

from __future__ import annotations

import numpy as np
from typing import Literal

from .nan_utils import _safe_matvec, _normalize, _scaled_sumsq
from .pca_pls import pca as _pca, pls as _pls
from .kernel import kernel_pls as _kernel_pls, pairwise_rbf

try:
    from sklearn.base import BaseEstimator as _BaseEstimator
    from sklearn.base import RegressorMixin as _RegressorMixin
    from sklearn.base import TransformerMixin as _TransformerMixin
except ImportError:  # pragma: no cover
    class _BaseEstimator:
        def get_params(self, deep=True):
            return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}

        def set_params(self, **params):
            for key, value in params.items():
                setattr(self, key, value)
            return self

    class _TransformerMixin:
        def fit_transform(self, X, y=None, **fit_params):
            return self.fit(X, y, **fit_params).transform(X)

    class _RegressorMixin:
        def score(self, X, y, sample_weight=None):
            y_pred = self.predict(X)
            mask = ~np.isnan(y_pred) & ~np.isnan(y)
            if not np.any(mask):
                return np.nan
            residual = y[mask] - y_pred[mask]
            ss_res = np.nansum(residual ** 2)
            ss_tot = np.nansum((y[mask] - np.nanmean(y[mask])) ** 2)
            return 1 - ss_res / ss_tot if ss_tot > 0 else 1.0


class PCA(_BaseEstimator, _TransformerMixin):
    """Scikit-learn style PCA that wraps the NA-safe `pca()` helper.

    Args:
        n_components: Number of components to extract when fitting.
        center: Whether to center columns before fitting.
        tol: Convergence tolerance forwarded to the backend PCA.
        maxiter: Maximum iterations to run per component.
        ncomp: Legacy alias for `n_components`, added for parity with the functional helper.

    Attributes:
        components_: Loadings for the fitted model.
        scores_: Projected samples returned by the helper.
        explained_: Number of scaled sum-of-squares per component.
        means_: Column means used for centering.
        n_components_: Actual number of components extracted.
        n_features_in_: Number of features in the training data.

    Note:
        The estimator preserves the MCAR-scaled inner products from the helper so the returned scores/loadings stay comparable to the functional API.
    Example:
        >>> import numpy as np
        >>> from missing_methods.sk import PCA
        >>> X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9]])
        >>> X[1, 0] = np.nan
        >>> estimator = PCA(ncomp=2)
        >>> _ = estimator.fit(X)
        >>> estimator.transform(X).shape
        (3, 2)
    """

    def __init__(self, n_components=None, center=True, tol=1e-06, maxiter=1000, ncomp=None):
        if ncomp is not None:
            n_components = ncomp
        self.n_components = n_components
        self.center = center
        self.tol = tol
        self.maxiter = maxiter

    def fit(self, X, y=None, **fit_params):
        result = _pca(
            X,
            ncomp=self.n_components,
            center=self.center,
            tol=self.tol,
            maxiter=self.maxiter,
        )
        self.components_ = result["loadings"]
        self.scores_ = result["scores"]
        self.explained_ = result["explained"]
        self.means_ = result["means"]
        self.n_components_ = self.scores_.shape[1]
        self.n_features_in_ = self.components_.shape[0]
        return self

    def transform(self, X):
        if not hasattr(self, "means_"):
            raise ValueError("PCA instance is not fitted yet")
        X = np.asarray(X, dtype=float)
        Xc = X - self.means_
        scores = np.column_stack(
            [_safe_matvec(Xc, self.components_[:, j]) for j in range(self.n_components_)]
        )
        return scores

    def inverse_transform(self, X_scores):
        if not hasattr(self, "components_"):
            raise ValueError("PCA instance is not fitted yet")
        return np.dot(X_scores, self.components_.T) + self.means_


class PLSRegressor(_BaseEstimator, _RegressorMixin, _TransformerMixin):
    """Scikit-learn style PLS regressor that wraps the NA-safe `pls()` helper.

    Args:
        n_components: Number of latent components to extract.
        center: Whether the helper should center X and Y.
        tol: Convergence tolerance forwarded to the helper.
        maxiter: Maximum iterations to run per component.
        ncomp: Legacy alias for `n_components`, added for parity with the functional helper.

    Attributes:
        weights_: X weight vectors for each component.
        loadings_x_: Loadings for the X block.
        loadings_y_: Loadings for the Y block.
        means_x_: Column means of the X block.
        means_y_: Column means of the Y block.
        n_components_: Number of components computed.
        n_features_in_: Number of features in the training X.

    Note:
        The estimator keeps the same MCAR-scaled cross-products provided by the functional helper, so X and Y are balanced even if their missingness patterns differ.
    Example:
        >>> import numpy as np
        >>> from missing_methods.sk import PLSRegressor
        >>> X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9]])
        >>> Y = np.array([[2.4], [0.6], [2.1]])
        >>> X[2, 1] = np.nan
        >>> Y[0, 0] = np.nan
        >>> estimator = PLSRegressor(ncomp=2)
        >>> _ = estimator.fit(X, Y)
        >>> estimator.predict(X).shape
        (3, 1)
    """

    def __init__(self, n_components=2, center=True, tol=1e-06, maxiter=1000, ncomp=None):
        if ncomp is not None:
            n_components = ncomp
        self.n_components = n_components
        self.center = center
        self.tol = tol
        self.maxiter = maxiter

    def fit(self, X, Y, **fit_params):
        result = _pls(
            X,
            Y,
            ncomp=self.n_components,
            center=self.center,
            tol=self.tol,
            maxiter=self.maxiter,
        )
        self.weights_ = result["weights"]
        self.loadings_x_ = result["loadings_x"]
        self.loadings_y_ = result["loadings_y"]
        self.means_x_ = result["means_x"]
        self.means_y_ = result["means_y"]
        self.n_components_ = self.weights_.shape[1]
        self.n_features_in_ = self.weights_.shape[0]
        return self

    def transform(self, X):
        if not hasattr(self, "weights_"):
            raise ValueError("PLSRegressor instance is not fitted yet")
        X = np.asarray(X, dtype=float)
        residual = X - self.means_x_
        mask = ~np.isnan(residual)
        scores = np.zeros((X.shape[0], self.n_components_), dtype=float)
        for comp in range(self.n_components_):
            t = _safe_matvec(residual, self.weights_[:, comp])
            scores[:, comp] = t
            recon = np.outer(t, self.loadings_x_[:, comp])
            residual[mask] -= recon[mask]
        return scores

    def predict(self, X):
        scores = self.transform(X)
        y_pred = np.zeros((X.shape[0], self.loadings_y_.shape[0]), dtype=float)
        for comp in range(self.n_components_):
            y_pred += np.outer(scores[:, comp], self.loadings_y_[:, comp])
        return y_pred + self.means_y_


pca = PCA
pls = PLSRegressor

class Normalizer(_BaseEstimator, _TransformerMixin):
    """Transformer that calls the NA-safe normalize helper in a scikit-learn style.

    Example:
        >>> import numpy as np
        >>> from missing_methods.sk import Normalizer
        >>> X = np.array([[3.0, np.nan], [0.0, 4.0]])
        >>> normalizer = Normalizer()
        >>> normalized = normalizer.fit_transform(X)
        >>> normalized.shape
        (2, 2)
    """

    def __init__(self, norm: Literal["l2"] = "l2"):
        if norm != "l2":
            raise ValueError("only 'l2' norm is supported")
        self.norm = norm

    def fit(self, arr: np.ndarray, y=None):
        arr = np.asarray(arr, dtype=float)
        self.n_features_in_ = arr.shape[1] if arr.ndim > 1 else 1
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        return _normalize(arr)


class StandardScaler(_BaseEstimator, _TransformerMixin):
    """Transformer that standardizes columns while respecting missing values.

    Example:
        >>> import numpy as np
        >>> from missing_methods.sk import StandardScaler
        >>> X = np.array([[1.0, 2.0], [np.nan, 4.0]])
        >>> scaler = StandardScaler()
        >>> transformed = scaler.fit_transform(X)
        >>> reconstructed = scaler.inverse_transform(transformed)
        >>> np.allclose(reconstructed, X, equal_nan=True)
        True
    """

    def __init__(self, *, scale: bool = True):
        self.scale = scale

    def fit(self, arr: np.ndarray, y=None):
        arr = np.asarray(arr, dtype=float)
        self.n_features_in_ = arr.shape[1] if arr.ndim > 1 else 1
        self.means_ = np.nanmean(arr, axis=0)
        residuals = arr - self.means_
        sumsq = _scaled_sumsq(residuals, axis=0, scale=self.scale)
        counts = (~np.isnan(arr)).sum(axis=0)
        denom = np.where(counts > 1, counts - 1, 1.0)
        variances = sumsq / denom
        stds = np.sqrt(variances)
        stds = np.where(np.isfinite(stds) & (stds != 0), stds, 1.0)
        self.scale_ = stds
        self.var_ = variances
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        self._validate_fitted()
        arr = np.asarray(arr, dtype=float)
        return (arr - self.means_) / self.scale_

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        self._validate_fitted()
        arr = np.asarray(arr, dtype=float)
        return arr * self.scale_ + self.means_

    def _validate_fitted(self):
        if not hasattr(self, "scale_"):
            raise ValueError("StandardScaler instance is not fitted yet")


class KernelPLSRegressor(_BaseEstimator, _RegressorMixin):
    """Kernel PLS using the NaN-aware RBF kernel helper.

    Args:
        n_components: Number of latent components to compute.
        gamma: Inverse kernel width passed to `kernel_pls`.
        center: Whether to center the Y block before fitting.
        tol: Convergence tolerance forwarded to the helper.
        maxiter: Maximum iterations per component.
        ncomp: Legacy alias for `n_components`.

    Attributes:
        scores_: Kernel scores for the fitted data.
        loadings_y_: Y loadings produced by the helper.
        means_y_: Column means of the response.
        kernel_: Centered training kernel matrix.
        gamma_: Gamma value actually used.
        coverage_: Observation coverage per training sample.

    Note:
        Predictions and transformations reuse the NaN-aware kernel built on the training data.
    Example:
        >>> import numpy as np
        >>> from missing_methods.sk import KernelPLSRegressor
        >>> X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9]])
        >>> Y = np.array([[2.4], [0.6], [2.1]])
        >>> X[1, 0] = np.nan
        >>> estimator = KernelPLSRegressor(n_components=2)
        >>> _ = estimator.fit(X, Y)
        >>> estimator.predict(X).shape
        (3, 1)
    """

    def __init__(self, n_components=2, gamma=None, center=True, tol=1e-06, maxiter=1000, ncomp=None):
        if ncomp is not None:
            n_components = ncomp
        self.n_components = n_components
        self.gamma = gamma
        self.center = center
        self.tol = tol
        self.maxiter = maxiter

    def fit(self, X, Y, **fit_params):
        result = _kernel_pls(
            X,
            Y,
            ncomp=self.n_components,
            gamma=self.gamma,
            center=self.center,
            tol=self.tol,
            maxiter=self.maxiter,
        )
        self.scores_ = result["scores"]
        self.loadings_y_ = result["loadings_y"]
        self.means_y_ = result["means_y"]
        self.weights_ = result["weights"]
        self.loadings_x_ = result["loadings_x"]
        self.kernel_ = result["kernel"]
        self.gamma_ = result["gamma"]
        self.coverage_ = result["coverage"]
        self._kernel_center = result["kernel_center"]
        self.n_components_ = self.scores_.shape[1]
        self.n_features_in_ = X.shape[1]
        self._X_fit = np.asarray(X, dtype=float)
        return self

    def transform(self, X=None):
        if not hasattr(self, "scores_"):
            raise ValueError("KernelPLSRegressor instance is not fitted yet")
        if X is None:
            return self.scores_
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self._X_fit.shape[1]:
            raise ValueError("New data must have the same feature count as the fit data")
        k_new, _ = pairwise_rbf(X, self._X_fit, gamma=self.gamma_)
        k_centered = k_new @ self._kernel_center
        residual = k_centered.copy()
        scores = np.zeros((X.shape[0], self.n_components_), dtype=float)
        for comp in range(self.n_components_):
            w = self.weights_[:, comp]
            t_comp = residual @ w
            scores[:, comp] = t_comp
            p = self.loadings_x_[:, comp]
            residual -= np.outer(t_comp, p)
        return scores

    def predict(self, X):
        scores = self.transform(X)
        y_pred = scores @ self.loadings_y_.T
        return y_pred + self.means_y_


__all__ = ["PCA", "PLSRegressor", "KernelPLSRegressor", "Normalizer", "StandardScaler", "pca", "pls"]
