"""Linear Discriminant Analysis with optional PCA-based missing value imputation."""

from __future__ import annotations

import numpy as np

from .impute import pca_impute, _impute_with_loadings
from .nan_utils import _validate_sample_weight
from .plotting import get_class_colors


def _validate_targets(y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError("y must be a 1-D array")
    if arr.size == 0:
        raise ValueError("y must not be empty")
    if np.issubdtype(arr.dtype, np.number) and np.any(np.isnan(arr.astype(float, copy=False))):
        raise ValueError("y cannot contain NaN values")
    classes, encoded = np.unique(arr, return_inverse=True)
    if classes.size < 2:
        raise ValueError("LDA requires at least two classes")
    return arr, classes, encoded


def _validate_prior(prior, class_weight_sums: np.ndarray) -> np.ndarray:
    if prior is None:
        total = float(np.sum(class_weight_sums))
        if total <= 0:
            return np.full(class_weight_sums.size, 1.0 / class_weight_sums.size)
        return class_weight_sums / total

    prior_arr = np.asarray(prior, dtype=float).reshape(-1)
    if prior_arr.size != class_weight_sums.size:
        raise ValueError("prior must have one value per class")
    if np.any(prior_arr < 0):
        raise ValueError("prior cannot contain negative values")
    total = float(np.sum(prior_arr))
    if total <= 0:
        raise ValueError("prior must sum to a positive value")
    return prior_arr / total


def _logsumexp(arr: np.ndarray, axis: int = 1) -> np.ndarray:
    arr_max = np.max(arr, axis=axis, keepdims=True)
    stabilized = arr - arr_max
    return (arr_max + np.log(np.sum(np.exp(stabilized), axis=axis, keepdims=True))).squeeze(axis)


def _compute_discriminants(
    X: np.ndarray,
    means: np.ndarray,
    precision: np.ndarray,
    log_prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    precision_means = np.linalg.solve(precision, means.T).T
    intercept = -0.5 * np.sum(means * precision_means, axis=1) + log_prior
    scores = X @ precision_means.T + intercept
    return scores, precision_means


def _prepare_predictors(
    X,
    *,
    impute_ncomp,
    impute_preprocessing,
    pca_tol,
    pca_maxiter,
    sample_weight,
):
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2-D array")
    result = pca_impute(
        X_arr,
        ncomp=impute_ncomp,
        preprocessing=impute_preprocessing,
        tol=pca_tol,
        maxiter=pca_maxiter,
        sample_weight=sample_weight,
    )
    return X_arr, result


def _unpack_lda_inputs(lda_result_or_means, covariance, prior, classes):
    if isinstance(lda_result_or_means, dict):
        if covariance is not None or prior is not None:
            raise ValueError(
                "When passing an lda result dictionary, covariance and prior must not be provided"
            )
        required_keys = ("means", "covariance", "prior")
        missing_keys = [key for key in required_keys if key not in lda_result_or_means]
        if missing_keys:
            raise ValueError(
                "lda result dictionary is missing required keys: " + ", ".join(missing_keys)
            )
        means = np.asarray(lda_result_or_means["means"], dtype=float)
        covariance = np.asarray(lda_result_or_means["covariance"], dtype=float)
        prior = np.asarray(lda_result_or_means["prior"], dtype=float)
        if classes is None and "classes" in lda_result_or_means:
            classes = lda_result_or_means["classes"]
    else:
        if covariance is None or prior is None:
            raise TypeError(
                "Expected either an lda result dictionary or means, covariance, and prior"
            )
        means = np.asarray(lda_result_or_means, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        prior = np.asarray(prior, dtype=float)

    if classes is None:
        classes = np.arange(means.shape[0])
    return means, covariance, prior, np.asarray(classes)


def lda(
    X,
    y,
    *,
    X_new=None,
    prior=None,
    regularization=1e-06,
    impute_ncomp=0.9,
    impute_preprocessing="standardize",
    pca_tol=1e-06,
    pca_maxiter=1000,
    sample_weight=None,
):
    """Fit multiclass Linear Discriminant Analysis (LDA).

    Missing values in predictors are handled by first applying ``pca_impute``.
    The classifier then estimates class means and a shared covariance matrix,
    and evaluates analytical LDA discriminant functions.

    Args:
        X: Predictor matrix with shape ``(n_samples, n_features)``.
        y: Class labels with shape ``(n_samples,)``.
        X_new: Optional matrix to classify using the fitted model.
        prior: Optional class prior probabilities of shape ``(n_classes,)``.
            If omitted, weighted empirical class frequencies are used.
        regularization: Non-negative ridge term added to the shared covariance
            diagonal for numerical stability.
        impute_ncomp: PCA imputation component control. ``0 < value < 1`` means
            explained-variance threshold, ``value >= 1`` means exact component count.
        impute_preprocessing: One of ``"standardize"``, ``"center"``, ``"none"``.
        pca_tol: Convergence tolerance for the PCA imputation stage.
        pca_maxiter: Maximum PCA iterations per component.
        sample_weight: Optional non-negative row weights.

    Returns:
        Dictionary containing model parameters and fitted predictions on training
        data. If ``X_new`` is supplied, predictions for new data are included.
    """
    if regularization < 0:
        raise ValueError("regularization must be non-negative")

    y_arr, classes, encoded = _validate_targets(y)
    X_arr, impute_result = _prepare_predictors(
        X,
        impute_ncomp=impute_ncomp,
        impute_preprocessing=impute_preprocessing,
        pca_tol=pca_tol,
        pca_maxiter=pca_maxiter,
        sample_weight=sample_weight,
    )

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must contain the same number of rows")

    weights = _validate_sample_weight(sample_weight, X_arr.shape[0])
    X_filled = impute_result["filled_X"]

    n_classes = classes.size
    n_features = X_filled.shape[1]

    class_weight_sums = np.zeros(n_classes, dtype=float)
    means = np.zeros((n_classes, n_features), dtype=float)
    for k in range(n_classes):
        mask = encoded == k
        wk = weights[mask]
        sw = float(np.sum(wk))
        class_weight_sums[k] = sw
        if sw <= 0:
            raise ValueError("Each class must have a positive total sample weight")
        means[k] = np.sum(X_filled[mask] * wk[:, np.newaxis], axis=0) / sw

    prior_arr = _validate_prior(prior, class_weight_sums)

    centered = X_filled - means[encoded]
    weighted_centered = centered * np.sqrt(weights)[:, np.newaxis]
    scatter = weighted_centered.T @ weighted_centered
    denom = np.sum(weights) - n_classes
    if denom <= 0:
        denom = max(float(np.sum(weights)), 1.0)
    covariance = scatter / denom
    if regularization > 0:
        covariance = covariance.copy()
        covariance.flat[:: covariance.shape[0] + 1] += regularization

    log_prior = np.log(np.clip(prior_arr, 1e-300, None))
    scores, precision_means = _compute_discriminants(X_filled, means, covariance, log_prior)
    log_norm = _logsumexp(scores, axis=1)
    posterior = np.exp(scores - log_norm[:, np.newaxis])
    fitted_idx = np.argmax(scores, axis=1)

    result = {
        "classes": classes,
        "prior": prior_arr,
        "means": means,
        "covariance": covariance,
        "precision_means": precision_means,
        "scores": scores,
        "posterior": posterior,
        "fitted": classes[fitted_idx],
        "impute_ncomp": impute_result["ncomp"],
        "impute_preprocessing": impute_result["preprocessing"],
        "impute_means": impute_result["means"],
        "impute_scales": impute_result["scales"],
        "impute_loadings": impute_result["loadings"],
        "pca_result": impute_result["pca_result"],
        "filled_X": X_filled,
    }

    if X_new is not None:
        X_new_arr = np.asarray(X_new, dtype=float)
        if X_new_arr.ndim != 2:
            raise ValueError("X_new must be a 2-D array")
        if X_new_arr.shape[1] != n_features:
            raise ValueError("X_new must have the same number of features as X")
        X_new_filled = _impute_with_loadings(
            X_new_arr,
            result["impute_means"],
            result["impute_scales"],
            result["impute_loadings"],
        )
        new_scores = X_new_filled @ precision_means.T + (
            -0.5 * np.sum(means * precision_means, axis=1) + log_prior
        )
        new_log_norm = _logsumexp(new_scores, axis=1)
        result["new_scores"] = new_scores
        result["new_posterior"] = np.exp(new_scores - new_log_norm[:, np.newaxis])
        result["new_fitted"] = classes[np.argmax(new_scores, axis=1)]
        result["new_filled_X"] = X_new_filled

    return result


def lda_pairwise_boundaries(
    lda_result_or_means,
    covariance: np.ndarray | None = None,
    prior: np.ndarray | None = None,
    classes=None,
):
    """Return exact pairwise LDA boundaries as ``normal.T @ x + offset = 0``."""
    means, covariance, prior, classes = _unpack_lda_inputs(
        lda_result_or_means,
        covariance,
        prior,
        classes,
    )

    if means.ndim != 2:
        raise ValueError("means must have shape (n_classes, n_features)")
    if covariance.shape != (means.shape[1], means.shape[1]):
        raise ValueError("covariance must have shape (n_features, n_features)")
    if prior.shape != (means.shape[0],):
        raise ValueError("prior must have shape (n_classes,)")

    precision_means = np.linalg.solve(covariance, means.T).T
    intercept = -0.5 * np.sum(means * precision_means, axis=1) + np.log(np.clip(prior, 1e-300, None))

    boundaries = []
    for i in range(means.shape[0] - 1):
        for j in range(i + 1, means.shape[0]):
            boundaries.append(
                {
                    "class_i": classes[i],
                    "class_j": classes[j],
                    "normal": precision_means[i] - precision_means[j],
                    "offset": intercept[i] - intercept[j],
                }
            )
    return boundaries


def plot_lda_boundaries(boundaries, *, xlim=None, ylim=None, ax=None, line_kwargs=None):
    """Plot analytical pairwise boundaries for 2-D LDA models.

    This helper only draws exact boundary lines and does not use grid-based
    region approximation.
    """
    try:
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plot_lda_boundaries requires matplotlib") from exc

    if ax is None:
        _, ax = plt.subplots()
    if line_kwargs is None:
        line_kwargs = {"color": "black", "linewidth": 1.0}

    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    for b in boundaries:
        normal = np.asarray(b["normal"], dtype=float)
        if normal.size != 2:
            raise ValueError("plot_lda_boundaries only supports 2-D boundaries")
        a1, a2 = normal
        c = float(b["offset"])

        if abs(a2) > 1e-12:
            x_vals = np.array([xlim[0], xlim[1]], dtype=float)
            y_vals = -(a1 * x_vals + c) / a2
            ax.plot(x_vals, y_vals, **line_kwargs)
        elif abs(a1) > 1e-12:
            x_val = -c / a1
            ax.plot([x_val, x_val], [ylim[0], ylim[1]], **line_kwargs)

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    return ax


def _intersect_interval(lo: float, hi: float, new_lo: float, new_hi: float) -> tuple[float, float]:
    return max(lo, new_lo), min(hi, new_hi)


def _line_box_t_interval(x0: np.ndarray, d: np.ndarray, xlim, ylim) -> tuple[float, float] | None:
    t_lo = -np.inf
    t_hi = np.inf

    for coord, delta, lo_b, hi_b in (
        (x0[0], d[0], float(xlim[0]), float(xlim[1])),
        (x0[1], d[1], float(ylim[0]), float(ylim[1])),
    ):
        if abs(delta) < 1e-14:
            if coord < lo_b or coord > hi_b:
                return None
            continue
        t1 = (lo_b - coord) / delta
        t2 = (hi_b - coord) / delta
        lo_i = min(t1, t2)
        hi_i = max(t1, t2)
        t_lo, t_hi = _intersect_interval(t_lo, t_hi, lo_i, hi_i)
        if t_lo > t_hi:
            return None
    return t_lo, t_hi


def plot_lda_boundary_segments(
    lda_result_or_means,
    covariance: np.ndarray | None = None,
    prior: np.ndarray | None = None,
    *,
    classes=None,
    xlim=None,
    ylim=None,
    ax=None,
    line_kwargs=None,
):
    """Plot only valid analytical LDA boundary segments for 2-D models.

    Unlike :func:`plot_lda_boundaries`, this helper clips each pairwise boundary
    to the region where the two involved classes are not dominated by any third
    class, matching the true decision boundary segments.
    """
    try:
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plot_lda_boundary_segments requires matplotlib") from exc

    means, covariance, prior, classes = _unpack_lda_inputs(
        lda_result_or_means,
        covariance,
        prior,
        classes,
    )
    if means.ndim != 2 or means.shape[1] != 2:
        raise ValueError("plot_lda_boundary_segments requires means with shape (n_classes, 2)")
    if covariance.shape != (2, 2):
        raise ValueError("plot_lda_boundary_segments requires a 2x2 covariance matrix")
    if prior.shape != (means.shape[0],):
        raise ValueError("prior must have one value per class")
    if classes.shape[0] != means.shape[0]:
        raise ValueError("classes must have one label per class mean")

    if ax is None:
        _, ax = plt.subplots()
    if line_kwargs is None:
        line_kwargs = {"color": "black", "linewidth": 1.2}

    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    w = np.linalg.solve(covariance, means.T).T
    b = -0.5 * np.sum(means * w, axis=1) + np.log(np.clip(prior, 1e-300, None))

    class_index = {cls: idx for idx, cls in enumerate(classes)}
    boundaries = lda_pairwise_boundaries(means, covariance, prior, classes=classes)

    for boundary in boundaries:
        i = class_index[boundary["class_i"]]
        j = class_index[boundary["class_j"]]
        normal = np.asarray(boundary["normal"], dtype=float)
        offset = float(boundary["offset"])
        norm2 = float(np.dot(normal, normal))
        if norm2 <= 1e-18:
            continue

        # Point on line normal.T @ x + offset = 0 and line direction.
        x0 = -offset * normal / norm2
        d = np.array([-normal[1], normal[0]], dtype=float)

        t_interval = _line_box_t_interval(x0, d, xlim, ylim)
        if t_interval is None:
            continue
        t_lo, t_hi = t_interval

        # Keep only t where class i/j are not dominated by any third class.
        for k in range(means.shape[0]):
            if k == i or k == j:
                continue
            alpha = float(np.dot(w[i] - w[k], d))
            beta = float(np.dot(w[i] - w[k], x0) + (b[i] - b[k]))
            if abs(alpha) < 1e-14:
                if beta < 0:
                    t_lo, t_hi = 1.0, 0.0
                    break
                continue
            t_cut = -beta / alpha
            if alpha > 0:
                t_lo, t_hi = _intersect_interval(t_lo, t_hi, t_cut, np.inf)
            else:
                t_lo, t_hi = _intersect_interval(t_lo, t_hi, -np.inf, t_cut)
            if t_lo > t_hi:
                break

        if t_lo > t_hi:
            continue

        p0 = x0 + t_lo * d
        p1 = x0 + t_hi * d
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], **line_kwargs)

    ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    return ax


def plot_lda_regions(
    lda_result_or_means,
    covariance: np.ndarray | None = None,
    prior: np.ndarray | None = None,
    *,
    classes=None,
    xlim=None,
    ylim=None,
    resolution=250,
    ax=None,
    alpha=0.2,
    draw_segments=True,
    line_kwargs=None,
    class_colors=None,
):
    """Plot 2-D LDA decision regions with optional analytical segment overlays.

    Region filling is grid-based (argmax of exact linear discriminants sampled
    over a grid). Boundary segments, when enabled, are drawn analytically via
    :func:`plot_lda_boundary_segments`. If ``class_colors`` is omitted, the
    region palette falls back to ``DEFAULT_CLASS_COLORS``.
    """
    try:
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")
        mcolors = importlib.import_module("matplotlib.colors")
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plot_lda_regions requires matplotlib") from exc

    means, covariance, prior, classes = _unpack_lda_inputs(
        lda_result_or_means,
        covariance,
        prior,
        classes,
    )

    if means.ndim != 2 or means.shape[1] != 2:
        raise ValueError("plot_lda_regions requires means with shape (n_classes, 2)")
    if covariance.shape != (2, 2):
        raise ValueError("plot_lda_regions requires a 2x2 covariance matrix")
    if prior.shape != (means.shape[0],):
        raise ValueError("prior must have one value per class")

    if ax is None:
        _, ax = plt.subplots()

    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    log_prior = np.log(np.clip(prior, 1e-300, None))
    w = np.linalg.solve(covariance, means.T).T
    b = -0.5 * np.sum(means * w, axis=1) + log_prior

    xs = np.linspace(float(xlim[0]), float(xlim[1]), int(resolution))
    ys = np.linspace(float(ylim[0]), float(ylim[1]), int(resolution))
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])

    scores = pts @ w.T + b
    winner = np.argmax(scores, axis=1).reshape(YY.shape)

    cmap = mcolors.ListedColormap(get_class_colors(means.shape[0], class_colors=class_colors))

    ax.pcolormesh(XX, YY, winner, shading="auto", cmap=cmap, alpha=alpha)

    if draw_segments:
        if line_kwargs is None:
            line_kwargs = {"color": "black", "linewidth": 1.2}
        plot_lda_boundary_segments(
            means,
            covariance,
            prior,
            classes=classes,
            xlim=xlim,
            ylim=ylim,
            ax=ax,
            line_kwargs=line_kwargs,
        )

    ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    return ax


__all__ = [
    "lda",
    "lda_pairwise_boundaries",
    "plot_lda_boundaries",
    "plot_lda_boundary_segments",
    "plot_lda_regions",
]
