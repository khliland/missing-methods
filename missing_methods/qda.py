"""Quadratic Discriminant Analysis with optional PCA-based missing-value imputation."""

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
        raise ValueError("QDA requires at least two classes")
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


def _unpack_qda_inputs(qda_result_or_means, covariances, prior, classes):
    if isinstance(qda_result_or_means, dict):
        if covariances is not None or prior is not None:
            raise ValueError(
                "When passing a qda result dictionary, covariances and prior must not be provided"
            )
        required_keys = ("means", "covariances", "prior")
        missing_keys = [key for key in required_keys if key not in qda_result_or_means]
        if missing_keys:
            raise ValueError(
                "qda result dictionary is missing required keys: " + ", ".join(missing_keys)
            )
        means = np.asarray(qda_result_or_means["means"], dtype=float)
        covariances = np.asarray(qda_result_or_means["covariances"], dtype=float)
        prior = np.asarray(qda_result_or_means["prior"], dtype=float)
        if classes is None and "classes" in qda_result_or_means:
            classes = qda_result_or_means["classes"]
    else:
        if covariances is None or prior is None:
            raise TypeError(
                "Expected either a qda result dictionary or means, covariances, and prior"
            )
        means = np.asarray(qda_result_or_means, dtype=float)
        covariances = np.asarray(covariances, dtype=float)
        prior = np.asarray(prior, dtype=float)

    if classes is None:
        classes = np.arange(means.shape[0])
    return means, covariances, prior, np.asarray(classes)


def _stable_covariance_inverse(cov: np.ndarray, regularization: float) -> tuple[np.ndarray, float, np.ndarray]:
    p = cov.shape[0]
    eye = np.eye(p, dtype=float)
    jitter = max(regularization, 0.0)
    for _ in range(6):
        cov_try = cov + jitter * eye if jitter > 0 else cov
        sign, logdet = np.linalg.slogdet(cov_try)
        if sign > 0:
            try:
                precision = np.linalg.solve(cov_try, eye)
            except np.linalg.LinAlgError:
                precision = np.linalg.lstsq(cov_try, eye, rcond=None)[0]
            return precision, float(logdet), cov_try
        jitter = max(1e-10, 10.0 * (jitter if jitter > 0 else 1e-10))

    raise np.linalg.LinAlgError("Unable to obtain positive-definite covariance matrix for QDA")


def _qda_scores_from_params(
    X: np.ndarray,
    means: np.ndarray,
    precisions: np.ndarray,
    log_dets: np.ndarray,
    log_prior: np.ndarray,
) -> np.ndarray:
    n = X.shape[0]
    m = means.shape[0]
    scores = np.empty((n, m), dtype=float)
    for k in range(m):
        xc = X - means[k]
        quad = np.sum((xc @ precisions[k]) * xc, axis=1)
        scores[:, k] = -0.5 * quad - 0.5 * log_dets[k] + log_prior[k]
    return scores


def qda(
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
    """Fit multiclass Quadratic Discriminant Analysis (QDA).

    Missing values in predictors are handled by first applying ``pca_impute``.
    QDA then estimates one covariance matrix per class and evaluates classwise
    quadratic discriminants.
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
    covariances = np.zeros((n_classes, n_features, n_features), dtype=float)
    precisions = np.zeros((n_classes, n_features, n_features), dtype=float)
    log_dets = np.zeros(n_classes, dtype=float)

    for k in range(n_classes):
        mask = encoded == k
        wk = weights[mask]
        Xk = X_filled[mask]
        sw = float(np.sum(wk))
        class_weight_sums[k] = sw
        if sw <= 0:
            raise ValueError("Each class must have a positive total sample weight")

        mu = np.sum(Xk * wk[:, np.newaxis], axis=0) / sw
        means[k] = mu

        xc = Xk - mu
        scatter = (wk[:, np.newaxis] * xc).T @ xc
        denom = sw - 1.0 if sw > 1.0 else 1.0
        cov = scatter / denom
        prec, logdet, cov_reg = _stable_covariance_inverse(cov, regularization)
        covariances[k] = cov_reg
        precisions[k] = prec
        log_dets[k] = logdet

    prior_arr = _validate_prior(prior, class_weight_sums)
    log_prior = np.log(np.clip(prior_arr, 1e-300, None))

    scores = _qda_scores_from_params(X_filled, means, precisions, log_dets, log_prior)
    log_norm = _logsumexp(scores, axis=1)
    posterior = np.exp(scores - log_norm[:, np.newaxis])
    fitted_idx = np.argmax(scores, axis=1)

    result = {
        "classes": classes,
        "prior": prior_arr,
        "means": means,
        "covariances": covariances,
        "precisions": precisions,
        "log_dets": log_dets,
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
        new_scores = _qda_scores_from_params(X_new_filled, means, precisions, log_dets, log_prior)
        new_log_norm = _logsumexp(new_scores, axis=1)
        result["new_scores"] = new_scores
        result["new_posterior"] = np.exp(new_scores - new_log_norm[:, np.newaxis])
        result["new_fitted"] = classes[np.argmax(new_scores, axis=1)]
        result["new_filled_X"] = X_new_filled

    return result


def qda_pairwise_conics(
    qda_result_or_means,
    covariances: np.ndarray | None = None,
    prior: np.ndarray | None = None,
    classes=None,
):
    """Return pairwise QDA conics as ``x.T @ A @ x + b.T @ x + c = 0``."""
    means, covariances, prior, classes = _unpack_qda_inputs(
        qda_result_or_means,
        covariances,
        prior,
        classes,
    )

    if means.ndim != 2:
        raise ValueError("means must have shape (n_classes, n_features)")
    if covariances.shape != (means.shape[0], means.shape[1], means.shape[1]):
        raise ValueError("covariances must have shape (n_classes, n_features, n_features)")
    if prior.shape != (means.shape[0],):
        raise ValueError("prior must have shape (n_classes,)")

    n_classes = means.shape[0]
    precisions = np.zeros_like(covariances)
    log_dets = np.zeros(n_classes, dtype=float)
    for k in range(n_classes):
        sign, logdet = np.linalg.slogdet(covariances[k])
        if sign <= 0:
            raise ValueError("Each covariance matrix must be positive definite")
        log_dets[k] = logdet
        try:
            precisions[k] = np.linalg.solve(covariances[k], np.eye(covariances.shape[1], dtype=float))
        except np.linalg.LinAlgError:
            precisions[k] = np.linalg.lstsq(covariances[k], np.eye(covariances.shape[1], dtype=float), rcond=None)[0]

    conics = []
    for i in range(n_classes - 1):
        for j in range(i + 1, n_classes):
            Pi = precisions[i]
            Pj = precisions[j]
            mui = means[i]
            muj = means[j]

            A = -0.5 * (Pi - Pj)
            b = Pi @ mui - Pj @ muj
            c = (
                -0.5 * (mui @ Pi @ mui - muj @ Pj @ muj)
                - 0.5 * (log_dets[i] - log_dets[j])
                + np.log(np.clip(prior[i], 1e-300, None))
                - np.log(np.clip(prior[j], 1e-300, None))
            )

            conics.append(
                {
                    "class_i": classes[i],
                    "class_j": classes[j],
                    "A": A,
                    "b": b,
                    "c": float(c),
                }
            )
    return conics


def _conic_value(x: np.ndarray, A: np.ndarray, b: np.ndarray, c: float) -> np.ndarray:
    xv = np.asarray(x, dtype=float)
    return np.einsum("...i,ij,...j->...", xv, A, xv) + xv @ b + float(c)


def _trace_conic_roots_x(
    A: np.ndarray,
    b: np.ndarray,
    c: float,
    *,
    xlim,
    ylim,
    n_samples: int,
    eps: float = 1e-12,
) -> list[np.ndarray]:
    a_xx = float(A[0, 0])
    a_xy = float(A[0, 1] + A[1, 0])
    a_yy = float(A[1, 1])
    d_x = float(b[0])
    d_y = float(b[1])
    f0 = float(c)

    xs = np.linspace(float(xlim[0]), float(xlim[1]), int(n_samples))
    branches = [[], []]

    for x in xs:
        qa = a_yy
        qb = a_xy * x + d_y
        qc = a_xx * x * x + d_x * x + f0

        roots: list[float] = []
        if abs(qa) <= eps:
            if abs(qb) > eps:
                roots = [(-qc / qb)]
        else:
            disc = qb * qb - 4.0 * qa * qc
            if disc >= -eps:
                disc = max(disc, 0.0)
                sqrt_disc = np.sqrt(disc)
                roots = [(-qb - sqrt_disc) / (2.0 * qa), (-qb + sqrt_disc) / (2.0 * qa)]
                roots.sort()

        if len(roots) == 1:
            y = roots[0]
            if float(ylim[0]) <= y <= float(ylim[1]):
                branches[0].append([x, y])
                branches[1].append([np.nan, np.nan])
            else:
                branches[0].append([np.nan, np.nan])
                branches[1].append([np.nan, np.nan])
            continue

        if len(roots) == 2:
            y0, y1 = roots
            if float(ylim[0]) <= y0 <= float(ylim[1]):
                branches[0].append([x, y0])
            else:
                branches[0].append([np.nan, np.nan])
            if float(ylim[0]) <= y1 <= float(ylim[1]):
                branches[1].append([x, y1])
            else:
                branches[1].append([np.nan, np.nan])
            continue

        branches[0].append([np.nan, np.nan])
        branches[1].append([np.nan, np.nan])

    out = []
    for branch in branches:
        arr = np.asarray(branch, dtype=float)
        if np.any(np.isfinite(arr[:, 0])):
            out.append(arr)
    return out


def _trace_conic_roots_y(
    A: np.ndarray,
    b: np.ndarray,
    c: float,
    *,
    xlim,
    ylim,
    n_samples: int,
    eps: float = 1e-12,
) -> list[np.ndarray]:
    a_xx = float(A[0, 0])
    a_xy = float(A[0, 1] + A[1, 0])
    a_yy = float(A[1, 1])
    d_x = float(b[0])
    d_y = float(b[1])
    f0 = float(c)

    ys = np.linspace(float(ylim[0]), float(ylim[1]), int(n_samples))
    branches = [[], []]

    for y in ys:
        qa = a_xx
        qb = a_xy * y + d_x
        qc = a_yy * y * y + d_y * y + f0

        roots: list[float] = []
        if abs(qa) <= eps:
            if abs(qb) > eps:
                roots = [(-qc / qb)]
        else:
            disc = qb * qb - 4.0 * qa * qc
            if disc >= -eps:
                disc = max(disc, 0.0)
                sqrt_disc = np.sqrt(disc)
                roots = [(-qb - sqrt_disc) / (2.0 * qa), (-qb + sqrt_disc) / (2.0 * qa)]
                roots.sort()

        if len(roots) == 1:
            x = roots[0]
            if float(xlim[0]) <= x <= float(xlim[1]):
                branches[0].append([x, y])
                branches[1].append([np.nan, np.nan])
            else:
                branches[0].append([np.nan, np.nan])
                branches[1].append([np.nan, np.nan])
            continue

        if len(roots) == 2:
            x0, x1 = roots
            if float(xlim[0]) <= x0 <= float(xlim[1]):
                branches[0].append([x0, y])
            else:
                branches[0].append([np.nan, np.nan])
            if float(xlim[0]) <= x1 <= float(xlim[1]):
                branches[1].append([x1, y])
            else:
                branches[1].append([np.nan, np.nan])
            continue

        branches[0].append([np.nan, np.nan])
        branches[1].append([np.nan, np.nan])

    out = []
    for branch in branches:
        arr = np.asarray(branch, dtype=float)
        if np.any(np.isfinite(arr[:, 0])):
            out.append(arr)
    return out


def _extract_segments(pts: np.ndarray, *, min_points: int = 2) -> list[np.ndarray]:
    finite = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    out: list[np.ndarray] = []
    start = None
    n = pts.shape[0]
    for i in range(n + 1):
        is_finite = i < n and finite[i]
        if is_finite and start is None:
            start = i
            continue
        if (not is_finite) and start is not None:
            if i - start >= min_points:
                out.append(pts[start:i].copy())
            start = None
    return out


def _stitch_segments(segments: list[np.ndarray], *, tol: float) -> list[np.ndarray]:
    if not segments:
        return []

    stitched = [seg.copy() for seg in segments]
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(stitched):
            j = i + 1
            while j < len(stitched):
                a = stitched[i]
                b = stitched[j]
                a0 = a[0]
                a1 = a[-1]
                b0 = b[0]
                b1 = b[-1]

                if np.linalg.norm(a1 - b0) <= tol:
                    stitched[i] = np.vstack([a, b])
                elif np.linalg.norm(a1 - b1) <= tol:
                    stitched[i] = np.vstack([a, b[::-1]])
                elif np.linalg.norm(a0 - b0) <= tol:
                    stitched[i] = np.vstack([a[::-1], b])
                elif np.linalg.norm(a0 - b1) <= tol:
                    stitched[i] = np.vstack([a[::-1], b[::-1]])
                else:
                    j += 1
                    continue

                stitched.pop(j)
                changed = True
                break
            if changed:
                break
            i += 1
    return stitched


def _plot_segmented_line(ax, pts: np.ndarray, *, line_kwargs):
    finite = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    if not np.any(finite):
        return

    start = None
    n = pts.shape[0]
    for i in range(n + 1):
        is_finite = i < n and finite[i]
        if is_finite and start is None:
            start = i
            continue
        if (not is_finite) and start is not None:
            if i - start >= 2:
                seg = pts[start:i]
                ax.plot(seg[:, 0], seg[:, 1], **line_kwargs)
            start = None


def _filter_branch_by_qda_dominance(
    branch: np.ndarray,
    *,
    means: np.ndarray,
    precisions: np.ndarray,
    log_dets: np.ndarray,
    log_prior: np.ndarray,
    class_i: int,
    class_j: int,
    dominance_tol: float,
) -> np.ndarray:
    finite = np.isfinite(branch[:, 0]) & np.isfinite(branch[:, 1])
    if not np.any(finite):
        return branch

    pts = branch.copy()
    scores = _qda_scores_from_params(pts[finite], means, precisions, log_dets, log_prior)
    pair_diff = np.abs(scores[:, class_i] - scores[:, class_j])

    n_classes = means.shape[0]
    if n_classes > 2:
        others = [k for k in range(n_classes) if k not in (class_i, class_j)]
        pair_min = np.minimum(scores[:, class_i], scores[:, class_j])
        other_max = np.max(scores[:, others], axis=1)
        keep = (pair_diff <= 1e-5) & ((pair_min - other_max) >= -float(dominance_tol))
    else:
        keep = pair_diff <= 1e-5

    finite_idx = np.flatnonzero(finite)
    drop_idx = finite_idx[~keep]
    pts[drop_idx] = np.array([np.nan, np.nan])
    return pts


def plot_qda_boundary_segments(
    qda_result_or_means,
    covariances: np.ndarray | None = None,
    prior: np.ndarray | None = None,
    *,
    classes=None,
    xlim=None,
    ylim=None,
    n_samples=1200,
    ax=None,
    line_kwargs=None,
    dominance_tol=1e-07,
    stitch_tolerance=None,
):
    """Plot pairwise QDA boundaries as analytically traced curve segments.

    This helper traces conic points by solving the pairwise QDA conic equation
    along one axis and filtering to points where no third class dominates.
    Unlike contour-based rendering, this does not evaluate a 2D contour grid for
    boundary overlays. To improve robustness for near-vertical branches, roots are
    traced along both x and y directions and nearby fragments can be stitched.
    """
    try:
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plot_qda_boundary_segments requires matplotlib") from exc

    means, covariances, prior, classes = _unpack_qda_inputs(
        qda_result_or_means,
        covariances,
        prior,
        classes,
    )

    if means.ndim != 2 or means.shape[1] != 2:
        raise ValueError("plot_qda_boundary_segments requires means with shape (n_classes, 2)")
    if covariances.shape != (means.shape[0], 2, 2):
        raise ValueError("plot_qda_boundary_segments requires covariances with shape (n_classes, 2, 2)")
    if prior.shape != (means.shape[0],):
        raise ValueError("prior must have shape (n_classes,)")
    if int(n_samples) < 100:
        raise ValueError("n_samples must be at least 100")

    if ax is None:
        _, ax = plt.subplots()

    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    n_classes = means.shape[0]
    precisions = np.zeros_like(covariances)
    log_dets = np.zeros(n_classes, dtype=float)
    for k in range(n_classes):
        sign, logdet = np.linalg.slogdet(covariances[k])
        if sign <= 0:
            raise ValueError("Each covariance matrix must be positive definite")
        log_dets[k] = logdet
        try:
            precisions[k] = np.linalg.solve(covariances[k], np.eye(2, dtype=float))
        except np.linalg.LinAlgError:
            precisions[k] = np.linalg.lstsq(covariances[k], np.eye(2, dtype=float), rcond=None)[0]

    log_prior = np.log(np.clip(prior, 1e-300, None))
    conics = qda_pairwise_conics(means, covariances, prior, classes=classes)
    class_index = {cls: idx for idx, cls in enumerate(classes)}

    if line_kwargs is None:
        line_kwargs = {"color": "black", "linewidth": 1.0}

    if stitch_tolerance is None:
        x_span = float(xlim[1]) - float(xlim[0])
        y_span = float(ylim[1]) - float(ylim[0])
        stitch_tolerance = 0.5 * np.hypot(x_span, y_span) / float(n_samples)
    stitch_tolerance = float(stitch_tolerance)

    for conic in conics:
        i = class_index[conic["class_i"]]
        j = class_index[conic["class_j"]]

        branches_x = _trace_conic_roots_x(
            conic["A"],
            conic["b"],
            conic["c"],
            xlim=xlim,
            ylim=ylim,
            n_samples=int(n_samples),
        )
        branches_y = _trace_conic_roots_y(
            conic["A"],
            conic["b"],
            conic["c"],
            xlim=xlim,
            ylim=ylim,
            n_samples=int(n_samples),
        )

        candidate_segments: list[np.ndarray] = []

        for branch in branches_x + branches_y:
            filtered = _filter_branch_by_qda_dominance(
                branch,
                means=means,
                precisions=precisions,
                log_dets=log_dets,
                log_prior=log_prior,
                class_i=i,
                class_j=j,
                dominance_tol=dominance_tol,
            )
            candidate_segments.extend(_extract_segments(filtered, min_points=2))

        segments = _stitch_segments(candidate_segments, tol=stitch_tolerance)

        for seg in segments:
            vals = _conic_value(seg, conic["A"], conic["b"], conic["c"])
            if vals.size and np.nanmax(np.abs(vals)) > 5e-4:
                continue
            ax.plot(seg[:, 0], seg[:, 1], **line_kwargs)

    ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    return ax


def plot_qda_regions(
    qda_result_or_means,
    covariances: np.ndarray | None = None,
    prior: np.ndarray | None = None,
    *,
    classes=None,
    xlim=None,
    ylim=None,
    resolution=250,
    ax=None,
    alpha=0.2,
    draw_pairwise=True,
    line_kwargs=None,
    class_colors=None,
):
    """Plot QDA decision regions with optional pairwise-conic boundary overlays.

    Region filling is grid-based. Pairwise conics are evaluated analytically and
    rendered via contouring, with optional masking to reduce third-class-dominated
    boundary pieces. If ``class_colors`` is omitted, the region palette falls
    back to ``DEFAULT_CLASS_COLORS``.
    """
    try:
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")
        mcolors = importlib.import_module("matplotlib.colors")
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plot_qda_regions requires matplotlib") from exc

    means, covariances, prior, classes = _unpack_qda_inputs(
        qda_result_or_means,
        covariances,
        prior,
        classes,
    )

    if means.ndim != 2 or means.shape[1] != 2:
        raise ValueError("plot_qda_regions requires means with shape (n_classes, 2)")
    if covariances.shape != (means.shape[0], 2, 2):
        raise ValueError("plot_qda_regions requires covariances with shape (n_classes, 2, 2)")
    if prior.shape != (means.shape[0],):
        raise ValueError("prior must have shape (n_classes,)")

    if ax is None:
        _, ax = plt.subplots()

    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    # Build QDA score parameters.
    n_classes = means.shape[0]
    precisions = np.zeros_like(covariances)
    log_dets = np.zeros(n_classes, dtype=float)
    for k in range(n_classes):
        sign, logdet = np.linalg.slogdet(covariances[k])
        if sign <= 0:
            raise ValueError("Each covariance matrix must be positive definite")
        log_dets[k] = logdet
        try:
            precisions[k] = np.linalg.solve(covariances[k], np.eye(2, dtype=float))
        except np.linalg.LinAlgError:
            precisions[k] = np.linalg.lstsq(covariances[k], np.eye(2, dtype=float), rcond=None)[0]

    log_prior = np.log(np.clip(prior, 1e-300, None))

    xs = np.linspace(float(xlim[0]), float(xlim[1]), int(resolution))
    ys = np.linspace(float(ylim[0]), float(ylim[1]), int(resolution))
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])

    scores = _qda_scores_from_params(pts, means, precisions, log_dets, log_prior)
    winner = np.argmax(scores, axis=1).reshape(YY.shape)

    cmap = mcolors.ListedColormap(get_class_colors(means.shape[0], class_colors=class_colors))

    ax.pcolormesh(XX, YY, winner, shading="auto", cmap=cmap, alpha=alpha)

    if draw_pairwise:
        conics = qda_pairwise_conics(means, covariances, prior, classes=classes)
        if line_kwargs is None:
            line_kwargs = {"colors": "black", "linewidths": 1.0}

        class_index = {cls: idx for idx, cls in enumerate(classes)}

        for conic in conics:
            i = class_index[conic["class_i"]]
            j = class_index[conic["class_j"]]
            A = conic["A"]
            b = conic["b"]
            c = conic["c"]

            # Evaluate exact conic F(x,y)=0.
            F = (
                A[0, 0] * XX * XX
                + (A[0, 1] + A[1, 0]) * XX * YY
                + A[1, 1] * YY * YY
                + b[0] * XX
                + b[1] * YY
                + c
            )

            # Dominance mask reduces portions dominated by third classes.
            if n_classes > 2:
                pair_min = np.minimum(scores[:, i], scores[:, j])
                other = [k for k in range(n_classes) if k not in (i, j)]
                other_max = np.max(scores[:, other], axis=1)
                keep = (pair_min - other_max) >= -1e-9
                F_plot = np.where(keep.reshape(F.shape), F, np.nan)
            else:
                F_plot = F

            ax.contour(XX, YY, F_plot, levels=[0.0], **line_kwargs)

    ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    return ax


__all__ = ["qda", "qda_pairwise_conics", "plot_qda_regions", "plot_qda_boundary_segments"]
