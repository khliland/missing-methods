import numpy as np

from .nan_utils import _validate_sample_weight
from .pca_pls import pca
from .impute import _preprocess_for_imputation


def _sigmoid(arr: np.ndarray) -> np.ndarray:
    clipped = np.clip(arr, -40.0, 40.0)
    return np.reciprocal(1.0 + np.exp(-clipped))


def _soft_threshold(arr: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(arr) * np.maximum(np.abs(arr) - threshold, 0.0)


def _validate_binary_target(y) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError("y must be a 1-D array")
    if np.any(np.isnan(arr.astype(float, copy=False))):
        raise ValueError("y cannot contain NaNs")
    classes = np.unique(arr)
    if classes.size != 2:
        raise ValueError("Only binary logistic regression is supported")
    encoded = (arr == classes[1]).astype(float)
    return encoded, classes



def _masked_effective_matrix(X: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(X)
    observed = mask.sum(axis=1)
    scale = np.divide(
        X.shape[1],
        observed,
        out=np.zeros(X.shape[0], dtype=float),
        where=observed > 0,
    )
    return np.where(mask, X, 0.0) * scale[:, np.newaxis]


def _masked_smooth_objective(
    Z: np.ndarray,
    y: np.ndarray,
    intercept: float,
    coef: np.ndarray,
    sample_weight: np.ndarray,
    l2_penalty: float,
) -> float:
    eta = intercept + Z @ coef
    probs = np.clip(_sigmoid(eta), 1e-12, 1.0 - 1e-12)
    loss = -np.sum(sample_weight * (y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))
    return loss + 0.5 * l2_penalty * np.dot(coef, coef)


def _masked_objective(
    Z: np.ndarray,
    y: np.ndarray,
    intercept: float,
    coef: np.ndarray,
    sample_weight: np.ndarray,
    l2_penalty: float,
    l1_penalty: float = 0.0,
) -> float:
    return _masked_smooth_objective(Z, y, intercept, coef, sample_weight, l2_penalty) + l1_penalty * np.sum(
        np.abs(coef)
    )


def _masked_gradient(
    Z: np.ndarray,
    y: np.ndarray,
    intercept: float,
    coef: np.ndarray,
    sample_weight: np.ndarray,
    l2_penalty: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    eta = intercept + Z @ coef
    probs = np.clip(_sigmoid(eta), 1e-12, 1.0 - 1e-12)
    residual = probs - y
    weighted_residual = sample_weight * residual
    grad_intercept = float(np.sum(weighted_residual))
    grad_coef = Z.T @ weighted_residual + l2_penalty * coef
    return grad_intercept, grad_coef, probs


def _refine_newton(
    Z: np.ndarray,
    y: np.ndarray,
    intercept: float,
    coef: np.ndarray,
    sample_weight: np.ndarray,
    l2_penalty: float,
    refinement_tol: float,
    refinement_maxiter: int,
) -> tuple[float, np.ndarray, bool, int]:
    converged = False
    n_iter = 0

    for n_iter in range(1, refinement_maxiter + 1):
        grad_intercept, grad_coef, probs = _masked_gradient(
            Z,
            y,
            intercept,
            coef,
            sample_weight,
            l2_penalty,
        )

        curvature = sample_weight * probs * (1.0 - probs)
        h00 = np.sum(curvature)
        h01 = Z.T @ curvature
        h11 = Z.T @ (curvature[:, np.newaxis] * Z)
        if l2_penalty > 0:
            h11.flat[:: h11.shape[0] + 1] += l2_penalty

        hessian = np.empty((coef.size + 1, coef.size + 1), dtype=float)
        hessian[0, 0] = h00
        hessian[0, 1:] = h01
        hessian[1:, 0] = h01
        hessian[1:, 1:] = h11
        gradient = np.concatenate(([grad_intercept], grad_coef))

        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        current_loss = _masked_smooth_objective(Z, y, intercept, coef, sample_weight, l2_penalty)
        step_scale = 1.0
        accepted = False
        for _ in range(20):
            candidate_intercept = intercept - step_scale * step[0]
            candidate_coef = coef - step_scale * step[1:]
            candidate_loss = _masked_smooth_objective(
                Z,
                y,
                candidate_intercept,
                candidate_coef,
                sample_weight,
                l2_penalty,
            )
            if np.isfinite(candidate_loss) and candidate_loss <= current_loss:
                intercept = candidate_intercept
                coef = candidate_coef
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            break
        if np.max(np.abs(step_scale * step)) < refinement_tol:
            converged = True
            break

    return intercept, coef, converged, n_iter


def _refine_proximal_gradient(
    Z: np.ndarray,
    y: np.ndarray,
    intercept: float,
    coef: np.ndarray,
    sample_weight: np.ndarray,
    l2_penalty: float,
    l1_penalty: float,
    refinement_tol: float,
    refinement_maxiter: int,
) -> tuple[float, np.ndarray, bool, int]:
    converged = False
    n_iter = 0

    for n_iter in range(1, refinement_maxiter + 1):
        grad_intercept, grad_coef, probs = _masked_gradient(
            Z,
            y,
            intercept,
            coef,
            sample_weight,
            l2_penalty,
        )
        curvature = sample_weight * probs * (1.0 - probs)
        h11 = Z.T @ (curvature[:, np.newaxis] * Z)
        if l2_penalty > 0:
            h11.flat[:: h11.shape[0] + 1] += l2_penalty

        local_lipschitz = max(float(np.linalg.norm(h11, ord=2)), 1e-12)
        step_size = 1.0 / local_lipschitz
        current_smooth = _masked_smooth_objective(Z, y, intercept, coef, sample_weight, l2_penalty)
        current_objective = current_smooth + l1_penalty * np.sum(np.abs(coef))
        accepted = False
        delta_intercept = 0.0
        delta_coef = np.zeros_like(coef)

        for _ in range(25):
            candidate_intercept = intercept - step_size * grad_intercept
            candidate_coef = _soft_threshold(coef - step_size * grad_coef, step_size * l1_penalty)
            delta_intercept = candidate_intercept - intercept
            delta_coef = candidate_coef - coef
            candidate_smooth = _masked_smooth_objective(
                Z,
                y,
                candidate_intercept,
                candidate_coef,
                sample_weight,
                l2_penalty,
            )
            quadratic_bound = (
                current_smooth
                + grad_intercept * delta_intercept
                + np.dot(grad_coef, delta_coef)
                + 0.5
                * (delta_intercept * delta_intercept + np.dot(delta_coef, delta_coef))
                / step_size
            )
            candidate_objective = candidate_smooth + l1_penalty * np.sum(np.abs(candidate_coef))
            if (
                np.isfinite(candidate_smooth)
                and candidate_smooth <= quadratic_bound + 1e-12
                and candidate_objective <= current_objective + 1e-12
            ):
                intercept = candidate_intercept
                coef = candidate_coef
                accepted = True
                break
            step_size *= 0.5

        if not accepted:
            break

        max_update = max(np.abs(delta_intercept), np.max(np.abs(delta_coef), initial=0.0))
        if max_update < refinement_tol:
            converged = True
            break

    return intercept, coef, converged, n_iter


def logistic(
    X,
    y,
    *,
    l2_penalty=1e-06,
    l1_penalty=0.0,
    impute_ncomp=None,
    impute_preprocessing="standardize",
    pca_tol=1e-06,
    pca_maxiter=1000,
    warm_start_solver="lbfgs",
    warm_start_maxiter=1000,
    refinement_tol=1e-06,
    refinement_maxiter=50,
    sample_weight=None,
):
    """Fit hybrid missing-robust binary logistic regression.

    The routine fills only missing entries using a low-rank PCA reconstruction,
    uses a dense sklearn solver for a warm start, and then refines the coefficients
    on the original matrix with MCAR-scaled masked updates. When ``l1_penalty`` is
    positive, the refinement switches to a proximal-gradient path so sparsity can be
    handled without relying on a smooth Newton step.

    Args:
        X: Predictor matrix with shape (n_samples, n_features).
        y: Binary target vector with shape (n_samples,).
        l2_penalty: Ridge penalty applied during the masked refinement.
        l1_penalty: Lasso penalty applied during masked proximal refinement.
        impute_ncomp: Number of PCA components used for warm-start imputation.
        impute_preprocessing: One of "standardize", "center", or "none".
        pca_tol: Convergence tolerance for the PCA imputation stage.
        pca_maxiter: Maximum PCA iterations per component.
        warm_start_solver: Dense sklearn solver used on the filled matrix.
        warm_start_maxiter: Maximum iterations for the warm-start solver.
        refinement_tol: Convergence tolerance for the masked Newton refinement.
        refinement_maxiter: Maximum refinement iterations.
        sample_weight: Optional row weights with shape (n_samples,).

    Returns:
        Dictionary containing fitted coefficients, probabilities, warm-start data,
        and imputation metadata.

    Example:
        >>> import numpy as np
        >>> from missing_methods.logistic import logistic
        >>> X = np.array([[1.2, 0.1], [0.8, np.nan], [-1.1, -0.2], [-0.9, -0.4]])
        >>> y = np.array([1, 1, 0, 0])
        >>> result = logistic(X, y, impute_ncomp=1)
        >>> result["probabilities"].shape
        (4,)
        >>> sparse = logistic(X, y, impute_ncomp=1, l1_penalty=0.05)
        >>> sparse["coef"].shape
        (2,)
        >>> weighted = logistic(X, y, impute_ncomp=1, sample_weight=np.array([1.0, 1.5, 1.0, 0.75]))
        >>> weighted["probabilities"].shape
        (4,)
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "logistic() requires scikit-learn. Install the optional dependency with 'pip install -e .[sklearn]'."
        ) from exc

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array")
    encoded_y, classes = _validate_binary_target(y)
    if X.shape[0] != encoded_y.shape[0]:
        raise ValueError("X and y must contain the same number of rows")
    if l2_penalty < 0:
        raise ValueError("l2_penalty must be non-negative")
    if l1_penalty < 0:
        raise ValueError("l1_penalty must be non-negative")
    row_weights = _validate_sample_weight(sample_weight, X.shape[0])

    transformed, means, scales = _preprocess_for_imputation(
        X,
        impute_preprocessing,
        sample_weight=row_weights,
    )
    max_rank = min(max(X.shape[0] - 1, 1), X.shape[1])
    if impute_ncomp is None:
        impute_ncomp = min(5, max_rank)
    impute_ncomp = int(impute_ncomp)
    if impute_ncomp <= 0:
        raise ValueError("impute_ncomp must be positive")
    impute_ncomp = min(impute_ncomp, max_rank)

    if np.isnan(X).any():
        pca_result = pca(
            transformed,
            ncomp=impute_ncomp,
            center=False,
            tol=pca_tol,
            maxiter=pca_maxiter,
            sample_weight=row_weights,
        )
        reconstructed = (pca_result["scores"] @ pca_result["loadings"].T) * scales + means
        X_filled = np.where(np.isnan(X), reconstructed, X)
    else:
        pca_result = None
        X_filled = X.copy()

    warm_kwargs = {
        "solver": warm_start_solver,
        "max_iter": int(warm_start_maxiter),
        "fit_intercept": True,
    }
    if l2_penalty > 0:
        warm_kwargs["C"] = 1.0 / l2_penalty
    else:
        warm_kwargs["C"] = np.inf
    warm_model = LogisticRegression(**warm_kwargs)
    warm_model.fit(X_filled, encoded_y, sample_weight=row_weights)

    intercept = float(warm_model.intercept_[0])
    coef = warm_model.coef_[0].astype(float, copy=True)
    warm_intercept = intercept
    warm_coef = coef.copy()

    Z = _masked_effective_matrix(X)
    if l1_penalty > 0:
        intercept, coef, converged, n_iter = _refine_proximal_gradient(
            Z,
            encoded_y,
            intercept,
            coef,
            row_weights,
            l2_penalty,
            l1_penalty,
            refinement_tol,
            int(refinement_maxiter),
        )
    else:
        intercept, coef, converged, n_iter = _refine_newton(
            Z,
            encoded_y,
            intercept,
            coef,
            row_weights,
            l2_penalty,
            refinement_tol,
            int(refinement_maxiter),
        )

    probabilities = _sigmoid(intercept + Z @ coef)
    return {
        "coef": coef,
        "intercept": intercept,
        "classes": classes,
        "probabilities": probabilities,
        "fitted": probabilities >= 0.5,
        "converged": converged,
        "n_iter": n_iter,
        "warm_start_coef": warm_coef,
        "warm_start_intercept": warm_intercept,
        "filled_X": X_filled,
        "l1_penalty": l1_penalty,
        "l2_penalty": l2_penalty,
        "impute_preprocessing": impute_preprocessing,
        "impute_ncomp": impute_ncomp,
        "impute_means": means,
        "impute_scales": scales,
        "pca_result": pca_result,
    }


__all__ = ["logistic"]
