import numpy as np

from .nan_utils import _normalize, _safe_crossprod, _safe_matvec, _scaled_sumsq


def pca(X, ncomp, center=True, tol=1e-06, maxiter=1000):
    """Compute PCA scores/loadings while gracefully handling missing entries.

    Args:
        X: Input data matrix with shape (n_samples, n_features).
        ncomp: Number of principal components to extract.
        center: Whether to subtract column means before decomposition.
        tol: Convergence tolerance for the NIPALS iterations.
        maxiter: Maximum number of iterations per component.

    Returns:
        Dictionary containing:
            scores: Projected samples, shape (n_samples, n_components).
            loadings: Feature directions, shape (n_features, n_components).
            explained: Sum of squares captured by each component.
            means: Column means used for centering (zeros if not centered).
            residual: Residual matrix after extracting the requested components.
    Note:
        Missing values are assumed MCAR so inner products are scaled by the observed proportions, keeping variance estimates neutral.
    Example:
        >>> import numpy as np
        >>> from missing_methods import pca
        >>> X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9]])
        >>> X[1, 0] = np.nan
        >>> result = pca(X, ncomp=2)
        >>> result["scores"].shape 
        (3, 2)
    """
    X = np.asarray(X, dtype=float)
    mask = ~np.isnan(X)
    # Keep track of valid entries so we only deflate and update observed cells.
    if center:
        means = np.nanmean(X, axis=0)
        means[np.isnan(means)] = 0.0
    else:
        means = np.zeros(X.shape[1], dtype=float)
    Xc = X - means

    # Preallocate outputs and keep the residual matrix for iterative deflation.
    scores = np.zeros((X.shape[0], ncomp), dtype=float)
    loadings = np.zeros((X.shape[1], ncomp), dtype=float)
    explained = np.zeros(ncomp, dtype=float)
    residual = Xc.copy()

    for comp in range(ncomp):
        start = residual[:, 0]
        start = np.nan_to_num(start)
        # Use the first column as an initial guess and compute the orthogonal weight vector.
        w = _normalize(_safe_crossprod(residual, start))
        if not np.any(w):
            break

        for _ in range(maxiter):
            t = _safe_matvec(residual, w)
            tt = _scaled_sumsq(t)
            if tt <= 0:
                break
            p = _safe_crossprod(residual, t) / tt
            w_new = _normalize(p)
            if np.nansum((w_new - w) ** 2) < tol ** 2:
                w = w_new
                break
            w = w_new

        # Extract component once convergence is reached, then deflate.
        t = _safe_matvec(residual, w)
        tt = _scaled_sumsq(t)
        if tt <= 0:
            break
        p = _safe_crossprod(residual, t) / tt
        scores[:, comp] = t
        loadings[:, comp] = p
        explained[comp] = tt

        recon = np.outer(t, p)
        # Deflate using the same mask to ensure we only subtract observed entries.
        residual[mask] -= recon[mask]

    return {
        "scores": scores,
        "loadings": loadings,
        "explained": explained,
        "means": means,
        "residual": residual,
    }


def pls(X, Y, ncomp, center=True, tol=1e-06, maxiter=1000):
    """Fit NIPALS-style PLS components while ignoring NaNs.

    Args:
        X: Predictor matrix with shape (n_samples, n_features_x).
        Y: Response matrix with shape (n_samples, n_features_y).
        ncomp: Number of latent components to compute.
        center: Whether to center both matrices before fitting.
        tol: Convergence tolerance for the alternating updates.
        maxiter: Maximum number of iterations for each component.

    Returns:
        Dictionary containing:
            scores: Common scores (t) for the X block.
            weights: Weights used to compute the X scores.
            loadings_x: Loadings for the X block.
            loadings_y: Loadings for the Y block.
            explained: Sum of squared X scores per component.
            means_x: Column means of the X block.
            means_y: Column means of the Y block.
            residual_x: Residual X matrix after deflation.
            residual_y: Residual Y matrix after deflation.

    Note:
        Cross-products are scaled by the proportion of observed entries (MCAR assumption) so that X and Y stay balanced despite differing missingness.
    Example:
        >>> import numpy as np
        >>> from missing_methods import pls
        >>> X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9]])
        >>> Y = np.array([[2.4], [0.6], [2.1]])
        >>> X[1, 0] = np.nan
        >>> Y[2, 0] = np.nan
        >>> result = pls(X, Y, ncomp=2)
        >>> result["scores"].shape 
        (3, 2)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    mask_x = ~np.isnan(X)
    mask_y = ~np.isnan(Y)
    if center:
        means_x = np.nanmean(X, axis=0)
        means_x[np.isnan(means_x)] = 0.0
        means_y = np.nanmean(Y, axis=0)
        means_y[np.isnan(means_y)] = 0.0
    else:
        means_x = np.zeros(X.shape[1], dtype=float)
        means_y = np.zeros(Y.shape[1], dtype=float)
    Xc = X - means_x
    Yc = Y - means_y

    # Keep track of both X and Y residuals so each block is deflated consistently.
    scores = np.zeros((X.shape[0], ncomp), dtype=float)
    weights = np.zeros((X.shape[1], ncomp), dtype=float)
    loadings_x = np.zeros((X.shape[1], ncomp), dtype=float)
    loadings_y = np.zeros((Y.shape[1], ncomp), dtype=float)
    explained = np.zeros(ncomp, dtype=float)
    residual_x = Xc.copy()
    residual_y = Yc.copy()

    for comp in range(ncomp):
        u = residual_y[:, 0]
        u = np.nan_to_num(u)
        # Alternating updates until convergence for current component.
        for _ in range(maxiter):
            w = _normalize(_safe_crossprod(residual_x, u))
            if not np.any(w):
                break
            t = _safe_matvec(residual_x, w)
            tt = _scaled_sumsq(t)
            if tt <= 0:
                break
            q = _safe_crossprod(residual_y, t)
            q = q / np.sqrt(np.nansum(q * q)) if np.nansum(q * q) > 0 else q
            u_new = _safe_matvec(residual_y, q)
            if np.nansum((u_new - u) ** 2) < tol ** 2:
                u = u_new
                break
            u = u_new
        w = _normalize(_safe_crossprod(residual_x, u))
        if not np.any(w):
            break
        t = _safe_matvec(residual_x, w)
        tt = _scaled_sumsq(t)
        if tt <= 0:
            break
        p = _safe_crossprod(residual_x, t) / tt
        c = _safe_crossprod(residual_y, t) / tt
        scores[:, comp] = t
        weights[:, comp] = w
        loadings_x[:, comp] = p
        loadings_y[:, comp] = c
        explained[comp] = tt

        recon_x = np.outer(t, p)
        recon_y = np.outer(t, c)
        residual_x[mask_x] -= recon_x[mask_x]
        residual_y[mask_y] -= recon_y[mask_y]

    return {
        "scores": scores,
        "weights": weights,
        "loadings_x": loadings_x,
        "loadings_y": loadings_y,
        "explained": explained,
        "means_x": means_x,
        "means_y": means_y,
        "residual_x": residual_x,
        "residual_y": residual_y,
    }
