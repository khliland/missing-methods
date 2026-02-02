import numpy as np

from .nan_utils import _normalize, _safe_crossprod, _safe_matvec


def pca(X, ncomp, center=True, tol=1e-06, maxiter=1000):
    """Compute PCA scores/loadings while gracefully handling missing entries."""
    X = np.asarray(X, dtype=float)
    mask = ~np.isnan(X)
    if center:
        means = np.nanmean(X, axis=0)
        means[np.isnan(means)] = 0.0
    else:
        means = np.zeros(X.shape[1], dtype=float)
    Xc = X - means

    scores = np.zeros((X.shape[0], ncomp), dtype=float)
    loadings = np.zeros((X.shape[1], ncomp), dtype=float)
    explained = np.zeros(ncomp, dtype=float)
    residual = Xc.copy()

    for comp in range(ncomp):
        start = residual[:, 0]
        start = np.nan_to_num(start)
        w = _normalize(_safe_crossprod(residual, start))
        if not np.any(w):
            break

        for _ in range(maxiter):
            t = _safe_matvec(residual, w)
            tt = np.nansum(t * t)
            if tt <= 0:
                break
            p = _safe_crossprod(residual, t) / tt
            w_new = _normalize(p)
            if np.nansum((w_new - w) ** 2) < tol ** 2:
                w = w_new
                break
            w = w_new

        t = _safe_matvec(residual, w)
        tt = np.nansum(t * t)
        if tt <= 0:
            break
        p = _safe_crossprod(residual, t) / tt
        scores[:, comp] = t
        loadings[:, comp] = p
        explained[comp] = tt

        recon = np.outer(t, p)
        residual[mask] -= recon[mask]

    return {
        "scores": scores,
        "loadings": loadings,
        "explained": explained,
        "means": means,
        "residual": residual,
    }


def pls(X, Y, ncomp, center=True, tol=1e-06, maxiter=1000):
    """Fit NIPALS-style PLS components while ignoring NaNs."""
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
        for _ in range(maxiter):
            w = _normalize(_safe_crossprod(residual_x, u))
            if not np.any(w):
                break
            t = _safe_matvec(residual_x, w)
            tt = np.nansum(t * t)
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
        tt = np.nansum(t * t)
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
