import numpy as np

from .nan_utils import _safe_correlation
from .pca_pls import pca


def mfa(blocks, ncomp=None, center=True, tol=1e-06, maxiter=1000):
    """Perform multiple factor analysis across aligned blocks while handling NaNs.

    Each block is centered (when ``center`` is True), scaled by its first eigenvalue,
    and concatenated before a global PCA extracts the common scores and loadings.
    The function keeps block-level loadings, correlation loadings, and variance
    summaries alongside the overall statistics returned by the PCA helper.

    Args:
        blocks: Sequence of 2-D arrays with the same number of rows (samples).
        ncomp: Number of global components to keep; defaults to min(n_samples - 1, total_features).
        center: Whether to center each block before the per-block PCA scaling.
        tol: Convergence tolerance forwarded to the internal PCA calls.
        maxiter: Max number of iterations per PCA component.

    Returns:
        Dictionary containing:
            scores: Global scores for all concatenated variables.
            loadings: Global loadings matching the concatenated feature axis.
            block_loadings: List of loadings per original block.
            correlation_loadings: Stacked correlation loadings between scaled block variables and global scores.
            block_correlation_loadings: List of per-block correlation loadings.
            explained: Sum of squares captured by each global component.
            explained_variance: Explained variance normalized by total variance.
            explained_cumulative: Cumulative explained variance.
            block_means: Means computed per block (used for centering).
            block_scales: Scaling applied to each block after centering.
            block_eigenvalues: First eigenvalues used for scaling each block.
            block_feature_counts: Original feature counts for each block.
            total_variance: Global variance of the concatenated, scaled blocks.
            n_samples: Number of samples (rows) across blocks.
            n_features: Total number of concatenated features.
    """
    if not blocks:
        raise ValueError("At least one block must be provided for MFA")

    prepared = []
    for block in blocks:
        arr = np.asarray(block, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        elif arr.ndim != 2:
            raise ValueError("Each block must be a 2-D array")
        prepared.append(arr)

    n_samples = prepared[0].shape[0]
    for arr in prepared:
        if arr.shape[0] != n_samples:
            raise ValueError("All blocks must have the same number of samples")

    scaled_blocks = []
    block_means = []
    block_scales = []
    block_eigenvalues = []
    block_feature_counts = []
    eps = np.finfo(float).eps

    for block in prepared:
        local_out = pca(block, ncomp=1, center=center, tol=tol, maxiter=maxiter)
        means = local_out["means"]
        eigenvalue = float(local_out["explained"][0]) if local_out["explained"].size else 0.0
        centered = block - means
        scale = 1.0 / np.sqrt(eigenvalue) if eigenvalue > eps else 1.0
        scaled_blocks.append(centered * scale)
        block_means.append(means)
        block_scales.append(scale)
        block_eigenvalues.append(eigenvalue)
        block_feature_counts.append(block.shape[1])

    total_features = sum(block_feature_counts)
    if total_features == 0:
        raise ValueError("Each block must contain at least one variable")

    concatenated = np.hstack(scaled_blocks)
    if ncomp is None:
        ncomp = min(max(n_samples - 1, 1), total_features)
    else:
        ncomp = int(ncomp)
        if ncomp <= 0:
            raise ValueError("ncomp must be positive")
        ncomp = min(ncomp, total_features)

    global_out = pca(concatenated, ncomp, center=False, tol=tol, maxiter=maxiter)
    scores = global_out["scores"]
    loadings = global_out["loadings"]
    explained = global_out["explained"]

    total_variance = np.nansum(concatenated * concatenated)
    variance_denom = total_variance if total_variance > eps else eps
    explained_variance = explained / variance_denom
    explained_cumulative = np.cumsum(explained_variance)

    block_loadings = []
    start = 0
    for count in block_feature_counts:
        end = start + count
        block_loadings.append(loadings[start:end])
        start = end

    block_correlation_loadings = []
    for block_scaled in scaled_blocks:
        correlations = np.zeros((block_scaled.shape[1], scores.shape[1]), dtype=float)
        for var_idx in range(block_scaled.shape[1]):
            col = block_scaled[:, var_idx]
            for comp_idx in range(scores.shape[1]):
                correlations[var_idx, comp_idx] = _safe_correlation(col, scores[:, comp_idx])
        block_correlation_loadings.append(correlations)

    correlation_loadings = np.vstack(block_correlation_loadings) if block_correlation_loadings else np.zeros((0, scores.shape[1]))

    return {
        "scores": scores,
        "loadings": loadings,
        "block_loadings": block_loadings,
        "correlation_loadings": correlation_loadings,
        "block_correlation_loadings": block_correlation_loadings,
        "explained": explained,
        "explained_variance": explained_variance,
        "explained_cumulative": explained_cumulative,
        "block_means": block_means,
        "block_scales": block_scales,
        "block_eigenvalues": block_eigenvalues,
        "block_feature_counts": block_feature_counts,
        "total_variance": total_variance,
        "n_samples": n_samples,
        "n_features": total_features,
    }
