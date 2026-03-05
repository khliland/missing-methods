import numpy as np
import pandas as pd

ALL_METHODS = ("PCA", "PLS", "MFA", "RV", "RV2")

DEFAULT_THRESHOLDS = {
    "min_row_coverage": 0.10,
    "min_col_coverage": 0.10,
    "min_xy_overlap": 0.10,
    "min_pairwise_overlap": 0.05,
}

def missingness_recommendations(
    X,
    Y=None,
    methods=None,
    thresholds=None,
    blocks=None,   # optional list of column indices for MFA
):
    """
    Generate method-specific missingness recommendations for
    PCA, PLS, MFA, RV, and RV2.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
    Y : np.ndarray or None
    methods : iterable of method names or None
    thresholds : dict or None
    blocks : list of index arrays (MFA only)

    Returns
    -------
    pandas.DataFrame

    Example:
    -------
        >>> import numpy as np
        >>> from missing_methods import missingness_recommendations
        >>> rng = np.random.default_rng(0)
        >>> X = rng.standard_normal((10, 4))
        >>> Y = rng.standard_normal((10, 4)) + 0.3 * X
        >>> X[[1, 3, 7], 2] = np.nan
        >>> Y[[0, 4, 9], 1] = np.nan
        >>> missingness_recommendations(X, Y)
    """

    if methods is None:
        methods = ALL_METHODS
    else:
        methods = tuple(methods)

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    else:
        thresholds = {**DEFAULT_THRESHOLDS, **thresholds}

    rows = []

    def add(method, condition, trigger, severity, message,
            action="", neutral=True):

        if method not in methods:
            return

        rows.append({
            "method": method,
            "condition": condition,
            "trigger": bool(trigger),
            "severity": severity if trigger else "ok",
            "message": message if trigger else "OK",
            "action": action if trigger else "",
            "neutral": neutral,
        })

    # ------------------------------------------------------------------
    # Compute missingness facts
    # ------------------------------------------------------------------
    X = np.asarray(X, dtype=float)
    maskX = ~np.isnan(X)

    n, p = X.shape
    row_prop_X = maskX.mean(axis=1)
    col_prop_X = maskX.mean(axis=0)

    # Fatal: fully missing rows / columns
    fully_missing_row_X = np.any(row_prop_X == 0)
    fully_missing_col_X = np.any(col_prop_X == 0)

    if fully_missing_row_X or fully_missing_col_X:
        for m in methods:
            add(
                m,
                "fully_missing_data",
                True,
                "abort",
                "X contains fully missing rows or columns",
                "Remove or explicitly impute before analysis",
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # X–Y overlap (PLS, RV)
    # ------------------------------------------------------------------
    if Y is not None:
        Y = np.asarray(Y, dtype=float)
        maskY = ~np.isnan(Y)

        row_overlap_XY = maskX.any(axis=1) & maskY.any(axis=1)
        prop_row_overlap_XY = row_overlap_XY.mean()
    else:
        prop_row_overlap_XY = 1.0

    # ------------------------------------------------------------------
    # MFA block coverage (optional)
    # ------------------------------------------------------------------
    block_min_prop = 1.0
    if blocks is not None:
        for cols in blocks:
            block_mask = maskX[:, cols]
            block_prop = block_mask.mean()
            block_min_prop = min(block_min_prop, block_prop)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    # ---- PCA ----------------------------------------------------------
    add(
        "PCA",
        "low_row_coverage",
        row_prop_X.min() < thresholds["min_row_coverage"],
        "warn",
        "Very sparse samples may inflate score variance",
        "Inspect score stability; interpret cautiously",
    )

    add(
        "PCA",
        "low_col_coverage",
        col_prop_X.min() < thresholds["min_col_coverage"],
        "warn",
        "Variables with very low coverage give unstable loadings",
        "Consider filtering variables",
    )

    # ---- PLS ----------------------------------------------------------
    add(
        "PLS",
        "low_row_coverage",
        row_prop_X.min() < thresholds["min_row_coverage"],
        "recommend",
        "Row-wise missingness is highly imbalanced",
        "Consider sample weighting by proportion observed",
        neutral=False,
    )

    add(
        "PLS",
        "no_xy_overlap",
        prop_row_overlap_XY < thresholds["min_xy_overlap"],
        "abort",
        "Too few rows have observed values in both X and Y",
        "Abort: cross-covariance undefined or unstable",
    )

    # ---- MFA ----------------------------------------------------------
    add(
        "MFA",
        "uneven_block_coverage",
        block_min_prop < thresholds["min_col_coverage"],
        "warn",
        "At least one MFA block has very low coverage",
        "Block eigenvalue scaling may be distorted",
    )

    # ---- RV / RV2 -----------------------------------------------------
    add(
        "RV",
        "low_xy_overlap",
        prop_row_overlap_XY < thresholds["min_pairwise_overlap"],
        "warn",
        "RV is based on very few shared observations",
        "RV / RV2 likely unstable",
    )

    add(
        "RV2",
        "low_xy_overlap",
        prop_row_overlap_XY < thresholds["min_pairwise_overlap"],
        "warn",
        "RV2 is based on very few shared observations",
        "RV / RV2 likely unstable",
    )

    # ------------------------------------------------------------------
    # Final DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows)

    severity_order = pd.CategoricalDtype(
        ["ok", "warn", "recommend", "abort"], ordered=True
    )
    df["severity"] = df["severity"].astype(severity_order)

    return df