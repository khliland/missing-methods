import numpy as np

import missing_methods as mm


def test_top_level_exports_are_available():
    expected = {
        "pca",
        "pls",
        "mfa",
        "rv",
        "rv2",
        "rv_list",
        "rv2_list",
        "normalize",
        "standardize",
        "yeo_johnson",
        "box_cox",
        "kernel_pls",
        "logistic",
        "pca_impute",
        "lda",
        "qda",
    }
    missing = [name for name in expected if not hasattr(mm, name)]
    assert not missing


def test_power_transforms_preserve_shape_and_nans():
    x_signed = np.array([[1.0, -2.0], [2.5, np.nan], [3.2, 0.3]])
    x_yj, lambdas_yj = mm.yeo_johnson(x_signed, return_lambdas=True)
    x_yj_new = mm.yeo_johnson(np.array([[1.4, -1.0], [np.nan, 0.7]]), lambdas=lambdas_yj)
    if isinstance(x_yj_new, tuple):
        x_yj_new = x_yj_new[0]

    x_pos = np.array([[1.0, 2.0], [1.8, np.nan], [3.1, 4.2]])
    x_bc, lambdas_bc = mm.box_cox(x_pos, return_lambdas=True)
    x_bc_new = mm.box_cox(np.array([[1.2, 2.4], [np.nan, 5.5]]), lambdas=lambdas_bc)
    if isinstance(x_bc_new, tuple):
        x_bc_new = x_bc_new[0]

    assert x_yj.shape == x_signed.shape
    assert x_bc.shape == x_pos.shape
    assert np.isnan(x_yj[1, 1])
    assert np.isnan(x_bc[1, 1])
    assert x_yj_new.shape == (2, 2)
    assert x_bc_new.shape == (2, 2)


def test_pca_impute_fills_missing_values():
    x = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
    result = mm.pca_impute(x, ncomp=1)

    assert result["filled_X"].shape == x.shape
    assert not np.isnan(result["filled_X"]).any()
