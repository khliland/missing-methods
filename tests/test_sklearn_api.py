import numpy as np

from missing_methods.sk import (
    BoxCoxTransformer,
    LDAClassifier,
    Normalizer,
    PCA,
    PCAImputer,
    PLSRegressor,
    QDAClassifier,
    StandardScaler,
    YeoJohnsonTransformer,
)


def test_sklearn_wrappers_import_and_basic_transform_paths():
    x = np.array([[1.0, -2.0], [2.0, np.nan], [3.0, 1.5]])
    yj = YeoJohnsonTransformer()
    xt = yj.fit_transform(x)
    xr = yj.inverse_transform(xt)

    assert xt.shape == x.shape
    assert np.allclose(xr, x, equal_nan=True)

    x_pos = np.array([[1.0, 2.0], [1.5, np.nan], [2.5, 3.5]])
    bc = BoxCoxTransformer()
    xt_pos = bc.fit_transform(x_pos)
    assert xt_pos.shape == x_pos.shape

    normalizer = Normalizer()
    scaler = StandardScaler()
    assert normalizer.fit_transform(x_pos).shape == x_pos.shape
    assert scaler.fit_transform(x_pos).shape == x_pos.shape


def test_sklearn_estimators_basic_fit_predict_paths():
    x = np.array([[0.1, 1.0], [0.0, 0.9], [1.1, -0.2], [1.0, -0.1]])
    x[1, 0] = np.nan
    y = np.array([0, 0, 1, 1])

    lda = LDAClassifier(impute_ncomp=1)
    qda = QDAClassifier(impute_ncomp=1)
    assert lda.fit(x, y).predict(x).shape == (4,)
    assert qda.fit(x, y).predict(x).shape == (4,)


def test_sklearn_pca_pls_and_imputer_paths():
    x = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9]])
    y = np.array([[2.4], [0.6], [2.1]])
    x[1, 0] = np.nan

    pca = PCA(ncomp=2)
    pls = PLSRegressor(ncomp=1)
    imputer = PCAImputer(ncomp=1)

    assert pca.fit(x).transform(x).shape == (3, 2)
    assert pls.fit(x, y).predict(x).shape == (3, 1)
    assert imputer.fit_transform(np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])).shape == (3, 3)
