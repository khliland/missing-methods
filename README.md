# missing-methods

`missing-methods` provides NA-aware PCA/PLS/MFA helpers plus RV/RV2 similarity scores that work even when the datasets contain `NaN` entries. It also ships preprocessing helpers (`normalize`, `standardize`) plus scikit-style `Normalizer`/`StandardScaler`, so you can assemble MCAR-aware pipelines. It's a lightweight alternative to full chemometrics toolkits and is suitable for exploratory analysis of paired data matrices.

## Installation

```bash
pip install -e .
```

Alternatively, install directly from GitHub:

```bash
pip install git+https://github.com/khliland/missing-methods.git
```

This project requires Python 3.10+ and depends only on `numpy` for the core helpers. The scikit-learn estimator wrappers (including the Normalizer/StandardScaler) need `scikit-learn`; install them with

```bash
pip install -e .[sklearn]
```

## Usage

```python
import missing_methods as mm
import numpy as np

X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9]])
Y = np.array([[2.4, 2.9], [0.6, 0.5], [2.1, 2.2]])

result = mm.pca(X, ncomp=2)
print(result["scores"].shape)  # -> (3, 2)
print("RV", mm.rv(X, Y))
```

A full set of examples for all included functions and scikit-learn wrappers is found in `examples/examples.ipynb`, while development testing against `hoggorm` is found in `examples/development_testing.ipynb`.

## Neutrality and missingness

All methods internally scale sums-of-squares and inner products by the proportion of observed entries so that the estimated variances/covariances stay unbiased under MCAR (missing completely at random). The same scaled geometry is used across PCA, PLS, MFA, and RV/RV2 so the similarity summaries stay comparable even when datasets have different missingness patterns.

## Scikit-learn wrappers

For users that prefer estimator classes, `missing_methods.sk` exposes scikit-learn-style estimators that delegate to the functional helpers while keeping the MCAR scaling. It now also re-exports the Normalizer/StandardScaler wrappers so you can drop them into `Pipeline`s alongside `PCA`/`PLS`.

```python
from missing_methods.sk import pca, pls, Normalizer, StandardScaler

estimator = pca(n_components=2)
estimator.fit(X)
scores = estimator.transform(X)

pls_estimator = pls(n_components=2)
pls_estimator.fit(X, Y)
Y_pred = pls_estimator.predict(X)

normalizer = Normalizer()
normalized = normalizer.fit_transform(X)
scaler = StandardScaler()
scaled = scaler.fit_transform(X)
```
