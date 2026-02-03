# missing-methods

`missing-methods` provides NA-aware PCA/PLS helpers plus RV/RV2 similarity scores that work even when the datasets contain `NaN` entries. It's a lightweight alternative to full chemometrics toolkits and is suitable for exploratory analysis of paired data matrices.

## Installation

```bash
pip install -e .
```

Alternatively, install directly from GitHub:

```bash
pip install git+https://github.com/khliland/missing-methods.git
```

This project requires Python 3.10+ and depends only on `numpy`.

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

The repository keeps `testing.ipynb` in `examples/` for interactive comparisons against `hoggorm` and missing-data scenarios.

## Neutrality and missingness

All methods internally scale sums-of-squares and inner products by the proportion of observed entries so that the estimated variances/covariances stay unbiased under MCAR (missing completely at random). The same scaled geometry is used across PCA, PLS, MFA, and RV/RV2 so the similarity summaries stay comparable even when datasets have different missingness patterns.
