# missing-methods

`missing-methods` provides NA-aware methods that work even when datasets contain `NaN` entries.

Included methods are grouped as:

- Decomposition / exploratory:  
	Principal Component Analysis (`pca`) and Multiple Factor Analysis (`mfa`)
- Component-based prediction:  
	Partial Least Squares (`pls`) and kernel Partial Least Squares (`kernel_pls`)
- Classification:  
	Logistic Regression (`logistic`), Linear Discriminant Analysis (`lda`), and Quadratic Discriminant Analysis (`qda`)
- Imputation:  
	Principal Component Analysis-based imputation (`pca_impute`)
- Similarity metrics:  
	RV and RV2 coefficients (`rv`, `rv2`, `rv_list`, `rv2_list`)
- Preprocessing utilities:  
	functional helpers (`normalize`, `standardize`, `yeo_johnson`, `box_cox`)
- Scikit-style preprocessing:  
	`PCAImputer`, `Normalizer`, `StandardScaler`, `YeoJohnsonTransformer`, and `BoxCoxTransformer`
- Scikit-style estimators:  
	`PCA`, `PLSRegressor`, `KernelPLSRegressor`, `LogisticClassifier`, `LDAClassifier`, and `QDAClassifier`

Minimal examples of all implemented methods are found in `examples/examples.ipynb`, while more elaborate examples are found in `examples/kernel_pls.ipynb` and `examples/UCI_wine.ipynb`.

## Installation

```bash
pip install -e .
```

Alternatively, install directly from GitHub:

```bash
pip install git+https://github.com/khliland/missing-methods.git
```

This project requires Python 3.10+ and depends only on `numpy` for the core helpers. The scikit-learn estimator wrappers plus the hybrid logistic regression warm start need `scikit-learn`; install them with

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

sample_weight = np.array([1.0, 0.75, 1.25])
weighted_pls = mm.pls(X, Y, ncomp=2, sample_weight=sample_weight)
weighted_rv = mm.rv(X, Y, sample_weight=sample_weight)

X_signed = np.array([[1.0, -2.0], [2.5, np.nan], [3.2, 0.3]])
X_signed_t, yj_lambdas = mm.yeo_johnson(X_signed, return_lambdas=True)
X_signed_new = np.array([[1.4, -1.0], [np.nan, 0.7]])
X_signed_new_t = mm.yeo_johnson(X_signed_new, lambdas=yj_lambdas)

X_pos = np.array([[1.0, 2.0], [1.8, np.nan], [3.1, 4.2]])
X_pos_t, bc_lambdas = mm.box_cox(X_pos, return_lambdas=True)
X_pos_new = np.array([[1.2, 2.4], [np.nan, 5.5]])
X_pos_new_t = mm.box_cox(X_pos_new, lambdas=bc_lambdas)

X_class = np.array([[1.2, 0.1], [0.8, np.nan], [-1.1, -0.2], [-0.9, -0.4]])
y_class = np.array([1, 1, 0, 0])
logit = mm.logistic(X_class, y_class, impute_ncomp=1)
sparse_logit = mm.logistic(X_class, y_class, impute_ncomp=1, l1_penalty=0.05)
print(logit["probabilities"].shape)  # -> (4,)

lda_result = mm.lda(X_class, y_class, impute_ncomp=1)
print(lda_result["posterior"].shape)  # -> (4, 2)

qda_result = mm.qda(X_class, y_class, impute_ncomp=1)
print(qda_result["posterior"].shape)  # -> (4, 2)

boundaries = mm.lda_pairwise_boundaries(lda_result)
print(len(boundaries))  # -> number of pairwise analytical boundaries

conics = mm.qda_pairwise_conics(qda_result)
print(len(conics))  # -> number of pairwise conics

class_colors = mm.get_class_colors(len(qda_result["classes"]))

# If class_colors is omitted, plot_lda_regions / plot_qda_regions fall back to
# mm.DEFAULT_CLASS_COLORS automatically.

# In 2D, draw analytically traced QDA boundary segments
# (no contour-based pairwise boundary rendering)
# mm.plot_qda_regions(qda_result)
# mm.plot_qda_boundary_segments(qda_result)

X_miss = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
imputed = mm.pca_impute(X_miss, ncomp=1)
print(imputed["filled_X"].shape)  # -> (3, 3)
```

A full set of examples for all included functions and scikit-learn wrappers is found in `examples/examples.ipynb`, while development testing against `hoggorm` is found in `examples/development_testing.ipynb`.

## Neutrality and missingness

All methods internally scale sums-of-squares and inner products by the proportion of observed entries so that the estimated variances/covariances stay unbiased under MCAR (missing completely at random). The same scaled geometry is used across PCA, PLS, MFA, and RV/RV2 so the similarity summaries stay comparable even when datasets have different missingness patterns.

## Scikit-learn wrappers

For users that prefer estimator classes, `missing_methods.sk` exposes scikit-learn-style wrappers that delegate to the functional helpers while keeping the MCAR scaling. On the preprocessing side this includes `PCAImputer`, `Normalizer`, `StandardScaler`, `YeoJohnsonTransformer`, and `BoxCoxTransformer`; on the modeling side it includes `PCA`, `PLSRegressor`, `KernelPLSRegressor`, `LogisticClassifier`, `LDAClassifier`, and `QDAClassifier`.

```python
from missing_methods.sk import PCA, PLSRegressor, KernelPLSRegressor, PCAImputer, LogisticClassifier, LDAClassifier, QDAClassifier, Normalizer, StandardScaler, YeoJohnsonTransformer, BoxCoxTransformer
import numpy as np
from sklearn.pipeline import Pipeline

X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9]])
Y = np.array([[2.4, 2.9], [0.6, 0.5], [2.1, 2.2]])
X_miss = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
X_class = np.array([[1.2, 0.1], [0.8, np.nan], [-1.1, -0.2], [-0.9, -0.4]])
y_class = np.array([1, 1, 0, 0])

estimator = PCA(n_components=2)
sample_weight = np.array([1.0, 0.75, 1.25])
estimator.fit(X)
scores = estimator.transform(X)
estimator.fit(X, sample_weight=sample_weight)
scores = estimator.transform(X)

pls_estimator = PLSRegressor(n_components=2)
pls_estimator.fit(X, Y)
Y_pred = pls_estimator.predict(X)
pls_estimator.fit(X, Y, sample_weight=sample_weight)
Y_pred = pls_estimator.predict(X)

kernel_estimator = KernelPLSRegressor(n_components=2)
kernel_estimator.fit(X, Y)
Y_pred_kernel = kernel_estimator.predict(X)
kernel_estimator.fit(X, Y, sample_weight=sample_weight)
Y_pred_kernel = kernel_estimator.predict(X)

classifier = LogisticClassifier(impute_ncomp=1)
classifier.fit(X_class, y_class)
class_probabilities = classifier.predict_proba(X_class)

sparse_classifier = LogisticClassifier(impute_ncomp=1, l1_penalty=0.05)
sparse_classifier.fit(X_class, y_class)

lda_classifier = LDAClassifier(impute_ncomp=1)
lda_classifier.fit(X_class, y_class)
lda_probabilities = lda_classifier.predict_proba(X_class)

qda_classifier = QDAClassifier(impute_ncomp=1)
qda_classifier.fit(X_class, y_class)
qda_probabilities = qda_classifier.predict_proba(X_class)

imputer = PCAImputer(ncomp=1)
imputer.fit(X_miss)
X_filled = imputer.transform(X_miss)

normalizer = Normalizer()
normalized = normalizer.fit_transform(X)
scaler = StandardScaler()
scaled = scaler.fit_transform(X)

power = YeoJohnsonTransformer()
X_power = power.fit_transform(X_class)
X_back = power.inverse_transform(X_power)

positive_pipeline = Pipeline([
	("boxcox", BoxCoxTransformer()),
	("scale", StandardScaler()),
])
X_positive = np.array([[1.0, 2.0], [1.5, 3.0], [2.5, np.nan]])
X_positive_t = positive_pipeline.fit_transform(X_positive)
```
