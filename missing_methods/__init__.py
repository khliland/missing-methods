"""High-level NA-aware decomposition, classification, and RV utilities."""

from .mfa import mfa
from .pca_pls import pca, pls
from .rv import rv, rv2, rv_list, rv2_list
from .preprocessing import standardize, normalize
from .assessment import missingness_recommendations
from .kernel import kernel_pls
from .logistic import logistic
from .impute import pca_impute
from .plotting import DEFAULT_CLASS_COLORS, get_class_colors
from .lda import lda, lda_pairwise_boundaries, plot_lda_boundaries, plot_lda_boundary_segments, plot_lda_regions
from .qda import qda, qda_pairwise_conics, plot_qda_regions, plot_qda_boundary_segments

__all__ = [
	"pca",
	"pls",
	"rv",
	"rv2",
	"rv_list",
	"rv2_list",
	"mfa",
	"standardize",
	"normalize",
	"missingness_recommendations",
	"kernel_pls",
	"logistic",
	"pca_impute",
	"DEFAULT_CLASS_COLORS",
	"get_class_colors",
	"lda",
	"lda_pairwise_boundaries",
	"plot_lda_boundaries",
	"plot_lda_boundary_segments",
	"plot_lda_regions",
	"qda",
	"qda_pairwise_conics",
	"plot_qda_regions",
	"plot_qda_boundary_segments",
]
