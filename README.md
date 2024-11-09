# Applied-Presentation-of-Statistical-Data-Analysis
This repository contains the scripts of the theoretical background and applied part of the bachelor thesis "Applied Presentation of Statistical Data Analysis" by Samuel Brinkmann.

## Theoretical Background Scripts

The following tables contain a mapping of Python scripts to the content in the theoretical background where their results appear.

### Violation of Assumptions

| **Method** | **Assumption** | **Figure** | **File** |
| --- | --- | --- | --- |
| Right-tailed one-sided t-test | Normality | Figure 2 | [t_test_no_normality.py](theoretical%20background%20scripts/t_test_no_normality.py) |
| Right-tailed one-sided t-test | Independence of oberservations | Table 1 | [t_test_no_independence.py](theoretical%20background%20scripts/t_test_no_independence.py) |
| Two-sided t-test for independent samples | Homoscedasticity | Table 2 | [t_test_no_homoscedasticity.py](theoretical%20background%20scripts/t_test_no_homoscedasticity.py) |
| ANCOVA | Linearity | Figure 3, Table 3-5 | [ancova_no_linearity.py](theoretical%20background%20scripts/ancova_no_linearity.py) |
| Mann-Whitney U test | Independence of observations | Table 6 | [mwu_test_no_independence.py](theoretical%20background%20scripts/mwu_test_no_independence.py) |
| EFA | Independence of observations | Figure 4 | [efa_no_independence.py](theoretical%20background%20scripts/efa_no_independence.py) |
| EFA | Linearity | Table 8-9 | [efa_no_linearity.py](theoretical%20background%20scripts/efa_no_linearity.py) |
| PCA | Homoscedasticity | Table 10-11 | [pca_no_homoscedasticity.py](theoretical%20background%20scripts/pca_no_homoscedasticity.py) |
| UMAP | Homoscedasticity | Figure 5 | [umap_no_homoscedasticity.py](theoretical%20background%20scripts/umap_no_homoscedasticity.py) |
| Pearson's Correlation | Linearity scope | Table 12, Figure 6 | [pearson_coefficient_no_linearity.py](theoretical%20background%20scripts/pearson_coefficient_no_linearity.py) |
| Spearman's Correlation | Monotonicity scope | Table 13-14 | [spearman_coefficient_no_monotonicity.py](theoretical%20background%20scripts/spearman_coefficient_no_monotonicity.py) |
| HDBSCAN | Homoscedasticity | Figure 7 | [hdbscan_no_homoscedasticity.py](theoretical%20background%20scripts/hdbscan_no_homoscedasticity.py) |
| LDA | Independence of observations | Figure 8 | [lda_no_independence.py](theoretical%20background%20scripts/lda_no_independence.py) |
| LDA | Linearity | Figure 9 | [lda_no_linearity.py](theoretical%20background%20scripts/lda_no_linearity.py) |
| ML model + Shapley value | Absence of multicollinearity | Figure 10 | [shapley_multicollinearity.py](theoretical%20background%20scripts/shapley_multicollinearity.py) |

### Checking and Meeting Assumptions

| **Assumption** | **Method** | **Figure** | **File** |
| --- | --- | --- | --- |
| Normality | Q-Q-Plot | Figure 11 | [normality_q_q_plot.py](theoretical%20background%20scripts/normality_q_q_plot.py) |
| Normality | Skewness and excess kurtosis | Table 15 | [normality_measures_of_shape.py](theoretical%20background%20scripts/normality_measures_of_shape.py) |
| Normality | Shapiro-Wilk test | Table 16 | [normality_shapiro_wilk_test.py](theoretical%20background%20scripts/normality_shapiro_wilk_test.py) |
| Normality | Logarithmic, Box-Cox, and Yue-Johnson transformations | Table 17 | [normality_shapiro_wilk_test_transformed.py](theoretical%20background%20scripts/normality_shapiro_wilk_test_transformed.py) |
| Homoscedasticity | Bartlett's and Levene's test | Table 18 | [homoscedasticity_bartlett_levene.py](theoretical%20background%20scripts/homoscedasticity_bartlett_levene.py) |
| Linearity | Ramsey RESET test | Table 19 | [linearity_ramsey_reset_test.py](theoretical%20background%20scripts/linearity_ramsey_reset_test.py) |
| Absence of multicollinearity | VIF | Table 20 | [multicollinearity_vif.py](theoretical%20background%20scripts/multicollinearity_vif.py) |

## Applied Part Scripts

A description of the used variables can be found [here](VARIABLES.md).
