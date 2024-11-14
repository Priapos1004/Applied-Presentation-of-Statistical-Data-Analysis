# Applied-Presentation-of-Statistical-Data-Analysis
This repository contains the scripts of the theoretical background and applied part of the bachelor thesis "Applied Presentation of Statistical Data Analysis" by Samuel Brinkmann.

## Python Environment

Python version 3.11.9 was used to run the scripts. Additionally, the following packages were installed using pip:

| **Library** | **Version** |
| --- | :---: |
| pandas | 2.2.3 |
| tqdm | 4.67.0 |
| scikit-learn | 1.5.2 |
| prince[^1] | 0.13.1 |
| xgboost | 2.1.2 |
| shap | 0.46.0 |
| factor-analyzer | 0.5.1 |
| matplotlib | 3.9.2 |
| statsmodels | 0.14.4 |
| umap-learn[^2] | 0.5.6 |
| hdbscan[^3] | 0.8.39 |
| horns[^4] | 0.6.0 |

You can also see [requirements.txt](requirements.txt) for the packages.

[^1]: M. Halford. *Prince*. MIT License. [Available at: https://github.com/MaxHalford/prince](https://github.com/MaxHalford/prince).

[^2]: L. McInnes, J. Healy, N. Saul, and L. Grossberger. *UMAP: Uniform Manifold Approximation and Projection*. The Journal of Open Source Software, 3(29), page 861, 2018.

[^3]: L. McInnes, J. Healy, and S. Astels. *hdbscan: Hierarchical density based clustering*. The Journal of Open Source Software, 2(11), 2017. doi: [10.21105/joss.00205](https://doi.org/10.21105%2Fjoss.00205).

[^4]: S. R. Mathias. *Horns: Horn's parallel analysis in Python*. 2024. [Available at: https://github.com/sammosummo/Horns](https://github.com/sammosummo/Horns).

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
| Normality | Logarithmic, Box-Cox, and Yeo-Johnson transformations | Table 17 | [normality_shapiro_wilk_test_transformed.py](theoretical%20background%20scripts/normality_shapiro_wilk_test_transformed.py) |
| Homoscedasticity | Bartlett's and Levene's test | Table 18 | [homoscedasticity_bartlett_levene.py](theoretical%20background%20scripts/homoscedasticity_bartlett_levene.py) |
| Linearity | Ramsey RESET test | Table 19 | [linearity_ramsey_reset_test.py](theoretical%20background%20scripts/linearity_ramsey_reset_test.py) |
| Absence of multicollinearity | VIF | Table 20 | [multicollinearity_vif.py](theoretical%20background%20scripts/multicollinearity_vif.py) |

## Applied Part Scripts

A description of the variables can be found [here](VARIABLES.md).

You can run the applied part's scripts with the command:

```shell
python applied\ part\ scripts/1_study.py
```

The scripts use an artificial dataset that retains the original structure of variables but is populated with randomly generated values. The ranges for these random values were estimated by ChatGPT-4, solely based on the variable descriptions provided, without incorporating any additional information or biases. The data generation script is named [generate_dataset.py](applied%20part%20scripts/generate_dataset.py) and complies with GDPR requirements.

You can modify the row number of the artificial dataset as well as the logging settings of the applied part's scripts in [config.py](applied%20part%20scripts/config.py).

### Removed Variables

Due to insufficient variation (i.e., where one category occurs in less than 0.5% of cases), the following binary variables were removed in the "Feature importance analysis" experiment.

#### Subset "over 65"

| **Variable** | **Note** |
| --- | --- |
| BAMOUNT_BINARY | - |
| SEMINARTYPE_i | Several seminartypes were removed |
| TEMPLATE_i | Several document templates were removed |
| firm_SUPPORT_REPRESENTATIVE_ratio, firm_REPRESENTATIVE_WrCStC | - |
| firm_COMPANYTYPE_i | Several company types were removed |
| COMMROLE_i_j | Several commitee and role combinations were removed |
| CATEGORY_i | Several contact reasons were removed, including all legal protection advice variables |
| CONTRIBUTION_IN_INSTALLMENTS | - |
| PAYDELAY_1A, PAYDELAY_1B, PAYDELAY_B | - |
| EMPLOYMENT_STATUS_i, EMPLOYMENT_TYPE_i | Several employment statuses and types were removed |

#### Subset "newcomer"

| **Variable** | **Note** |
| --- | --- |
| SEMINARTYPE_i | Several seminartypes were removed |
| TEMPLATE_i | Several document templates were removed |
| firm_REPRESENTATIVE_WrCStC, firm_VALID_TARIFF | - |
| firm_COMPANYTYPE_i, firm_SECTOR_i | Several company types and sectors were removed |
| COMMROLE_i_j | Several commitee and role combinations were removed |
| CATEGORY_i | Several contact reasons were removed, including the CATEGORY_other_legal_protection variable |
| CONTRIBUTION_exemption, CONTRIBUTION_IN_INSTALLMENTS | - |
| EMPLOYMENT_STATUS_i | Several employment statuses were removed |

Additionally, two BAMOUNT_i variables were removed due to insufficient variation.
