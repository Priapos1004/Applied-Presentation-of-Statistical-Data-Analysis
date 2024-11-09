from pca_no_homoscedasticity import data_correlated_mv, data_scaled_corr_mv
from scipy.stats import bartlett, levene

# Perform Bartlett's test of sphericity
bartlett_test_unscaled = bartlett(data_correlated_mv['X1'], data_correlated_mv['X2'], data_correlated_mv['X3'])
# Perform Levene's test for equal variances
levene_test_unscaled = levene(data_correlated_mv['X1'], data_correlated_mv['X2'], data_correlated_mv['X3'])
print(f'Data unscaled: bartlett\'s test p-value {bartlett_test_unscaled.pvalue} and levene\'s test p-value {levene_test_unscaled.pvalue}')

# Perform Bartlett's test of sphericity
bartlett_test_scaled = bartlett(data_scaled_corr_mv[:, 0], data_scaled_corr_mv[:, 1], data_scaled_corr_mv[:, 2])
# Perform Levene's test for equal variances
levene_test_scaled = levene(data_scaled_corr_mv[:, 0], data_scaled_corr_mv[:, 1], data_scaled_corr_mv[:, 2])
print(f'Data scaled: bartlett\'s test p-value {bartlett_test_scaled.pvalue} and levene\'s test p-value {levene_test_scaled.pvalue:.4f}')
