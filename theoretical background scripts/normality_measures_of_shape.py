from scipy.stats import kurtosis, skew
from t_test_no_normality import data_normal, data_skewed

print(f'Data skewed: skewness {skew(data_skewed):.4f} and excess kurtosis {kurtosis(data_skewed, fisher=True):.4f}')
print(f'Data normal: skewness {skew(data_normal):.4f} and excess kurtosis {kurtosis(data_normal, fisher=True):.4f}')
