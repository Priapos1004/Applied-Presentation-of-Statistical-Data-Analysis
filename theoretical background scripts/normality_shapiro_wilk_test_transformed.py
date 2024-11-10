import numpy as np
from scipy import stats
from t_test_no_normality import data_skewed

# 1. Apply Logarithmic Transformation
data_log = np.log(data_skewed)
# Shapiro-Wilk test for logarithmic transformed data
stat_log, p_log = stats.shapiro(data_log)
print(f'Log-transformed data: statistics {stat_log:.4f} and p-value={p_log:.4f}')

# 2. Apply Box-Cox Transformation
data_boxcox, lambda_boxcox = stats.boxcox(data_skewed)
# Shapiro-Wilk test for Box-Cox transformed data
stat_boxcox, p_boxcox = stats.shapiro(data_boxcox)
print(f'Box-Cox transformed data: statistics {stat_boxcox:.4f}, p-value {p_boxcox:.4f}, and lambda {lambda_boxcox:.2f}')

# 3. Apply Johnson SU Transformation
data_yeojohnson, lambda_yeojohnson = stats.yeojohnson(data_skewed)
# Shapiro-Wilk test for Johnson-transformed data
stat_yeojohnson, p_yeojohnson = stats.shapiro(data_yeojohnson)
print(f'Johnson transformed data: statistics {stat_yeojohnson:.4f}, p-value {p_yeojohnson:.4f}, and lambda {lambda_yeojohnson:.2f}')
