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
# Johnsonsu.fit returns four parameters (a, b, loc, scale)
params = stats.johnsonsu.fit(data_skewed)
# Transform data using the fitted Johnson parameters
data_johnson_transformed = stats.johnsonsu.rvs(*params, size=len(data_skewed))
# Check for any infinite or NaN values
data_johnson_transformed = data_johnson_transformed[np.isfinite(data_johnson_transformed)]
# Shapiro-Wilk test for Johnson-transformed data
stat_johnson, p_johnson = stats.shapiro(data_johnson_transformed)
print(f'Johnson transformed data: statistics {stat_johnson:.4f} and p-value {p_johnson:.4f}')
