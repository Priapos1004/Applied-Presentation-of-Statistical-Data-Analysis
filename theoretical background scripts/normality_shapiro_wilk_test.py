from scipy import stats
from t_test_no_normality import data_normal, data_skewed

# Shapiro-Wilk test
stat1, p1 = stats.shapiro(data_skewed)
stat2, p2 = stats.shapiro(data_normal)

print(f'Data skewed: statistics {stat1:.4f} and p-value {p1:.4f}')
print(f'Data normal: statistics {stat2:.4f} and p-value {p2:.4f}')
