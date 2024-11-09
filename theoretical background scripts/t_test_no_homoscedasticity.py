import numpy as np
from scipy import stats

# Set the random seed for reproducibility
np.random.seed(694)

n1 = 15
data_normal1 = np.random.normal(loc=0, scale=4, size=n1)
data_normal1 = [round(elem) for elem in data_normal1]
# [0, 4, -2, 4, 0, -2, -5, 6, 6, 4, 3, 1, -5, -1, 1]

n2 = 25
data_normal2 = np.random.normal(loc=8, scale=13, size=n2)
data_normal2 = [round(elem) for elem in data_normal2]
# [27, -27, 20, -17, 19, -15, -8, 23, 13, 21, 18, 19, 37, -14, 21, 10, -7, -13, -13, 6, 34, 28, -10, 20, 1]

# Perform two-sided t-test for independent samples
t_stat, p_value = stats.ttest_ind(data_normal1, data_normal2)
