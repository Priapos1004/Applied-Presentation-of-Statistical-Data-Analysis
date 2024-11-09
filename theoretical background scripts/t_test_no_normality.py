import numpy as np
from scipy import stats

# number of observations
n = 20

# Generate a skewed distribution (e.g., exponential distribution)
np.random.seed(42) # Seed for reproducibility
data_skewed = 2600 - (np.random.exponential(scale=2, size=n) * 300).round()  # Right-skewed data

# Generate a normal distribution with the same mean and std as data_skewed
np.random.seed(27) # Seed for reproducibility
data_normal = np.random.normal(loc=0, scale=np.std(data_skewed), size=n).round()
data_normal += np.mean(data_skewed) - np.mean(data_normal) # Adjust mean

# Define the threshold for the one-sided t-test
threshold = 1900

# Perform a right-tailed t-test against the threshold for both samples
t_stat_skewed, p_value_skewed = stats.ttest_1samp(data_skewed, threshold, alternative="greater")
t_stat_normal, p_value_normal = stats.ttest_1samp(data_normal, threshold, alternative="greater")
