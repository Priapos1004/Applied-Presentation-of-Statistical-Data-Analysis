import numpy as np
from scipy import stats

# Setting a seed for reproducibility
np.random.seed(42)

# Parameters
n_simulations = 10000
sample_size = 30
alpha = 0.05  # Significance level

# Simulate independent data and dependent data
independent_rejections = 0
dependent_rejections = 0

for _ in range(n_simulations):
    # Independent data
    data_independent = np.random.normal(loc=0, scale=1, size=sample_size)
    _, p_val_independent = stats.ttest_1samp(data_independent, popmean=0, alternative="greater")
    
    # Check if we reject the null hypothesis
    if p_val_independent < alpha:
        independent_rejections += 1

    # Generate correlated data using multivariate normal distribution
    mean = np.zeros(sample_size)
    cov = np.full((sample_size, sample_size), 0.5)  # Correlation of 0.5
    np.fill_diagonal(cov, 1)  # Variance of 1 on the diagonal
    data_dependent = np.random.multivariate_normal(mean, cov)
    _, p_val_dependent = stats.ttest_1samp(data_dependent, popmean=0, alternative="greater")
    
    # Check if we reject the null hypothesis
    if p_val_dependent < alpha:
        dependent_rejections += 1

# Calculate Type I error rate
type_1_error_independent = independent_rejections / n_simulations
type_1_error_dependent = dependent_rejections / n_simulations
