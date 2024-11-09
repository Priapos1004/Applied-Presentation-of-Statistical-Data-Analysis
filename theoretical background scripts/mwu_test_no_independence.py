import numpy as np
from scipy.stats import mannwhitneyu

# Setting a seed for reproducibility
np.random.seed(42)

# Simulation parameters
n_simulations = 1000
n_samples = 300
rho = 0.005  # autocorrelation parameter
alpha = 0.05 # significance level

# Mean vector for the multivariate normal distribution
mean_vector = np.zeros(n_samples)
# Covariance matrix for the multivariate normal distribution
cov_matrix = rho * np.ones((n_samples, n_samples)) + (1 - rho) * np.eye(n_samples)

# Storage for results
p_values_independent = []
p_values_dependent = []

# Simulation loop
for _ in range(n_simulations):
    # Scenario 1: Independent data
    group1_indep = np.random.normal(loc=0, scale=1, size=n_samples)
    group2_indep = np.random.normal(loc=0, scale=1, size=n_samples)

    # Perform the MWU test on the independent data
    _, p_indep = mannwhitneyu(group1_indep, group2_indep, alternative='two-sided')
    p_values_independent.append(p_indep)
    
    # Scenario 2: Dependent data (introducing autocorrelation)
    group1_dep = np.random.multivariate_normal(mean_vector, cov_matrix)
    group2_dep = np.random.multivariate_normal(mean_vector, cov_matrix)
    
    # Perform the MWU test on the dependent data
    _, p_dep = mannwhitneyu(group1_dep, group2_dep, alternative='two-sided')
    p_values_dependent.append(p_dep)

# Calculate the type I error rates
significant_indep = np.mean(np.array(p_values_independent) < alpha)
significant_dep = np.mean(np.array(p_values_dependent) < alpha)
