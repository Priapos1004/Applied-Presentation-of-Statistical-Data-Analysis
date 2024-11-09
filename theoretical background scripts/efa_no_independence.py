import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for simulation
n = 300  # Number of observations
n_factors = 1  # Number of latent factors for PAF
n_runs = 500  # Number of runs to compare
rho_values = np.linspace(0, 0.9, 20)  # Range of autocorrelation values to test

# Storage for explained variance
explained_variance_means = []

# Loop through different rho values
for rho in tqdm(rho_values, desc="Autocorrelation values"):
    # Storage for explained variance in this rho iteration
    var_dep = []

    # Covariance matrix for dependency
    mean_vector = np.zeros(n)
    cov_matrix = rho * np.ones((n, n)) + (1 - rho) * np.eye(n)

    # Run multiple simulations
    for _ in range(n_runs):
        # Generate dependent data
        latent_factor_dep = np.random.multivariate_normal(mean_vector, cov_matrix)
        variable1_dep = (latent_factor_dep * 100 + np.random.normal(400, 100, n)).astype(int)
        variable2_dep = (latent_factor_dep * 4 + 15 + np.random.normal(20, 4, n)).astype(int)
        data_dep = pd.DataFrame({'Variable1': variable1_dep, 'Variable2': variable2_dep})
        
        # Scale data to satisfy homoscedasticity
        scaler_dep = StandardScaler()
        data_dep_scaled = scaler_dep.fit_transform(data_dep)
        
        # Perform factor analysis
        fa_dep = FactorAnalyzer(n_factors=n_factors, method='principal', rotation=None)
        fa_dep.fit(data_dep_scaled)
        var_dep.append(fa_dep.get_factor_variance()[1][0])
    
    # Store the mean explained variance for this rho
    explained_variance_means.append(np.mean(var_dep))


if __name__=='__main__':
    # Plot explained variance for autocorrelation level
    plt.figure(figsize=(10, 6))
    plt.plot(rho_values, explained_variance_means, label='Explained Variance', marker='o')
    plt.xlabel('Autocorrelation Coefficient (rho)')
    plt.ylabel('Mean Explained Variance')
    plt.grid(True)
    plt.show()
