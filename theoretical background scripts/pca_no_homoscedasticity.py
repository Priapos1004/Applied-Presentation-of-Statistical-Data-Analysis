import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Define the number of samples
n_samples = 100

# Parameters for the multivariate normal distribution
mean = [0, 0, 0]  # Mean vector for X1, X2, and X3
cov = [[1, 9, 1], [9, 100, 1], [1, 1, 1000]]  # Covariance matrix

# Generate synthetic correlated data using a multivariate normal distribution
data_mv = multivariate_normal.rvs(mean=mean, cov=cov, size=n_samples)
data_correlated_mv = pd.DataFrame(data_mv, columns=['X1', 'X2', 'X3'])

# Apply PCA without scaling
pca_correlated_mv = PCA(n_components=3)
principal_components_correlated_mv = pca_correlated_mv.fit_transform(data_correlated_mv)

# Apply PCA with scaling (standardisation)
scaler = StandardScaler()
data_scaled_corr_mv = scaler.fit_transform(data_correlated_mv)
pca_scaled_corr_mv = PCA(n_components=3)
principal_components_scaled_corr_mv = pca_scaled_corr_mv.fit_transform(data_scaled_corr_mv)

# Loadings and explained variance for unscaled PCA
loadings_unscaled = pca_correlated_mv.components_
explained_variance_no_scaling_corr_mv = pca_correlated_mv.explained_variance_ratio_

# Loadings and explained variance for scaled PCA
loadings_scaled = pca_scaled_corr_mv.components_
explained_variance_with_scaling_corr_mv = pca_scaled_corr_mv.explained_variance_ratio_
