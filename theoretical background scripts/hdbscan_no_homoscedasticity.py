import hdbscan
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Generate data: concentric circles (for complex non-linear shapes) and blob
n_samples = 300
X1, _ = make_circles(n_samples=n_samples, factor=0.5, noise=0.05)  # Circular data with noise
X2, _ = make_blobs(n_samples=n_samples, centers=[[3, 3]], cluster_std=0.3)  # Blob data for contrast

# Combine the data
X = np.vstack((X1, X2))

# Shift variables for only positive values (stylistic choice)
X[:, 0] += 2
X[:, 1] += 2

# Scale Y-axis for variation
X[:, 1] *= 100

# Apply HDBSCAN without scaling
hdbscan_no_scaling = hdbscan.HDBSCAN(min_samples=5, min_cluster_size=10)
labels_no_scaling = hdbscan_no_scaling.fit_predict(X)

# Scale the features and apply HDBSCAN again
X_scaled = StandardScaler().fit_transform(X)
hdbscan_scaled = hdbscan.HDBSCAN(min_samples=5, min_cluster_size=10)
labels_scaled = hdbscan_scaled.fit_predict(X_scaled)


if __name__=='__main__':
    # Create subplots to show the comparison side by side
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Original data plot
    axs[0].scatter(X[:, 0], X[:, 1], c='blue', marker='o', edgecolor='k')
    axs[0].set_xlabel("X1")
    axs[0].set_ylabel("X2")

    # HDBSCAN without scaling
    axs[1].scatter(X[labels_no_scaling != -1, 0], X[labels_no_scaling != -1, 1], c=labels_no_scaling[labels_no_scaling != -1], cmap='plasma', marker='o', edgecolor='k')
    axs[1].scatter(X[labels_no_scaling == -1, 0], X[labels_no_scaling == -1, 1], c='black', marker='x', label='Noise')
    axs[1].set_xlabel("X1")
    axs[1].set_ylabel("X2")

    # HDBSCAN after scaling
    axs[2].scatter(X_scaled[labels_scaled != -1, 0], X_scaled[labels_scaled != -1, 1], c=labels_scaled[labels_scaled != -1], cmap='plasma', marker='o', edgecolor='k')
    axs[2].scatter(X_scaled[labels_scaled == -1, 0], X_scaled[labels_scaled == -1, 1], c='black', marker='x', label='Noise')
    axs[2].set_xlabel("Scaled X1")
    axs[2].set_ylabel("Scaled X2")

    # Show the plots
    plt.tight_layout()
    plt.show()
