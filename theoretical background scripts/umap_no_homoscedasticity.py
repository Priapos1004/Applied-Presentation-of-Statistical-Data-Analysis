import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Generate data: two concentric 3D spheres and a 3D blob
n_samples = 300
# Sphere 1 (radius 1)
sphere_1 = np.random.normal(0, 0.2, size=(n_samples, 3))
sphere_1 = sphere_1 / np.linalg.norm(sphere_1, axis=1)[:, np.newaxis]
# Sphere 2 (radius 2)
sphere_2 = np.random.normal(0, 0.2, size=(n_samples, 3))
sphere_2 = 2 * (sphere_2 / np.linalg.norm(sphere_2, axis=1)[:, np.newaxis])

# Blob data for contrast
X_blob, y_blob = make_blobs(n_samples=n_samples, centers=[[3, 3, 3]], cluster_std=0.3)

# Combine the datasets
X = np.vstack((sphere_1, sphere_2, X_blob))
y = np.hstack((np.zeros(n_samples), np.ones(n_samples), np.full(n_samples, 2)))

# Distort X-axis
X[:, 0] *= 100

# UMAP without scaling
umap_model_no_scaling = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding_no_scaling = umap_model_no_scaling.fit_transform(X)

# Scale features and apply UMAP
X_scaled = StandardScaler().fit_transform(X)
umap_model_scaling = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding_scaling = umap_model_scaling.fit_transform(X_scaled)

if __name__=='__main__':
    # Create subplots for comparison
    fig = plt.figure(figsize=(18, 6))

    # Original data plot (3D)
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, marker='o', edgecolor='k', cmap='plasma')
    ax.set_xlabel("X1", labelpad=-30)
    ax.set_ylabel("X2", labelpad=-30)
    ax.set_zlabel("X3", labelpad=-30)

    # UMAP without scaling (2D)
    ax2 = fig.add_subplot(132)
    ax2.scatter(embedding_no_scaling[:, 0], embedding_no_scaling[:, 1], c=y, cmap='plasma', marker='o', edgecolor='k')
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")

    # UMAP after scaling (2D)
    ax3 = fig.add_subplot(133)
    ax3.scatter(embedding_scaling[:, 0], embedding_scaling[:, 1], c=y, cmap='plasma', marker='o', edgecolor='k')
    ax3.set_xlabel("UMAP 1")
    ax3.set_ylabel("UMAP 2")

    # Show plots side by side
    plt.tight_layout()
    plt.show()
