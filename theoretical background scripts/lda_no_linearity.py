import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Generate non-linear data: two circles, one inside the other
X, y = make_circles(n_samples=500, factor=0.3, noise=0.05, random_state=42)

# Apply StandardScaler to standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit LDA to the data
lda = LinearDiscriminantAnalysis()
lda.fit(X_scaled, y)

# Predict labels for the original data
y_pred = lda.predict(X_scaled)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)

# Create a meshgrid for plotting the decision boundary
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Predict the class labels for each point in the meshgrid
Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


if __name__=='__main__':
    # Plot the decision boundary and the data points
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)

    # Correct the class labels
    plt.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1], color='red', label='Class 0 (Inner Circle)')
    plt.scatter(X_scaled[y == 0][:, 0], X_scaled[y == 0][:, 1], color='blue', label='Class 1 (Outer Circle)')

    # Add plot labels
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.legend()
    plt.grid(True)
    plt.show()
