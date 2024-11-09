import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for simulation
sample_sizes = [300, 450, 600]  # Different numbers of observations per class
n_runs = 100  # Number of runs to compare
rho_values = np.linspace(0, 0.9, 15)  # Range of autocorrelation values to test
test_size = 0.2  # Proportion of data for testing

# Storage for final accuracy scores
accuracy_scores = []

# Loop over each sample size
for i, n in enumerate(sample_sizes):
    print(f"Running simulation for {n} samples per class")

    # Mean vectors of classes
    mean_vector_class1 = np.zeros(n)
    mean_vector_class2 = np.ones(n)

    # Storage for accuracy scores
    train_accuracy_means = []
    test_accuracy_means = []
    independent_test_accuracy_means = []

    # Generate independent test data (rho = 0)
    cov_matrix_independent = np.eye(n)

    latent_factor_ind_class1 = np.random.multivariate_normal(mean_vector_class1, cov_matrix_independent)
    variable1_ind_class1 = (latent_factor_ind_class1 * 100 + np.random.normal(400, 100, n)).astype(int)
    variable2_ind_class1 = (latent_factor_ind_class1 * 4 + 15 + np.random.normal(20, 4, n)).astype(int)
    independent_data_class1 = np.column_stack((variable1_ind_class1, variable2_ind_class1))

    latent_factor_ind_class2 = np.random.multivariate_normal(mean_vector_class2, cov_matrix_independent)
    variable1_ind_class2 = (latent_factor_ind_class2 * 100 + np.random.normal(400, 100, n)).astype(int)
    variable2_ind_class2 = (latent_factor_ind_class2 * 4 + 15 + np.random.normal(20, 4, n)).astype(int)
    independent_data_class2 = np.column_stack((variable1_ind_class2, variable2_ind_class2))

    # Combine the independent test data and create labels
    X_test_independent = np.vstack((independent_data_class1, independent_data_class2))
    y_test_independent = np.hstack((np.zeros(n), np.ones(n)))

    # Loop through different rho values
    for rho in tqdm(rho_values, desc=f"Autocorrelation values for {n} samples"):
        # Storage for accuracy scores in this rho iteration
        train_acc_dep = []
        test_acc_dep = []
        test_acc_independent = []

        # Covariance matrix for dependency
        cov_matrix = rho * np.ones((n, n)) + (1 - rho) * np.eye(n)

        # Run multiple simulations
        for _ in range(n_runs):
            # Generate dependent data class1
            latent_factor_dep_class1 = np.random.multivariate_normal(mean_vector_class1, cov_matrix)
            variable1_dep_class1 = (latent_factor_dep_class1 * 100 + np.random.normal(400, 100, n)).astype(int)
            variable2_dep_class1 = (latent_factor_dep_class1 * 4 + 15 + np.random.normal(20, 4, n)).astype(int)
            data_dep_class1 = np.column_stack((variable1_dep_class1, variable2_dep_class1))

            # Generate dependent data class2
            latent_factor_dep_class2 = np.random.multivariate_normal(mean_vector_class2, cov_matrix)
            variable1_dep_class2 = (latent_factor_dep_class2 * 100 + np.random.normal(400, 100, n)).astype(int)
            variable2_dep_class2 = (latent_factor_dep_class2 * 4 + 15 + np.random.normal(20, 4, n)).astype(int)
            data_dep_class2 = np.column_stack((variable1_dep_class2, variable2_dep_class2))

            # Combine the data and create labels
            X_auto = np.vstack((data_dep_class1, data_dep_class2))
            y_auto = np.hstack((np.zeros(n), np.ones(n)))

            # Split data into training and testing sets (autocorrelated test set)
            X_train, X_test, y_train, y_test = train_test_split(X_auto, y_auto, test_size=test_size, random_state=42)

            # Apply StandardScaler on the training data and scale the test data using the same scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Apply Fisher's LDA
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_train_scaled, y_train)

            # Predictions on both train and test sets
            y_train_pred = lda.predict(X_train_scaled)
            y_test_pred = lda.predict(X_test_scaled)
            y_test_ind_pred = lda.predict(scaler.transform(X_test_independent))

            # Calculate accuracy for train, autocorrelated test, and independent test
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            independent_test_acc = accuracy_score(y_test_independent, y_test_ind_pred)

            # Store accuracies
            train_acc_dep.append(train_acc)
            test_acc_dep.append(test_acc)
            test_acc_independent.append(independent_test_acc)

        train_accuracy_means.append(np.mean(train_acc_dep))
        test_accuracy_means.append(np.mean(test_acc_dep))
        independent_test_accuracy_means.append(np.mean(test_acc_independent))


    accuracy_scores.append({
        'Train Accuracy': train_accuracy_means,
        'Autocorrelated Test Accuracy': test_accuracy_means,
        'Independent Test Accuracy': independent_test_accuracy_means
    })

# 'accuracy_scores' could be used for plotting
