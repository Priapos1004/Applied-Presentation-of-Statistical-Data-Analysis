import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate data
uses = np.random.randint(1, 500, size=n_samples)  # Number of uses before product broke
time_owned = np.random.uniform(0.1, 10, size=n_samples)  # Time owned in years

# Returned: 1 for first year, 2 for second year, 3 for never returned after purchase (2-year guarantee)
returned = np.random.choice([1, 2, 3], size=n_samples, p=[0.2, 0.3, 0.5])

# Age of the buyer (randomly generated between 18 and 70 years)
age = np.random.randint(18, 70, size=n_samples)

# Number of products previously purchased by the buyer
previous_customer = np.random.randint(0, 10, size=n_samples)

# Price of the product (in euros)
price_uncorrelated = np.random.uniform(20, 1200, size=n_samples)
price_correlated = (uses + time_owned * 65 + np.random.uniform(-30, 30, size=n_samples)) # highly correlated

# Create a binary target variable `future_customer` based on some of the features + some random noise
future_customer = np.where(np.random.rand(n_samples) < 0.01, 
                           np.random.rand(n_samples) > 0.5, 
                           (uses > np.mean(uses)) & (previous_customer > np.percentile(previous_customer, 25)) & (time_owned > np.mean(time_owned)))
future_customer = future_customer.astype(int)  # Convert to binary (1 or 0)

# Create DataFrame
data = pd.DataFrame({
    'uses': uses,
    'time_owned': time_owned,
    'returned': returned,
    'age': age,
    'previous_customer': previous_customer,
    'price_uncorrelated': price_uncorrelated,
    'price_correlated': price_correlated,
    'future_customer': future_customer,
})

# Train-test split
y = data['future_customer']

X1 = data.drop(columns=['future_customer', 'price_uncorrelated'])
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y, test_size=0.2, random_state=42)

X2 = data.drop(columns=['future_customer', 'price_correlated'])
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, test_size=0.2, random_state=42)

# Train XGBoost models
model1 = XGBClassifier(random_state=42)
model1.fit(X_train1, y_train1)

model2 = XGBClassifier(random_state=42)
model2.fit(X_train2, y_train2)

# SHAP explanations
# Important: SHAP is used on test data because 
# we want to see the learned general patterns 
# and avoid seeing overfitted patterns
explainer1 = shap.Explainer(model1, seed=42)
shap_values1 = explainer1(X_test1)

explainer2 = shap.Explainer(model2, seed=42)
shap_values2 = explainer2(X_test2)

if __name__=='__main__':
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # Increased figure size for more space

    # Plot SHAP summary for model 1 (price_correlated) using legacy method without ax argument
    plt.sca(axes[0])  # Set current axis
    shap.summary_plot(shap_values1, X_test1, show=False)

    # Change the automatic SHAP caption for the first plot
    axes[0].set_xticks([-4, -2, 0, 2, 4])
    axes[0].set_xlabel("Shapley value")

    # Hide the color bar for the first plot
    cbar = plt.gcf().get_axes()[-1]  # Access the color bar
    cbar.set_visible(False)  # Hide the color bar

    # Plot SHAP summary for model 2 (price_uncorrelated) using legacy method without ax argument
    plt.sca(axes[1])  # Set current axis
    shap.summary_plot(shap_values2, X_test2, show=False)

    # Change the automatic SHAP caption for the second plot
    axes[1].set_xticks([-4, -2, 0, 2, 4])
    axes[1].set_xlabel("Shapley value")

    # Adjust layout to avoid overlap
    plt.subplots_adjust(left=0.26, right=0.96, top=0.9, bottom=0.15, wspace=0.7)
    plt.show()
