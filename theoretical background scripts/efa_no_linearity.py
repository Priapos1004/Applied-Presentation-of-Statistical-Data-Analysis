import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from horns import parallel_analysis
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(2)

# Number of observations
n = 100 

# Generate latent variable for durability (related to uses, time_owned, and returned)
durability_factor = np.random.normal(0, 1, n)

# Variables derived from the durability latent factor
uses_linear = (durability_factor * 100 + np.random.normal(400, 100, n)).astype(int)
time_owned_linear = np.clip((durability_factor * 0.5 + np.random.normal(1.5, 0.5, n)), 0.5, 3.0).round(2)
returned_linear = np.select([durability_factor > 0.5, durability_factor < -0.5], [1, 2], default=3)

# Introduce a non-linear relationship for the derived variables
uses_nonlinear = np.clip(((durability_factor ** 2) * 300 + np.random.normal(500, 50, n)), 200, 1200).astype(int)
time_owned_nonlinear = np.clip((durability_factor * 0.5 + np.random.normal(1.5, 0.5, n)), 0.5, 3.0).round(2)
returned_nonlinear = np.select([durability_factor > 0.5, durability_factor < -0.5], [1, 2], default=3)


# Generate latent variable for user characteristics (related to age, previous_customer, and price)
user_factor = np.random.normal(0, 1, n)

# Variables derived from the user latent factor
age = (user_factor * 10 + 35 + np.random.normal(10, 5, n)).astype(int)
previous_customer = (user_factor * 2 + np.random.normal(3, 1, n)).astype(int)
price = (user_factor * 50 + 200 + np.random.normal(50, 20, n)).round(2)

# Ensure previous_customer has non-negative values
previous_customer = np.clip(previous_customer, 0, None)

# Create a DataFrame
df_correlated = pd.DataFrame({
    'product_id': np.random.randint(0, 11, size=n),
    'uses_linear': uses_linear,
    'time_owned_linear': time_owned_linear,
    'returned_linear': returned_linear,
    'uses_nonlinear': uses_nonlinear,
    'time_owned_nonlinear': time_owned_nonlinear,
    'returned_nonlinear': returned_nonlinear,
    'age': age,
    'previous_customer': previous_customer,
    'price': price
})

def run_EFA(data):
    # Step 1: Parallel Analysis
    # Run parallel analysis to determine the number of factors
    n_factors = parallel_analysis(
        data=data,
        analysis_type="fa",
    )

    # Step 2: EFA (Exploratory Factor Analysis)
    # Run EFA using principal axis factoring and promax rotation
    fa = FactorAnalyzer(n_factors=n_factors, method='principal', rotation='promax')
    fa.fit(data)

    # Step 3: EFA Results
    # Get the factor loadings
    loadings = pd.DataFrame(fa.loadings_, 
                            index=['uses', 'time_owned', 'returned', 'age', 'previous_customer', 'price'], 
                            columns=[f'Factor{i+1}' for i in range(n_factors)])
    return loadings


# Formating data frame
data_linear = df_correlated[['uses_linear', 'time_owned_linear', 'returned_linear', 'age', 'previous_customer', 'price']].to_numpy()
data_nonlinear = df_correlated[['uses_nonlinear', 'time_owned_nonlinear', 'returned_nonlinear', 'age', 'previous_customer', 'price']].to_numpy()

# Scale data to satisfy homoscedasticity assumption
scaler_linear = StandardScaler()
data_linear_scaled = scaler_linear.fit_transform(data_linear)
scaler_nonlinear = StandardScaler()
data_nonlinear_scaled = scaler_nonlinear.fit_transform(data_nonlinear)

loadings_linear = run_EFA(data_linear_scaled)
loadings_nonlinear = run_EFA(data_nonlinear_scaled)
