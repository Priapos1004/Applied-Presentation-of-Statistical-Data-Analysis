import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Set random seed for reproducibility
np.random.seed(42)

# Define the number of samples per group
n_samples = 100

# Generate 'time owned' as in the ANCOVA example
time_owned_ancova = np.concatenate([
    np.random.uniform(0.5, 2, n_samples),
    np.random.uniform(1, 5, n_samples),
    np.random.uniform(5, 15, n_samples)
])
# Generate 'time owned' uniformly distributed
time_owned_uni = np.random.uniform(0.5, 15, 3*n_samples)

# Generate base durability
durability = np.random.normal(2200, 100, 3*n_samples)

def reciprocal_func(time_owned):
    values = np.where(
        np.arange(len(time_owned)) % 4 == 0,  # Condition for indices
        1000/time_owned - 800,  # Operation for indices divisible by 4
        1000/time_owned - 1600   # Operation for other indices
    )
    return values

# Create a DataFrame
def create_df(durability: np.ndarray, time_owned: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({
        'Time_Owned': time_owned,
        'Durability_linear': durability - 80.0 * time_owned,
        'Durability_quadratic': durability - 8.0 * time_owned**2,
        'Durability_quadratic_2': durability - 8.0 * (time_owned - 7)**2,
        'Durability_reciprocal': durability + reciprocal_func(time_owned),
        'Durability_sine': durability - 200 * np.sin(time_owned),
    })
    return df

df_ancova = create_df(durability, time_owned_ancova)
df_uni = create_df(durability, time_owned_uni)

# Get column names
durability_column_names = [col for col in df_ancova.columns if "Durability" in col]

# 'spearmanr' returns a tuple of spearman's coefficient and p-value of hypothesis test -> only selecting the coefficient
durability_spearman_coef_ancova = [spearmanr(df_ancova['Time_Owned'], df_ancova[durability_col])[0] for durability_col in durability_column_names]
durability_spearman_coef_uni = [spearmanr(df_uni['Time_Owned'], df_uni[durability_col])[0] for durability_col in durability_column_names]
