import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Set random seed for reproducibility
np.random.seed(42)

# Define the number of samples per group
n_samples = 100

# Generate age groups
age_groups = ['Teenager', 'Working Adult', 'Retired Adult']
age_group_data = np.repeat(age_groups, n_samples)

# Generate 'time owned' as covariate
time_owned = np.concatenate([
    np.random.uniform(0.5, 2, n_samples),  # Teenagers have owned the product for a shorter time
    np.random.uniform(1, 5, n_samples),    # Working adults have owned it for a moderate time
    np.random.uniform(5, 15, n_samples)    # Retired adults have owned it for a long time
])

# Generate base durability for groups
durability = np.concatenate([
        np.random.normal(2200, 100, n_samples),  # durability for teenagers
        np.random.normal(2200, 100, n_samples),  # durability for working adults
        np.random.normal(2200, 100, n_samples)   # durability for retired adults
    ])

# Create a DataFrame
df = pd.DataFrame({
    'Age_Group': age_group_data,
    'Time_Owned': time_owned,
    'Durability_linear': durability - 80.0 * time_owned,
    'Durability_nonlinear': durability - 8.0 * time_owned**2
})

def get_adjusted_means(
        data: pd.DataFrame,
        dependent_variable: str,
        model
    ) -> pd.DataFrame:
    # Create a DataFrame for raw means
    raw_means_df = data.groupby('Age_Group')[dependent_variable].mean().reset_index()
    raw_means_df.columns = ['Age_Group', 'Raw_Durability']

    # Create a DataFrame for adjusted means
    means_df = pd.DataFrame({
        'Age_Group': data['Age_Group'].unique(),
        'Time_Owned': [data['Time_Owned'].mean()] * len(data['Age_Group'].unique())
    })
    means_df['Adjusted_Durability'] = model.predict(means_df)

    # Combine both raw and adjusted means into a single DataFrame for comparison
    combined_means_df = pd.merge(raw_means_df, means_df[['Age_Group', 'Adjusted_Durability']], on='Age_Group')
    return combined_means_df

def fit_ancova(
        data: pd.DataFrame,
        dependent_variable: str
    ) -> tuple[float, pd.DataFrame]:
    # Fit the ANCOVA model
    model = ols(f'{dependent_variable} ~ C(Age_Group) + Time_Owned', data=data).fit()
    # Generate the ANCOVA table
    anova_table = sm.stats.anova_lm(model, typ=2)
    # Get p-value
    p_value = anova_table.loc['C(Age_Group)', 'PR(>F)']

    adjusted_means_df = get_adjusted_means(data, dependent_variable, model)
    return p_value, adjusted_means_df

p_value_linear, adjusted_means_df_linear = fit_ancova(df, "Durability_linear")
p_value_nonlinear, adjusted_means_df_nonlinear = fit_ancova(df, "Durability_nonlinear")
