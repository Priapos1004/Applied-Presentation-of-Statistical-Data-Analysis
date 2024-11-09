import pandas as pd
from shapley_multicollinearity import data
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create the first DataFrame with price_uncorrelated
data_uncorrelated = data[['uses', 'time_owned', 'returned', 'age', 'previous_customer', 'price_uncorrelated']]
# Calculate VIF for each feature
vif_data_uncorrelated = pd.DataFrame()
vif_data_uncorrelated['Feature'] = data_uncorrelated.columns
vif_data_uncorrelated['VIF'] = [variance_inflation_factor(data_uncorrelated.values, i) for i in range(len(data_uncorrelated.columns))]
print(f'VIF for price_uncorrelated case:\n{vif_data_uncorrelated}\n')

# Create the second DataFrame with price_correlated
data_correlated = data[['uses', 'time_owned', 'returned', 'age', 'previous_customer', 'price_correlated']]
# Calculate VIF for each feature
vif_data_correlated = pd.DataFrame()
vif_data_correlated['Feature'] = data_correlated.columns
vif_data_correlated['VIF'] = [variance_inflation_factor(data_correlated.values, i) for i in range(len(data_correlated.columns))]
print(f'VIF for price_correlated case:\n{vif_data_correlated}\n')
