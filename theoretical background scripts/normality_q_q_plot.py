import matplotlib.pyplot as plt
import scipy.stats as stats
from t_test_no_normality import data_normal, data_skewed

# Create a figure with 2 subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Q-Q plot for skewed data
stats.probplot(data_skewed, dist="norm", plot=axes[0])
axes[0].set_ylabel('Sample Quantiles')
axes[0].set_xlabel('Theoretical Quantiles')
axes[0].set_title('')

# Q-Q plot for normally distributed data
stats.probplot(data_normal, dist="norm", plot=axes[1])
axes[1].set_ylabel('')
axes[1].set_xlabel('Theoretical Quantiles')
axes[1].set_title('')

# Display the plots
plt.tight_layout()
plt.show()
