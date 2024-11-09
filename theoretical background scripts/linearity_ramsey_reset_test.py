import statsmodels.api as sm
from ancova_no_linearity import df
from statsmodels.stats.diagnostic import linear_reset

# Adding constant for intercept
X = sm.add_constant(df['Time_Owned'])

# Fitting the linear regression model
model_linear = sm.OLS(df['Durability_linear'], X).fit()
# Performing the RESET test
reset_test_linear = linear_reset(model_linear, power=2, use_f=True)
print(f'Data linear: Ramsey RESET test p-value {reset_test_linear.pvalue:.4f}')

# Fitting the linear regression model
model_nonlinear = sm.OLS(df['Durability_nonlinear'], X).fit()
# Performing the RESET test
reset_test_nonlinear = linear_reset(model_nonlinear, power=2, use_f=True)
print(f'Data nonlinear: Ramsey RESET test p-value {reset_test_nonlinear.pvalue}')
