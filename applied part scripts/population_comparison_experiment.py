import pandas as pd
import numpy as np
from utils_analysis import setup_logger
from scipy.stats import chi2_contingency, anderson, boxcox, yeojohnson, levene, ks_2samp, mannwhitneyu, kurtosis, skew

logger = setup_logger(__name__, __file__)

def run_anderson(data: pd.Series) -> float:
    """
    Calculate lowest significance level at which the 
    data could be considered normally distributed

    Parameters
    ----------
    data : pd.Series
        data that shall be checked

    Returns
    -------
    lowest_significance_level : float
        lowest significance level at which the 
        data could be considered normally distributed and 10 
        if significance level is above 0.15
    """
    result = anderson(data)
    lowest_significance_level = 1000
    ad_statistic = result.statistic
    for significance_level, critical_value in zip(result.significance_level, result.critical_values):
        if ad_statistic < critical_value:
            lowest_significance_level = significance_level
    return lowest_significance_level / 100

def run_population_comparison_experiment(X: pd.DataFrame, y: pd.Series, subset: str):
    logger.info(f'##### Starting to analyse subset "{subset}" #####')
    binary_columns = [col for col in X.columns if 'legal_protection' in col]
    non_binary_columns = ['IN_CONTACT', 'STATUS_SENTIMENT', 'STRIKE_MONEY', 'STRIKE_LEN']
    significance_level = 0.05
    binary_results = {}
    # Investigate binary columns
    for binary_col in binary_columns:
        logger.info(f'Check binary column: {binary_col}')
        contingency_table = pd.crosstab(y, X[binary_col])
        # Perform the Chi-square test
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
        logger.debug(f'Chi-square test statistic: {chi2}')
        logger.info(f'p-value: {p}')
        logger.debug(f'Degrees of freedom: {dof}')
        logger.debug(f'Observed frequencies (True values):\n{contingency_table}\n')
        percent_contigency_table = contingency_table.T/contingency_table.T.sum()
        logger.debug(f'Observed percentages (True values):\n{percent_contigency_table}\n')
        logger.debug(f'Expected frequencies:\n{expected_df}\n')
        percent_expected_df = expected_df.T/expected_df.T.sum()
        logger.debug(f'Expected percentages:\n{percent_expected_df}\n')
        binary_results[binary_col] = {'test stat': chi2, 'p-value': p}

    pd.DataFrame(binary_results).T.to_csv(f'binary_comparison_{subset}.csv')

    non_binary_results = {}
    # Investigate non-binary columns
    for non_binary_col in non_binary_columns:
        for lag in [3, 6, 12]:
            column_name = f'lag{lag}_{non_binary_col}'
            churned_X = X[y == 1][column_name]
            not_churned_X = X[y == 0][column_name]
            logger.info(f'Check non-binary column: {column_name}')
            anderson_lowest_churned = run_anderson(churned_X)
            anderson_lowest_not_churned = run_anderson(not_churned_X)
            logger.debug(f'Anderson-Darling test churned p-value: {anderson_lowest_churned}')
            logger.debug(f'Anderson-Darling test not churned p-value: {anderson_lowest_not_churned}')
            normality_assumption = (anderson_lowest_churned <= significance_level) and (anderson_lowest_not_churned <= significance_level)
            logger.info(f'Normality assumption satisfied: {normality_assumption}')
            
            levene_p = levene(churned_X, not_churned_X).pvalue
            homoscedasticity_assumption = levene_p > significance_level
            logger.debug(f'Levene test p-value: {levene_p}')
            logger.info(f'Homoscedasticity assumption satisfied: {homoscedasticity_assumption}')
            
            _, ks_p = ks_2samp(churned_X, not_churned_X)
            equal_shape_assumption = ks_p > significance_level
            logger.debug(f'Kolmogorow-Smirnov test p-value: {ks_p}')
            logger.info(f'Equal shape assumption satisfied: {equal_shape_assumption}')

            if equal_shape_assumption:
                mwu_stat, mwu_p = mannwhitneyu(churned_X, not_churned_X, alternative='two-sided')
                logger.debug(f'MWU test: p-value {mwu_p} and statistic {mwu_stat}')
                equal_mwu = mwu_p > significance_level
                logger.info(f'MWU test not rejected: {equal_mwu}')
            else:
                mwu_stat, mwu_p = np.nan, np.nan
                equal_mwu = np.nan

            # logarithmic transformation
            churned_X_log = np.log(churned_X+1)
            not_churned_X_log = np.log(not_churned_X+1)
            normal_churned_log = run_anderson(churned_X_log)
            normal_not_churned_log = run_anderson(not_churned_X_log)

            # Box-Cox transformation
            churned_X_boxcox, lambda_churned_boxcox = boxcox(churned_X+1)
            not_churned_X_boxcox, lambda_not_churned_boxcox = boxcox(churned_X+1)
            normal_churned_boxcox = run_anderson(churned_X_boxcox)
            normal_not_churned_boxcox = run_anderson(not_churned_X_boxcox)

            # Yeo-Johnson transformation
            churned_X_yeojohnson, lambda_churned_yeojohnson = yeojohnson(churned_X)
            not_churned_X_yeojohnson, lambda_not_churned_yeojohnson = yeojohnson(not_churned_X)
            normal_churned_yeojohnson = run_anderson(churned_X_yeojohnson)
            normal_not_churned_yeojohnson = run_anderson(not_churned_X_yeojohnson)
            
            mean_churned = churned_X.mean()
            mean_not_churned = not_churned_X.mean()
            logger.debug(f'Mean churned {mean_churned:.4f} and not churned {mean_not_churned:.4f}')
            mean_churned_nz = churned_X[churned_X != 0].mean()
            mean_not_churned_nz = not_churned_X[not_churned_X != 0].mean()
            mean_diff_nz = mean_not_churned_nz/mean_churned_nz
            logger.info(f'Mean non_zero churned {mean_churned_nz:.4f} and not churned {mean_not_churned_nz:.4f} (diff: {mean_diff_nz:.4f})')
            median_churned = churned_X.median()
            median_not_churned = not_churned_X.median()
            logger.debug(f'Median churned {median_churned:.4f} and not churned {median_not_churned:.4f}')
            min_churned = churned_X.min()
            min_not_churned = not_churned_X.min()
            logger.debug(f'Min churned {min_churned:.4f} and not churned {min_not_churned:.4f}')
            max_churned = churned_X.max()
            max_not_churned = not_churned_X.max()
            logger.debug(f'Max churned {max_churned:.4f} and not churned {max_not_churned:.4f}')
            skew_churned = skew(churned_X)
            skew_not_churned = skew(not_churned_X)
            logger.debug(f'Skewness churned {skew_churned:.4f} and not churned {skew_not_churned:.4f}')
            kurtosis_churned = kurtosis(churned_X, fisher=True)
            kurtosis_not_churned = kurtosis(not_churned_X, fisher=True)
            logger.debug(f'Excess kurtosis churned {kurtosis_churned:.4f} and not churned {kurtosis_not_churned:.4f}\n')

            nz_ratio_churned = len(churned_X[churned_X != 0])/len(churned_X)
            nz_ratio_not_churned = len(not_churned_X[not_churned_X != 0])/len(not_churned_X)
            if nz_ratio_churned != 0: 
                nz_ratio_diff = nz_ratio_not_churned/nz_ratio_churned
            else:
                nz_ratio_diff = np.nan
            logger.info(f'Non-zero ratio of values: churned {nz_ratio_churned:.4f} and not churned {nz_ratio_not_churned:.4f} (diff: {nz_ratio_diff:.4f})')

            logger.debug(f'churned value percent:\n{churned_X.value_counts()/len(churned_X)}\n')
            logger.debug(f'not churned value percent:\n{not_churned_X.value_counts()/len(not_churned_X)}\n')
            logger.debug(f'churned value count:\n{churned_X.value_counts()}\n')
            logger.debug(f'not churned value count:\n{not_churned_X.value_counts()}\n')

            non_binary_results[column_name] = {
                'anderson p-value churned': anderson_lowest_churned,
                'anderson p-value not churned': anderson_lowest_not_churned,
                'normality assumption': normality_assumption,
                'levene p-value': levene_p,
                'homoscedasticity assumption': homoscedasticity_assumption,
                'ks p-value': ks_p,
                'equal shape assumption': equal_shape_assumption,
                'equal mwu': equal_mwu,
                'mwu stat': mwu_stat,
                'mwu p-value': mwu_p,
                'mean churned': mean_churned,
                'mean not churned': mean_not_churned,
                'meadian churned': median_churned,
                'median not churned': median_not_churned,
                'min churned': min_churned,
                'min not churned': min_not_churned,
                'max churned': max_churned,
                'max not churned': max_not_churned,
                'skewness churned': skew_churned,
                'skewness not churned': skew_not_churned,
                'excess kurtosis churned': kurtosis_churned,
                'excess kurtosis not churned': kurtosis_not_churned,
                'lambda churned boxcox': lambda_churned_boxcox,
                'lambda not churned boxcox': lambda_not_churned_boxcox,
                'lambda churned yeojohnson': lambda_churned_yeojohnson,
                'lambda not churned yeojohnson': lambda_not_churned_yeojohnson,
                'normal churned log': normal_churned_log,
                'normal not churned log': normal_not_churned_log,
                'normal churned boxcox': normal_churned_boxcox,
                'normal not churned boxcox': normal_not_churned_boxcox,
                'normal churned johnson': normal_churned_yeojohnson,
                'normal not churned johnson': normal_not_churned_yeojohnson,
                'nz ratio churned': nz_ratio_churned,
                'nz ratio not churned': nz_ratio_not_churned,
                'nz ratio diff': nz_ratio_diff,
                'nz mean churned': mean_churned_nz,
                'nz mean not churned': mean_not_churned_nz,
                'nz mean diff': mean_diff_nz,
            }
    pd.DataFrame(non_binary_results).T.to_csv(f'non_binary_comparison_{subset}.csv')
