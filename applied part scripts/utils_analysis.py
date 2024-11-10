import logging
import os
import sys

import numpy as np
import pandas as pd
from config import LOG_FOLDER_PATH, LOG_LEVEL, LOG_TO_FILE
from prince import MCA
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

### Setup Functions ###

def setup_logger(name=__name__, file=__file__):
    logger = logging.getLogger(name)

    if LOG_LEVEL == "debug":
        logger.setLevel(logging.DEBUG)
    elif LOG_LEVEL == "info":
        logger.setLevel(logging.INFO)
    elif LOG_LEVEL == "warning":
        logger.setLevel(logging.WARNING)
    elif LOG_LEVEL == "error":
        logger.setLevel(logging.ERROR)
    else:
        raise ValueError(f"Log level '{LOG_LEVEL}' invalid!")

    # To prevent double logging output
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.propagate = False

    # Console handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    # File handler (if enabled)
    if LOG_TO_FILE:
        log_file_path = LOG_FOLDER_PATH + f"{os.path.basename(file).replace('.py', '')}.log"
        f_handler = logging.FileHandler(log_file_path, mode="w")
        f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)
    return logger

### Analysis Functions ###

UTILS_ANALYSIS_LOGGER = setup_logger(__name__, __file__)

def get_subset(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    if subset_name == 'over 65':
        return df[df['AGE'] > 65]
    elif subset_name == 'newcomer':
        return df[df['MEMBERSHIP_LENGTH'] <= 3]
    elif subset_name == 'all':
        return df
    else:
        raise ValueError(f'"{subset_name}" is not a valid subset name!')

def check_multicollinearity(X: pd.DataFrame, logger=UTILS_ANALYSIS_LOGGER) -> pd.DataFrame:
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    # VIF values can be NA or inf if there are perfect collinearities
    # or constant variables -> remove those values
    len_before_na = len(vif_data)
    vif_data.dropna(subset=['VIF'], inplace=True)
    len_before_inf = len(vif_data)
    vif_data.drop(vif_data.index[~np.isfinite(vif_data['VIF'])], inplace=True)
    logger.debug(f'Columns with VIF NA {len_before_na-len_before_inf}, inf {len_before_inf-len(vif_data)}, and other {len(vif_data)}')
    vif_data.sort_values(by=["VIF"], ascending=False, inplace=True)
    return vif_data

def find_best_k(X: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Find best feature number for PCA / MCA

    Parameters
    ----------
    X : pd.DataFrame
        X data to use
    method: str
        method to use

    Returns
    -------
    results_df : pd.DataFrame
        dataframe with columns 'PC', 'explained cumulative variance', 'explained variance' containing the first 3 PCs and the PCs for above 80%, 85%, 90%, 95%, 98%, and 99.5% explained cumulative variance
    """
    if method == "pca":
        pca = PCA(n_components=None, random_state=42)
        pca.fit(X)
        explained_ratio = pca.explained_variance_ratio_
        cumulative_explained_ratio = [sum(explained_ratio[:idx]) for idx in range(1, len(explained_ratio)+1)]
    elif method == "mca":
        mca = MCA(n_components=len(X.columns), one_hot=False, random_state=42)
        mca.fit(X)
        explained_ratio = mca.percentage_of_variance_ / 100
        cumulative_explained_ratio = mca.cumulative_percentage_of_variance_ / 100
    else:
        raise ValueError(f'Method "{method}" is not supported.')

    results_df = pd.DataFrame([[i+1, cumulative_explained_ratio[i], explained_ratio[i]] for i in range(min(3, len(X.columns)))], columns=["PC", "explained cumulative variance", "explained variance"])

    thresholds = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 0.995]
    
    if len(X.columns) > 3:
        thresholds = [elem for elem in thresholds if elem > cumulative_explained_ratio[2]]
        if len(thresholds) > 0:
            for i in range(3, len(cumulative_explained_ratio)-1):
                if cumulative_explained_ratio[i] > thresholds[0]:
                    results_df.loc[len(results_df)] = [i+1, cumulative_explained_ratio[i], explained_ratio[i]]
                    thresholds = [elem for elem in thresholds if elem > cumulative_explained_ratio[i]]
                
                if not thresholds:
                    break

    results_df['PC'] = results_df['PC'].astype(int)
    results_df['explained cumulative variance'] = results_df['explained cumulative variance'].astype(float)
    results_df['explained variance'] = results_df['explained variance'].astype(float)

    return results_df
