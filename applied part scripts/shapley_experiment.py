import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from factor_analyzer.factor_analyzer import calculate_kmo
from prince import MCA
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from utils_analysis import check_multicollinearity, find_best_k, setup_logger
from xgboost import XGBClassifier

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
logger = setup_logger(__name__, __file__)

def run_pca(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[PCA, pd.DataFrame, pd.DataFrame]:
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
    X_test_scaled = pd.DataFrame(scaler.fit_transform(test_data), columns=test_data.columns)
    logger.info('Scaled data for pca')
    # Check for sufficient linearity
    kmo_result = calculate_kmo(X_train_scaled)
    logger.info(f'KMO test result: {round(kmo_result[1], 3)}')
    # Find best number of principal components
    pca_result = find_best_k(X_train_scaled, method='pca')
    logger.debug(f'PCA result:\n{pca_result}\n\n')
    # Apply PCA to data
    k_pca = pca_result.PC.iloc[-4] # Select PC for 95% explained varaince
    pca = PCA(n_components=k_pca, random_state=42)
    X_train_scaled_uncorrelated = pca.fit_transform(X_train_scaled)
    X_test_scaled_uncorrelated = pca.transform(X_test_scaled)
    return pca, pd.DataFrame(X_train_scaled_uncorrelated, columns=[f'PCA_{i}' for i in range(X_train_scaled_uncorrelated.shape[1])]), pd.DataFrame(X_test_scaled_uncorrelated, columns=[f'PCA_{i}' for i in range(X_test_scaled_uncorrelated.shape[1])])

def run_mca(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[MCA, pd.DataFrame, pd.DataFrame]:
    # Find best number of principal components
    mca_result = find_best_k(train_data, method='mca')
    logger.debug(f'MCA result:\n{mca_result}\n\n')
    # Apply PCA to non-binary data
    k_mca = mca_result.PC.iloc[-4] # Select PC for 95% explained inertia
    mca = MCA(n_components=k_mca, one_hot=False, random_state=42)
    X_train_uncorrelated = mca.fit_transform(train_data)
    X_test_uncorrelated = mca.transform(test_data)
    return mca, pd.DataFrame(X_train_uncorrelated.values, columns=[f'MCA_{i}' for i in range(X_train_uncorrelated.shape[1])]), pd.DataFrame(X_test_uncorrelated.values, columns=[f'MCA_{i}' for i in range(X_test_uncorrelated.shape[1])])

def plot_shapley_values(shap_values, X_test, image_name: str):
    # Set up figure for the SHAP summary plot
    _, ax = plt.subplots(figsize=(6, 6))
    # Generate SHAP summary plot
    shap.summary_plot(shap_values, X_test, show=False, plot_size=(6, 6))
    # Customize x-axis ticks and labels
    ax.set_xlabel("Shapley value")
    ax.set_xticks([-4, -2, 0, 2, 4])
    # Adjust layout to optimize spacing
    plt.tight_layout()
    plt.show()

def get_classification_report(model, X_test, y_test, prob_thresholds: list[float], name: str):
    for prob_threshold in prob_thresholds:
        # Make predictions on the test data
        y_pred = (model.predict_proba(X_test)[:, 1] >= prob_threshold).astype(int)
        # Generate classification report
        logger.info(f'Classification report ({name}, prob threshold: {prob_threshold}):\n{classification_report(y_test, y_pred)}\n')

def get_original_shap_values(shap_values, loadings: pd.DataFrame) -> pd.DataFrame:
    shap_values_df = pd.DataFrame(shap_values.values, columns=shap_values.feature_names)
    original_shap_values = pd.DataFrame(np.dot(shap_values_df, loadings.T), columns = loadings.index)
    return original_shap_values

def get_top_loadings(loadings: pd.DataFrame, name: str, component_names: list[str]):
    for component in component_names:
        logger.info(f'{name} loadings {component}:\n{loadings.loc[loadings[component].abs().nlargest(15).index, component]}\n')

def apply_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, loadings: pd.DataFrame, name: str, subset: str):
    if 'over 65' in subset:
        ml_model = XGBClassifier(random_state=42, max_depth=7, scale_pos_weight=2, gamma=10, n_estimators=500, learning_rate=0.01)
        prob_thresholds = [0.5]
    elif 'newcomer' in subset:
        ml_model = XGBClassifier(random_state=42, max_depth=7, scale_pos_weight=3, gamma=10, n_estimators=400, learning_rate=0.005)
        prob_thresholds = [0.65]
    else:
        raise ValueError(f'no ML model for subset "{subset}"')

    ml_model.fit(X_train, y_train)
    explainer = shap.Explainer(ml_model, seed=42)
    shap_values = explainer(X_test)
    get_classification_report(ml_model, X_test=X_train, y_test=y_train, prob_thresholds=prob_thresholds, name=name+' train')
    get_classification_report(ml_model, X_test=X_test, y_test=y_test, prob_thresholds=prob_thresholds, name=name+' test')

    # Calculate feature importance of original features
    original_shap_values = get_original_shap_values(shap_values, loadings)
    logger.info(f'Calculation original feature importance:\n{original_shap_values.abs().mean().sort_values(ascending=False).head(10)}\n')

    # Get loadings of top 10 components with highest absolute Shapley values
    shap_values_df = pd.DataFrame(shap_values.values, columns=shap_values.feature_names)
    component_importance = shap_values_df.abs().mean()
    component_names = component_importance.sort_values(ascending=False).head(10).index.tolist()
    get_top_loadings(loadings=loadings, name=name, component_names=component_names)
    return shap_values

def run_shapley_experiment(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, binary_columns: list[str], non_binary_columns: list[str], combined_columns_train: list[str], subset: str):
    logger.info(f'##### Starting to analyse subset "{subset}" #####')
    # Check multicollinearity [takes very long]
    vif_result = check_multicollinearity(X_train, logger=logger)
    logger.info(f'VIF result:\n{vif_result.head(10)}\n\n')
    # Run PCA
    pca, X_train_scaled_uncorrelated, X_test_scaled_uncorrelated = run_pca(train_data=X_train[non_binary_columns], test_data=X_test[non_binary_columns])
    logger.info('Applied PCA to non-binary data')
    # Run MCA
    mca, X_train_binary_uncorrelated, X_test_binary_uncorrelated = run_mca(train_data=X_train[binary_columns], test_data=X_test[binary_columns])
    logger.info('Applied MCA to binary data')
    # Combine data
    combined_X_train = pd.concat([X_train_scaled_uncorrelated, X_train_binary_uncorrelated], axis=1)
    combined_X_test = pd.concat([X_test_scaled_uncorrelated, X_test_binary_uncorrelated], axis=1)
    logger.info('Combined uncorrelated binary and non-binary data')
    # Check multicollinearity
    vif_result_pca_mca = check_multicollinearity(combined_X_train, logger=logger)
    logger.info(f'VIF result (PCA + MCA):\n{vif_result_pca_mca.head(10)}\n\n')
    pca_loadings = pd.DataFrame(pca.components_.T, columns=[f'PCA_{i}' for i in range(pca.n_components_)], index=non_binary_columns)
    mca_loadings = mca.column_contributions_
    mca_loadings.columns = [f'MCA_{i}' for i in range(X_train_binary_uncorrelated.shape[1])]
    # Calculate norms for each component (column-wise)
    pca_component_magnitudes = np.linalg.norm(pca_loadings, axis=0)
    mca_component_magnitudes = np.linalg.norm(mca_loadings, axis=0)
    # Display or print the magnitudes
    logger.debug(f'PCA component magnitudes: {pca_component_magnitudes}')
    logger.debug(f'MCA component magnitudes: {mca_component_magnitudes}')
    # Scale MCA loadings
    scaled_mca_loadings = mca_loadings / np.linalg.norm(mca_loadings, axis=0)
    scaled_mca_component_magnitudes = np.linalg.norm(scaled_mca_loadings, axis=0)
    logger.debug(f'scaled MCA component magnitudes: {scaled_mca_component_magnitudes}')
    # Combine loadings
    pca_mca_loadings = pd.concat([pca_loadings, scaled_mca_loadings], axis=1).fillna(0)
    # SHAP for PCA+MCA
    shap_values = apply_xgboost(X_train=combined_X_train, y_train=y_train, X_test=combined_X_test, y_test=y_test, loadings=pca_mca_loadings, name='PCA + MCA', subset=subset)
    plot_shapley_values(shap_values, combined_X_test, image_name=f'pcamca_{subset}')
