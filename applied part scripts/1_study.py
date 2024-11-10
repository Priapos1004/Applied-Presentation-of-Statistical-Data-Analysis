import pandas as pd
from generate_dataset import generate_dataset
from shapley_experiment import run_shapley_experiment
from population_comparison_experiment import run_population_comparison_experiment
from sklearn.model_selection import train_test_split
from utils_analysis import get_subset, setup_logger

logger = setup_logger(__name__, __file__)

def run_study(df_old: pd.DataFrame, df_new: pd.DataFrame, subsets: list[str]):
    for subset in subsets:
        logger.info(f'##### Starting to analyse subset "{subset}" #####')
        subset_df_old = get_subset(df_old, subset_name=subset)
        logger.info(f'Containing {len(subset_df_old)} of {len(df_old)} members ({(len(subset_df_old)/len(df_old)*100):.2f}%) (old data)')
        subset_df_new = get_subset(df_new, subset_name=subset)
        logger.info(f'Containing {len(subset_df_new)} of {len(df_new)} members ({(len(subset_df_new)/len(df_new)*100):.2f}%) (new data)')
        # Split old data into train and test data
        # VIF can only handle int64 and float64, not Int64 and Float64
        X = subset_df_old.drop(columns=['EXIT']).astype('float64')
        y = subset_df_old['EXIT'].astype('int64')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y.to_list(), random_state=42)
        logger.debug(f'Row number: train {len(y_train)} and test {len(y_test)}')
        logger.debug(f'Churn percentage: train {(sum(y_train)/len(y_train)*100):.2f}% and test {(sum(y_test)/len(y_test)*100):.2f}%')
        # Get binary (with at least 0.5% variations) and non-binary columns
        binary_columns = [col for col in X_train.columns if (X_train[col].nunique() == 2) and (X_train[col].mean()<0.995) and (X_train[col].mean()>0.005)]
        non_binary_columns = [col for col in X_train.columns if X_train[col].nunique() > 2]
        combined_columns = binary_columns + non_binary_columns
        logger.debug(f'Binary columns (number: {len(binary_columns)}):\n{binary_columns}\n')
        logger.debug(f'Non-binary columns (number: {len(non_binary_columns)}):\n{non_binary_columns}\n')
        # X and y values of new data
        X_new = subset_df_new.drop(columns=['EXIT']).astype('float64')
        y_new = subset_df_new['EXIT'].astype('int64')
        logger.debug(f'Churn rate y_new: {(y_new.mean()*100):.4f}%')

        # Run first experiment
        if subset != 'all':
            # Run experiment with train-test split of first reference date
            run_shapley_experiment(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, binary_columns=binary_columns, non_binary_columns=non_binary_columns, combined_columns_train=combined_columns, subset=subset+'_train_test')
            # Run experiment with both reference dates
            run_shapley_experiment(X_train=X, y_train=y, X_test=X_new, y_test=y_new, binary_columns=binary_columns, non_binary_columns=non_binary_columns, combined_columns_train=combined_columns, subset=subset)
            logger.info('Finished shapley experiment')

        # Run second experiment
        run_population_comparison_experiment(X=X, y=y, subset=subset+'_old')
        run_population_comparison_experiment(X=X_new, y=y_new, subset=subset+'_new')
        logger.info('Finished churned experiment\n\n')

if __name__=='__main__':
    # artifical dataset with reference date 01.01.2023
    final_df_old = generate_dataset()
    # artifical dataset with reference date 01.09.2023
    final_df_new = generate_dataset()
    run_study(final_df_old, final_df_new, subsets=['over 65', 'newcomer', 'all'])