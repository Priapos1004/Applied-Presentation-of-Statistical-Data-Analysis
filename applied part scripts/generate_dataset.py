import numpy as np
import pandas as pd
from tqdm import trange
from config import SIZE

# Set random seed for reproducibility
np.random.seed(42)

# The ranges are estimated by GPT4o without additional knowledge about the data
DATA = [
    # Basis Data
    ['SEX', '-', 'binary', 0, 1],
    ['AGE', '-', 'int', 18, 80],
    ['BIRTHMONTH', '-', 'int', 1, 12],
    ['ADVERT_personal', '-', 'binary', 0, 1],
    ['ADVERT_union', '-', 'binary', 0, 1],
    ['ADVERT_none', '-', 'binary', 0, 1],
    ['ONLINEENTRY', '-', 'binary', 0, 1],
    ['MEMBERSHIP_LENGTH', '-', 'int', 0, 50],
    ['BANK_A', '-', 'binary', 0, 1],
    ['BANK_B', '-', 'binary', 0, 1],
    ['POSTCODE_current', '-', 'int', 1000, 9999],
    ['EXIT', '-', 'binary', 0, 1],

    # Benefits Data
    ['BAMOUNT_i', 'sum', 'float', 0, 5000],
    ['BAMOUNT_TOTAL', 'sum', 'float', 0, 50000],
    ['BAMOUNT_BINARY', 'max', 'binary', 0, 1],

    # Seminar Data
    ['SEMINARTYPE_i', 'max', 'binary', 0, 1],
    ['SEMINAR_TOTAL', 'sum', 'int', 0, 50],

    # Strike Data
    ['STRIKE_LEN', 'sum', 'int', 0, 30],
    ['STRIKE_MONEY', 'sum', 'float', 0, 10000],

    # Documents Data
    ['TEMPLATE_1', 'max', 'binary', 0, 1],
    ['TEMPLATE_2', 'max', 'binary', 0, 1],
    ['TEMPLATE_3', 'max', 'binary', 0, 1],
    ['TEMPLATE_4', 'max', 'binary', 0, 1],
    ['TEMPLATE_5', 'max', 'binary', 0, 1],
    ['TEMPLATE_6', 'max', 'binary', 0, 1],
    ['TEMPLATE_7', 'max', 'binary', 0, 1],
    ['TEMPLATE_8', 'max', 'binary', 0, 1],
    ['TEMPLATE_9', 'max', 'binary', 0, 1],
    ['TEMPLATE_10', 'max', 'binary', 0, 1],
    ['TEMPLATE_11', 'max', 'binary', 0, 1],
    ['TEMPLATE_12', 'max', 'binary', 0, 1],
    ['TEMPLATE_13', 'max', 'binary', 0, 1],
    ['TEMPLATE_14', 'max', 'binary', 0, 1],
    ['TEMPLATE_15', 'max', 'binary', 0, 1],

    # Firm Data
    ['firm_ACTION_i', 'max', 'binary', 0, 1],
    ['firm_CONTACTPERSON', 'max', 'binary', 0, 1],
    ['firm_REPRESENTATIVE_ratio', 'mean', 'float', 0, 1],
    ['firm_SUPPORT_REPRESENTATIVE_ratio', 'mean', 'float', 0, 1],
    ['firm_COMPANYTYPE_i', 'max', 'binary', 0, 1],
    ['firm_EMPLOYEES_ratio', 'mean', 'float', 0, 1],
    ['firm_EMPLOYEES_F', 'mean', 'float', 0, 1],
    ['firm_EMPLOYEES_M', 'mean', 'float', 0, 1],
    ['firm_SECTOR_i', 'max', 'binary', 0, 1],
    ['firm_REPRESENTATIVE_WrC', 'max', 'binary', 0, 1],
    ['firm_REPRESENTATIVE_StC', 'max', 'binary', 0, 1],
    ['firm_REPRESENTATIVE_WrCStC', 'max', 'binary', 0, 1],
    ['firm_YOUTHRATIO', 'mean', 'float', 0, 1],
    ['firm_MEMBERS_TOTAL', 'mean', 'int', 0, 1000],
    ['firm_MEMBERS_F', 'mean', 'float', 0, 1],
    ['firm_MEMBERS_M', 'mean', 'float', 0, 1],
    ['firm_MEMBERS_FULLTIME', 'mean', 'float', 0, 1],
    ['firm_MEMBERS_PARTTIME', 'mean', 'float', 0, 1],
    ['firm_VALID_TARIFF', 'max', 'binary', 0, 1],

    # Committee Data
    ['COMMROLE_i_j', 'max', 'binary', 0, 1],
    ['COMMROLE_999999', 'max', 'binary', 0, 1],
    ['RANK_max', 'max', 'int', 0, 10],
    ['RANK_sum', 'sum', 'int', 0, 100],

    # Contact Data
    ['IN_CONTACT', 'sum', 'binary', 0, 50],
    ['CATEGORY_Assign_and_Restore_payment', 'max', 'binary', 0, 1],
    ['CATEGORY_social_legal_protection', 'max', 'binary', 0, 1],
    ['CATEGORY_labour_legal_protection', 'max', 'binary', 0, 1],
    ['CATEGORY_other_legal_protection', 'max', 'binary', 0, 1],
    ['CATEGORY_update_data', 'max', 'binary', 0, 1],
    ['CATEGORY_i', 'max', 'binary', 0, 1],
    ['STATUS_SENTIMENT', 'mean', 'float', -1, 1],

    # Dynamic Data
    ['POSTCODE_change', 'sum', 'binary', 0, 1],
    ['FIRM_change', 'sum', 'binary', 0, 1],
    ['CONTRIBUTION', 'mean', 'float', 0, 500],
    ['CONTRIBUTION_exemption', 'max', 'binary', 0, 1],
    ['CONTRIBUTION_CYCLE_Y', 'max', 'binary', 0, 1],
    ['CONTRIBUTION_CYCLE_H', 'max', 'binary', 0, 1],
    ['CONTRIBUTION_CYCLE_Q', 'max', 'binary', 0, 1],
    ['CONTRIBUTION_CYCLE_M', 'max', 'binary', 0, 1],
    ['CONTRIBUTION_IN_INSTALLMENTS', 'max', 'binary', 0, 1],
    ['PAYDELAY_1A', 'max', 'binary', 0, 1],
    ['PAYDELAY_1B', 'max', 'binary', 0, 1],
    ['PAYDELAY_B', 'max', 'binary', 0, 1],
    ['EMPLOYMENT_DEGREE', 'mean', 'float', 0, 1],
    ['EMPLOYMENT_STATUS_i', 'max', 'binary', 0, 1],
    ['EMPLOYMENT_TYPE_i', 'max', 'binary', 0, 1],
    ['INCOME', 'mean', 'float', 0, 10000],
    ['TARIFFNEGO', 'max', 'binary', 0, 1],
    ['UNDER_TARIFF', 'max', 'binary', 0, 1],
    ['MAGAZINE', 'max', 'binary', 0, 1],
]

def get_values(var_type: str, var_min, var_max, size: int) -> np.ndarray:
    if var_type == 'float':
        values = np.random.uniform(var_min, var_max, size=size)
    elif var_type == 'int':
        values = np.random.randint(var_min, var_max+1, size=size)
    else: # var_type == 'binary'
        values = np.random.choice([0,1], size=size)
    return values

def get_values_list(var_type: str, var_min, var_max, size: int, number: int):
    return [get_values(var_type, var_min, var_max, size) for _ in range(number)]

def generate_dataset(size: int = SIZE) -> pd.DataFrame:
    # Define column names
    columns = ['Variable', 'Aggregate Function', 'Data Type', 'Min Estimated Value', 'Max Estimated Value']
    # Create variable DataFrame
    variable_df = pd.DataFrame(DATA, columns=columns)

    # Generate union dataset
    final_dict = {}
    for idx in trange(len(variable_df), desc='Generate variables'):
        var_name = variable_df['Variable'].iloc[idx]
        var_min = variable_df['Min Estimated Value'].iloc[idx]
        var_max = variable_df['Max Estimated Value'].iloc[idx]
        var_type = variable_df['Data Type'].iloc[idx]
        var_agg = variable_df['Aggregate Function'].iloc[idx]

        if var_agg == '-':
            final_dict[var_name] = get_values(var_type, var_min, var_max, size)
        elif var_agg == 'mean':
            final_dict['lag3_'+var_name] = get_values('float', var_min, var_max, size)
            final_dict['lag6_'+var_name] = get_values('float', var_min, var_max, size)
            final_dict['lag12_'+var_name] = get_values('float', var_min, var_max, size)
        elif var_agg == 'sum':
            value_lag3 = np.sum(get_values_list(var_type, var_min, var_max, size, number=3), axis=0)
            value_lag6 = value_lag3 + np.sum(get_values_list(var_type, var_min, var_max, size, number=3), axis=0)
            value_lag12 = value_lag6 + np.sum(get_values_list(var_type, var_min, var_max, size, number=6), axis=0)
            final_dict['lag3_'+var_name] = value_lag3
            final_dict['lag6_'+var_name] = value_lag6
            final_dict['lag12_'+var_name] = value_lag12
        else: # var_agg == 'max'
            value_lag3 = get_values(var_type, var_min, var_max, size)
            value_lag6 = get_values(var_type, var_min, var_max, size)
            value_lag12 = get_values(var_type, var_min, var_max, size)
            final_dict['lag3_'+var_name] = value_lag3
            final_dict['lag6_'+var_name] = np.maximum(value_lag3, value_lag6)
            final_dict['lag12_'+var_name] = np.maximum.reduce([value_lag3, value_lag6, value_lag12])

    return pd.DataFrame(final_dict)
