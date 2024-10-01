import pandas as pd

from typing import Callable

def feature_engineering_imaging(data: pd.DataFrame):
    """
    Function to perform feature engineering on imaging data using PyRadiomics.
    
    Parameters:
    - data (DataFrame): The input imaging data.
    
    Returns:
    - radiomics_features (DataFrame): DataFrame containing radiomics features.
    """
    
    raise NotImplementedError


symp_cols = [
    'esas_pain',
    'esas_tiredness',
    'esas_nausea',
    'esas_depression',
    'esas_anxiety',
    'esas_drowsiness',
    'esas_appetite',
    'esas_well_being',
    'esas_shortness_of_breath',
    'patient_ecog',
]
lab_cols = [
    'alanine_aminotransferase',
    'albumin',
    'alkaline_phosphatase',
    'aspartate_aminotransferase',
    #'basophil',
    'bicarbonate',
    'chloride',
    'creatinine',
    'eosinophil',
    'glucose',
    'hematocrit',
    'hemoglobin',
    'lactate_dehydrogenase',
    'lymphocyte',
    'magnesium',
    'mean_corpuscular_hemoglobin',
    'mean_corpuscular_hemoglobin_concentration',
    'mean_corpuscular_volume',
    'mean_platelet_volume',
    'monocyte',
    'neutrophil',
    'phosphate',
    'platelet',
    'potassium',
    'red_blood_cell',
    'red_cell_distribution_width',
    'sodium',
    'total_bilirubin',
    'white_blood_cell',
]
symp_change_cols = [f'{col}_change' for col in symp_cols]
lab_change_cols = [f'{col}_change' for col in lab_cols]


def feature_engineering_clinical(data: pd.DataFrame, cfg): #, function: Callable
    """
    Function to perform feature engineering on clinical data using custom functions.
    
    Parameters:
    - data (DataFrame): The input clinical data.
    - function_file (str): The path to the file containing customized feature engineering functions.
    
    Returns:
    - clinical_features (DataFrame): DataFrame containing clinical features.
    """
    import numpy as np
    from functools import partial
    # import yaml
    # from constants import lab_cols, lab_change_cols, symp_cols, symp_change_cols

    # with open('config.yaml') as file:
    #     cfg = yaml.safe_load(file)
        
    # align_on = cfg['align_on'] 
    main_date_col = cfg['main_date_col'] 

    def get_change_since_prev_session(data: pd.DataFrame) -> pd.DataFrame:
        """Get change since last session"""
        cols = symp_cols + lab_cols
        change_cols = symp_change_cols + lab_change_cols
        result = []
        for mrn, group in data.groupby('mrn'):
            group[cols] = group[cols].apply(pd.to_numeric) # convert all columns of DataFrame # pd.to_numeric(s, errors='coerce')
            change = group[cols] - group[cols].shift()
            result.append(change.reset_index().to_numpy())
        result = np.concatenate(result)

        result = pd.DataFrame(result, columns=['index']+change_cols).set_index('index')
        result.index = result.index.astype(int)
        data = pd.concat([data, result], axis=1)

        return data

    def get_missingness_features(data: pd.DataFrame) -> pd.DataFrame:
        
        target_cols = symp_cols + lab_cols + lab_change_cols + symp_change_cols
        
        for col in target_cols: data[f'{col}_is_missing'] = data[col].isnull()
            
        return data

    ###############################################################################
    # Time
    ###############################################################################
    def get_visit_month_feature(data, col: str = 'treatment_date'):
        # convert to cyclical features
        month = data[col].dt.month - 1
        data['visit_month_sin'] = np.sin(2*np.pi*month/12)
        data['visit_month_cos'] = np.cos(2*np.pi*month/12)
        return data

    def get_days_since_last_event(data, main_date_col: str = 'treatment_date', event_date_col: str = 'treatment_date'):
        if main_date_col == event_date_col:
            return (data[main_date_col] - data[event_date_col].shift()).dt.days
        else:
            return (data[main_date_col] - data[event_date_col]).dt.days

    def get_years_diff(data, col1: str, col2: str):
        return data[col1].dt.year - data[col2].dt.year

    ###############################################################################
    # Treatment
    ###############################################################################
    def get_line_of_therapy(data):
        # identify line of therapy (the nth different palliative intent treatment taken)
        # NOTE: all other intent treatment are given line of therapy of 0. Usually (not always but oh well) once the first
        # palliative treatment appears, the rest of the treatments remain palliative
        new_regimen = (data['first_treatment_date'] != data['first_treatment_date'].shift())
        palliative_intent = data['intent'] == 'PALLIATIVE'
        return (new_regimen & palliative_intent).cumsum()


    ###############################################################################
    # Drug dosages
    ###############################################################################
    def get_perc_ideal_dose_given(data, drug_to_dose_formula_map: dict[str, str]):
        """Convert given dose as a percentage of ideal (recommended) dose

        data must have weight, body surface area, age, female, and creatinine columns along with the dosage columns
        """
        result = {}
        for drug, dose_formula in drug_to_dose_formula_map.items():
            dose_col = f'drug_{drug}_given_dose'
            if dose_col not in data.columns: continue
            ideal_dose = get_ideal_dose(data, drug, dose_formula)
            perc_ideal_dose_given = data[dose_col] / ideal_dose # NOTE: 0/0 = np.nan, x/0 = np.inf
            perc_ideal_dose_given = perc_ideal_dose_given
            result[drug] = perc_ideal_dose_given
        result = pd.DataFrame(result)
        result.columns = '%_ideal_dose_given_' + result.columns
        return result

    def get_ideal_dose(data, drug: str, dose_formula: str):
        col = f'drug_{drug}_regimen_dose'
        carboplatin_dose_formula = ('min(regimen_dose * 150, regimen_dose * (((140-age[yrs]) * weight [kg] * 1.23 * '
                                    '(0.85 if female) / creatinine [umol/L]) + 25))')
        if dose_formula == 'regimen_dose': 
            return data[col]
        
        elif dose_formula == 'regimen_dose * bsa': 
            return data[col] * data['body_surface_area']
        
        elif dose_formula == 'regimen_dose * weight': 
            return data[col] * data['weight']
        
        elif dose_formula == carboplatin_dose_formula:
            return pd.concat([data[col] * 150, data[col] * (get_creatinine_clearance(data) + 25)], axis=1).min(axis=1)
        
        else:
            raise ValueError(f'Ideal dose formula {dose_formula} not supported')

    ###############################################################################
    # Special Formulas
    ###############################################################################
    def get_creatinine_clearance(data):
        return (140 - data['age']) * data['weight'] * 1.23 * data['female'].replace({True: 0.85, False: 1}) / data['creatinine']
    
    
    ######### Process feature engineering ##########
    data['treatment_date'] = pd.to_datetime(data['treatment_date'], errors='coerce')
    data['first_treatment_date'] = pd.to_datetime(data['first_treatment_date'], errors='coerce')
    
    data = get_change_since_prev_session(data)
    data = get_missingness_features(data)
    data = get_visit_month_feature(data, col=main_date_col)
    data['line_of_therapy'] = data.groupby('mrn', group_keys=False).apply(get_line_of_therapy)
    data['days_since_starting_treatment'] = (data[main_date_col] - data['first_treatment_date']).dt.days
    get_days_since_last_treatment = partial(
        get_days_since_last_event, main_date_col=main_date_col, event_date_col='treatment_date'
    )
    data['days_since_last_treatment'] = data.groupby('mrn', group_keys=False).apply(get_days_since_last_treatment)

    return data

    # raise NotImplementedError
    
# ###############################################################################
# # Engineering Features
# ###############################################################################
# def custom_clinical_features(df: pd.DataFrame):
#     """
#     Perform customized feature engineering on the given DataFrame.

#     Parameters:
#     df (pd.DataFrame): The input DataFrame.

#     Returns:
#     pd.DataFrame: The transformed DataFrame with engineered features.
#     """
#     import numpy as np
#     from functools import partial
#     import yaml
#     from constants import lab_cols, lab_change_cols, symp_cols, symp_change_cols

#     with open('config.yaml') as file:
#         cfg = yaml.safe_load(file)
        
#     # align_on = cfg['align_on'] 
#     main_date_col = cfg['main_date_col'] 

#     def get_change_since_prev_session(df: pd.DataFrame) -> pd.DataFrame:
#         """Get change since last session"""
#         cols = symp_cols + lab_cols
#         change_cols = symp_change_cols + lab_change_cols
#         result = []
#         for mrn, group in df.groupby('mrn'):
#             group[cols] = group[cols].apply(pd.to_numeric) # convert all columns of DataFrame # pd.to_numeric(s, errors='coerce')
#             change = group[cols] - group[cols].shift()
#             result.append(change.reset_index().to_numpy())
#         result = np.concatenate(result)

#         result = pd.DataFrame(result, columns=['index']+change_cols).set_index('index')
#         result.index = result.index.astype(int)
#         df = pd.concat([df, result], axis=1)

#         return df

#     def get_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
        
#         target_cols = symp_cols + lab_cols + lab_change_cols + symp_change_cols
        
#         for col in target_cols: df[f'{col}_is_missing'] = df[col].isnull()
            
#         return df

#     ###############################################################################
#     # Time
#     ###############################################################################
#     def get_visit_month_feature(df, col: str = 'treatment_date'):
#         # convert to cyclical features
#         month = df[col].dt.month - 1
#         df['visit_month_sin'] = np.sin(2*np.pi*month/12)
#         df['visit_month_cos'] = np.cos(2*np.pi*month/12)
#         return df

#     def get_days_since_last_event(df, main_date_col: str = 'treatment_date', event_date_col: str = 'treatment_date'):
#         if main_date_col == event_date_col:
#             return (df[main_date_col] - df[event_date_col].shift()).dt.days
#         else:
#             return (df[main_date_col] - df[event_date_col]).dt.days

#     def get_years_diff(df, col1: str, col2: str):
#         return df[col1].dt.year - df[col2].dt.year

#     ###############################################################################
#     # Treatment
#     ###############################################################################
#     def get_line_of_therapy(df):
#         # identify line of therapy (the nth different palliative intent treatment taken)
#         # NOTE: all other intent treatment are given line of therapy of 0. Usually (not always but oh well) once the first
#         # palliative treatment appears, the rest of the treatments remain palliative
#         new_regimen = (df['first_treatment_date'] != df['first_treatment_date'].shift())
#         palliative_intent = df['intent'] == 'PALLIATIVE'
#         return (new_regimen & palliative_intent).cumsum()


#     ###############################################################################
#     # Drug dosages
#     ###############################################################################
#     def get_perc_ideal_dose_given(df, drug_to_dose_formula_map: dict[str, str]):
#         """Convert given dose as a percentage of ideal (recommended) dose

#         df must have weight, body surface area, age, female, and creatinine columns along with the dosage columns
#         """
#         result = {}
#         for drug, dose_formula in drug_to_dose_formula_map.items():
#             dose_col = f'drug_{drug}_given_dose'
#             if dose_col not in df.columns: continue
#             ideal_dose = get_ideal_dose(df, drug, dose_formula)
#             perc_ideal_dose_given = df[dose_col] / ideal_dose # NOTE: 0/0 = np.nan, x/0 = np.inf
#             perc_ideal_dose_given = perc_ideal_dose_given
#             result[drug] = perc_ideal_dose_given
#         result = pd.DataFrame(result)
#         result.columns = '%_ideal_dose_given_' + result.columns
#         return result

#     def get_ideal_dose(df, drug: str, dose_formula: str):
#         col = f'drug_{drug}_regimen_dose'
#         carboplatin_dose_formula = ('min(regimen_dose * 150, regimen_dose * (((140-age[yrs]) * weight [kg] * 1.23 * '
#                                     '(0.85 if female) / creatinine [umol/L]) + 25))')
#         if dose_formula == 'regimen_dose': 
#             return df[col]
        
#         elif dose_formula == 'regimen_dose * bsa': 
#             return df[col] * df['body_surface_area']
        
#         elif dose_formula == 'regimen_dose * weight': 
#             return df[col] * df['weight']
        
#         elif dose_formula == carboplatin_dose_formula:
#             return pd.concat([df[col] * 150, df[col] * (get_creatinine_clearance(df) + 25)], axis=1).min(axis=1)
        
#         else:
#             raise ValueError(f'Ideal dose formula {dose_formula} not supported')

#     ###############################################################################
#     # Special Formulas
#     ###############################################################################
#     def get_creatinine_clearance(df):
#         return (140 - df['age']) * df['weight'] * 1.23 * df['female'].replace({True: 0.85, False: 1}) / df['creatinine']
    

#     df = get_visit_month_feature(df, col=main_date_col)
#     df['line_of_therapy'] = df.groupby('mrn', group_keys=False).apply(get_line_of_therapy)
#     df['days_since_starting_treatment'] = (df[main_date_col] - df['first_treatment_date']).dt.days
#     get_days_since_last_treatment = partial(
#         get_days_since_last_event, main_date_col=main_date_col, event_date_col='treatment_date'
#     )
#     df['days_since_last_treatment'] = df.groupby('mrn', group_keys=False).apply(get_days_since_last_treatment)

#     return df