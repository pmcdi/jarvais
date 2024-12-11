import pandas as pd
from pandas.api.types import is_numeric_dtype

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

def replace_missing(data: pd.DataFrame, categorical_columns: list, continuous_columns: list, config: dict):
    """
    Replace missing values in the dataset based on specified strategies.

    This method handles missing values in both continuous and categorical columns.
    The strategies for handling missing values are specified in the configuration.

    Continuous columns:
        - 'median': Replace missing values with the median of the column.
        - 'mean': Replace missing values with the mean of the column.
        - 'mode': Replace missing values with the mode of the column.
        - If an unknown strategy is specified, the median is used by default.

    Categorical columns:
        - A specified string from the configuration is used to replace missing values.
        - If 'knn' is specified, KNN imputation is used to fill missing values.

    Note:
        KNN imputation for categorical variables is performed using integer encoding and
        inverse transformation. This can be computationally expensive for large datasets.
    """

    for cont in continuous_columns: 
        strategy = 'median' if cont not in config['missingness_strategy']['continuous'].keys() else config['missingness_strategy']['continuous'][cont]
        if strategy.lower() == 'median':
            replace_with = data[cont].median()
        elif strategy.lower() == 'mean':
            replace_with = data[cont].mean()
        elif strategy.lower() == 'mode':
            replace_with = data[cont].mode()
        else:
            print(f'Unknown value {strategy} provided to replace {cont}. Using median')
            replace_with = data[cont].median()
        data[cont] = data[cont].fillna(replace_with)

    for cat in categorical_columns: # For categorical var, replaces with string provided in config(default Unknown)
        filler = 'Unknown' if cat not in config['missingness_strategy']['categorical'].keys() else config['missingness_strategy']['categorical'][cat]
        if config['missingness_strategy']['categorical'][cat].lower() != 'knn':
            data[cat] = data[cat].fillna(filler)

    data_to_use = data[categorical_columns + continuous_columns]
    for cat in categorical_columns:
        if config['missingness_strategy']['categorical'][cat].lower() == 'knn':
            print(f'Using KNN to fill missing values in {cat}, this may take a while...\n')
            data[cat] = knn_impute_categorical(data_to_use, categorical_columns)[cat]    

    return data 

def knn_impute_categorical(data: pd.DataFrame, categorical_columns: list):
    """
    Perform KNN imputation on categorical columns. 
    From https://www.kaggle.com/discussions/questions-and-answers/153147

    Args:
        data (DataFrame): The data containing categorical columns.
        categorical_columns (list): List of categorical column names.

    Returns:
        DataFrame: Data with imputed categorical columns.
    """
    mm = MinMaxScaler()
    mappin = {}

    def find_category_mappings(df, variable):
        return {k: i for i, k in enumerate(df[variable].dropna().unique(), 0)}

    def integer_encode(df, variable, ordinal_mapping):
        df[variable] = df[variable].map(ordinal_mapping)

    df = data.copy()
    for variable in categorical_columns:
        mappings = find_category_mappings(df, variable)
        mappin[variable] = mappings

    for variable in categorical_columns:
        integer_encode(df, variable, mappin[variable])

    scaled_data = mm.fit_transform(df)
    knn_imputer = KNNImputer()
    knn_imputed = knn_imputer.fit_transform(scaled_data)
    df.iloc[:, :] = mm.inverse_transform(knn_imputed)
    for col in df.categorical_columns:
        df[col] = round(df[col]).astype('int')

    for col in categorical_columns:
        inv_map = {v: k for k, v in mappin[col].items()}
        df[col] = df[col].map(inv_map)

    return df

def get_outliers(data: pd.DataFrame, categorical_columns: list):
    """
    Perform outlier analysis on categorical columns. 

    Args:
        data (DataFrame): The data containing categorical columns.
        categorical_columns (list): List of categorical column names.

    Returns:
        DataFrame: Data with imputed categorical columns.
    """

    mapping = {}
    
    for cat in categorical_columns:
        category_counts = data[cat].value_counts()
        threshold = int(len(data)*.01)
        outliers = category_counts[category_counts < threshold].index.tolist()

        mapping[cat] = {}

        for _cat in data[cat].unique():
            if _cat in outliers:
                mapping[cat][f'{_cat}'] = 'Other'
            else:
                mapping[cat][f'{_cat}'] = f'{_cat}'

        if len(outliers) > 0:
            outliers = [f'{o}: {category_counts[o]} out of {data[cat].count()}' for o in outliers]
            print(f'  - Outliers found in {cat}: {outliers}')
            return f'  - Outliers found in {cat}: {outliers}\n', mapping
        else:
            print(f'  - No Outliers found in {cat}')
            return f'  - No Outliers found in {cat}\n', mapping
    
def infer_types(data: pd.DataFrame):
    """
    Infer and categorize column data types in the dataset.
    Adapted from https://github.com/tompollard/tableone/blob/main/tableone/preprocessors.py

    This method analyzes the dataset to categorize columns as either 
    continuous or categorical based on their data types and unique value proportions.

    Assumptions:
        - All non-numerical and non-date columns are considered categorical.
        - Boolean columns are not considered numerical but categorical.
        - Numerical columns with a unique value proportion below a threshold are 
        considered categorical.

    The method also applies a heuristic to detect and classify ID columns 
    as categorical if they have a low proportion of unique values.
    """

    date_columns = [
        col for col in data.select_dtypes(include=['object']).columns
        if pd.to_datetime(data[col], format='mixed', errors='coerce').notna().any()
    ]

    # assume all non-numerical and date columns are categorical
    numeric_cols = set([col for col in data.columns if is_numeric_dtype(data[col])])
    numeric_cols = set([col for col in numeric_cols if data[col].dtype != bool]) # Filter out boolean 
    likely_cat = set(data.columns) - numeric_cols
    likely_cat = list(likely_cat - set(date_columns))

    # check proportion of unique values if numerical
    for var in numeric_cols:
        likely_flag = 1.0 * data[var].nunique()/data[var].count() < 0.025
        if likely_flag:
            likely_cat.append(var)

    likely_cat = [cat for cat in likely_cat if data[cat].nunique()/data[cat].count() < 0.2] # Heuristic targeted at detecting ID columns

    categorical_columns = likely_cat
    continuous_columns = list(set(data.columns) - set(likely_cat) - set(date_columns))

    return categorical_columns, continuous_columns, date_columns

