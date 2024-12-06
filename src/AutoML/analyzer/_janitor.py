from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

def knn_impute_categorical(data, columns):
    """
    Perform KNN imputation on categorical columns. 
    From https://www.kaggle.com/discussions/questions-and-answers/153147

    Args:
        data (DataFrame): The data containing categorical columns.
        columns (list): List of categorical column names.

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
    for variable in columns:
        mappings = find_category_mappings(df, variable)
        mappin[variable] = mappings

    for variable in columns:
        integer_encode(df, variable, mappin[variable])

    scaled_data = mm.fit_transform(df)
    knn_imputer = KNNImputer()
    knn_imputed = knn_imputer.fit_transform(scaled_data)
    df.iloc[:, :] = mm.inverse_transform(knn_imputed)
    for col in df.columns:
        df[col] = round(df[col]).astype('int')

    for col in columns:
        inv_map = {v: k for k, v in mappin[col].items()}
        df[col] = df[col].map(inv_map)

    return df

def get_outliers(data, categorical_columns):
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
    
    return "No outliers found\n", mapping