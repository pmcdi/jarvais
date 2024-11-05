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

def mrmr_reduction(task, X, y, k):
    from mrmr import mrmr_classif, mrmr_regression

    mrmr_method = mrmr_classif if task in ['binary', 'multiclass'] else mrmr_regression
    selected_features = mrmr_method(X=X, y=y, K=k, n_jobs=1)
    return X[selected_features]

def var_reduction(X, y):
    from sklearn.feature_selection import VarianceThreshold

    selector = VarianceThreshold()
    _ = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support(indices=True)]
    return X[selected_features]

def kbest_reduction(task, X, y, k):
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression

    if task in ['binary', 'multiclass']:
        f_method = f_classif
    else:
        f_method = f_regression

    selector = SelectKBest(score_func=f_method, k=k)
    _ = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support(indices=True)]
    return X[selected_features]

def chi2_reduction(X, y, k):
    from sklearn.feature_selection import SelectKBest, chi2

    selector = SelectKBest(score_func=chi2, k=k)
    _ = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support(indices=True)]
    return X[selected_features]

def generate_report_pdf(outlier_analysis=None,
                        multiplots=None,
                        categorical_columns=None,
                        output_dir: str = "./"):
    import os
    import pandas as pd
    from fpdf import FPDF
    from fpdf.enums import Align

    from .pdf import add_outlier_analysis, add_multiplots, add_table
    """
    Generate a PDF report of the analysis with plots and tables.

    This method creates a PDF report containing various analysis results, 
    including outlier analysis, correlation plots, multiplots, and a summary table.

    The report is structured as follows:
        - Cover page with the report title.
        - Outlier analysis text.
        - Pair plot, Pearson correlation, and Spearman correlation plots.
        - Multiplots for categorical vs. continuous variable relationships.
        - A tabular summary of the analysis results.

    The PDF uses custom fonts and is saved in the specified output directory.
    """

    # Instantiate PDF
    pdf = FPDF()
    pdf.add_page()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Adding unicode fonts
    font_path = os.path.join(script_dir, 'fonts/DejaVuSans.ttf')
    pdf.add_font("dejavu-sans", style="", fname=font_path)
    font_path = os.path.join(script_dir, 'fonts/DejaVuSans-Bold.ttf')
    pdf.add_font("dejavu-sans", style="b", fname=font_path)
    pdf.set_font('dejavu-sans', '', 24)  

    # Title
    pdf.write(5, "Analysis Report\n\n")

    # Add outlier analysis
    if outlier_analysis:
        pdf = add_outlier_analysis(pdf, outlier_analysis)
    
    # Add page-wide pairplots
    pdf.image(os.path.join(output_dir, 'pairplot.png'), Align.C, w=pdf.epw-20)
    pdf.add_page()

    # Add correlation plots
    pdf.image(os.path.join(output_dir, 'pearson_correlation.png'), Align.C, h=pdf.eph/2)
    pdf.image(os.path.join(output_dir, 'spearman_correlation.png'), Align.C, h=pdf.eph/2)

    # Add multiplots
    if multiplots and categorical_columns:
        pdf = add_multiplots(pdf, multiplots, categorical_columns)

    # Add demographic breakdown "table one"
    path_tableone = os.path.join(output_dir, 'tableone.csv')
    if os.path.exists(path_tableone):
        csv_df = pd.read_csv(path_tableone, na_filter=False).astype(str)
        pdf = add_table(pdf, csv_df)

    # Save PDF
    pdf.output(os.path.join(output_dir, 'analysis_report.pdf'))
