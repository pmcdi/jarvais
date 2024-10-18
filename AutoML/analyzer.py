import os, yaml

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tableone import TableOne
import numpy as np
from pandas.api.types import is_numeric_dtype

from .utils.plot import plot_one_multiplot, plot_corr, plot_pairplot, plot_umap
from .utils.functional import knn_impute_categorical, get_outliers, generate_report_pdf

from typing import Union




from umap import UMAP

from joblib import Parallel, delayed

class Analyzer():
    def __init__(self,
                 data: pd.DataFrame, 
                 target_variable: Union[str, None] = None,
                 config_file: Union[str, os.PathLike, None] = None,
                 output_dir: Union[str, os.PathLike] = '.'):
        """

        Initializes the Analyzer with the provided data.

        Parameters:
        -----------
        data : pd.DataFrame
            The input data to analyze.
        target_variable : str, optional
            The target variable in the dataset. Default is None.
        config_file : Union[str, os.PathLike, None], optional
            Path to a yaml file that contains settings for janitor. Default is None where janitor will produce one automaticaly.
        output_dir: Union[str, os.PathLike], optional
            Output directory for analysis. Default is current directory.

        """

        self.data = data
        self.target_variable = target_variable

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if output_dir == '.':
            print('Using current directory as output directory\n')
            
        self.output_dir = output_dir

        if config_file is not None:
            if os.path.isfile(config_file):
                with open(config_file, 'r') as file:
                    self.config = yaml.safe_load(file)
            else:
                raise ValueError('Config file does not exist')
        else:
            self.config = None

        self.outlier_analysis = '' # Used later when writing to PDF
    
    def _replace_missing(self):
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

        for cont in self.continuous_columns: 
            strategy = 'median' if cont not in self.config['missingness_strategy']['continuous'].keys() else self.config['missingness_strategy']['continuous'][cont]
            if strategy.lower() == 'median':
                replace_with = self.data[cont].median()
            elif strategy.lower() == 'mean':
                replace_with = self.data[cont].mean()
            elif strategy.lower() == 'mode':
                replace_with = self.data[cont].mode()
            else:
                print(f'Unknown value {strategy} provided to replace {cont}. Using median')
                replace_with = self.data[cont].median()
            self.data[cont] = self.data[cont].fillna(replace_with)

        for cat in self.categorical_columns: # For categorical var, replaces with string provided in config(default Unknown)
            filler = 'Unknown' if cat not in self.config['missingness_strategy']['categorical'].keys() else self.config['missingness_strategy']['categorical'][cat]
            if self.config['missingness_strategy']['categorical'][cat].lower() != 'knn':
                self.data[cat] = self.data[cat].fillna(filler)

        data_to_use = self.data[self.categorical_columns + self.continuous_columns]
        for cat in self.categorical_columns:
            if self.config['missingness_strategy']['categorical'][cat].lower() == 'knn':
                print(f'Using KNN to fill missing values in {cat}, this may take a while...\n')
                self.data[cat] = knn_impute_categorical(data_to_use, self.categorical_columns)[cat]                

    def _infer_types(self):
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

        # assume all non-numerical and date columns are categorical
        numeric_cols = set([col for col in self.data.columns if is_numeric_dtype(self.data[col])])
        numeric_cols = set([col for col in numeric_cols if self.data[col].dtype != bool]) # Filter out boolean 
        likely_cat = set(self.data.columns) - numeric_cols
        likely_cat = list(likely_cat - set(self.date_columns))

        # check proportion of unique values if numerical
        for var in numeric_cols:
            likely_flag = 1.0 * self.data[var].nunique()/self.data[var].count() < 0.025
            if likely_flag:
                likely_cat.append(var)

        likely_cat = [cat for cat in likely_cat if self.data[cat].nunique()/self.data[cat].count() < 0.2] # Heuristic targeted at detecting ID columns
        self.categorical_columns = likely_cat
        self.continuous_columns = list(set(self.data.columns) - set(likely_cat) - set(self.date_columns))
        
    def _run_janitor(self):
        """
        Perform data cleaning and type inference on the dataset.

        This method checks for a configuration file. If none exists, it uses heuristics to 
        infer data types and clean the dataset by identifying date columns, differentiating
        between categorical and continuous columns, and detecting outliers in categorical columns.

        The process includes:
            - Detecting date columns by attempting to convert string columns to datetime.
            - Inferring types of columns as categorical or continuous.
            - Converting non-numeric values in continuous columns to NaN.
            - Identifying columns with all NaN values, likely indicating ID columns.
            - Detecting outliers in categorical columns, which are values occurring in 
            less than 1% of the dataset, and mapping them to 'Other'.
            - Storing the results in a configuration dictionary for further use.

        If a configuration file is present, it applies mappings from the file to clean the dataset.
        """

        if self.config is None:
            self.config = {}
            columns = {}
            
            # Detect dates
            self.date_columns = []
            for col in self.data.columns:
                if self.data[col].dtype == 'object':
                    try:
                        self.data[col] = pd.to_datetime(self.data[col])
                        self.date_columns.append(col)
                    except ValueError:
                        pass
            columns['date'] = self.date_columns

            # Find categorical vs continous 
            self._infer_types()
            self.data.loc[:, self.continuous_columns] = self.data.loc[:, self.continuous_columns].apply(lambda x: pd.to_numeric(x, errors='coerce')) # Replace all non numerical values with NaN
            
            nan_ = self.data.apply(lambda col: col.isna().all())
            nan_columns = nan_[nan_].index.tolist()
            if len(nan_columns) > 0:
                print("Columns that are all NaN(probably ID columns) dropping...: ", nan_columns)
                self.continuous_columns = list(set(self.continuous_columns) - set(nan_columns))

            print(f'Config file not found, used a heuristic to define categorical and continuous columns. Please review!\nCategorical: {self.categorical_columns}\n\nContinuous: {self.continuous_columns}\n')

            columns['categorical'] = self.categorical_columns
            columns['continuous'] = self.continuous_columns
            columns['other'] = nan_columns

            self.config['columns'] = columns

            # Cleans the data by either checking for outliers in categorical variables or applying mappings from the provided file.
            print('Config file not found, checking for outliers in categorical variables...')

            outlier_analysis, mapping = get_outliers(self.data, self.categorical_columns)
            
            self.outlier_analysis += outlier_analysis
            self.config['mapping'] = mapping
        else:
            self.continuous_columns = self.config['columns']['continuous']
            self.categorical_columns = self.config['columns']['categorical']
            self.data.loc[:, self.continuous_columns] = self.data.loc[:, self.continuous_columns].map(lambda x: pd.to_numeric(x, errors='coerce')) # Replace all non numerical values with NaN

        self.data[self.categorical_columns].astype('category')

        print('Applying changes from config file...\n')

        for key in self.config['mapping'].keys():
            assert key in self.data.columns, f"{key} in mapping file not found data"
            self.data.loc[:, [key]] = self.data.loc[:, key].replace(self.config['mapping'][key])

    def _create_multiplots(self):
        """
        Generate and save multiplots for categorical and continuous variables.

        This method creates multiplots for each categorical variable against all continuous variables. 
        It saves the plots as PNG files in a specified output directory. The multiplots consist of a 
        pie chart showing the distribution of the categorical variable and violin plots showing the 
        relationship between the categorical and each continuous variable.

        Steps:
            - Create a directory for saving multiplots if it does not exist.
            - Prepare data for pie chart plotting by grouping and sorting values.
            - Generate a pie chart for each categorical variable to show distribution.
            - Generate violin plots for each continuous variable against the categorical variable.
            - Adjust font size based on the number of categories.
            - Save plots to files and store file paths in the multiplots list.
        """

        self.multiplots = [] # Used to save in PDF later

        if not os.path.exists(os.path.join(self.output_dir, 'multiplots')): # To save multiplots
            os.mkdir(os.path.join(self.output_dir, 'multiplots'))

        self.multiplots = Parallel(n_jobs=-1)(delayed(plot_one_multiplot)(self.data, self.umap_data, var, self.continuous_columns, self.output_dir) for var in self.categorical_columns)

    def run(self):

        """
        Runs the data cleaning and visualization process.
        """

        self._run_janitor()

        # Create Table One
        df_keep = self.data[self.continuous_columns + self.categorical_columns]

        self.mytable = TableOne(df_keep, categorical=self.categorical_columns, pval=False)
        print(self.mytable.tabulate(tablefmt = "fancy_grid"))
        self.mytable.to_csv(os.path.join(self.output_dir, 'tableone.csv'))

        # Apply missingness replacement and save updated data
        if not 'missingness_strategy' in self.config.keys():
            print('Applying default missingness to of unknown and median, change strategy in config file and run again if needed...\n')
            self.config['missingness_strategy'] = {}
            self.config['missingness_strategy']['categorical'] = {cat :'Unknown' for cat in self.categorical_columns} # Defining default replacement for each missing categorical variable
            self.config['missingness_strategy']['continuous'] = {cont :'median' for cont in self.continuous_columns} # Defining default replacement for each missing continuous variable

            with open(os.path.join(self.output_dir, 'config.yaml'), 'w') as f:
                yaml.dump(self.config, f)

        # Clean it up
        self._replace_missing()
        self.data.to_csv(os.path.join(self.output_dir, 'updated_data.csv'))

        # Create Plots 
        size = len(self.continuous_columns)*1.5

        # Correlation Plots
        p_corr = self.data[self.continuous_columns].corr(method="pearson")
        s_corr = self.data[self.continuous_columns].corr(method="spearman")
        plot_corr(p_corr, size, file_name='pearson_correlation.png', output_dir=self.output_dir)
        plot_corr(s_corr, size, file_name='spearman_correlation.png', output_dir=self.output_dir)

        # UMAP reduced data + Plots
        self.umap_data = UMAP(n_components=2).fit_transform(self.data[self.continuous_columns])
        plot_umap(self.umap_data, output_dir=self.output_dir)

        # plot pairplot
        if len(self.continuous_columns) > 10: # Keep only the top ten correlated pairs in the pair plot
            corr_pairs = s_corr.abs().unstack().sort_values(kind="quicksort", ascending=False).drop_duplicates()
            top_10_pairs = corr_pairs[corr_pairs < 1].nlargest(5)
            columns_to_plot = list(set([index for pair in top_10_pairs.index for index in pair]))
        else:
            columns_to_plot = self.continuous_columns

        if self.target_variable in self.categorical_columns: # No point in using target as hue if its not a categorical variable
            plot_pairplot(self.data, columns_to_plot, output_dir=self.output_dir, target_variable=self.target_variable)     
        else:
            plot_pairplot(self.data, columns_to_plot, output_dir=self.output_dir)         

        # Create Multiplots
        self._create_multiplots()

        # Create Output PDF
        generate_report_pdf(outlier_analysis=self.outlier_analysis, 
                            multiplots=self.multiplots, 
                            categorical_columns=self.categorical_columns, 
                            output_dir=self.output_dir)

