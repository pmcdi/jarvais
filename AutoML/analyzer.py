import os, warnings, yaml

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tableone import TableOne
import numpy as np
from datetime import datetime

from pandas.api.types import is_numeric_dtype

from typing import Union

from fpdf import FPDF
from fpdf.enums import Align

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

class AutoMLAnalyzer():
    
    def __init__(self,
                 data: pd.DataFrame, 
                 target_variable: Union[str, None] = None,
                 config_file: Union[str, os.PathLike, None] = None,
                 output_dir: Union[str, os.PathLike] = '.'):
        """

        Initializes the AutoMLAnalyzer with the provided data.

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
    
    def _knn_impute_categorical(self, data, columns):
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
            strategy = self.config['missingness_strategy']['continuous'][cont]
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
            if self.config['missingness_strategy']['categorical'][cat].lower() != 'knn':
                self.data[cat] = self.data[cat].fillna(self.config['missingness_strategy']['categorical'][cat])

        data_to_use = self.data[self.categorical_columns + self.continuous_columns]
        for cat in self.categorical_columns:
            if self.config['missingness_strategy']['categorical'][cat].lower() == 'knn':
                print(f'Using KNN to fill missing values in {cat}, this may take a while...\n')
                self.data[cat] = self._knn_impute_categorical(data_to_use, self.categorical_columns)[cat]                

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
            self.data.loc[:, self.continuous_columns] = self.data.loc[:, self.continuous_columns].map(lambda x: pd.to_numeric(x, errors='coerce')) # Replace all non numerical values with NaN
            
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

            mapping = {} # Make a mapping file based on heuristic to be updated later

            for cat in self.categorical_columns:
                category_counts = self.data[cat].value_counts()
                threshold = int(len(self.data)*.01)
                outliers = category_counts[category_counts < threshold].index.tolist()

                mapping[cat] = {}

                for _cat in self.data[cat].unique():
                    if _cat in outliers:
                        mapping[cat][f'{_cat}'] = 'Other'
                    else:
                        mapping[cat][f'{_cat}'] = f'{_cat}'

                if len(outliers) > 0:
                    outliers = [f'{o}: {category_counts[o]} out of {self.data[cat].count()}' for o in outliers]
                    print(f'  - Outliers found in {cat}: {outliers}')
                    self.outlier_analysis += f'  - Outliers found in {cat}: {outliers}\n'
                else:
                    print(f'  - No Outliers found in {cat}')
                    self.outlier_analysis += f'  - No Outliers found in {cat}\n'

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

        def prep_for_pie(df, label):

            # Prepares data for pie plotting by grouping and sorting values.

            data = df.groupby(label).size().sort_values(ascending=False)

            labels = data.index.tolist()
            values = data.values.tolist()
            
            return labels, values

        df_counts = pd.DataFrame(columns=["counts"])
        for var in self.categorical_columns:

            num_categories = len(self.data[var].unique())

            sns.set_theme(style="white")
            labels, values = prep_for_pie(self.data, var)

            # save counts to dataframe
            df_counts = pd.concat([df_counts, 
                                pd.DataFrame(index=["", var], data=["", ""], columns=["counts"]), 
                                pd.DataFrame(index=labels, data=values, columns=["counts"])])

            # only write % if big enough
            def autopct(pct):
                return ('%1.1f%%' % pct) if pct > 3.5 else ''


            def calculate_fontsize(num_categories):
                base_fontsize = 16
                min_fontsize = 8
                return max(base_fontsize - num_categories * 1.5, min_fontsize)
            
            fontsize = calculate_fontsize(num_categories)

            n = len(self.continuous_columns)

            rows = int(np.ceil(np.sqrt(n)))
            cols = int(np.ceil((n) / rows))
            fig, ax = plt.subplots(rows, cols, figsize=(24, 18)) 
            ax = ax.flatten() 

            ax[0].pie(values, 
                        labels=labels, 
                        autopct=autopct, 
                        startangle=90,
                        counterclock=False,
                        textprops={'fontsize': fontsize},
                        colors=plt.cm.Set2.colors) # 90 = 12 o'clock, 0 = 3 o'clock, 180 = 9 o'clock

            ax[0].set_title(f"{var} Distribution. N: {self.data[var].count()}")
            
            for i in range(1, n):
                sns.violinplot(x=var, y=self.continuous_columns[i], data=self.data, ax=ax[i], inner="point")
                ax[i].tick_params(axis='x', labelrotation=67.5)
                ax[i].set_title(f"{var} vs {self.continuous_columns[i]}")
            
            # Hide any empty subplots
            for j in range(n, len(ax)):
                ax[j].axis('off')

            plt.tight_layout()
            
            plt.savefig(os.path.join(self.output_dir, 'multiplots', f'{var}_multiplots.png'))
            self.multiplots.append(os.path.join(self.output_dir, 'multiplots', f'{var}_multiplots.png'))
            plt.close()

    def _create_output_pdf(self):
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

        pdf = FPDF()

        pdf.add_page()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        font_path = os.path.join(script_dir, 'fonts/DejaVuSans.ttf')
        pdf.add_font("dejavu-sans", style="", fname=font_path)
        
        font_path = os.path.join(script_dir, 'fonts/DejaVuSans-Bold.ttf')
        pdf.add_font("dejavu-sans", style="b", fname=font_path)

        pdf.set_font('dejavu-sans', '', 24)  
        pdf.write(5, "Analysis Report\n\n")

        if self.outlier_analysis != '':
            pdf.set_font('dejavu-sans', '', 12)  
            pdf.write(5, f"Outlier Analysis:\n")
            pdf.set_font('dejavu-sans', '', 10) 
            pdf.write(5, self.outlier_analysis)
        
        pdf.image(os.path.join(self.output_dir, 'pairplot.png'), Align.C, w=pdf.epw-20)

        pdf.add_page()
        pdf.image(os.path.join(self.output_dir, 'pearson_correlation.png'), Align.C, h=pdf.eph/2)
        pdf.image(os.path.join(self.output_dir, 'spearman_correlation.png'), Align.C, h=pdf.eph/2)

        for plot in self.multiplots:
            pdf.add_page()
            pdf.image(plot, keep_aspect_ratio=True, w=pdf.epw-20, h=pdf.eph)

        pdf.add_page()
        pdf.set_font('dejavu-sans', 'b', 14)  
        pdf.write_html(self.mytable.tabulate(tablefmt = "html"), table_line_separators=True)

        pdf.output(os.path.join(self.output_dir, 'analysis_report.pdf'))

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

        self._replace_missing()
        self.data.to_csv(os.path.join(self.output_dir, 'updated_data.csv'))

        # Create Plots
        
        if self.target_variable in self.categorical_columns: # No point in using target as hue if its not a categorical variable
            g = sns.pairplot(df_keep, hue=self.target_variable)
        else:
            g= sns.pairplot(df_keep)   
        g.figure.suptitle("Pair Plot", y=1.08)  

        figure_path = os.path.join(self.output_dir, 'pairplot.png')
        plt.savefig(figure_path)
        plt.close()

        pearson_correlation = self.data[self.continuous_columns].corr(method='pearson')

        plt.figure(figsize=(16,12))
        mask = np.triu(np.ones_like(pearson_correlation, dtype=bool)) # Keep only lower triangle
        np.fill_diagonal(mask, False)
        g = sns.heatmap(pearson_correlation, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidth=.5, fmt="1.2f")
        plt.title(f'Pearson Correlation Matrix')
        plt.tight_layout()

        figure_path = os.path.join(self.output_dir, 'pearson_correlation.png')
        plt.savefig(figure_path)
        plt.close()

        spearman_correlation = self.data[self.continuous_columns].corr(method='spearman')

        plt.figure(figsize=(16,12))
        mask = np.triu(np.ones_like(spearman_correlation, dtype=bool)) # Keep only lower triangle
        np.fill_diagonal(mask, False)
        g = sns.heatmap(spearman_correlation, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidth=.5, fmt="1.2f")
        plt.title(f'Spearman Correlation Matrix')
        plt.tight_layout()

        figure_path = os.path.join(self.output_dir, 'spearman_correlation.png')
        plt.savefig(figure_path)
        plt.close()

        self._create_multiplots()

        # Create Output PDF
        self._create_output_pdf()


  
