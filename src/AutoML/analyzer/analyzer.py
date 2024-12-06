import yaml, warnings, logging
from pathlib import Path

import pandas as pd
from tableone import TableOne

from ._janitor import replace_missing, get_outliers, infer_types

from ..utils.plot import plot_one_multiplot, plot_corr, plot_pairplot, plot_umap
from ..utils.pdf import generate_analysis_report_pdf

from typing import Union

from umap import UMAP

from joblib import Parallel, delayed

logging.basicConfig(filename=(Path.cwd() / "warnings.log"), level=logging.INFO)

def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    log_message = f"{category.__name__}: {message} in {filename} at line {lineno}"
    logging.warning(log_message)

warnings.showwarning = custom_warning_handler

class Analyzer():
    def __init__(self,
                 data: pd.DataFrame, 
                 target_variable: Union[str, None] = None,
                 one_hot_encode: bool = False,
                 config_file: Union[str, Path, None] = None,
                 output_dir: Union[str, Path] = Path.cwd()):
        """

        Initializes the Analyzer with the provided data.

        Parameters:
        -----------
        data : pd.DataFrame
            The input data to analyze.
        target_variable : str, optional
            The target variable in the dataset. Default is None.
        one_hot_encode : bool, optional
            One Hot Encode the data. Default is False
        config_file : Union[str, os.PathLike, None], optional
            Path to a yaml file that contains settings for janitor. Default is None where janitor will produce one automaticaly.
        output_dir: Union[str, os.PathLike], optional
            Output directory for analysis. Default is current directory.

        """

        self.data = data
        self.target_variable = target_variable
        self.one_hot_encode = one_hot_encode
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.resolve() == Path.cwd():
            print('Using current directory as output directory\n')
         
        self.output_dir = output_dir

        if config_file is not None:
            config_file = Path(config_file) 
            if config_file.is_file():  
                with config_file.open('r') as file: 
                    self.config_file = yaml.safe_load(file)  
            else:
                raise ValueError(f'Config file does not exist at {config_file}')
        else:
            self.config = None

        self.outlier_analysis = '' # Used later when writing to PDF           
        
    def _create_config(self):
        """
        Creates and saves a configuration file for column types, outlier handling, and missing value strategies.

        Steps:
        1. **Infer Column Types**: Identifies categorical, continuous, and date columns using `infer_types`.
        2. **Handle NaN Columns**: Drops columns entirely filled with NaN and updates the continuous column list.
        3. **Outlier Detection**: Identifies outliers in categorical columns and stores the mappings.
        4. **Missing Value Strategy**: Sets default imputation strategies for categorical and continuous variables.
        """

        print('Config file not found. Creating custom...')

        self.config = {}
        columns = {}    

        self.categorical_columns, self.continuous_columns, self.date_columns = infer_types(self.data)
        self.data.loc[:, self.continuous_columns] = self.data.loc[:, self.continuous_columns].apply(lambda x: pd.to_numeric(x, errors='coerce')) # Replace all non numerical values with NaN
        
        nan_ = self.data.apply(lambda col: col.isna().all())
        nan_columns = nan_[nan_].index.tolist()
        if len(nan_columns) > 0:
            print("Columns that are all NaN(probably ID columns) dropping...: ", nan_columns)
            self.continuous_columns = list(set(self.continuous_columns) - set(nan_columns))

        print(f'Used a heuristic to define categorical and continuous columns. Please review!\nCategorical: {self.categorical_columns}\n\nContinuous: {self.continuous_columns}')

        columns['categorical'] = self.categorical_columns
        columns['continuous'] = self.continuous_columns
        columns['date'] = self.date_columns
        columns['other'] = nan_columns

        self.config['columns'] = columns

        outlier_analysis, mapping = get_outliers(self.data, self.categorical_columns)
        
        self.outlier_analysis += outlier_analysis
        self.config['mapping'] = mapping

        self.config['missingness_strategy'] = {}
        self.config['missingness_strategy']['categorical'] = {cat :'Unknown' for cat in self.categorical_columns} # Defining default replacement for each missing categorical variable
        self.config['missingness_strategy']['continuous'] = {cont :'median' for cont in self.continuous_columns} # Defining default replacement for each missing continuous variable
    
    def _apply_config(self):

        print('Applying changes from config...\n')

        for key in self.config['mapping'].keys():
            assert key in self.data.columns, f"{key} in mapping file not found data"
            self.data.loc[:, [key]] = self.data.loc[:, key].replace(self.config['mapping'][key])

        self.data = replace_missing(self.data, self.categorical_columns, self.continuous_columns, self.config)

        if self.one_hot_encode:
            self.data = pd.get_dummies(
                self.data, 
                columns=[cat for cat in self.categorical_columns if cat != self.target_variable])
        
        self.data.to_csv(self.output_dir / 'updated_data.csv')

    def _create_multiplots(self):
        """
        Generate and save multiplots for each categorical variable against all continuous variables. 
        """

        self.multiplots = [] # Used to save in PDF later

        (self.output_dir / 'multiplots').mkdir(parents=True, exist_ok=True)

        self.multiplots = Parallel(n_jobs=-1)(delayed(plot_one_multiplot)(self.data, self.umap_data, var, self.continuous_columns, self.output_dir) for var in self.categorical_columns)

    def run(self):

        """
        Runs the data cleaning and visualization process.
        """

        if self.config is None:
            self._create_config()
        else:
            self.continuous_columns = self.config['columns']['continuous']
            self.categorical_columns = self.config['columns']['categorical']
            self.data.loc[:, self.continuous_columns] = self.data.loc[:, self.continuous_columns].map(lambda x: pd.to_numeric(x, errors='coerce')) # Replace all non numerical values with NaN

        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)

        # Create Table One
        df_keep = self.data[self.continuous_columns + self.categorical_columns]

        self.mytable = TableOne(df_keep, categorical=self.categorical_columns, pval=False)
        print(self.mytable.tabulate(tablefmt = "fancy_grid"))
        self.mytable.to_csv(self.output_dir / 'tableone.csv')

        self._apply_config()

        # PLOTS
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
        generate_analysis_report_pdf(outlier_analysis=self.outlier_analysis, 
                            multiplots=self.multiplots, 
                            categorical_columns=self.categorical_columns, 
                            output_dir=self.output_dir)
    
    @classmethod
    def dry_run(cls, data: pd.DataFrame, output_dir: str | Path = Path.cwd()):

        output_dir = Path(output_dir)

        analyzer = cls(data, output_dir=output_dir)  
        analyzer._create_config()  

        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(analyzer.config, f)

        return analyzer.config  


