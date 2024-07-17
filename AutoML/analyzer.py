import os, warnings

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tableone import TableOne
import numpy as np

from pandas.api.types import is_numeric_dtype

from typing import Union

from fpdf import FPDF
from fpdf.enums import Align

class AutoMLAnalyzer():
    
    def __init__(self,
                 data: pd.DataFrame, 
                 target_variable: Union[str, None] = None,
                 categorical: Union[list, None] = None,
                 mapping_file: Union[str, os.PathLike, None] = None,
                 output_dir: Union[str, os.PathLike] = '.'):
        """

        Initializes the AutoMLAnalyzer with the provided data and target variable.
        Optionally, a mapping file can be provided for data cleaning.

        Parameters:
        -----------
        data : pd.DataFrame
            The input data to analyze.
        target_variable : str, optional
            The target variable in the dataset. Default is None.
        categorical: Union[list, None], optional 
            Categorical columns in data. Default is None, and categorical and numerical columns will be infered. 
            If provided remaining columns are infered as numerical.
        mapping_file : Union[str, os.PathLike, None], optional
            Path to a CSV file that contains mappings for data cleaning. Default is None.
        output_dir: Union[str, os.PathLike], optional
            Output directory for analysis. Default is current directory.

        Raises:
        -------
        ValueError
            If the provided mapping file does not exist.

        """

        self.data = data
        self.target_variable = target_variable

        if os.path.exists(output_dir):
            if output_dir == '.':
                print('Using current directory as output directory\n')
            self.output_dir = output_dir
        else:
            raise ValueError('Output path not found')

        if mapping_file is not None:
            if os.path.isfile(mapping_file):
                self.mapping_file = pd.read_csv(mapping_file, header=None) # Ignore headers assume 1st column -> 2nd column
            else:
                raise ValueError('Mapping file does not exist')
        else:
            self.mapping_file = mapping_file

        if categorical is None:
            self._infer_types()
            warnings.warn(f'Categorical columns not defined, used a heuristic to define categorical and continuous columns. Please review!\nCategorical: {self.categorical_columns}\nContinuous: {self.continuous_columns}')
        else:
            self.categorical_columns = categorical
            self.continuous_columns = list(set(self.data.columns) - set(categorical))
            self.data[self.continuous_columns] = self.data[self.continuous_columns].applymap(lambda x: pd.to_numeric(x, errors='coerce')) # Replace all non numerical values with NaN

        self.outlier_analysis = '' # Used later to write to PDF

    def _infer_types(self):
        # adapted from https://github.com/tompollard/tableone/blob/main/tableone/preprocessors.py

        # assume all non-numerical and date columns are categorical
        numeric_cols = set([col for col in self.data.columns if is_numeric_dtype(self.data[col])])
        date_cols = set(self.data.select_dtypes(include=[np.datetime64]).columns)
        likely_cat = set(self.data.columns) - numeric_cols
        likely_cat = list(likely_cat - date_cols)

        # check proportion of unique values if numerical
        for var in self.data._get_numeric_data().columns:
            likely_flag = 1.0 * self.data[var].nunique()/self.data[var].count() < 0.01
            if likely_flag:
                likely_cat.append(var)

        self.continuous_columns = list(set(self.data.columns) - set(likely_cat))
        self.categorical_columns = likely_cat

    def _run_janitor(self):
        
        # Cleans the data by either checking for outliers in categorical variables or applying mappings from the provided file.
        
        if self.mapping_file is None:
            print('Mapping file not found checking for outliers in categorical variables...')

            for cat in self.categorical_columns:
                category_counts = self.data[cat].value_counts()
                threshold = int(category_counts.mean()*.05)
                outliers = category_counts[category_counts < threshold].index.tolist()

                if len(outliers) > 0:
                    outliers = [f'{o}: {category_counts[o]} out of {self.data[cat].count()}' for o in outliers]
                    print(f'  - Outliers found in {cat}: {outliers}')
                    self.outlier_analysis += f'  - Outliers found in {cat}: {outliers}\n'
                else:
                    print(f'  - No Outliers found in {cat}')
                    self.outlier_analysis += f'  - No Outliers found in {cat}\n'
        else:
            print('Applying changes from mapping file...\n')

            self.mapping_file.columns = ['original', 'standard']
            self.mapping_file = self.mapping_file.dropna(subset=['original'])
            self.mapping_file = self.mapping_file[self.mapping_file['standard'] != '']

            key_value_pairs = dict(zip(self.mapping_file['original'], self.mapping_file['standard']))

            self.data = self.data.replace(key_value_pairs)

    def _create_multiplots(self):

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

            cont_columns = [cont for cont in self.data.columns if cont not in self.categorical_columns]
            n = len(cont_columns)

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

            ax[0].set_title(f"{var} Distribution")
            
            for i in range(1, n):

                # # calculate medians and number of observations
                # medians = self.data.groupby(var)[cont_columns[i]].mean().values
                # nobs = self.data[var].value_counts().values
                # nobs = [str(x) for x in nobs.tolist()]
                # nobs = ["n: " + i for i in nobs]
            
                sns.violinplot(x=var, y=cont_columns[i], data=self.data, ax=ax[i], inner="point")
                ax[i].tick_params(axis='x', labelrotation=67.5)
                ax[i].set_title(f"{var} vs {cont_columns[i]}")

                # # Add text to the figure
                # pos = range(len(nobs))
                # for tick in pos:
                #     ax[i].text(pos[tick], medians[tick], nobs[tick],
                #                 horizontalalignment='center',
                #                 size='small',
                #                 color='black')

            
            # Hide any empty subplots
            for j in range(n, len(ax)):
                ax[j].axis('off')

            plt.tight_layout()
            
            # plt.show()
            plt.savefig(os.path.join(self.output_dir, 'multiplots', f'{var}_multiplots.png'))
            self.multiplots.append(os.path.join(self.output_dir, 'multiplots', f'{var}_multiplots.png'))
            plt.close()

    def _create_output_pdf(self):

        pdf = FPDF()

        pdf.add_page()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        font_path = os.path.join(script_dir, 'fonts/DejaVuSans.ttf')
        pdf.add_font("dejavu-sans", style="", fname=font_path)
        
        font_path = os.path.join(script_dir, 'fonts/DejaVuSans-Bold.ttf')
        pdf.add_font("dejavu-sans", style="b", fname=font_path)

        pdf.set_font('dejavu-sans', '', 24)  
        pdf.write(5, "Analysis Report\n\n")

        if self.mapping_file is None:
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
            pdf.image(plot, Align.C, w=pdf.epw-20, h=pdf.eph)

        pdf.add_page()
        pdf.set_font('dejavu-sans', 'b', 14)  
        pdf.write_html(self.mytable.tabulate(tablefmt = "html"), table_line_separators=True)

        pdf.output(os.path.join(self.output_dir, 'analysis_report.pdf'))

    def run(self):

        """
        Runs the data cleaning and visualization process.
        """

        self._run_janitor()

        # Create plots

        self.mytable = TableOne(self.data, categorical=self.categorical_columns, pval=False)
        print(self.mytable.tabulate(tablefmt = "fancy_grid"))
        self.mytable.to_csv(os.path.join(self.output_dir, 'tableone.csv'))
        
        if self.target_variable in self.categorical_columns: # No point in using target as hue if its not a categorical variable
            g = sns.pairplot(self.data, hue=self.target_variable)
        else:
            g= sns.pairplot(self.data)   
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

        self._create_output_pdf()


  
