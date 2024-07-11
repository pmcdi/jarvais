import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tableone import TableOne
import numpy as np

from pandas.api.types import is_numeric_dtype, is_string_dtype

from typing import Union

from fpdf import FPDF
from fpdf.enums import Align

class AutoMLAnalyzer():
    
    def __init__(self,
                 data: pd.DataFrame, 
                 target_variable: Union[str, None] = None,
                 mapping_file: Union[str, os.PathLike, None] = None,
                 output_dir: Union[str, os.PathLike] = '.'):
        """

        Initializes the AutoMLAnalyzer with the provided data and target variable.
        Optionally, a mapping file can be provided for data cleaning.

        Parameters:
        -----------
        data : pd.DataFrame
            The input data to analyze.
        target_variable : str
            The target variable in the dataset.
        mapping_file : Union[str, os.PathLike, None], optional
            Path to a CSV file that contains mappings for data cleaning. Default is None.

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

        self.numeric_columns = [col for col in self.data.columns if is_numeric_dtype(self.data[col])]
        self.categorical_columns = [col for col in self.data.columns if is_string_dtype(self.data[col])]

        if self.target_variable is not None and 1.*self.data[self.target_variable].nunique()/self.data[self.target_variable].count() < 0.04:
            # Heuristic to check if target variable is categorical, https://stackoverflow.com/questions/35826912/what-is-a-good-heuristic-to-detect-if-a-column-in-a-pandas-dataframe-is-categori
            self.categorical_columns.append(self.target_variable)

        self.outlier_analysis = '' # Used later to write to PDF

    def _run_janitor(self):
        
        # Cleans the data by either checking for outliers in categorical variables or applying mappings from the provided file.
        
        if self.mapping_file is None:
            print('Mapping file not found checking for outliers in categorical variables...')

            for cat in self.categorical_columns:
                category_counts = self.data[cat].value_counts()
                threshold = int(category_counts.mean()*.05)
                outliers = category_counts[category_counts < threshold].index.tolist()

                if len(outliers) > 0:
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

            fig, ax = plt.subplots()

            ax.pie(values, 
                        labels=labels, 
                        autopct=autopct, 
                        startangle=90,
                        counterclock=False,
                        textprops={'fontsize': fontsize},
                        colors=plt.cm.Set2.colors) # 90 = 12 o'clock, 0 = 3 o'clock, 180 = 9 o'clock

            ax.set_title(f"{var} Distribution")
            plt.tight_layout()
            
            fig.savefig(os.path.join(self.output_dir, f'{var}_piechart.png'))
            plt.close(fig)
            self.multiplots.append(os.path.join(self.output_dir, f'{var}_piechart.png'))
            
            for cont in [cont for cont in self.data.columns if cont not in self.categorical_columns]:

                sns.violinplot(x=var, y=cont, data=self.data)
                plt.xticks(rotation=67.5)
                plt.title(f"{var} vs {cont}")
                plt.tight_layout()

                # plt.show()
                plt.savefig(os.path.join(self.output_dir, f'{var}_{cont}_multiplots.png'))
                self.multiplots.append(os.path.join(self.output_dir, f'{var}_{cont}_multiplots.png'))
                plt.close()

    def _create_output_pdf(self):

        pdf = FPDF()

        pdf.add_page()

        pdf.set_font('Times', '', 24)  
        pdf.write(5, "Analysis Report\n\n")

        if self.mapping_file is None:
            pdf.set_font('Times', '', 12)  
            pdf.write(5, f"Outlier Analysis:\n")
            pdf.set_font('Times', '', 10) 
            pdf.write(5, self.outlier_analysis)
        
        pdf.image(os.path.join(self.output_dir, 'pairplot.png'), Align.C, w=pdf.epw-20)

        pdf.add_page()
        pdf.image(os.path.join(self.output_dir, 'pearson_correlation.png'), Align.C, h=pdf.eph/2)
        pdf.image(os.path.join(self.output_dir, 'spearman_correlation.png'), Align.C, h=pdf.eph/2)

        if len(self.multiplots) > 0: # Don't want to add a page if not needed
            pdf.add_page()
            for plot in self.multiplots:
                pdf.image(plot, Align.C, w=pdf.epw-20, h=pdf.eph/2)

        pdf.add_page()
        pdf.set_font('Times', '', 14)  
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

        pearson_correlation = self.data[self.numeric_columns].corr(method='pearson')

        plt.figure(figsize=(8, 6))
        mask = np.triu(np.ones_like(pearson_correlation, dtype=bool)) # Keep only lower triangle
        np.fill_diagonal(mask, False)
        sns.heatmap(pearson_correlation, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidth=.5, square=True)
        plt.title(f'Pearson Correlation Matrix')
        plt.tight_layout()

        figure_path = os.path.join(self.output_dir, 'pearson_correlation.png')
        plt.savefig(figure_path)
        plt.close()

        spearman_correlation = self.data[self.numeric_columns].corr(method='spearman')

        plt.figure(figsize=(8, 6))
        mask = np.triu(np.ones_like(spearman_correlation, dtype=bool)) # Keep only lower triangle
        np.fill_diagonal(mask, False)
        sns.heatmap(spearman_correlation, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidth=.5, square=True)
        plt.title(f'Spearman Correlation Matrix')
        plt.tight_layout()

        figure_path = os.path.join(self.output_dir, 'spearman_correlation.png')
        plt.savefig(figure_path)
        plt.close()

        self._create_multiplots()

        self._create_output_pdf()


  
