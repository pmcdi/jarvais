import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tableone import TableOne

from pandas.api.types import is_numeric_dtype, is_string_dtype

from typing import Union

class AutoMLAnalyzer():
    def __init__(self,
                 data: pd.DataFrame, 
                 target_variable: str,
                 mapping_file: Union[str, os.PathLike, None] = None):
        
        self.data = data
        self.target_variable = target_variable

        if mapping_file is not None:
            if os.path.isfile(mapping_file):
                self.mapping_file = pd.read_csv(mapping_file, header=None) # Ignore headers assume 1st column -> 2nd column
            else:
                raise ValueError('Mapping file does not exist')
        else:
            self.mapping_file = mapping_file

        self.numeric_columns = [col for col in self.data.columns if is_numeric_dtype(self.data[col])]
        self.categorical_columns = [col for col in self.data.columns if is_string_dtype(self.data[col])]

        if 1.*self.data[self.target_variable].nunique()/self.data[self.target_variable].count() < 0.05:
            # Heuristic to check if target variable is categorical, https://stackoverflow.com/questions/35826912/what-is-a-good-heuristic-to-detect-if-a-column-in-a-pandas-dataframe-is-categori
            self.categorical_columns.append(self.target_variable)

    def _run_janitor(self):
        
        if self.mapping_file is None:
            print('Mapping file not found checking for outliers in categorical variables...')

            for cat in self.categorical_columns:
                category_counts = self.data[cat].value_counts()
                threshold = int(category_counts.mean()*.05)
                outliers = category_counts[category_counts < threshold].index.tolist()

                if len(outliers) > 0:
                    print(f'  - Outliers found in {cat}: {outliers}')
                else:
                    print(f'  - No Outliers found in {cat}')
        else:
            print('Applying changes from mapping file...\n')

            self.mapping_file.columns = ['original', 'standard']
            self.mapping_file = self.mapping_file.dropna(subset=['original'])
            self.mapping_file = self.mapping_file[self.mapping_file['standard'] != '']

            key_value_pairs = dict(zip(self.mapping_file['original'], self.mapping_file['standard']))

            self.data = self.data.replace(key_value_pairs)

    def _create_pie_plot(self):

        def prep_for_pie(df, label):
            # df[value] = pd.to_numeric(df[value])

            data = df.groupby(label).size().sort_values(ascending=False)

            labels = data.index.tolist()
            values = data.values.tolist()
            
            return labels, values

        df_counts = pd.DataFrame(columns=["counts"])
        for var in self.categorical_columns:

            num_categories = len(self.data[var].unique())

            if num_categories < 20: # Only plot for small number of categories

                sns.set(style="white")
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
                
                if 'Age' in self.categorical_columns:
                    fig, ax = plt.subplots(1, 2, figsize=(12,6), dpi=150)

                    ax[0].pie(values, 
                            labels=labels, 
                            autopct=autopct, 
                            startangle=90,
                            counterclock=False,
                            textprops={'fontsize': fontsize},
                            colors=plt.cm.Set2.colors) # 90 = 12 o'clock, 0 = 3 o'clock, 180 = 9 o'clock
                    ax[0].set_title(f"{var} Distribution")

                    sns.violinplot(x=var, y="Age", data=self.data, ax=ax[1])
                    ax[1].set_title(f"{var} vs Age")

                else:
                    fig, ax = plt.subplots(figsize=(6,6))
                    ax.pie(values, 
                            labels=labels, 
                            autopct=autopct, 
                            startangle=90,
                            counterclock=False,
                            textprops={'fontsize': fontsize},
                            colors=plt.cm.Set2.colors) # 90 = 12 o'clock, 0 = 3 o'clock, 180 = 9 o'clock
                    ax.set_title(f"{var} Distribution")

                plt.show()
                # fig.savefig(f"multiplots/{var}_multiplots.png")

    def run(self):

        self._run_janitor()

        # Create plots

        mytable = TableOne(self.data, categorical=self.categorical_columns, pval=False)
        print(mytable.tabulate(tablefmt = "fancy_grid"))

        self._create_pie_plot()
        
        sns.pairplot(self.data, hue=self.target_variable)
        plt.show()
        
        pearson_correlation = self.data[self.numeric_columns].corr(method='pearson')

        plt.figure(figsize=(8, 6))
        sns.heatmap(pearson_correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Pearson Correlation of Performance Metrics with Acceptability')
        plt.show()

        spearman_coreelation = self.data[self.numeric_columns].corr(method='spearman')

        plt.figure(figsize=(8, 6))
        sns.heatmap(spearman_coreelation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Spearman Correlation of Performance Metrics with Acceptability')
        plt.show()