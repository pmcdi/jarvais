import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os

class ModelWrapper:
    def __init__(self, predictor, feature_names, target_variable=None):
        self.ag_model = predictor
        self.feature_names = feature_names
        self.target_variable = target_variable
        if target_variable is None and predictor.problem_type != 'regression':
            print("Since target_class not specified, SHAP will explain predictions for each class")
    
    def predict_proba(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1,-1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        preds = self.ag_model.predict_proba(X)
        return preds
    
def plot_feature_importance(predictor, X_test, y_test, 
                            output_dir: str = "./"):
        """
        Plots the feature importance with standard deviation and p-value significance.
        """
        df = predictor.feature_importance(pd.concat([X_test, y_test], axis=1))

        # Plotting
        fig, ax = plt.subplots(figsize=(20, 12), dpi=72)

        # Adding bar plot with error bars
        bars = ax.bar(df.index, df['importance'], yerr=df['stddev'], capsize=5, color='skyblue', edgecolor='black')

        # Adding p_value significance indication
        for i, (bar, p_value) in enumerate(zip(bars, df['p_value'])):
            height = bar.get_height()
            significance = '*' if p_value < 0.05 else ''
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, significance, ha='center', va='bottom', fontsize=12, color='red')

        # Labels and title
        ax.set_xlabel('Feature')
        ax.set_ylabel('Importance')
        ax.set_title('Feature Importance with Standard Deviation and p-value Significance')
        ax.axhline(0, color='grey', linewidth=0.8)
        ax.set_xticks(np.arange(len(df.index.values)))
        ax.set_xticklabels(df.index.values, rotation=45)

        # Save plot
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.show()

def plot_shap_values(predictor, X_train, X_test, 
                     max_display: int = 10,
                     output_dir: str = "./"):
    # import shap only at function call
    import shap
    
    predictor = ModelWrapper(predictor, X_train.columns)
    # sample 100 samples from training set to create baseline
    background_data = shap.sample(X_train, 100) 
    shap_exp = shap.KernelExplainer(predictor.predict_proba, background_data)

    # sample 100 samples from test set to evaluate with shap values
    test_data = shap.sample(X_test, 100) 

    # Compute SHAP values for the test set
    shap_values = shap_exp(test_data)

    # Generate and save the SHAP explanation plots
    
    # this is commented out as beeswarm is missing `ax` parameter
    # fig, ax = plt.subplots(figsize=(20, 12), dpi=300)    
    # shap.plots.beeswarm(shap_values[...,1], max_display=max_display, show=False, ax=ax)
    # fig.savefig(os.path.join(output_dir, 'shap_beeswarm.png'))
    # plt.close()

    fig, ax = plt.subplots(figsize=(20, 12), dpi=72)
    shap.plots.heatmap(shap_values[...,1], max_display=max_display, show=True, ax=ax)
    fig.savefig(os.path.join(output_dir, 'shap_heatmap.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(20, 12), dpi=72)
    shap.plots.bar(shap_values[...,1], max_display=max_display, show=False, ax=ax)
    fig.savefig(os.path.join(output_dir, 'shap_barplot.png'))
    plt.close()

def prep_for_pie(df, label):
    # Prepares data for pie plotting by grouping and sorting values.
    data = df.groupby(label).size().sort_values(ascending=False)

    labels = data.index.tolist()
    values = data.values.tolist()
    
    return labels, values

def plot_one_multiplot(data, umap_data, var, continuous_columns, 
                       output_dir: str = "./"):
    from scipy.stats import f_oneway, ttest_ind

    num_categories = len(data[var].unique())

    sns.set_theme(style="white")
    labels, values = prep_for_pie(data, var)

    # only write % if big enough
    def autopct(pct):
        return ('%1.1f%%' % pct) if pct > 3.5 else ''

    def calculate_fontsize(num_categories):
        base_fontsize = 16
        min_fontsize = 8
        return max(base_fontsize - num_categories * 1.5, min_fontsize)
    
    fontsize = calculate_fontsize(num_categories)

    # setting number of rows/columns for subplots
    n = len(continuous_columns) + 2
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil((n) / rows))
    scaler = 6

    # create subplot grid
    fig, ax = plt.subplots(rows, cols, figsize=(rows*scaler, cols*scaler)) 
    ax = ax.flatten() 

    # Pie plot of categorical variable
    ax[0].pie(values, 
                labels=labels, 
                autopct=autopct, 
                startangle=90,
                counterclock=False,
                textprops={'fontsize': fontsize},
                colors=plt.cm.Set2.colors) # 90 = 12 o'clock, 0 = 3 o'clock, 180 = 9 o'clock
    ax[0].set_title(f"{var} Distribution. N: {data[var].count()}")
    
    # UMAP colored by variable
    sns.scatterplot(x=umap_data[:,0], y=umap_data[:,1], hue=data[var], alpha=.7, ax=ax[1])
    ax[1].set_title(f'UMAP of Continuous Variables with {var}')
    if data[var].nunique() > 5: # Puts legend under plot if there are too many categories
        ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    p_values = {}

    # Calculate p-values
    for col in continuous_columns:
        unique_values = data[var].unique()
        
        # If binary classification, use t-test
        if len(unique_values) == 2:
            group1 = data[data[var] == unique_values[0]][col]
            group2 = data[data[var] == unique_values[1]][col]
            _, p_value = ttest_ind(group1, group2, equal_var=False)
        
        # For more than two categories, use ANOVA
        else:
            groups = [data[data[var] == value][col] for value in unique_values]
            _, p_value = f_oneway(*groups)
        
        p_values[col] = p_value

    # Sort the continuous columns by p-value (ascending order)
    sorted_columns = sorted(p_values, key=p_values.get)

    for i, col in enumerate(sorted_columns):
        sns.violinplot(x=var, y=col, data=data, ax=ax[i+2], inner="point")
        ax[i+2].tick_params(axis='x', labelrotation=67.5)
        ax[i+2].set_title(f"{var} vs {col} (p-value: {p_values[col]:.4f})")

    # Turn off unused axes
    for j in range(n, len(ax)):
        fig.delaxes(ax[j])  # Turn off unused axes

    plt.tight_layout()
    
    # save multiplot
    multiplot_path = os.path.join(output_dir, 'multiplots', f'{var}_multiplots.png')
    plt.savefig(multiplot_path)
    plt.close()
    
    # return path to figure for PDF
    return multiplot_path

def plot_corr(corr, size, 
              file_name: str = 'correlation_matrix.png',
              output_dir: str = "./"):
    

    fig, ax = plt.subplots(1, 1, figsize=(size, size))
    mask = np.triu(np.ones_like(corr, dtype=bool)) # Keep only lower triangle
    np.fill_diagonal(mask, False)
    g = sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidth=.5, fmt="1.2f", ax=ax)
    plt.title(f'Pearson Correlation Matrix')
    plt.tight_layout()

    figure_path = os.path.join(output_dir, file_name)
    fig.savefig(figure_path)
    plt.close()

def plot_pairplot(data, columns_to_plot,
                  output_dir: str = "./", 
                  target_variable: str = None):
    hue = None
    if target_variable is not None:
        columns_to_plot += [target_variable]
        hue = target_variable

    g = sns.pairplot(data[columns_to_plot], hue=hue)   
    g.figure.suptitle("Pair Plot", y=1.08)  

    figure_path = os.path.join(output_dir, 'pairplot.png')
    plt.savefig(figure_path)
    plt.close()

def plot_umap(umap_data, output_dir: str = "./"):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=72)
    g = sns.scatterplot(x=umap_data[:,0], y=umap_data[:,1], alpha=.7, ax=ax)
    plt.title(f'UMAP of Continuous Variables')

    figure_path = os.path.join(output_dir, 'umap_continuous_data.png')
    fig.savefig(figure_path)
    plt.close()