from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, r2_score, root_mean_squared_error
from sklearn.calibration import calibration_curve

from sksurv.nonparametric import kaplan_meier_estimator

from itertools import combinations

from .functional import auprc, bootstrap_metric

sns.set_theme(style="darkgrid", font="Arial")

# ANALYZER

def prep_for_pie(df, label):
    # Prepares data for pie plotting by grouping and sorting values.
    data = df.groupby(label).size().sort_values(ascending=False)

    labels = data.index.tolist()
    values = data.values.tolist()
    
    return labels, values

def plot_one_multiplot(data, umap_data, var, continuous_columns, 
                       output_dir: str | Path = Path.cwd()):
    from scipy.stats import f_oneway, ttest_ind

    output_dir = Path(output_dir)

    num_categories = len(data[var].unique())

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
    
    # save multiplot
    multiplot_path = output_dir / 'multiplots' / f'{var}_multiplots.png'
    plt.savefig(multiplot_path)
    plt.close()
    
    # return path to figure for PDF
    return multiplot_path

def plot_corr(corr, size, 
              file_name: str = 'correlation_matrix.png',
              output_dir: str | Path = Path.cwd()):
    
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(1, 1, figsize=(size, size))
    mask = np.triu(np.ones_like(corr, dtype=bool)) # Keep only lower triangle
    np.fill_diagonal(mask, False)
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidth=.5, fmt="1.2f", ax=ax)
    plt.title(f'Correlation Matrix')
    plt.tight_layout()

    figure_path = output_dir / file_name
    fig.savefig(figure_path)
    plt.close()

def plot_frequency_table(data: pd.DataFrame, columns: list, output_dir: str | Path):

    frequency_dir = Path(output_dir) / 'frequency_tables'
    frequency_dir.mkdir(parents=True, exist_ok=True)

    for column_1, column_2 in combinations(columns, 2):
        heatmap_data = pd.crosstab(data[column_1], data[column_2])
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='d', linewidth=.5)
        plt.title(f'Frequency Table for {column_1} and {column_2}')
        plt.xlabel(column_2)
        plt.ylabel(column_1)
        plt.savefig(frequency_dir / f'{column_1}_vs_{column_2}.png')
        plt.close()

def plot_pairplot(data, columns_to_plot,
                  output_dir: str | Path = Path.cwd(), 
                  target_variable: str = None):
    
    output_dir = Path(output_dir)

    hue = target_variable
    if target_variable is not None:
        columns_to_plot += [target_variable] 

    sns.set_theme(style="darkgrid", font="Arial")
    g = sns.pairplot(data[columns_to_plot], hue=hue)   
    g.figure.suptitle("Pair Plot", y=1.08)  

    figure_path = output_dir / 'pairplot.png'
    plt.savefig(figure_path)
    plt.close()

def plot_umap(umap_data, 
              output_dir: str | Path = Path.cwd()):
   
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=72)
    sns.scatterplot(x=umap_data[:,0], y=umap_data[:,1], alpha=.7, ax=ax)
    plt.title(f'UMAP of Continuous Variables')

    figure_path = output_dir / 'umap_continuous_data.png'
    fig.savefig(figure_path)
    plt.close()

def plot_kaplan_meier_by_category(data_x: pd.DataFrame, data_y: pd.DataFrame, categorical_columns: list, output_dir: str | Path):
    """
    Plots Kaplan-Meier survival curves for each category in the specified categorical columns.

    Parameters:
    - data_x: pandas DataFrame containing features, including categorical columns.
    - data_y: pandas structured array with 'event' and 'time' as columns.
    - categorical_columns: list of categorical column names in data_x to iterate over.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    for cat_col in categorical_columns:
        plt.figure(figsize=(10, 6))
        plt.title(f"Kaplan-Meier Survival Curve by {cat_col}")

        # Get unique categories for the current column
        unique_categories = data_x[cat_col].unique()

        # Plot survival curves for each category
        for category in unique_categories:
            mask_category = data_x[cat_col] == category
            try: # To catch when there are not enough samples for category
                time_category, survival_prob_category, conf_int = kaplan_meier_estimator(
                    data_y["event"][mask_category].astype(bool),
                    data_y["time"][mask_category],
                    conf_type="log-log",
                )
    
                plt.step(
                    time_category,
                    survival_prob_category,
                    where="post",
                    label=f"{cat_col} = {category}"
                )
                plt.fill_between(
                    time_category,
                    conf_int[0],
                    conf_int[1],
                    alpha=0.25,
                    step="post"
                )
            except Exception as _:
                pass

        # Customize plot appearance
        plt.ylim(0, 1)
        plt.ylabel(r"Estimated Probability of Survival $\hat{S}(t)$")
        plt.xlabel("Time $t$")
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.savefig(output_dir / f'kaplan_meier_{cat_col}.png')
        plt.close()

# EXPLAINER

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

        if self.ag_model.can_predict_proba:
            preds = self.ag_model.predict_proba(X)
        else:
            preds = self.ag_model.predict(X)
        return preds
    
def plot_feature_importance(df, output_dir: str | Path, model_name: str=''):
    """
    Plots the feature importance with standard deviation and p-value significance.
    """

    output_dir = Path(output_dir)

    # Plotting
    fig, ax = plt.subplots(figsize=(20, 12), dpi=72)

    # Adding bar plot with error bars
    bars = ax.bar(df.index, df['importance'], yerr=df['stddev'], capsize=5, color='skyblue', edgecolor='black')

    # Adding p_value significance indication
    if 'p_value' in df.columns:
        for bar, p_value in zip(bars, df['p_value']):
            height = bar.get_height()
            significance = '*' if p_value < 0.05 else ''
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.002, significance, 
                    ha='center', va='bottom', fontsize=10, color='red')

    # Labels and title
    ax.set_xlabel('Feature', fontsize=14)
    ax.set_ylabel('Importance', fontsize=14)
    ax.set_title(f'Feature Importance with Standard Deviation and p-value Significance ({model_name})', fontsize=16)
    ax.axhline(0, color='grey', linewidth=0.8)

    # Customize x-axis
    ax.set_xticks(np.arange(len(df.index.values)))
    ax.set_xticklabels(df.index.values, rotation=60, ha='right', fontsize=10)

    # Add gridlines
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add legend for significance at the top right
    significance_patch = plt.Line2D([0], [0], color='red', marker='*', linestyle='None', markersize=10, label='p < 0.05')
    ax.legend(handles=[significance_patch], loc='upper right', fontsize=12)

    # Adjust layout and save
    plt.tight_layout()
    fig.savefig(output_dir / 'feature_importance.png')
    plt.close()

def plot_shap_values(predictor, X_train, X_test, 
                     max_display: int = 10,
                     output_dir: str | Path = Path.cwd()):
    # import shap only at function call
    import shap

    output_dir = Path(output_dir)
    
    predictor = ModelWrapper(predictor, X_train.columns)
    # sample 100 samples from training set to create baseline
    background_data = shap.sample(X_train, 100) 
    shap_exp = shap.KernelExplainer(predictor.predict_proba, background_data)

    # sample 100 samples from test set to evaluate with shap values
    test_data = shap.sample(X_test, 100) 

    # Compute SHAP values for the test set
    shap_values = shap_exp(test_data)
    # print(shap_values[...,1])
    # Generate and save the SHAP explanation plots

    sns.set_theme(style="darkgrid", font="Arial")

    fig, ax = plt.subplots(figsize=(20, 12), dpi=72)
    shap.plots.heatmap(shap_values[...,1], max_display=max_display, show=False, ax=ax)
    fig.savefig(output_dir / 'shap_heatmap.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(20, 12), dpi=72)
    shap.plots.bar(shap_values[...,1], max_display=max_display, show=False, ax=ax)
    fig.savefig(output_dir / 'shap_barplot.png')
    plt.close()

def plot_violin_of_bootsrapped_metrics(predictor, X_test, y_test, X_val, y_val, X_train, y_train, output_dir: str | Path = Path.cwd()):
        
    output_dir = Path(output_dir)
        
    # Define metrics based on the problem type
    if predictor.problem_type == 'regression':
        metrics = [('R Squared', r2_score), ('Root Mean Squared Error', root_mean_squared_error)]
    else:
        metrics = [('AUROC', roc_auc_score), ('AUPRC', auprc)]

    # Prepare lists for DataFrame
    results = []

    # Loop through models and metrics to compute bootstrapped values
    for model_name in predictor.model_names():
        
        if predictor.problem_type == 'regression':
            y_pred_test = predictor.predict(X_test, model=model_name)
            y_pred_val = predictor.predict(X_val, model=model_name)
            y_pred_train = predictor.predict(X_train, model=model_name)
        else:
            y_pred_test = predictor.predict_proba(X_test, model=model_name).iloc[:, 1]
            y_pred_val = predictor.predict_proba(X_val, model=model_name).iloc[:, 1]
            y_pred_train = predictor.predict_proba(X_train, model=model_name).iloc[:, 1]

        for metric_name, metric_func in metrics:
            
            test_values = bootstrap_metric(y_test.to_numpy(), y_pred_test.to_numpy(), metric_func)
            results.extend([(model_name, metric_name, 'Test', value) for value in test_values])
            
            val_values = bootstrap_metric(y_val.to_numpy(), y_pred_val.to_numpy(), metric_func)
            results.extend([(model_name, metric_name, 'Validation', value) for value in val_values])
            
            train_values = bootstrap_metric(y_train.to_numpy(), y_pred_train.to_numpy(), metric_func)
            results.extend([(model_name, metric_name, 'Train', value) for value in train_values])

    # Create a results DataFrame
    result_df = pd.DataFrame(results, columns=['model', 'metric', 'data_split', 'value'])

     # Sort models by median metric value within each combination of metric and data_split
    model_order_per_split = {}
    for split in ['Test', 'Validation', 'Train']:
        split_order = (
            result_df[result_df['data_split'] == split]
            .groupby(['metric', 'model'])['value']
            .median()
            .reset_index()
            .sort_values(by=['metric', 'value'], ascending=[True, False])
            .groupby('metric')['model']
            .apply(list)
            .to_dict()
        )
        model_order_per_split[split] = split_order

    # Function to create violin plots for a specific data split
    def create_violin_plot(data_split, save_path):
        sns.set_theme(style="darkgrid", font="Arial")
        subset = result_df[result_df['data_split'] == data_split]
        g = sns.FacetGrid(
            subset,
            col="metric",
            margin_titles=True,
            height=4,
            aspect=1.5,
            xlim=(0,1.1)
        )

        # Create violin plots with sorted models
        def violin_plot(data, **kwargs):
            metric = data.iloc[0]['metric']
            order = model_order_per_split[data_split].get(metric, None)
            sns.violinplot(data=data, x="value", y="model", linewidth=1, order=order, **kwargs)

        g.map_dataframe(violin_plot)

        # Adjust the titles and axis labels
        g.set_titles(col_template="{col_name}")
        g.set_axis_labels("Metric Value", "Model")

        # Add overall title and adjust layout
        g.figure.suptitle(f"Model Performance of {data_split} Data (Bootstrapped)", fontsize=16)
        g.tight_layout(w_pad=0.5, h_pad=1)

        # Save the plot
        g.savefig(save_path, dpi=500)
        plt.close()

    # Generate and save plots for each data split
    create_violin_plot('Test', output_dir / 'test_metrics_bootstrap.png')
    create_violin_plot('Validation', output_dir / 'validation_metrics_bootstrap.png')
    create_violin_plot('Train', output_dir / 'train_metrics_bootstrap.png')

def plot_regression_diagnostics(y_true, y_pred, output_dir: str | Path = Path.cwd()):
    """
    Generates diagnostic plots for a regression model.

    Parameters:
    -----------
    model: Fitted regression model
        The regression model to evaluate.
    """

    output_dir = Path(output_dir)
    
    # Predict the target values
    residuals = y_true - y_pred

    def plot_regression_line(y_true, y_pred, xlabel='True Values', ylabel='Predictions', title='True vs Predicted Values'):
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="darkgrid", font="Arial")
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
        sns.lineplot(x=y_true, y=y_true, color='red')  # Perfect prediction line
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(output_dir / 'true_vs_predicted.png')
        plt.close()

    def plot_residuals(y_true, y_pred, title='Residual Plot'):
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="darkgrid", font="Arial")
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.title(title)
        plt.savefig(output_dir / 'residual_plot.png')
        plt.close()
    def plot_residual_histogram(residuals, title='Histogram of Residuals'):
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="darkgrid", font="Arial")
        sns.histplot(residuals, kde=True, bins=30)
        plt.xlabel('Residuals')
        plt.title(title)
        plt.savefig(output_dir / 'residual_hist.png')
        plt.close()

    # Call all the plotting functions
    plot_regression_line(y_true, y_pred)
    plot_residuals(y_true, y_pred)
    plot_residual_histogram(residuals)

def plot_classification_diagnostics(y_true, y_pred, y_val, y_val_pred, y_train, y_train_pred, output_dir: str | Path = Path.cwd()):
    """
    Generates diagnostic plots for a classification model.

    """

    output_dir = Path(output_dir)

    plot_epic_copy(y_true.to_numpy(), y_pred.to_numpy(), y_val.to_numpy(), y_val_pred.to_numpy(), y_train.to_numpy(), y_train_pred.to_numpy() , output_dir)

    conf_matrix = confusion_matrix(y_true, y_pred.apply(lambda x: 1 if x >= 0.5 else 0))

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()

### MODEL EVALUATION COPY OF EPIC PLOTS

def _bin_class_curve(y_true, y_pred):
    sort_ix = np.argsort(y_pred, kind="mergesort")[::-1]
    y_true = np.array(y_true)[sort_ix]
    y_pred = np.array(y_pred)[sort_ix]

    # Find where the threshold changes
    distinct_ix = np.where(np.diff(y_pred))[0]
    threshold_idxs = np.r_[distinct_ix, y_true.size - 1]

    # Add up the true positives and infer false ones
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    return fps, tps, y_pred[threshold_idxs]

def plot_epic_copy(y_test, y_pred, y_val, y_val_pred, y_train, y_train_pred, output_dir: str | Path = Path.cwd()):

    output_dir = Path(output_dir)

    # Compute test metrics
    fpr_test, tpr_test, thresholds_roc_test = roc_curve(y_test, y_pred)
    roc_auc_test = roc_auc_score(y_test, y_pred)
    precision_test, recall_test, thresholds_pr_test = precision_recall_curve(y_test, y_pred)
    average_precision_test = auprc(y_test, y_pred)
    prob_true_test, prob_pred_test = calibration_curve(y_test, y_pred, n_bins=10, strategy='uniform')

    # Compute validation metrics
    fpr_val, tpr_val, thresholds_roc_val = roc_curve(y_val, y_val_pred)
    roc_auc_val = roc_auc_score(y_val, y_val_pred)
    precision_val, recall_val, thresholds_pr_val = precision_recall_curve(y_val, y_val_pred)
    average_precision_val = auprc(y_val, y_val_pred)
    prob_true_val, prob_pred_val = calibration_curve(y_val, y_val_pred, n_bins=10, strategy='uniform')
    
    # Compute train metrics
    fpr_train, tpr_train, thresholds_roc_train = roc_curve(y_train, y_train_pred)
    roc_auc_train = roc_auc_score(y_train, y_train_pred)
    precision_train, recall_train, thresholds_pr_train = precision_recall_curve(y_train, y_train_pred)
    average_precision_train = auprc(y_train, y_train_pred)
    prob_true_train, prob_pred_train = calibration_curve(y_train, y_train_pred, n_bins=10, strategy='uniform')

    # Compute confidence intervals
    roc_conf_test = [round(val, 2) for val in np.percentile(bootstrap_metric(y_test, y_pred, roc_auc_score), (2.5, 97.5))]
    roc_conf_val = [round(val, 2) for val in np.percentile(bootstrap_metric(y_val, y_val_pred, roc_auc_score), (2.5, 97.5))]
    roc_conf_train = [round(val, 2) for val in np.percentile(bootstrap_metric(y_train, y_train_pred, roc_auc_score), (2.5, 97.5))]

    precision_conf_test = [round(val, 2) for val in np.percentile(bootstrap_metric(y_test, y_pred, auprc), (2.5, 97.5))]
    precision_conf_val = [round(val, 2) for val in np.percentile(bootstrap_metric(y_val, y_val_pred, auprc), (2.5, 97.5))]
    precision_conf_train = [round(val, 2) for val in np.percentile(bootstrap_metric(y_train, y_train_pred, auprc), (2.5, 97.5))]

    # Set Seaborn style
    sns.set_theme(style="darkgrid", font="Arial")

    plt.figure(figsize=(37.5, 10))

    # 1. ROC Curve
    plt.subplot(2, 5, 1)
    sns.lineplot(x=fpr_test, y=tpr_test, label=f"Test AUROC = {roc_auc_test:.2f} {roc_conf_test}", color="blue")
    sns.lineplot(x=fpr_val, y=tpr_val, label=f"Validation AUROC = {roc_auc_val:.2f} {roc_conf_val}", color="orange")
    sns.lineplot(x=fpr_train, y=tpr_train, label=f"Train AUROC = {roc_auc_train:.2f} {roc_conf_train}", color="green")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title("ROC Curve")
    plt.legend()

    # 2. Precision-Recall Curve
    plt.subplot(2, 5, 2)
    sns.lineplot(x=recall_test, y=precision_test, label=f"Test AUC-PR = {average_precision_test:.2f} {precision_conf_test}", color="blue")
    sns.lineplot(x=recall_val, y=precision_val, label=f"Validation AUC-PR = {average_precision_val:.2f} {precision_conf_val}", color="orange")
    sns.lineplot(x=recall_train, y=precision_train, label=f"Train AUC-PR = {average_precision_train:.2f} {precision_conf_train}", color="green")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    # 3. Calibration Curve
    plt.subplot(2, 5, 6)
    sns.lineplot(x=prob_pred_test, y=prob_true_test, label="Test Calibration Curve", color="blue", marker='o')
    sns.lineplot(x=prob_pred_val, y=prob_true_val, label="Validation Calibration Curve", color="orange", marker='o')
    sns.lineplot(x=prob_pred_train, y=prob_true_train, label="Train Calibration Curve", color="green", marker='o')
    sns.lineplot(x=[0, 1], y=[0, 1], linestyle="--", label="Perfect Calibration", color="gray")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Probability")
    plt.title("Calibration Curve")
    plt.legend()

    # 4. Sensitivity vs Flag Rate
    fps, tps, _ = _bin_class_curve(y_test, y_pred)
    sens_test = tps / sum(y_test)
    flag_rate_test = (tps + fps) / len(y_test)

    fps, tps, _ = _bin_class_curve(y_val, y_val_pred)
    sens_val = tps / sum(y_val)
    flag_rate_val = (tps + fps) / len(y_val)

    fps, tps, _ = _bin_class_curve(y_train, y_train_pred)
    sens_train = tps / sum(y_train)
    flag_rate_train = (tps + fps) / len(y_train)

    plt.subplot(2, 5, 7)
    sns.lineplot(x=flag_rate_test, y=sens_test, label="Test", color="blue")
    sns.lineplot(x=flag_rate_val, y=sens_val, label="Validation", color="orange")
    sns.lineplot(x=flag_rate_train, y=sens_train, label="Train", color="green")
    plt.xlabel('Flag Rate')
    plt.ylabel('Sensitivity')
    plt.title('Sensitivity/Flag Curve')
    plt.legend()

    # 5. Sensitivity, Specificity, PPV by Threshold
    # Test Metrics
    sensitivity_test = tpr_test
    specificity_test = 1 - fpr_test
    ppv_test = precision_test[:-1]

    # Validation Metrics
    sensitivity_val = tpr_val
    specificity_val = 1 - fpr_val
    ppv_val = precision_val[:-1]

    # Train Metrics
    sensitivity_train = tpr_train
    specificity_train = 1 - fpr_train
    ppv_train = precision_train[:-1]

    # Plot Test Metrics
    plt.subplot(2, 5, 3)
    sns.lineplot(x=thresholds_roc_test, y=sensitivity_test, label="Test Sensitivity", color="blue")
    sns.lineplot(x=thresholds_roc_test, y=specificity_test, label="Test Specificity", color="green")
    sns.lineplot(x=thresholds_pr_test, y=ppv_test, label="Test PPV", color="magenta")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Test Metrics by Threshold")
    plt.legend()

    # Plot Validation Metrics
    plt.subplot(2, 5, 4)
    sns.lineplot(x=thresholds_roc_val, y=sensitivity_val, label="Validation Sensitivity", linestyle="--", color="orange")
    sns.lineplot(x=thresholds_roc_val, y=specificity_val, label="Validation Specificity", linestyle="--", color="darkgreen")
    sns.lineplot(x=thresholds_pr_val, y=ppv_val, label="Validation PPV", linestyle="--", color="pink")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Validation Metrics by Threshold")
    plt.legend()

    # Plot Train Metrics
    plt.subplot(2, 5, 5)
    sns.lineplot(x=thresholds_roc_train, y=sensitivity_train, label="Train Sensitivity", linestyle=":", color="purple")
    sns.lineplot(x=thresholds_roc_train, y=specificity_train, label="Train Specificity", linestyle=":", color="brown")
    sns.lineplot(x=thresholds_pr_train, y=ppv_train, label="Train PPV", linestyle=":", color="cyan")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Train Metrics by Threshold")
    plt.legend()

    # 6. Histogram of Predicted Probabilities
    def _get_highest_bin_count(values, bins):
        counts, _ = np.histogram(values, bins=bins)
        return counts.max()

    # Example for y_pred and y_val_pred
    highest_bin_count = max(
        _get_highest_bin_count(y_pred[y_test == 0], bins=20),
        _get_highest_bin_count(y_pred[y_test == 1], bins=20),
        _get_highest_bin_count(y_val_pred[y_val == 0], bins=20),
        _get_highest_bin_count(y_val_pred[y_val == 1], bins=20),
        _get_highest_bin_count(y_train_pred[y_train == 0], bins=20),
        _get_highest_bin_count(y_train_pred[y_train == 1], bins=20)
    )
    highest_bin_count += highest_bin_count//20
    
    plt.subplot(2, 5, 8)
    sns.histplot(y_pred[y_test == 0], bins=20, alpha=0.7, label="Test Actual False", color='blue', kde=False)
    sns.histplot(y_pred[y_test == 1], bins=20, alpha=0.7, label="Test Actual True", color='magenta', kde=False)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Histogram of Predicted Probabilities")
    plt.ylim(0, highest_bin_count)
    plt.legend()

    plt.subplot(2, 5, 9)
    sns.histplot(y_val_pred[y_val == 0], bins=20, alpha=0.5, label="Validation Actual False", color='orange', kde=False)
    sns.histplot(y_val_pred[y_val == 1], bins=20, alpha=0.5, label="Validation Actual True", color='pink', kde=False)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Histogram of Predicted Probabilities")
    plt.ylim(0, highest_bin_count)
    plt.legend()

    plt.subplot(2, 5, 10)
    sns.histplot(y_train_pred[y_train == 0], bins=20, alpha=0.5, label="Train Actual False", color='green', kde=False)
    sns.histplot(y_train_pred[y_train == 1], bins=20, alpha=0.5, label="Train Actual True", color='purple', kde=False)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Histogram of Predicted Probabilities")
    plt.ylim(0, highest_bin_count)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'model_evaluation.png')
    plt.close()
