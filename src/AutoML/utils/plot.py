import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

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
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.002, significance, 
                    ha='center', va='bottom', fontsize=10, color='red')

        # Labels and title
        ax.set_xlabel('Feature', fontsize=14)
        ax.set_ylabel('Importance', fontsize=14)
        ax.set_title('Feature Importance with Standard Deviation and p-value Significance', fontsize=16)
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
        fig.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()

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
    # print(shap_values[...,1])
    # Generate and save the SHAP explanation plots

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

def plot_clustering_diagnostics(model, X: np.ndarray, cluster_labels: np.ndarray):
    """
    Generates diagnostic plots for a clustering model.

    Parameters:
    -----------
    model: Fitted clustering model
        The clustering model to evaluate.
    X: np.ndarray
        Feature matrix.
    cluster_labels: np.ndarray
        Cluster labels assigned by the clustering model.
    """
    
    def plot_pca_clusters(X, cluster_labels):
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=cluster_labels, palette='viridis')
        plt.title('PCA of Clusters')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

    def plot_silhouette_analysis(X, cluster_labels):
        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        
        plt.figure(figsize=(10, 6))
        y_lower = 10
        for i in range(len(np.unique(cluster_labels))):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values)
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        plt.axvline(x=silhouette_avg, color="red", linestyle="--")
        plt.xlabel("Silhouette Coefficient Values")
        plt.ylabel("Cluster")
        plt.title("Silhouette Analysis")
        plt.show()

    plot_pca_clusters(X, cluster_labels)
    plot_silhouette_analysis(X, cluster_labels)

def plot_regression_diagnostics(y_true, y_pred, output_dir):
    """
    Generates diagnostic plots for a regression model.

    Parameters:
    -----------
    model: Fitted regression model
        The regression model to evaluate.
    """
    
    # Predict the target values
    residuals = y_true - y_pred

    def plot_regression_line(y_true, y_pred, xlabel='True Values', ylabel='Predictions', title='True vs Predicted Values'):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
        sns.lineplot(x=y_true, y=y_true, color='red')  # Perfect prediction line
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(os.path.join(output_dir, 'true_vs_predicted.png'))
        plt.close()

    def plot_residuals(y_true, y_pred, title='Residual Plot'):
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.title(title)
        plt.savefig(os.path.join(output_dir, 'residual_plot.png'))
        plt.close()
    def plot_residual_histogram(residuals, title='Histogram of Residuals'):
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, bins=30)
        plt.xlabel('Residuals')
        plt.title(title)
        plt.savefig(os.path.join(output_dir, 'residual_hist.png'))
        plt.close()

    # Call all the plotting functions
    plot_regression_line(y_true, y_pred)
    plot_residuals(y_true, y_pred)
    plot_residual_histogram(residuals)

def plot_classification_diagnostics(y_true, y_pred, y_val, y_val_pred, output_dir):
    """
    Generates diagnostic plots for a classification model.

    """

    plot_epic_copy(y_true.to_numpy(), y_pred.to_numpy(), y_val.to_numpy(), y_val_pred.to_numpy(), output_dir)

    conf_matrix = confusion_matrix(y_true, y_pred.apply(lambda x: 1 if x >= 0.5 else 0))

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

### MODEL EVALUATION COPY OF EPIC PLOTS

def _bootstrap(y_test, y_pred, f, nsamples=100):
        values = []
        for _ in range(nsamples):
            idx = np.random.randint(len(y_test), size=len(y_test))
            pred_sample = y_pred[idx]
            y_test_sample = y_test[idx]
            val = f(y_test_sample.ravel(), pred_sample.ravel())
            values.append(val)
        return [round(val, 2) for val in np.percentile(values, (2.5, 97.5))]

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

def plot_epic_copy(y_test, y_pred, y_val, y_val_pred, output_dir):
    # Compute test metrics
    fpr_test, tpr_test, thresholds_roc_test = roc_curve(y_test, y_pred)
    roc_auc_test = roc_auc_score(y_test, y_pred)
    precision_test, recall_test, thresholds_pr_test = precision_recall_curve(y_test, y_pred)
    average_precision_test = average_precision_score(y_test, y_pred)
    prob_true_test, prob_pred_test = calibration_curve(y_test, y_pred, n_bins=10, strategy='uniform')

    # Compute validation metrics
    fpr_val, tpr_val, thresholds_roc_val = roc_curve(y_val, y_val_pred)
    roc_auc_val = roc_auc_score(y_val, y_val_pred)
    precision_val, recall_val, thresholds_pr_val = precision_recall_curve(y_val, y_val_pred)
    average_precision_val = average_precision_score(y_val, y_val_pred)
    prob_true_val, prob_pred_val = calibration_curve(y_val, y_val_pred, n_bins=10, strategy='uniform')
    
    roc_conf_test = _bootstrap(y_test, y_pred, roc_auc_score)
    roc_conf_val = _bootstrap(y_val, y_val_pred, roc_auc_score)
    precision_conf_test = _bootstrap(y_test, y_pred, average_precision_score)
    precision_conf_val = _bootstrap(y_val, y_val_pred, average_precision_score)
    
    # Set Seaborn style
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(20, 10))

    # 1. ROC Curve
    plt.subplot(2, 4, 1)
    sns.lineplot(x=fpr_test, y=tpr_test, label=f"Test AUROC = {roc_auc_test:.2f} {roc_conf_test}", color="blue")
    sns.lineplot(x=fpr_val, y=tpr_val, label=f"Validation AUROC = {roc_auc_val:.2f} {roc_conf_val}", color="orange")
    plt.fill_between(fpr_test, tpr_test - (roc_conf_test[1] - roc_auc_test), tpr_test + (roc_conf_test[1] - roc_auc_test), color='blue', alpha=0.2)
    plt.fill_between(fpr_val, tpr_val - (roc_conf_val[1] - roc_auc_val), tpr_val + (roc_conf_val[1] - roc_auc_val), alpha=0.2, color='orange')
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title("ROC Curve")
    plt.legend()

    # 2. Precision-Recall Curve
    plt.subplot(2, 4, 2)
    sns.lineplot(x=recall_test, y=precision_test, label=f"Test AUC-PR = {average_precision_test:.2f} {precision_conf_test}", color="blue")
    sns.lineplot(x=recall_val, y=precision_val, label=f"Validation AUC-PR = {average_precision_val:.2f} {precision_conf_test}", color="orange")
    plt.fill_between(recall_test, precision_test - (precision_conf_test[1] - average_precision_test), precision_test + (precision_conf_test[1] - average_precision_test), color='blue', alpha=0.2)
    plt.fill_between(recall_val, precision_val - (precision_conf_val[1] - average_precision_val), precision_val + (precision_conf_val[1] - average_precision_val), color='orange', alpha=0.2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    # 3. Calibration Curve
    plt.subplot(2, 4, 3)
    sns.lineplot(x=prob_pred_test, y=prob_true_test, label="Test Calibration Curve", color="blue", marker='o')
    sns.lineplot(x=prob_pred_val, y=prob_true_val, label="Validation Calibration Curve", color="orange", marker='o')
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

    plt.subplot(2, 4, 4)
    sns.lineplot(x=flag_rate_test, y=sens_test, label="Test", color="blue")
    sns.lineplot(x=flag_rate_val, y=sens_val, label="Validation", color="orange")
    plt.xlabel('Flag Rate')
    plt.ylabel('Sensitivity')
    plt.title('Sensitivity/Flag Curve')
    plt.legend()

    # 5. Sensitivity, Specificity, PPV by Threshold
    sensitivity_test = tpr_test
    specificity_test = 1 - fpr_test
    ppv_test = precision_test[:-1]
    sensitivity_val = tpr_val
    specificity_val = 1 - fpr_val
    ppv_val = precision_val[:-1]
    plt.subplot(2, 4, 5)
    sns.lineplot(x=thresholds_roc_test, y=sensitivity_test, label="Test Sensitivity", color="blue")
    sns.lineplot(x=thresholds_roc_test, y=specificity_test, label="Test Specificity", color="green")
    sns.lineplot(x=thresholds_pr_test, y=ppv_test, label="Test PPV", color="magenta")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Metrics by Threshold")
    plt.legend()

    plt.subplot(2, 4, 6)
    sns.lineplot(x=thresholds_roc_val, y=sensitivity_val, label="Validation Sensitivity", linestyle="--", color="orange")
    sns.lineplot(x=thresholds_roc_val, y=specificity_val, label="Validation Specificity", linestyle="--", color="darkgreen")
    sns.lineplot(x=thresholds_pr_val, y=ppv_val, label="Validation PPV", linestyle="--", color="pink")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Metrics by Threshold")
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
        _get_highest_bin_count(y_val_pred[y_val == 1], bins=20)
    )
    highest_bin_count += highest_bin_count//20
    
    plt.subplot(2, 4, 7)
    sns.histplot(y_pred[y_test == 0], bins=20, alpha=0.7, label="Test Actual False", color='blue', kde=False)
    sns.histplot(y_pred[y_test == 1], bins=20, alpha=0.7, label="Test Actual True", color='magenta', kde=False)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Histogram of Predicted Probabilities")
    plt.ylim(0, highest_bin_count)
    plt.legend()

    plt.subplot(2, 4, 8)
    sns.histplot(y_val_pred[y_val == 0], bins=20, alpha=0.5, label="Validation Actual False", color='orange', kde=False)
    sns.histplot(y_val_pred[y_val == 1], bins=20, alpha=0.5, label="Validation Actual True", color='pink', kde=False)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Histogram of Predicted Probabilities")
    plt.ylim(0, highest_bin_count)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_evaluation.png'))
    plt.close()