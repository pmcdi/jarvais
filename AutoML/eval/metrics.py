import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score, confusion_matrix, roc_curve, auc

# [ ] Precision Recall curve for classification

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

def plot_classification_diagnostics(model, X_test, y_test, data_columns):
    """
    Generates diagnostic plots for a classification model.

    Parameters:
    -----------
    model: Fitted classification model
        The classification model to evaluate.
    X: np.ndarray
        Feature matrix.
    y_true: np.ndarray
        True labels.
    """

    def plot_confusion_matrix(model):
        y_pred = model.predict(X_test)

        conf_matrix = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()


    def plot_roc_auc(model):
        y_pred_proba = model.predict_proba(X_test)[:, 1] 

        # ROC Curve for best subset
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'{model}:\n Area Under the Receiver Operating Characteristic (AUROC)', fontsize=15)
        plt.legend(loc="lower right")
        plt.show()

    plot_confusion_matrix(model)
    plot_roc_auc(model)

    def plot_feature_importance(model, feature_names):
        importance = model.feature_importances_
        indices = np.argsort(importance)

        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.barh(range(len(indices)), importance[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(model, data_columns)

def plot_regression_diagnostics(model, X_test, y_test, data_columns):
    """
    Generates diagnostic plots for a regression model.

    Parameters:
    -----------
    model: Fitted regression model
        The regression model to evaluate.
    """
    
    # Predict the target values
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    def plot_regression_line(y_true, y_pred, xlabel='True Values', ylabel='Predictions', title='True vs Predicted Values'):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
        sns.lineplot(x=y_true, y=y_true, color='red')  # Perfect prediction line
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def plot_residuals(y_true, y_pred, title='Residual Plot'):
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.title(title)
        plt.show()

    def plot_residual_histogram(residuals, title='Histogram of Residuals'):
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, bins=30)
        plt.xlabel('Residuals')
        plt.title(title)
        plt.show()

    # Call all the plotting functions
    plot_regression_line(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    plot_residual_histogram(residuals)
