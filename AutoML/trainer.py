import yaml
import numpy as np
import pandas as pd

from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import train_test_split, HalvingGridSearchCV, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.base import clone

from tabulate import tabulate

from .hparams import get_hyperparameters
from .eval import plot_classification_diagnostics, plot_clustering_diagnostics, plot_regression_diagnostics

class AutoMLSupervised():
    def __init__(self, 
                 task: str = "classifier",
                 cv: int = 5,
                 scaling_method: str='standard',
                 search: str='gridsearch',
                 config_path: str='configs/config.yaml'):
        """
        Initialize the AutoMLTrainer class with specified configurations.

        Parameters
        ----------
        task : str, default='classifier'
            The type of task to handle. Options are: 'classifier', 'regression'.
        cv : int, default=5
            Number of cross-validation folds.
        scaling_method : str, default='standard'
            Normalization method. Options are: 'standard', 'minmax', 'none'.
        search : str, default='gridsearch'
            Hyperparameter search method. Options are: 'halvinggrid', 'grid', 'random'.
        config_path : str, default='config.yaml'
            Path to the YAML configuration file.

        Raises
        ------
        ValueError
            If the task parameter is not one of the specified options.
        """
        self.cv = cv
        self.task = task.lower()
        self.search = search.lower()
        self.scaling_method = scaling_method.lower()

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        if task not in config:
            raise ValueError("Invalid task parameter. Choose one of: 'classifier', 'regression'")

        task_config = config[task]
        self.models = {name: eval(model)() for name, model in task_config['models'].items()}
        self.metrics = task_config['metrics']

        if not self.models or not self.metrics:
            raise ValueError("Models, metrics, or hparams configuration is missing or incorrect.")
        
        self.best_model = None # Definition for plotting check later

    def _get_hparams(self, model: str):
        """
        Returns hyperparameters for finetuning.
        """
        if model in self.hparams:
            return self.hparams[model]
        else:
            raise NotImplementedError(f"No hyperparameters defined for model: {model}")

    def _get_nparams(self, hparams):
        """
        Calculates the total number of hyperparameter combinations.
        """
        if isinstance(hparams, dict):
            n = 1
            for h in hparams:
                n *= len(hparams[h])
        elif isinstance(hparams, list):
            n = [1. for i in range(len(hparams))]
            for i, hparam in enumerate(hparams):
                for h in hparam:
                    n[i] *= len(hparam[h])
            n = sum(n)
        return n

    def _get_model(self, model, hparams, kwargs):
        """
        Returns the appropriate search model.
        """
        if 'halvinggrid' in self.search:
            return HalvingGridSearchCV(model, hparams, factor=2, scoring=self.metrics[-1], cv=self.cv, error_score=0.)
        elif 'grid' in self.search:
            return GridSearchCV(model, hparams, **kwargs)
        elif 'random' in self.search:
            return RandomizedSearchCV(model, hparams, **kwargs)

    def _normalize(self, X, y, 
                  method: str = 'standard'):
        """
        params
        ------
        method
            Which normalization method to use? One of ['standard', 'minmax']

        returns
        -------
        X
        """
        # initialize self.profile containing mean and variance for each feature
        self.profile = {} 

        # set method
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            return X, y

        # convert to numpy array
        if isinstance (X, (pd.DataFrame, pd.Series)):
            X_arr = X.to_numpy()
        else:
            X_arr = X        

        # transform
        X = scaler.fit_transform(X_arr)

        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy(dtype='float64')

        return X, y

    def _evaluate_model(self, k, kwargs):
        model = self.models[k]

        # get+count hparams
        hparams = self._get_hparams(k)
        n_params = self._get_nparams(hparams)

        # perform hparam search
        print(f"\nIterating search through {n_params} hyperparameters for {k}.")

        model = self._get_model(model, hparams, kwargs)
        clf = model.fit(self.X_train, self.y_train)

        result = {
            'model_key': k,
            'cv_results': clf.cv_results_,
            'best_estimator': clone(clf.best_estimator_),
            'scores': {},
            'refit_time': clf.refit_time_,
            'scorer': clf.scorer_,
            'train_score': {},
            'test_score': {}
        }

        # Save metrics
        for metric in clf.scorer_:
            train_score = clf.scorer_[metric](clf, self.X_train, self.y_train)
            result['train_score'][metric] = train_score
            result['scores'][metric] = clf.scorer_[metric](clf, self.X_test, self.y_test)

        return result

    def _display_results(self, result):
        headers = ["Metric", "Train Score", "Test Score"]
        table = []

        for idx, metric in enumerate(result['scores']):
            train_score = result['train_score'][metric]
            test_score = result['scores'][metric]
            if idx == len(result['scores'])-1: # USING LAST METRIC FOR PERFORMANCE, NEED TO CHANGE
                self.best_metric[result['model_key']] = test_score
            table.append([metric, f"{train_score:.4f}", f"{test_score:.4f}"])

        print(f"<< {result['model_key']} >> -- took {result['refit_time']:.5f}s to refit.")
        print(tabulate(table, headers, tablefmt="pretty"))
    
    def plot_results(self, model: str = None):

        if self.best_model is None:
            raise Exception('Best model not defined, run fit first')
        
        if model is None:
            model_to_plot = self.best_model
        else:
            model_to_plot = self.best_models[model]

        model_to_plot.fit(self.X_train, self.y_train)

        if self.task == 'classifier':
            plot_classification_diagnostics(model_to_plot, self.X_test, self.y_test, self.data_columns)
        elif self.task == 'regression':
            plot_regression_diagnostics(model_to_plot, self.X_test, self.y_test, self.data_columns)
   
    def fit(self, data, target_variable=None, 
            test_size: float = 0.2,
            n_iter: int = 25,
            exclude: list = [], stratify_on=''):
        """
        params
        ------
        test_size
            float representing fraction of dataset to hold-out as the test set
        """

        exclude.append(target_variable)
        X = data.drop(columns=exclude) # exclude others marked and target
        
        self.hparams = get_hyperparameters(data, self.task)

        self.data_columns = X.columns

        y = data[target_variable]

        if y.value_counts().min() > 1: # Meaning it can be used to stratify, if this condition is not met train_test_split produces - ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.
            if stratify_on is not None:
                stratify_col = y.astype(str) + '_' + data[stratify_on].astype(str) 
            else:
                stratify_col = y                
        else:
            if stratify_on is not None:
                stratify_col = data[stratify_on]
            else:
                stratify_col = None

        assert X.ndim == 2

        X, y = self._normalize(X, y, method=self.scaling_method) 

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, stratify=stratify_col, random_state=42)

        print(f"Event rates: {self.y_train.sum()/len(self.y_train):.3f} (train) {self.y_test.sum()/len(self.y_test):.3f} (test)")
        print(f"Train/test: {len(self.y_train)}/{len(self.y_test)}")

        # initialize saving best_models, best_params
        self.best_models, self.best_params, self.scores, self.best_metric, self.cv_results = {}, {}, {}, {}, {}
        
        kwargs = {'scoring': self.metrics, 
                  'n_jobs': -1, 
                  'cv': self.cv, 
                  'refit': self.metrics[-1], 
                  'error_score': 0.,
                  'return_train_score': True}
        
        if 'random' in self.search:
            kwargs['n_iter'] = n_iter

        with ThreadPoolExecutor(max_workers=None) as executor:
            future_to_key = {executor.submit(self._evaluate_model, *[k, kwargs]): k for k in self.models}
            for future in as_completed(future_to_key):
                k = future_to_key[future]
                try:
                    result = future.result()
                    self.cv_results[result['model_key']] = result['cv_results']
                    self.best_models[result['model_key']] = result['best_estimator']
                    self.scores[result['model_key']] = result['scores']

                    self._display_results(result)

                except Exception as exc:
                    print(f"Model {k} generated an exception: {exc}")
        
        best_idx = np.argmax(list(self.best_metric.values()))            
        model_k = list(self.best_metric.keys())[best_idx]
        self.best_model = self.best_models[model_k]
        print(f"\n\nBest model: {self.best_model} with {self.metrics[len(self.metrics)-1]} {self.best_metric[model_k]:.4f}")
        print(f"Parameters: {self.best_model.get_params()}")

        return

if __name__ == '__main__':

    # Example usage

    automl = AutoMLSupervised(task="classifier")

    df = pd.read_csv('data/MAR_per_OAR_for_patients.csv')

    X = df[['95HD', 'VolDice', 'SurfDice','JaccardIndex', 'APL', 'FNPL', 'FNV']]
    y = df['MAR'] > 3.5

    automl.fit(X, y)

