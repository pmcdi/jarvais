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

class AutoMLTrainer():
    def __init__(self, 
                 task: str = "classifier",
                 cv: int = 5,
                 scaling_method: str='standard',
                 search: str='gridsearch',
                 config_path: str='configs/config.yaml'):
        """
        Initialize the AutoML class with specified configurations.

        Parameters
        ----------
        task : str, default='classifier'
            The type of task to handle. Options are: 'classifier', 'regression', 'cluster'.
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
            raise ValueError("Invalid task parameter. Choose one of: 'classifier', 'regression', 'cluster'")

        task_config = config[task]
        self.models = {name: eval(model)() for name, model in task_config['models'].items()}
        self.metrics = task_config['metrics']
        self.hparams = task_config['hparams']

        if not self.models or not self.metrics or not self.hparams:
            raise ValueError("Models, metrics, or hparams configuration is missing or incorrect.")

    def get_hparams(self, model: str):
        """
        Returns hyperparameters for finetuning.
        """
        if model in self.hparams:
            return self.hparams[model]
        else:
            raise NotImplementedError(f"No hyperparameters defined for model: {model}")

    def get_nparams(self, hparams: dict):
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

    def tune_hparams(self, model, hparams, kwargs, X, y):
        """
        Tune hyperparameters and fit the model.
        """
        model = self.get_model(model, hparams, kwargs)
        return model.fit(X, y)

    def get_model(self, model, hparams, kwargs):
        """
        Returns the appropriate search model.
        """
        if 'halvinggrid' in self.search:
            return HalvingGridSearchCV(model, hparams, factor=2, scoring=self.metrics[-1], cv=self.cv, error_score=0., verbose=1)
        elif 'grid' in self.search:
            return GridSearchCV(model, hparams, verbose=1, **kwargs)
        elif 'random' in self.search:
            return RandomizedSearchCV(model, hparams, verbose=1, **kwargs)
        
    def scale(self, scaler, X, name='y'):
        """
        Scale data `X` using `scaler` and save profile of mean/variance for each feature
        """

        # convert to numpy array
        if isinstance (X, (pd.DataFrame, pd.Series)):
            X_arr = X.to_numpy()
        else:
            X_arr = X

        # add 2nd dimension
        if X.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        # transform
        X_arr = scaler.fit_transform(X_arr)

        # record scaling profile
        if isinstance(X, pd.DataFrame):
            for n, feature in enumerate(X.columns):
                self.profile[feature] = {'mean': scaler.mean_[n], 'var': scaler.var_[n]}
        elif isinstance(X, pd.Series) or X.ndim == 1:
            self.profile[X.name] = {'mean': scaler.mean_[0], 'var': scaler.var_[0]}
        else:
            for n in range(X.shape[1]):
                self.profile[name + "_" + str(n)] = {'mean': scaler.mean_[n], 'var': scaler.var_[n]}

        return X_arr

    def normalize(self, X, y, 
                  method: str = 'standard'):
        """
        params
        ------
        method
            Which noramlization method to use? One of ['standard', 'minmax']

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

        # scale using the chosen scaler
        X = self.scale(scaler, X, name='X')
        if 'class' not in self.task:
            y = self.scale(scaler, y, name='y')
        elif isinstance(X, (pd.DataFrame, pd.Series)):
            y = y.to_numpy(dtype='float64')

        # assert 0 < y.sum() < len(y) ??????????????????

        return X, y

    def evaluate_model(self, k, X_train, X_test, y_train, y_test, kwargs):
        model = self.models[k]

        # get+count hparams
        hparams = self.get_hparams(k)
        n_params = self.get_nparams(hparams)

        # perform hparam search
        print(f"\nIterating search through {n_params} hyperparameters for {k}.")
        clf = self.tune_hparams(model, hparams, kwargs, X_train, y_train)

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
            train_score = clf.scorer_[metric](clf, X_train, y_train)
            result['train_score'][metric] = train_score
            result['scores'][metric] = clf.scorer_[metric](clf, X_test, y_test)

        return result
        
    def fit(self, X, y, 
            test_size: float = 0.2,
            n_iter: int = 25,):
        """
        params
        ------
        X
            independent variables
        y
            dependent variable / endpoint to predict
        test_size
            float representing fraction of dataset to hold-out as the test set
        """
        assert X.ndim == 2

        X, y = self.normalize(X, y, method=self.scaling_method) 

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
        print(f"Event rates: {y_train.sum()/len(y_train):.3f} (train) {y_test.sum()/len(y_test):.3f} (test)")
        print(f"Train/test: {len(y_train)}/{len(y_test)}")

        # initialize saving best_models, best_params
        self.best_models, self.best_params, self.scores, self.auroc, self.cv_results = {}, {}, {}, {}, {}
        
        kwargs = {'scoring': self.metrics, 
                  'n_jobs': -1, 
                  'cv': self.cv, 
                  'refit': self.metrics[-1], 
                  'error_score': 0.,
                  'return_train_score': True}
        
        if 'random' in self.search:
            kwargs['n_iter'] = n_iter

        with ThreadPoolExecutor(max_workers=None) as executor:
            future_to_key = {executor.submit(self.evaluate_model, *[k, X_train, X_test, y_train, y_test, kwargs]): k for k in self.models}
            for future in as_completed(future_to_key):
                k = future_to_key[future]
                try:
                    result = future.result()
                    self.cv_results[result['model_key']] = result['cv_results']
                    self.best_models[result['model_key']] = result['best_estimator']
                    self.scores[result['model_key']] = result['scores']

                    print(f"<< {result['model_key']} >> -- took {result['refit_time']:.5f}s to refit.")
                    print(f"metric              |      train |     test |")

                    for metric in result['scores']:
                        train_score = result['train_score'][metric]
                        test_score = result['scores'][metric]
                        if "roc_auc" == metric:
                            self.auroc[result['model_key']] = test_score
                        print(f"{metric:<20}:   {train_score:>8.4f} {test_score:>10.4f}")

                except Exception as exc:
                    print(f"Model {k} generated an exception: {exc}")
        
        best_idx = np.argmax(list(self.auroc.values()))            
        model_k = list(self.auroc.keys())[best_idx]
        self.best_model = self.best_models[model_k]
        print(f"\n\nBest model: {self.best_model} with AUROC {self.auroc[model_k]:.4f}")
        print(f"Parameters: {self.best_model.get_params()}")

        return

if __name__ == '__main__':

    # Example usage

    automl = AutoMLTrainer(task="classifier")

    df = pd.read_csv('data/MAR_per_OAR_for_patients.csv')

    X = df[['95HD', 'VolDice', 'SurfDice','JaccardIndex', 'APL', 'FNPL', 'FNV']]
    y = df['MAR'] > 3.5

    automl.fit(X, y)

