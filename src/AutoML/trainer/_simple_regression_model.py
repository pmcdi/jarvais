import pandas as pd
from numpy import float32, ndarray

from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator

from sklearn.linear_model import LinearRegression, LogisticRegression

class SimpleRegressionModel(AbstractModel):
    """
    A simple regression model wrapped in an AutoGluon model class.

    It can handle `binary`, `multiclass`, `regression` tasks.
    """
    def __init__(self, **kwargs):
        # Simply pass along kwargs to parent, and init our internal `_feature_generator` variable to None
        super().__init__(**kwargs)
        self._feature_generator = None

    # The `_preprocess` method takes the input data and transforms it to the internal representation usable by the model.
    # `_preprocess` is called by `preprocess` and is used during model fit and model inference.
    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> ndarray:
        # print(f'Entering the `_preprocess` method: {len(X)} rows of data (is_train={is_train})')
        X = super()._preprocess(X, **kwargs)

        if is_train:
            # X will be the training data.
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            # This converts categorical features to numeric via stateful label encoding.
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)

        # Add a fillna call to handle missing values.
        return X.fillna(0).to_numpy(dtype=float32)

    # The `_fit` method takes the input training data (and optionally the validation data) and trains the model.
    def _fit(self,
            X: pd.DataFrame,  # training data
            y: pd.Series,  # training labels
            **kwargs):

        # Store the feature names before transforming to numpy
        feature_names = X.columns if isinstance(X, pd.DataFrame) else range(X.shape[1])

        # Make sure to call preprocess on X near the start of `_fit`.
        X = self.preprocess(X, is_train=True)

        # This fetches the user-specified (and default) hyperparameters for the model.
        params = self._get_model_params()

        if self.problem_type == 'regression':
            self.model = LinearRegression()
        else:
            self.model = LogisticRegression(**params)
        self.model.fit(X, y)

        # Print the coefficients and the intercept after training
        coefs = self.model.coef_.flatten()
        intercept = self.model.intercept_ if self.problem_type == 'regression' else self.model.intercept_[0]

        # Create a DataFrame to display feature names with their corresponding coefficients
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefs
        })

        # Print the coefficients along with their corresponding features
        # print("\nSimple Regression Model Coefficients:")
        # print(coef_df.sort_values(by='Coefficient', key=abs, ascending=False).to_string(index=False))  # sort by coefficient value for better readability

        # # Print the intercept separately
        # print(f"\nSimple Regression Model Intercept: {intercept}")

        # print('Exiting the `_fit` method')

    # The `_set_default_params` method defines the default hyperparameters of the model.
    def _set_default_params(self):
        default_params = {
            'solver': 'lbfgs',  # Solver to use in the optimization problem
            'max_iter': 1000,    # Maximum number of iterations for convergence
            'random_state': 0,   # Set the random seed
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    # The `_get_default_auxiliary_params` method defines model-agnostic parameters such as maximum memory usage and valid input column dtypes.
    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=['int', 'float', 'category'],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
