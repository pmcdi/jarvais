import abc
from typing import Callable

from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn import metrics

_RocRegionResults = namedtuple("_RocRegionResults", ("lower_tpr", "lower_fpr", "upper_tpr", "upper_fpr"))
ValueWithCI = namedtuple("ValueWithCI", ("value", "lower", "upper"))

class ConfidenceParam(abc.ABC):
    """
    An object that holds parameters for confidence interval and confidence region calculations.
    Inherit from this class to define defaults for a particular plot or calculation.

    Parameters
    ----------
    conf : None | float | dict
        None means no confidence values will be plotted. A float will use default methods at the given confidence
        level.
        Entries of 'level' in a dict will be for the confidence level.
        Entries of 'region' and 'interval' are functions whose signatures vary based on child class usage. See the
        defaults for examples.

        Values for the level must be bounded by 0 and 1 (exclusive), where 0.95 represents a 95% confidence level.
    """

    # Method signatures for these depend on the nature of the calculation and may vary with implementation
    region: Callable
    interval: Callable
    level: float

    _default_region: Callable = None
    _default_interval: Callable = None
    _default_level: float = 0.95

    _region_allowed: bool = True
    _interval_allowed: bool = True

    @abc.abstractmethod
    def __init__(self, conf: None | float | dict) -> None:
        self.region = self._default_region
        self.interval = self._default_interval
        self.level = self._default_level
        self.apply_conf(_get_specific_conf_dict(conf))
        super().__init__()

    def apply_conf(self, conf: dict) -> None:
        """
        Takes a confidence dictionary and applies the relevant attributes.
        """
        if "level" in conf and _validate_level(conf["level"]):
            self.level = conf["level"]
        # Built this way for ease in future enhancements
        for func in ("region", "interval"):
            if func not in conf:
                continue
            if not getattr(self, f"_{func}_allowed"):
                raise ValueError(f"{type(self)} does not support the parameter '{func}'")
            setattr(self, func, conf[func])

def _get_specific_conf_dict(conf: None | float | dict) -> dict:
    """Given the flexible conf argument, parse it according to its actual type"""
    if conf is None:
        return {}
    if isinstance(conf, (float, int)):
        return {"level": conf}
    return conf

def _validate_level(level: float) -> bool:
    """Validate a float 'level' parameter"""
    if level <= 0 or level >= 1:
        raise ValueError("Specified confidence level must be between 0 and 1 (exclusive).")
    return True

def simultaneous_joint_confidence(conf: ConfidenceParam, y_true, y_proba):
    """
    Applies the
    `Macskassy-Provost <https://archive.nyu.edu/jspui/bitstream/2451/27802/2/CPP-07-04.pdf>`_ 'Simultaneous Joint Confidence Regions'
    method for finding a confidence region on the ROC curve.

    Parameters
    -----------
    conf: parameters.ConfidenceParam
        An instance of ConfidenceParam that holds the desired level.
    y_true:
        A 1xN array of groundtruth labels, where 0 is a negative observation and 1 is a positive observation.
    y_proba:
        A 1xN array of predictions, ranging from 0 to 1 inclusive. Second columns (i.e. [:, 1]) of the standard
        output of `sklearn's predict_proba function
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba>`_.

    Returns
    -------
    thresholds:
        A 1xk array of parameterized threshold values at which the listed TPR and FPR are achieved
    tpr:
        A 1xk array of true positive rates at the corresponding threshold
    fpr:
        A 1xk array of false positive rates at the corresponding threshold
    results:
        A namedtuple containing the following entries:

        - lower_tpr: A 1xk array of predicted lower bounds on the true positive rate at the corresponding threshold
          given the input confidence level.
        - lower_fpr: A 1xk array of predicted lower bounds on the false positive rate at the corresponding threshold
          given the input confidence level.
        - upper_tpr: A 1xk array of predicted upper bounds on the true positive rate at the corresponding threshold
          given the input confidence level.
        - upper_fpr: A 1xk array of predicted upper bounds on the false positive rate at the corresponding threshold
          given the input confidence level.
    """  # noqa
    alpha = 1 - conf.level
    # Given alpha, get the Kolmogorov-Smirnov critical distance
    c_alpha = np.sqrt(-(np.log(alpha / 2) / 2))

    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()
    if isinstance(y_proba, pd.Series):
        y_proba = y_proba.to_numpy()

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_proba)

    # Adjust c_alpha to the sample size. N for these is the number of cases, not total observations
    ks_truepos = np.nan
    ks_falsepos = np.nan
    if not y_true.sum() == 0:
        ks_truepos = c_alpha / np.sqrt(y_true.sum())
    if not y_true.sum() == y_true.shape[0]:
        ks_falsepos = c_alpha / np.sqrt(y_true.shape[0] - y_true.sum())

    # Shift and clip the values so they are in the range [0,1]
    upper_tpr = np.clip(tpr + ks_truepos, 0, 1)
    upper_fpr = np.clip(fpr - ks_falsepos, 0, 1)
    lower_tpr = np.clip(tpr - ks_truepos, 0, 1)
    lower_fpr = np.clip(fpr + ks_falsepos, 0, 1)
    return thresholds, tpr, fpr, _RocRegionResults(lower_tpr, lower_fpr, upper_tpr, upper_fpr)


def hanley_mcneill_confidence(conf: ConfidenceParam, y_true, y_proba) -> ValueWithCI:
    """
    Applies the `Hanley-McNeill <https://pubs.rsna.org/doi/pdf/10.1148/radiology.143.1.7063747>`_ method for finding a
    confidence interval on the AUROC.

    Parameters
    -----------
    conf: parameters.ConfidenceParam
        An instance of ConfidenceParam that holds the desired level.
    y_true:
        A 1xN array of groundtruth labels, where 0 is a negative observation and 1 is a positive observation.
    y_proba:
        A 1xN array of predictions, ranging from 0 to 1 inclusive. Second columns (i.e. [:, 1]) of the standard output
        of
        `sklearn's predict_proba function <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba>`_.

    Returns
    -------
    auc: float
        The area under the ROC curve
    lower: float
        A lower bound on the expected value of the AUROC at the given confidence level
    upper: float
        An upper bound on the expected value of the AUROC at the given confidence level
    """  # noqa
    z_score = norm.ppf(0.5 + conf.level / 2)
    auc = metrics.roc_auc_score(y_true, y_proba)
    # Get the number of positive and negative cases
    m = y_true.sum().astype("int64")
    n = y_true.shape[0] - m
    # Calculate the distributional parameters assuming a neg-exp distribution of scores
    P_xxy = auc / (2 - auc)
    P_xyy = 2 * auc**2 / (1 + auc)
    # Calculate the standard error given the distributional parameters
    first = auc * (1 - auc)
    second = (m - 1) * (P_xxy - auc**2)
    third = (n - 1) * (P_xyy - auc**2)
    sigma = np.sqrt((first + second + third) / (m * n))
    # Apply the z score and standard error to find the interval
    lower = max(auc - z_score * sigma, 0)
    upper = min(auc + z_score * sigma, 1)
    return ValueWithCI(auc, lower, upper)


def logit_interval(conf: ConfidenceParam, theta, n) -> ValueWithCI:
    """
    Applies the `Logit <https://en.wikipedia.org/wiki/Logit>`_ interval for a given set of observations of a binary
    random variable.

    Parameters
    -----------
    conf: parameters.ConfidenceParam
        An instance of ConfidenceParam that holds the desired level.
    theta: float or numpy.array
        The rate of positive cases in the sample. If a numpy array is passed, the calculation will be applied
        array-wise.
    n: float or numpy.array
        The number of observations in the sample. If a numpy array is passed, the calculation will be applied
        array-wise.

    Returns
    -------
    theta: float
        The area under the provided theta
    lower: float
        A lower bound on the expected value of theta at the given confidence level
    upper: float
        An upper bound on the expected value of theta at the given confidence level
    """
    alpha = 1 - conf.level
    # Get the variance of the statistic in log-space
    tau = 1 / np.sqrt(n * theta * (1 - theta))
    # Calculate the adjusted p value for the log-space distribution
    eta = np.log(theta / (1 - theta))
    # Convert the confidence level to a z score in log-space
    phi_tau = norm.ppf(1 - alpha / 2) * tau
    # Bound the log-space estimator and re-adjust it to linear space
    lower = np.exp(eta - phi_tau) / (1 + np.exp(eta - phi_tau))
    upper = np.exp(eta + phi_tau) / (1 + np.exp(eta + phi_tau))
    return ValueWithCI(theta, lower, upper)


def agresti_coull_interval(conf: ConfidenceParam, theta, n) -> ValueWithCI:
    """
    Applies the `Agresti-Coull <http://users.stat.ufl.edu/~aa/articles/agresti_coull_1998.pdf>`_ interval for a given
    set of observations of a binary random variable.

    Parameters
    -----------
    conf: parameters.ConfidenceParam
        An instance of ConfidenceParam that holds the desired level.
    theta: float or numpy.array
        The rate of positive cases in the sample. If a numpy array is passed, the calculation will be applied
        array-wise.
    n: float or numpy.array
        The number of observations in the sample. If a numpy array is passed, the calculation will be applied
        array-wise.

    Returns
    -------
    theta: float
        The area under the provided theta
    lower: float
        A lower bound on the expected value of theta at the given confidence level
    upper: float
        An upper bound on the expected value of theta at the given confidence level
    """
    z_score = norm.ppf(0.5 + conf.level / 2)
    # Shift the values of N and p to build a better estimator
    n_adjusted = n + z_score**2
    p_adjusted = (theta * n + z_score**2 / 2) / n_adjusted
    # Treat the adjusted values as standard inputs into a binomial bound estimate
    interval = z_score * np.sqrt(p_adjusted * (1 - p_adjusted) / n_adjusted)
    return ValueWithCI(p_adjusted, p_adjusted - interval, p_adjusted + interval)
