"""
Validation utils
"""

import numpy as np
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split, ParameterSampler
# from autolrn.regression.timeseries.ts_utils import TS_split
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import joblib as jl
from sklearn.utils.validation import check_X_y, check_array, column_or_1d
from sklearn.metrics import mean_squared_error, mean_absolute_error

from scipy.stats import expon as sp_exp
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_unif
from scipy.stats import beta as sp_beta
from scipy.stats import reciprocal as sp_reproc


# https://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-optimization
class loguniform(sp_reproc):
    """A class supporting log-uniform random variables.

    Parameters
    ----------
    low : float
        The minimum value
    high : float
        The maximum value

    Methods
    -------
    rvs(self, size=None, random_state=None)
        Generate log-uniform random variables

    The most useful method for Scikit-learn usage is highlighted here.
    For a full list, see
    `scipy.stats.reciprocal
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.reciprocal.html>`_.
    This list includes all functions of ``scipy.stats`` continuous
    distributions such as ``pdf``.

    Notes
    -----
    This class generates values between ``low`` and ``high`` or

        low <= loguniform(low, high).rvs() <= high

    The logarithmic probability density function (PDF) is uniform. When
    ``x`` is a uniformly distributed random variable between 0 and 1, ``10**x``
    are random variales that are equally likely to be returned.

    This class is an alias to ``scipy.stats.reciprocal``, which uses the
    reciprocal distribution:
    https://en.wikipedia.org/wiki/Reciprocal_distribution

    Examples
    --------

    >>> from sklearn.utils.fixes import loguniform
    >>> rv = loguniform(1e-3, 1e1)
    >>> rvs = rv.rvs(random_state=42, size=1000)
    >>> rvs.min()  # doctest: +SKIP
    0.0010435856341129003
    >>> rvs.max()  # doctest: +SKIP
    9.97403052786026
    """


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true = column_or_1d(y_true)
    if isinstance(y_pred, Series):
        y_pred = y_pred.to_numpy()
    if isinstance(y_pred, DataFrame):
        y_pred = y_pred["y_pred"].to_numpy()

    epsilon = np.finfo(np.float64).eps
    mape = np.mean(np.abs(y_true - y_pred)/np.maximum(np.abs(y_true), epsilon))
    return mape * 100


def accuracy(y_true, y_pred, metric="rmse"):
    score = 0
    if metric == 'mae':
        score = mean_absolute_error(y_true, y_pred)
    elif metric == 'mape':
        score = mean_absolute_percentage_error(y_true, y_pred)
    elif metric == 'rmse':
        score = np.sqrt(mean_squared_error(y_true, y_pred))
    else:
        raise ValueError(
            f"'{metric}'' is not a valid metric, valid metrics are"
            " ['mae', 'mape', 'rmse']")
    return score


def _fit_and_score_candidates(
    candidate, X, y, n_iter=1, n_split=3, test_size=1, fh=1, param_setting=None,
    metric="rmse"):

    n_samples = X.shape[0]
    n_folds = n_split + 1

    # test_size = test_size if test_size is not None \
    #         else n_samples // n_folds
    # for how the method is built, you must set test_size == fh (forecast horizon)
    # or restrict the forecast horizon to the test size...
    if test_size < fh:
        test_size = fh
    elif test_size > fh:
        fh = test_size
    else:
        pass

    # Make sure we have enough samples for the given split parameters
    if n_folds > n_samples:
        raise ValueError(
            (f"Cannot have number of folds={n_folds} greater"
             f" than the number of samples={n_samples}."))
    if n_samples - (test_size * n_split) <= 0:
        raise ValueError(
            (f"Too many splits={n_split} for number of samples"
             f"={n_samples} with test_size={test_size}."))

    # check param_setting is not None and is a dict() w rsv_distros values
    if param_setting is not None and isinstance(param_setting, dict):
        candidate.set_params(**param_setting)
    
    cv_results = []

    ### you should turn this into a function
    print(f"iteration: {n_iter}")
    # for fold in range(n_split):
    #     print(f"\tfold: {fold + 1}")
    for fold in range(1, n_split + 1):
        print(f"\tfold: {fold}")
        train_size=int(X.shape[0] - (n_split + 1 - (fold + 1))*test_size)
        print("\ttrain_size=", train_size)
        X_train, y_train = X[: train_size], y[: train_size]

        # prediction intervals - error should increase linearly
        X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(
            X_train, y_train, shuffle=False, test_size=test_size, random_state=42)

        print("\tX_train_cv.shape=", X_train_cv.shape)
        print("\ty_train_cv.shape=", y_train_cv.shape)
        print("\ty_val_cv.shape=", y_val_cv.shape)

        # candidate_ = clone(candidate)
        candidate.fit(X_train_cv, y_train_cv)

        # X_val_cv is useless here in recursive forecasting
        # print("fh in '_fit_and_score_candidates'=", fh)  
        y_pred_cv = candidate.predict(fh)
        if isinstance(y_pred_cv, Series):
            y_pred_cv = y_pred_cv.to_numpy()
        if isinstance(y_pred_cv, DataFrame):
            y_pred_cv = y_pred_cv["y_pred"]

        score = accuracy(y_val_cv, np.array(y_pred_cv), metric)
        print(f"\t{metric} = {score:.3f}")
        cv_results.append(score)

        print()
        
    ps_score, ps_score_std = np.mean(cv_results), np.std(cv_results)
    print(
        f'\tCurrent cross validation {metric} score: {ps_score:.3f} ± {ps_score_std:.3f}')

    print()
    print()

    return ps_score, ps_score_std, param_setting


def score_models(y_test=None, forecasts=None, metric="rmse"):

    y_test = column_or_1d(y_test)
    scores = {}

    # if "y_test" in forecasts.columns:
    #     del forecasts["y_test"]

    for k, v in forecasts.items():
        if k == "y_pred":
            k = "target_y_pred"
        # scores[k] = mean_absolute_percentage_error(y_test, v)
        scores[k] = accuracy(y_test, v, metric)
    print()

    for k, v in scores.items():
        print(k, ": ", v)

    return scores


def check_back_transformer(t):
    # check t is instance of Pipeline, Transformer or numpy ufunc
    if t is not None:
        transformer = None
    else:
        raise AttributeError(
            "object should be a Pipeline, a Transformer or a numpy ufunc.")

    if isinstance(t, Pipeline):
        # ok, that's a Pipeline
        pass
    elif hasattr(t, "transform") or hasattr(t, "fit_transform"):
        # ok, you got a Transformer instance
        pass
    elif isinstance(t, np.ufunc):
        # ok, you have a universal function, make sure it's the right
        # inverse function :)
        t = FunctionTransformer(func=None, inverse_func=t)
    else:
        raise TypeError(
                "You should pass in a Pipeline or a Transformer instance"
                " or a numpy ufunc.")
    return t


second_half = sp_unif(0.5, 0.5)


tslas_param_distros = dict(
    alpha=loguniform(1e-3, 1e0)
    )


fasts_param_distros = dict(
    lasso_alpha=tslas_param_distros['alpha'],
    learning_rate=loguniform(1e-4, 1e0),
    n_estimators=sp_randint(10, 100),  
    subsample=second_half,
    max_features=[None, .75, .5, 'log2'],
    min_samples_leaf = sp_randint(1, 5),
    max_depth=sp_randint(3, 7),  
    )


class TSRandomizedSearchCV(BaseEstimator):
    
    def __init__(self, estimator=None, n_split=3, param_distro=None, 
        n_iter=10, fh=7, test_size=7, n_jobs=1, refit=True, metric="rmse"):
        self.estimator = estimator
        self.n_split = n_split
        self.param_distro = param_distro
        self.n_iter = n_iter
        self.fh = fh
        self.test_size = test_size
        self.n_jobs = n_jobs
        self.refit = refit
        self.metric = metric

        
    # pass train/test size here?
    def fit(self, X, y=None, **fit_kws):
        
        # # à post
        # estimator is gonna do this check inside '_fit_and_score_candidates'
        # X, y = check_X_y(X, y)
            
        # More check here!
        
        # n_iter = 15
        list_of_params = list(ParameterSampler(self.param_distro, self.n_iter))

        best_param_setting = dict()
        # the lower, the better
        self.best_score = 10**4
        # the lower, the better
        self.best_score_std = 10**4
        self.best_params_ = dict()


        cv_rscv = dict()
        
        base_estimator = clone(self.estimator)
        
        if int(X.shape[0] - self.n_split*self.test_size) <= 0:
            raise ValueError(
                f"Too many splits={self.n_split} for number of samples"
                 f"={X.shape[0]} with test_size={test_size}.")
            
        parallel = jl.Parallel(n_jobs=self.n_jobs,
                            # pre_dispatch=self.pre_dispatch
                            # prefer="threads"
                   )
        
        # parallelized rscv w recursive walk-forward valid.
        with parallel:
            out = parallel(
                jl.delayed(_fit_and_score_candidates)(
                    base_estimator, X, y, n_iter, self.n_split, 
                    self.test_size, self.fh, param_setting, self.metric)
                for n_iter, param_setting in enumerate(list_of_params, start=1)
            )

        # check results (out) then find best score/params
        # out should be of size 3*n_iter
        # print(out) 
        for element in out:
            if element[0] < self.best_score:
                self.best_score = element[0]
                self.best_score_std = element[1]
                best_param_setting = element[2]
        
        msg = f"Best validation {self.metric} score overall"
        print(
            f'{msg}: {self.best_score:.3f} ± {self.best_score_std:.3f}')
        
        if self.refit:
            # best_param_setting = param_setting
            self.best_params_['best_estimator'] = self.estimator.set_params(**best_param_setting)
            # scores on validation set
            self.best_params_[f'best_{self.metric}'] = self.best_score
            self.best_params_[f'best_{self.metric}_std'] = self.best_score_std
            self.best_params_["best_param_setting"] = best_param_setting
            print("Refitting best estimator...")
            self.best_params_['best_estimator'].fit(X, y)
            
        return self