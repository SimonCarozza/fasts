import numpy as np
from pandas import DataFrame
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, column_or_1d, check_is_fitted
from .. valid import score_models, check_back_transformer, mean_absolute_percentage_error
import re
import copy
from .. utils import plot_score_bar

from sklearnex import patch_sklearn
patch_sklearn()


# make results reproducible
seed = 12345  # 0, 42, 137, 


class BaseRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, 
                 lasso_alpha = 1.0,
                 loss='huber', 
                 learning_rate=0.1,
                 n_estimators=20,
                 criterion='friedman_mse', 
                 max_features='auto',
                 subsample=.3, 
                 min_samples_leaf=1, 
                 min_samples_split=2,
                 max_depth=3,
                 # gb_alpha = 0.95,
                 quantile_reg = False
                ):
        # self.ens = ens
        
        # LASSO's params
        self.lasso_alpha = lasso_alpha
        
        # let's fix width of prediction intervals
        # self.gb_alpha = gb_alpha
        self.loss = loss  # 'ls',
        self.learning_rate = learning_rate  # 0.1 
        self.criterion = criterion  # 'mse',
        self.max_depth = max_depth  # None,
        self.max_features = max_features  # 'auto',
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf  # 1,
        self.min_samples_split = min_samples_split  # 2,
        self.n_estimators = n_estimators  # 100,
        self.quantile_reg = quantile_reg
        
        # REForestRegressor params (LASSO + FOREST)
        
    def fit(self, X, y):
        self.X_cols_ = X.columns
        X, y = check_X_y(X, y)
        
        self.md_params = dict()

        self.md_params["lasso"]=dict()

        lasso = Lasso(alpha=self.lasso_alpha, random_state=seed)
        # set params of sub-estimators (LASSO)
        self.lasso_ = lasso.fit(X, y)
        self.n_cols_las = X.shape[1]
        print("X cols", self.n_cols_las)
    
        # you should take features w non-zero coeffs 
        # abs_importances = np.abs(sfm_est.coef_[0])
        # importances = 100*(abs_importances/np.max(abs_importances))
        lasso_coefs = self.lasso_.coef_
        #### maybe you should review this
        # you're making a boolean mask off coefficients
        self.indices_ = np.abs(lasso_coefs) > 0
        self.X_sel_ = X[:, self.indices_]
        self.n_cols_for = self.X_sel_.shape[1]
        if self.n_cols_for == 0:
            raise ValueError("Found array with 0 feature(s).")
        else:
            print("X_sel_ cols", self.n_cols_for)
    
        self.lasso_residuals_ = y - self.lasso_.predict(X)
        self.md_params["lasso"]["model"]=self.lasso_
        self.md_params["lasso"]["residuals"]=self.lasso_residuals_
        self.md_params["lasso"]["mean_residuals"]=np.mean(self.lasso_residuals_)

        self.y_ = y

        # you're fitting REGradientBoostingRegressor, not GradientBoostingRegressor
        gbr = GradientBoostingRegressor(
            loss = self.loss,  # 'ls',
            learning_rate = self.learning_rate,
            criterion = self.criterion,  # 'mse',
            max_depth = self.max_depth,  # None,
            max_features = self.max_features,  # 'auto',
            subsample = self.subsample,
            min_samples_leaf = self.min_samples_leaf,  # 1,
            min_samples_split = self.min_samples_split,  # 2,
            n_estimators = self.n_estimators,  # 100,
            # alpha = self.gb_alpha,
            warm_start=True,
            random_state=seed)

        trees = ["grad_boost"]

        self.md_params[trees[0]]=dict()
        self.md_params[trees[0]]["model"] = gbr
        

        # this is a bit of a stretch
        # case loss = 'quantile', quantile_reg can be False
        # quantile_reg = True, loss must be = 'quantile' for upper_95, else 'ls'
        if self.quantile_reg:
            trees.append("grad_boost_qtl")
            self.md_params[trees[1]]=dict()
            if self.loss != 'quantile':
                # gb_alpha is useless in this case
                gbr.set_params(loss='ls')
                # you'll remember that you're doing quantile reg
                self.loss = 'quantile'
            gbr_q95 = clone(gbr)
            gbr_q95.set_params(loss=self.loss, alpha=0.95)
            self.md_params[trees[1]]["model"] = gbr_q95
            print("with quantile regression.")
        
        for gb in trees:
            self.md_params[gb]["model"] = self.md_params[gb]["model"].fit(
                self.X_sel_, self.lasso_residuals_)
            # adjusting for bias in prediction
            # gbr has been trained upon lasso_residuals_
            gbr_residuals = self.lasso_residuals_ - self.md_params[gb]["model"].predict(self.X_sel_)
            # bias (if abs(mean) > 0)
            # https://otexts.com/fpp3/diagnostics.html#diagnostics
            # self.mean_gbr_residuals_ = np.mean(gbr_residuals)
            self.md_params[gb]["mean_residuals"]=np.mean(gbr_residuals)

        print("self.gbr_fitted_.n_features =", 
            self.md_params["grad_boost"]["model"].n_features_)
        
        return self
    
    # compatible w sklearn's API;
    # skipping prediction intervals for standard regression
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        # useless?
        check_is_fitted(self.md_params["grad_boost"]["model"])
        
        self.lasso_preds_ = self.md_params["lasso"]["model"].predict(X)
    
        self.gbr_preds_ = self.md_params["grad_boost"]["model"].predict(self.X_sel_)
        self.predictions_ = self.gbr_preds_ + self.lasso_preds_
        
        return self.predictions_


def mean_model(y_train=None, fh=1):
    y_train = column_or_1d(y_train)
    y_pred_mean = np.mean(y_train)
    return np.ones(fh, dtype = int)*y_pred_mean

def naive_model(y_train=None, fh=1):
    y_train = column_or_1d(y_train)
    naive_preds = np.ones(fh, dtype = int)*y_train[-1]
    return naive_preds

def seasonal_naive(y_train=None, season=1, fh=1):
    # ideally, season could also be a string in
    # ['hourly', 'daily'. 'weekly', 'monthly', 'yearly']
    # season == 1 --> simple naive
    y_train = column_or_1d(y_train)
    if season <= len(y_train):
        snaive_preds = y_train[-season:]
        if season <= fh:
            while len(snaive_preds) != fh:
                snaive_preds = np.append(snaive_preds, snaive_preds)[:fh]
        else:
            # snaive_preds = snaive_preds[-fh:]
            snaive_preds = snaive_preds[:fh]
    else:
        raise ValueError(
            "season's value can't be greater than series' length.")
    # additional condition(s)
    return snaive_preds

def drift(y_train=None, fh=1):
    y_train = column_or_1d(y_train)
    drift_preds = np.array([])
    for i in range(fh):
        y_new = round(y_train[-1] 
            + i*((y_train[-1] - y_train[0])/(len(y_train)-1)), 2)
        # print(y_new)
        drift_preds = np.append(drift_preds, y_new)
    return drift_preds


def back_transform_forecasts(
    bck_transformer, forecasts:DataFrame) -> DataFrame:

    t = check_back_transformer(bck_transformer)
    forecasts = forecasts.apply(
        lambda x: t.inverse_transform(
            x.values.reshape(-1, 1)).ravel())
    # basic checks
    # if not "y_pred" in target_forecasts.columns:
    if not "y_pred" in forecasts.columns:
        raise AttributeError(
            "'target_forecasts' should have an 'y_pred' column")

    return forecasts


class ForecastMixin:

    def forecasts_w_pred_ints(self, y_pred):

        if not isinstance(self, (FASTSRegressor, TSLasso)):
            raise TypeError(
                "valid types for target_estimator are "
                "['FASTSRegressor', 'TSLasso']")
        
        check_is_fitted(self)
        
        if not self.pred_int_:
            raise ValueError(
                "Forecasts were made without calculating prediction intervals")
    
        # check y_pred is series/ np array, std_err != 0

        if isinstance(self, FASTSRegressor):
            std_err = self.regbr_stderr_
        else:
            std_err = self.stderr_

        forecasts_df = DataFrame()
        forecasts_df['y_pred'] = y_pred
        forecasts_df['low_80'] = y_pred - 1.28*std_err
        forecasts_df['up_80'] = y_pred + 1.28*std_err
        forecasts_df['low_95'] = y_pred - 1.96*std_err
        forecasts_df['up_95'] = y_pred + 1.96*std_err

        return forecasts_df

    def compare_to_baselines(
            self, y_train, y_test=None, season=None, fh=1, bar_plot=False,
            bck_transformer=None, metric="rmse", prune=False):
        """
        next: customize score
        """

        check_is_fitted(self)

        forecasts = DataFrame()

        print()
        if not isinstance(self, (FASTSRegressor, TSLasso)):
            raise TypeError(
                "valid types for target_estimator are "
                "['FASTSRegressor', 'TSLasso']")
        elif isinstance(self, FASTSRegressor):

            if self.quantile_reg:
                print(
                    "FASTSRegressor with quantile regression performed.")
                forecasts = self.predict(fh, pred_int=True, prune=prune)
            else:
                print("FASTSRegressor here.")
                forecasts = self.predict(fh)
        elif isinstance(self, TSLasso):
            print("TSLasso here.")
            forecasts = self.predict(fh, pred_int=True, prune=prune)
        else:
            pass

        baselines = ['mean', 'naive', 'drift']

        forecasts['mean'] = mean_model(y_train, fh)
        forecasts['naive'] = naive_model(y_train, fh)
        forecasts['drift'] = drift(y_train, fh)
        if season is not None:
            forecasts['snaive'] = seasonal_naive(y_train, season, fh)
            baselines.append('snaive')

        # back_transform forecasts ...
        # https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/pipeline.py#L51
        if bck_transformer is not None:
            # # directly pass original test series slice
            if y_test is not None:
                forecasts["y_test"] = column_or_1d(y_test)
            forecasts = back_transform_forecasts(
                bck_transformer, forecasts)

        scores = None
        if y_test is not None:
            if "y_test" in forecasts.columns:
                y_test = forecasts["y_test"].to_numpy()
                # print("y_test (after back_transform_forecasts)", y_test)
                del forecasts["y_test"]

            candidates = ["y_pred"] + baselines
            # print(candidates)
            scores = score_models(y_test, forecasts[candidates], metric)
            if bar_plot:
                plot_score_bar(scores, metric)

        simple_forecasts = forecasts[baselines]
        target_forecasts = forecasts.drop(baselines, axis=1)

        return scores, simple_forecasts, target_forecasts

    # may throw "Found input variables with inconsistent numbers of samples:
    # (e.g.) [879, 7]"
    # tweaked .score() to make class compatible w sklearn
    def score(self, X, y, fh=0, metric="rmse", pred_int=False):
        
        # basically, here X is useless, but you could use it to check
        # y's size
        
        check_is_fitted(self)
        
         # fh > 1 to make time series forecasts
        if fh == 0:
            raise ValueError("Forecast horizon should be at least of length == 1")
        
        if len(y) != fh:
            raise AttributeError("test set must be of same length of observed set")
        
        if not pred_int:
            y_pred = self.predict(fh)
        else:
            y_pred = self.predict(fh, pred_int)['y_pred']
        
        score = vd.accuracy(y, y_pred, metric)
        
        return score


def bootstrap_residuals(
    y_pred, y_residuals, fh=1, n_series=100, prune=False) -> DataFrame:

    # make sure in np random choice the random assignment will be 
    # the same every time
    rng = np.random.default_rng(seed)

    y_pred = column_or_1d(y_pred)
    y_residuals = column_or_1d(y_residuals)

    bootstr_ts_df = DataFrame()
    bootstr_ts_df['y_pred'] = y_pred
    
    for i in range(n_series):
        bootstr_res = rng.choice(y_residuals, fh)
        bootstr_ts_df[f"ts_{i+1}"] = np.add(np.array(y_pred), bootstr_res)

    if prune:
        bootstr_ts_df_pruned = bootstr_ts_df.copy(deep=True)
        for i in range(fh):
            bootstr_ts_df_pruned = bootstr_ts_df_pruned.loc[
            :,~(bootstr_ts_df_pruned == bootstr_ts_df.T.max()[i]).any()]
            bootstr_ts_df_pruned = bootstr_ts_df_pruned.loc[
            :,~(bootstr_ts_df_pruned == bootstr_ts_df.T.min()[i]).any()]

    return bootstr_ts_df if not prune else bootstr_ts_df_pruned


from typing import List

def _forecast_series(estimator, window:List[float], fh:int) -> list:
    y_pred = list()

    if estimator is None:
        raise AttributeError("'estimator' cannot be of NoneType")
    elif not isinstance(estimator, (GradientBoostingRegressor, Lasso)):
        raise TypeError(
            "valid types for 'estimator' are "
            "['GradientBoostingRegressor', 'Lasso']")

    if not window:
        raise ValueError("window should be a list at least one element")

    for i in range(fh):
        y_pred.append(estimator.predict([window])[0])
        window = window[:-1]
        window = [y_pred[-1]] + window

    return y_pred


class FASTSRegressor(BaseRegressor, ForecastMixin):

    def __init__(self, 
                 lasso_alpha = 1.0,
                 loss='huber', 
                 learning_rate = 0.1,
                 n_estimators=20,
                 criterion='friedman_mse', 
                 max_features='auto',
                 subsample=.3, 
                 min_samples_leaf=1, 
                 min_samples_split=2,
                 max_depth=3,
                 # gb_alpha = 0.95,
                 quantile_reg = False
                ):
        super().__init__(
            lasso_alpha=lasso_alpha,
            loss=loss, 
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            criterion=criterion, 
            max_features=max_features,
            subsample=subsample, 
            min_samples_leaf=min_samples_leaf, 
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            # gb_alpha=gb_alpha, 
            quantile_reg=quantile_reg)
    
    def predict(self, fh=0, pred_int=False, prune=False):
        
        # md_params = dict()
        md_params = self.md_params.copy()
        
        check_is_fitted(self)
        check_is_fitted(md_params["grad_boost"]["model"])
        
        # fh > 1 to make time series forecasts
        if fh == 0:
            raise ValueError("Forecast horizon should be at least of length == 1")

        print()
        print("=== Trying to make forecasts w FASTSRegressor...")
        print()
            
        self.refst_pred_ = None
        
        # assuming you have 1-d time series -- you could use column_or_1d
        y_ = column_or_1d(self.y_)
        y_pred = []
        
        print()
        y_pred_res = []
        print()
        print("test windows:")
        last_window = y_[-self.n_cols_las:][::-1].tolist()
        print(last_window)
        print("length of first Lasso test window:", len(last_window))
        # y_ --> self.lasso_residuals_
        # last_window_res = self.lasso_residuals_[-self.n_cols_for:][::-1].tolist()
        last_window_res = md_params["lasso"]["residuals"][-self.n_cols_for:][::-1].tolist()
        print(last_window_res)
        print("length of first GradBoost test window (residuals):", 
            len(last_window_res))

        md_params["lasso"]["y_pred"] = y_pred
        md_params["lasso"]["window"] = last_window
        md_params["grad_boost"]["y_pred"] = y_pred_res
        md_params["grad_boost"]["window"] = last_window_res
        # print(last_window)
        # print()

        # print("GB's window length:", len(last_window_res))

        self.pred_int_ = False
        if pred_int == True:
            if not self.quantile_reg:
                raise ValueError("You should have performed quantile regression "
                                 "to calculate prediction intervals.")
            else:
                print()
                print("... gonna predict quantiles as well.")
                print()

                self.pred_int_ = pred_int
                y_pred_gbq_95 = []
                # in any case
                last_window_gbq_95 = list(md_params["grad_boost"]["window"])
                print(
                    f"length of first vanilla GradBoost test window before quant. reg:", 
                    len(last_window))
                print(last_window_gbq_95)
                print(
                    f"length of first GradBoost quantile test window (0.95):", 
                    len(last_window_gbq_95))

                md_params["grad_boost_qtl"]["window"] = last_window_gbq_95
                md_params["grad_boost_qtl"]["y_pred"] = y_pred_gbq_95
        else:
            if self.quantile_reg and "grad_boost_qtl" in md_params:
                del md_params["grad_boost_qtl"] 

        # little check
        # !? this can be True and work anyway!?
        if md_params["grad_boost"]["model"].n_features_ != self.n_cols_for:
            print("self.n_cols_for =", self.n_cols_for)
            print("self.gbr_fitted_.n_features =", 
                md_params["grad_boost"]["model"].n_features_)
            raise ValueError(
                "ValueError: Number of features of the model must match the input.")

        print()
        for k, v in md_params.items():
            print()
            print(f"You're here: '{k}' loop...")
            # print(md_params[k]["window"])
            # print()
            md_params[k]["y_pred"] = _forecast_series(
                md_params[k]["model"], md_params[k]["window"], fh)
        print()

        # adjusting for bias
        y_pred = np.array(
            md_params["lasso"]["y_pred"]) - md_params["lasso"]["mean_residuals"]
        y_pred_res = np.array(
            md_params["grad_boost"]["y_pred"]) - md_params["grad_boost"]["mean_residuals"]

        self.regbr_stderr_ = 0
        if pred_int == True:

            y_pred_gbq_95 = np.array(
                md_params["grad_boost_qtl"]["y_pred"]) - md_params["grad_boost_qtl"]["mean_residuals"]
            
            bootstr_ts_df = bootstrap_residuals(
                    y_pred, md_params["lasso"]["residuals"], fh, prune=prune)
            
            las_var = bootstr_ts_df.T.var()
            std_gbr = np.subtract(y_pred_gbq_95, y_pred_res)/1.96
            gbr_var = np.power(std_gbr, 2)
            self.regbr_stderr_ = np.sqrt((las_var + gbr_var)/2*fh)
            self.regbr_pred_ = self.forecasts_w_pred_ints(y_pred)
        else:
            forecasts_df = DataFrame()
            forecasts_df['y_pred'] = np.add(y_pred, y_pred_res)
            self.regbr_pred_ = forecasts_df
        
        return self.regbr_pred_  


# use it for comparison purpose
class TSLasso(BaseEstimator, ForecastMixin):

    def __init__(self, 
        # subestimator=Lasso(random_state=seed), 
        alpha=1.0):
        # self.subestimator=subestimator
        self.subestimator=Lasso(random_state=seed)
        self.alpha=alpha

    # def get_params(self, deep=True):
    #     return {"alpha": self.alpha}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    def fit(self, X, y):
        # if not isinstance(self.subestimator, Lasso):
        #     raise TypeError("Valid type for subestimator is Lasso.")
        self.X_, self.y_ = check_X_y(X, y)
        # set params of sub-estimators (LASSO)
        lasso = self.subestimator.set_params(alpha=self.alpha)
        self.lasso_ = lasso.fit(X, y)
        self.lasso_residuals_ = y - self.lasso_.predict(X)
        self.mean_lasso_residuals_ = np.mean(self.lasso_residuals_)
        return self
    
    def predict(self, fh=0, pred_int=False, prune=False):
        
        check_is_fitted(self)
        
        # fh > 1 to make time series forecasts
        if fh == 0:
            raise ValueError(
                "Forecast horizon should be at least of length == 1")
        
        # assuming you have 1-d time series
        y_ = column_or_1d(self.y_)
        
        self.n_cols = len(self.lasso_.coef_)
        last_window = y_[-self.n_cols:][::-1].tolist()
        
        self.pred_int_ = False
        
        y_pred = _forecast_series(self.lasso_, last_window, fh)
        # adjusting for bias
        y_pred = np.array(y_pred) - self.mean_lasso_residuals_
            
        # using boostrapped residuals (which should be uncorrelated) 
        # to calculate Error STD
        self.stderr_ = 0
        if pred_int == True:
            self.pred_int_ = pred_int
            bootstr_ts_df = bootstrap_residuals(
                y_pred, self.lasso_residuals_, fh, prune=prune)
            
            self.stderr_ = bootstr_ts_df.T.std()
            self.pred_ = self.forecasts_w_pred_ints(y_pred)
        else:
            forecasts_df = DataFrame()
            forecasts_df['y_pred'] = y_pred
            self.pred_ = forecasts_df
        
        return self.pred_ 
