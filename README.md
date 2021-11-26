# fasts
 Fast automatic time series forecasting

 **fasts** is a lightweight toy-tool to forecast future values of time series with a trend.

## How to use fasts

 You can define a `FASTSRegressor` instance, train and compare it to baselines on test data with automatically computed scores and even on a bar-plot and then make forecasts in a recursive fashion, provided a forecast horizon is specified.

 You can also perform a quantile regression and display **prediction intervals** on a plot.

``` python
from fasts import estimators as es
from fasts import valid as vd
from fasts import utils as us
from fasts.datasets import load_data

fh=test_size=28

posts = load_data("passengers.csv")

X, y = us.embed_to_X_y(pd.Series(posts), n_lags=21, name='posts')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, shuffle=False)

fasts = es.FASTSRegressor(quantile_reg=True)

fasts.fit(X_train, y_train)

tgt_preds = fasts.predict(fh, pred_int=True)

# display an actual vs. forecast plot w zoom in 
us.plot_forecasts(
    passengers[:-fh],
    tgt_preds[tgt_preds > 0].fillna(0),
    passengers.index, 
    y_test=passengers[-fh:],
    return_pred_int=True)
```

 You can do **hyperparameter optimization** by running a `TSRandomizedSearchCV` instance, which can then refit the best model on the train data you provided.

``` python
fasts_params = vd.fasts_param_distros

rscv = vd.TSRandomizedSearchCV(
    estimator=fasts, n_split=3, param_distro=fasts_params, fh=fh, 
    test_size=test_size, n_iter=15, n_jobs=-1
)

rscv.fit(X_train, y_train)

best_fasts = rscv.best_params_['best_estimator']

# Compare best estimator to baselines on test data...
scores, spreds, tgt_preds = best_fasts.compare_to_baselines(
    y_train, y_test, season=12, fh=fh, bar_plot=True
)

us.plot_forecasts(
    passengers[:-fh],
    tgt_preds[tgt_preds > 0].fillna(0),
    passengers.index, 
    y_test=passengers[-fh:],
    simple_preds=spreds,
    return_pred_int=True)
```

 The idea behind **fasts** is to avoid the hassle of analyzing ACF and PACF plots and detrending data. This doesn't mean you should skip a thorough time series analysis, which can help yu craft better models though. 

 I tried to keep a certain level of compatibility with `scikit-learn` to make `fasts` more appetible to users familiar with that library, but full compatibility requires further development on my part.

 ## DISCLAIMER

 As a toy-project, fasts **has not been tested following a [TDD](https://en.wikipedia.org/wiki/Test-driven_development) approach**, so it's not guaranteed to be stable and is **not aimed to nor ready for production**. Please use tested and maintained packages for reaserch and business.
