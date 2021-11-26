import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
pd.options.plotting.backend = "plotly"
import datetime
from time import time
import warnings
warnings.filterwarnings('ignore')
from fasts import estimators as es
from fasts import valid as vd
from fasts import utils as us
from fasts.datasets import load_data

print()
print()

print("*** Forecasts of Medium Posts using FASTSRegressor")
print()
print()

posts = load_data("mediumposts.csv")

print(posts)
print()

n_jobs=-2
n_iter=25
n_lags=10
# forecasting horizon
fh=28 
test_size=28

X, y = us.embed_to_X_y(posts, n_lags=n_lags, name='posts')

print(X.tail(3))
print(y.tail(3))
print()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, shuffle=False)

fasts = es.FASTSRegressor()
fasts_params = vd.fasts_param_distros

print("=== TS Randomized Search CV...")

rscv = vd.TSRandomizedSearchCV(
    estimator=fasts, n_split=3, param_distro=fasts_params, fh=fh, 
    test_size=test_size, n_iter=n_iter, n_jobs=n_jobs
)

t0 = time()
rscv.fit(X_train, y_train)
t1 = time()

print(f"Full time of Search: {(t1 - t0):.2f} s")
print()

best_fasts = rscv.best_params_['best_estimator']

print("=== Compare best estimator to baselines on test data...")

_, spreds, tgt_preds = best_fasts.compare_to_baselines(
    y_train, y_test, season=7, fh=fh, bar_plot=True)

_, spreds, tgt_preds = best_fasts.compare_to_baselines(
    y_train, y_test, season=7, fh=fh, bar_plot=True, metric="mape")

print()
print("=== Plot train forecasts against true test data and simple preds...")

us.plot_forecasts(
    posts[800:-fh], 
    tgt_preds[tgt_preds > 0].fillna(0),
    posts.index[800:], 
    y_test=y_test,
    simple_preds=spreds)

print()
print("=== Fit best model upon full X, y (using best parameters from rdn search)...")

t0 = time()
best_fasts.fit(X, y)
t1 = time()

print(f"Time of fitting: {(t1 - t0):.2f} s")
print()
print("=== Compare best estimator baselines...")

_, spreds_full, tgt_fcs_full = best_fasts.compare_to_baselines(
    y, season=7, fh=fh, bar_plot=True)

best_fasts.compare_to_baselines(
    y, season=7, fh=fh, bar_plot=True, metric="mape");

print()

full_series_index = pd.date_range(start=posts.index[0], periods=(len(posts) + fh))
print(full_series_index)

print()
print("=== Plot forecasts against simple preds...")

# it would be useful to write a zoom util to avoid slicing...
us.plot_forecasts(
    posts[650:], 
    tgt_fcs_full[tgt_fcs_full > 0].fillna(0),
    full_series_index[650:], 
    simple_preds=spreds_full
)

