"""
utils module
"""

import numpy as np
from pandas import Series, DataFrame, concat
from .. valid import mean_absolute_percentage_error
from sklearn.utils.validation import check_is_fitted
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def series_to_supervised(data, n_lags=1, n_var=1, dropnan=True):
    # check data is a pandas DataFrame w index and numeric series
    
    cols = list()
    # input sequence (t-1, ... t-n_lags)
    for i in range(n_lags, 0, -1):
        cols.append(data.shift(-i))
    # forecast sequence (t, t+1, ... t+n_var)
    for i in range(0, n_var):
        cols.append(data.shift(i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


def embed_to_X_y(data, n_lags=1, n_var=1, dropnan=True, name='ts'):
    
    emb_data = series_to_supervised(data, n_lags, n_var, dropnan)
    
    lags = [f'lag_{l}' for l in range(1, emb_data.shape[1])]
    
    X = DataFrame(emb_data[:,1:], columns=lags) 
    y = Series(emb_data[:,0], name=name)
    
    return (X, y)


def plot_score_bar(scores: dict, metric="rmse"):
    plt.figure(figsize=(12, 6))
    clr = 'red' if scores["target_y_pred"] > np.min(np.stack(scores.values())) else 'green'
    plt.bar("target_y_pred", scores["target_y_pred"], width=.8, color=clr)
    scores_min = np.min(np.stack(scores.values()))
    scores.pop("target_y_pred")
    plt.bar(scores.keys(), scores.values(), width=.8, color="orange")
    plt.axhline(y = scores_min, color ="black", linestyle ="--")
    plt.title(f"Score ({metric.upper()})", fontsize=14)
    plt.show()


def plot_forecasts(
    series, forecasts_df, 
    series_fcs_index=None,
    train_label='orig_scale_posts', forec_label='forecasts',
    y_test=None,
    return_pred_int=False,
    simple_preds=None,
    title='Actual vs Forecast plot',
    y_label='values'):

    """
    series_fcs_index: index of a full series-plus-forecasts series
    """

    # Point forecast
    forecasts = forecasts_df['y_pred']

    # check series, forecasts_df, ...

    if series_fcs_index is None:
        series_fcs_index = np.arange(len(series) + forecasts_df.shape[0])
    
    full_time_series_df = DataFrame(index=series_fcs_index)
    full_time_series_df[train_label] = np.concatenate(
        (series.values, np.full(len(forecasts), np.nan)))
    full_time_series_df[forec_label] = np.concatenate(
        (np.full(len(series), np.nan), forecasts))
    
    plt.figure(figsize=(15, 6))
    plt.plot(full_time_series_df[forec_label], 
        color="coral", label=forec_label, linewidth=3)
    plt.plot(full_time_series_df[train_label], 
        color="seagreen", label='train')
    # draw vertical line to split train data from forecasts
    plt.axvline(x=series_fcs_index[-len(forecasts)-1], 
        color="black", linestyle="--")
    
    # royalblue
    if y_test is not None and isinstance(y_test , Series):
        full_time_series_df['y_test'] = np.concatenate(
            (np.full(len(series), np.nan), y_test.values))
        plt.plot(
            full_time_series_df['y_test'], 
            color="cornflowerblue", label='test', linewidth=3)

    # 'lightseagreen'
    colors = ['gray', 'orchid', 'purple', 'goldenrod']

    if simple_preds is not None and isinstance(simple_preds, DataFrame):
        for i, (k, v) in enumerate(simple_preds.items()):
            full_time_series_df[k] = np.concatenate(
                (np.full(len(series), np.nan), v))
            plt.plot(full_time_series_df[k], colors[i], label=k)

    if return_pred_int:
        low_80 = forecasts_df['low_80'] 
        up_80 = forecasts_df['up_80'] 
        low_95 = forecasts_df['low_95'] 
        up_95 = forecasts_df['up_95']
        full_time_series_df['low_80'] = np.concatenate(
            (np.full(len(series), np.nan), low_80))
        full_time_series_df['up_80'] = np.concatenate(
            (np.full(len(series), np.nan), up_80))
        full_time_series_df['low_95'] = np.concatenate(
            (np.full(len(series), np.nan), low_95))
        full_time_series_df['up_95'] = np.concatenate(
            (np.full(len(series), np.nan), up_95))

        print()
        print("full_time_series_df")
        print(full_time_series_df.tail(5))

        plt.fill_between(
            full_time_series_df.index, 
            full_time_series_df['low_95'].values, 
            full_time_series_df['up_95'].values,
            alpha=0.2, 
            label="95% prediction intervals")
        plt.fill_between(
            full_time_series_df.index, 
            full_time_series_df['low_80'].values, 
            full_time_series_df['up_80'].values, 
            alpha=0.2,
            label="80% prediction intervals")
    plt.title(title, fontsize=14)
    plt.ylabel(y_label)
    plt.legend()    
    plt.tight_layout()
    plt.show()