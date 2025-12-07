# src/baselines.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def time_split(series: pd.Series, test_size: int = 24):
    """
    Split a univariate series into train/test by last N points.
    """
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    return train, test

def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Mean Absolute Percentage Error (robust to scaling).
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    # Avoid division by zero
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0

def evaluate_forecast(test: pd.Series, pred: pd.Series) -> dict:
    """
    Return MAE, RMSE, MAPE as a dict.
    """
    mae = mean_absolute_error(test, pred)
    rmse = np.sqrt(np.mean((np.array(test) - np.array(pred))**2))
    mp = mape(test, pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mp}

def fit_sarima_forecast(train: pd.Series, test: pd.Series,
                        order=(1,1,1), seasonal_order=(0,1,1,12)):
    """
    Fit SARIMAX with given (p,d,q) and (P,D,Q,s). Default seasonality s=12 for monthly.
    """
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=len(test)).predicted_mean
    return res, pred

def fit_holt_winters_forecast(train: pd.Series, test: pd.Series,
                              trend="add", seasonal="add", seasonal_periods=12):
    """
    Fit Holt-Winters (Triple Exponential Smoothing) with additive components (robust for gold).
    """
    hw = ExponentialSmoothing(train, trend=trend, seasonal=seasonal,
                              seasonal_periods=seasonal_periods, initialization_method="estimated")
    res = hw.fit()
    pred = res.forecast(len(test))
    return res, pred

def plot_forecasts(train: pd.Series, test: pd.Series, preds: dict, title="Baselines forecast comparison"):
    """
    Plot train/test with multiple forecast series for comparison.
    preds: dict of {label: pd.Series}
    """
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train.values, label="Train", color="gray")
    plt.plot(test.index, test.values, label="Test (Actual)", color="black")
    for label, series in preds.items():
        plt.plot(series.index, series.values, label=label, linestyle="--")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
