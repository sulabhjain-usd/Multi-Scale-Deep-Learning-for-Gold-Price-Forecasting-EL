import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
def fit_models(train: pd.DataFrame, target_col: str = "close", order=(1,1,1)):
    """
    Fit a simple ARIMA model on the training set.
    Returns the fitted model.
    """
    series = train[target_col].dropna()
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit


def forecast_models(model_fit, steps: int = 10):
    """
    Forecast future values using the fitted ARIMA model.
    """
    forecast = model_fit.forecast(steps=steps)
    return forecast



def eval_metrics(actual, forecast):
    """
    Evaluate forecast accuracy with MAE and RMSE.
    """
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)   # always returns MSE
    rmse = np.sqrt(mse)                          # take square root manually
    return {"MAE": mae, "RMSE": rmse}



def plot_fcst_vs_actual(actual: pd.Series, forecast: pd.Series, title: str = "Forecast vs Actual"):
    """
    Plot forecasted values against actual values.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual.values, label="Actual")
    plt.plot(forecast.index, forecast.values, label="Forecast", linestyle="--")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
