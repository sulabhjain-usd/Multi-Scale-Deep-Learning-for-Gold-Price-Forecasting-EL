# src/diagnostics.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

def acf_pacf_plots(series: pd.Series, max_lag: int = 48, title_prefix: str = ""):
    """
    Plot ACF and PACF for a given series, with optional title prefix.
    Use returns (diff/log returns) when series is non-stationary.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), lags=max_lag, ax=axes[0])
    plot_pacf(series.dropna(), lags=max_lag, ax=axes[1], method="ywm")
    axes[0].set_title(f"{title_prefix} ACF")
    axes[1].set_title(f"{title_prefix} PACF")
    plt.tight_layout()
    plt.show()

def stl_decompose(series: pd.Series, period: int, title: str = ""):
    """
    STL decomposition for trend/seasonality visualization.
    Period for monthly is typically 12 (yearly seasonality).
    """
    result = STL(series.dropna(), period=period, robust=True).fit()
    fig = result.plot()
    fig.set_size_inches(10, 6)
    fig.suptitle(title or "STL decomposition", fontsize=12)
    plt.tight_layout()
    plt.show()
    return result

def prepare_stationary(series: pd.Series, use_log: bool = True, diff_order: int = 1) -> pd.Series:
    """
    Create a more stationary series by (optional) log-transform and differencing.
    """
    s = series.copy()
    if use_log:
        s = np.log(s)
    for _ in range(diff_order):
        s = s.diff()
    return s
