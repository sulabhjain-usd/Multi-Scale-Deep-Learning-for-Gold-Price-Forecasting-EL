# src/regimes_vol.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rolling_stats(series: pd.Series, windows=(6, 12, 24)):
    """
    Compute rolling mean and std over given window sizes (in periods of the input frequency).
    For monthly data: 12 ≈ 1-year, 24 ≈ 2-year.
    """
    df = pd.DataFrame({"value": series})
    for w in windows:
        df[f"roll_mean_{w}"] = series.rolling(w).mean()
        df[f"roll_std_{w}"] = series.rolling(w).std()
    return df

def realized_vol(series: pd.Series, window: int = 12):
    """
    Realized volatility using log returns and rolling std.
    """
    log_ret = np.log(series).diff()
    vol = log_ret.rolling(window).std() * np.sqrt(window)  # annualize-ish for monthly
    return vol

def plot_regime_vol(series: pd.Series, windows=(12, 24), title="Regime & Volatility overview"):
    """
    Visualize rolling mean/std and realized volatility as regime/volatility proxies.
    """
    df = rolling_stats(series, windows=windows)
    vol = realized_vol(series, window=12)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(series.index, series.values, label="Close", color="black")
    axes[0].set_title("Price (level)")

    for w in windows:
        axes[1].plot(df.index, df[f"roll_mean_{w}"], label=f"Rolling mean {w}")
    axes[1].legend()
    axes[1].set_title("Rolling means (regime proxy)")

    for w in windows:
        axes[2].plot(df.index, df[f"roll_std_{w}"], label=f"Rolling std {w}", alpha=0.7)
    axes[2].plot(vol.index, vol.values, label="Realized vol (12m log-ret std)", color="red")
    axes[2].legend()
    axes[2].set_title("Rolling std and realized volatility")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def zscore_regimes(series: pd.Series, window: int = 24, threshold: float = 1.0):
    """
    Rolling z-score as a simple regime change indicator.
    Flags points where z-score magnitude exceeds threshold.
    """
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()
    z = (series - roll_mean) / roll_std
    regimes = pd.Series(np.where(np.abs(z) > threshold, 1, 0), index=series.index)  # 1 = extreme regime
    return pd.DataFrame({"value": series, "zscore": z, "regime_flag": regimes})
