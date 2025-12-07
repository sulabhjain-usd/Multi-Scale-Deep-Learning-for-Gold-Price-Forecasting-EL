import pandas as pd
import numpy as np

def add_returns_features(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")

    # Percentage return
    df["pct_return"] = df[price_col].pct_change()

    # Log return
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))

    # Rolling statistics
    df["rolling_mean_5"] = df["pct_return"].rolling(window=5).mean()
    df["rolling_std_5"] = df["pct_return"].rolling(window=5).std()

    return df
