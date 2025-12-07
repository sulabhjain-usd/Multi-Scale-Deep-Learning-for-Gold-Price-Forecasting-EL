# src/io_monthly.py
import pandas as pd

def load_monthly_gold_csv(path: str) -> pd.DataFrame:
    """
    Load monthly gold CSV with semicolon separators and parse the date column.
    Ensures a proper datetime index and numeric columns.
    """
    df = pd.read_csv(
        path,
        sep=";",
        parse_dates=["Date"],
        dayfirst=False,  # dates are YYYY.MM.DD
    )

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]
    # Set datetime index
    df = df.set_index("date").sort_index()

    # Coerce numerics safely (handles mixed string floats)
    num_cols = ["open", "high", "low", "close", "volume"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop any fully empty rows
    df = df.dropna(how="all")

    return df
