import pandas as pd
from pathlib import Path

def load_gold_csv(path: str, chunksize: int = None, parse_dates: bool = True):
    """
    Load a gold price CSV file with optional chunking and datetime parsing.
    Assumes the file uses semicolon separators.
    """
    if chunksize:
        chunks = []
        for chunk in pd.read_csv(path, sep=";", chunksize=chunksize):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(path, sep=";")

    if parse_dates:
        # Adjust column names based on your file
        if "Date" in df.columns:
            dt_col = "Date"
        elif "datetime" in df.columns:
            dt_col = "datetime"
        else:
            raise ValueError("No datetime column found")

        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
        df = df.sort_values(dt_col).set_index(dt_col)

    return df


def save_processed(df: pd.DataFrame, path: str):
    """
    Save a processed DataFrame to Parquet format.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
