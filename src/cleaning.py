import pandas as pd

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning on a DataFrame:
    - Strip whitespace from column names
    - Convert column names to lowercase
    - Drop duplicate rows
    - Reset index
    """
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Drop duplicates
    df = df.drop_duplicates()

    # Reset index
    df = df.reset_index(drop=True)

    return df
