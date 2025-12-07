import pandas as pd

def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Check for missing values in each column of the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.Series
        Count of missing values per column.
    """
    return df.isnull().sum()


def check_data_types(df: pd.DataFrame) -> pd.Series:
    """
    Check the data types of each column in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.Series
        Data types of each column.
    """
    return df.dtypes


def check_unique_values(df: pd.DataFrame) -> pd.Series:
    """
    Check the number of unique values in each column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.Series
        Count of unique values per column.
    """
    return df.nunique()
