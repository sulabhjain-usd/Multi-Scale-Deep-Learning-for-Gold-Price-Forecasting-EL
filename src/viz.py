import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_time_series(df: pd.DataFrame, column: str, title: str = "Time Series Plot"):
    """
    Plot a time series column from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a datetime index.
    column : str
        Column name to plot.
    title : str
        Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[column], label=column)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_histogram(df: pd.DataFrame, column: str, bins: int = 30, title: str = "Histogram"):
    """
    Plot a histogram of a column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to plot.
    bins : int
        Number of bins.
    title : str
        Title of the plot.
    """
    plt.figure(figsize