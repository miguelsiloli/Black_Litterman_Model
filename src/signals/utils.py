import pandas as pd
from scipy.stats import zscore


def compute_z_scores(df):
    """
    Computes z-scores for each column in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with the z-scores for each column.
    """
    # Compute z-scores for each column
    z_scores_df = df.apply(zscore, nan_policy="omit")

    return z_scores_df


def concat_dataframes(dataframes):
    """
    Formats the index of each DataFrame in the list to 'yyyy-mm' format and reindexes them to align properly.

    Parameters:
    dataframes (list of pd.DataFrame): List of DataFrames to format and reindex.

    Returns:
    pd.DataFrame: The concatenated DataFrame with aligned indices.
    """
    # Create a union of all indices
    all_indices = pd.Index([])
    for df in dataframes:
        df.index = pd.to_datetime(df.index).strftime("%Y-%m")
        all_indices = all_indices.union(df.index)

    # Reindex each DataFrame
    formatted_dataframes = [df.reindex(all_indices) for df in dataframes]

    # Concatenate the DataFrames
    concatenated_df = pd.concat(formatted_dataframes, axis=1, join="outer")

    return concatenated_df


def average_pooling(df, col_name="average"):
    """
    Performs average pooling by computing the average of all columns
    and returns a single column with the same index.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with a single column containing the average of all columns.
    """
    # Compute the average of all columnsa
    avg_df = pd.DataFrame()
    avg_df[col_name] = df.mean(axis=1)  #

    return avg_df


def apply_diff_to_columns(df, periods=1):
    """
    Applies the diff method to each column in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    periods (int): The number of periods to use for the diff calculation.

    Returns:
    pd.DataFrame: A DataFrame with the diff applied to each column.
    """
    # Apply diff to each column
    diff_df = df.diff(periods=periods)

    return diff_df


def calculate_score(row: float, sentiment_value: float) -> int:
    """
    Calculates a sentiment-based score for an asset based on the sentiment value and the asset's performance.

    Parameters:
    ----------
    row : float
        The asset's performance value (or its correlation with sentiment).
    sentiment_value : float
        The sentiment value (e.g., the sentiment PCA score).

    Returns:
    -------
    int
        - 1 if the product of the sentiment value and the asset performance is greater than 1.
        - -1 if the product is less than -1.
        - 0 otherwise (indicating no strong signal).

    Example:
    --------
    >>> score = calculate_score(0.5, 1.2)
    >>> print(score)  # Outputs: 1

    Notes:
    ------
    - This function is used to calculate the final asset signal based on the sentiment data.
    - The signal is capped between -1 and 1.
    """
    value = sentiment_value * row
    if value > 1:
        return 1
    elif value < -1:
        return -1
    else:
        return 0
