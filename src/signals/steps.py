from sklearn.decomposition import PCA
import pandas as pd


def compute_1d_pca(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs Principal Component Analysis (PCA) on the numeric columns of the input DataFrame, reducing the data to a single dimension.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing numeric and non-numeric columns. Only numeric columns will be used for PCA.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the 1-dimensional PCA result, with a single column named 'sentiment_signal'.

    Steps:
    -----
    1. Selects only numeric columns from the input DataFrame.
    2. Performs PCA to reduce the selected columns to 1 dimension.
    3. Returns a new DataFrame containing the PCA result.

    Example:
    --------
    >>> df = pd.DataFrame({
    >>>     'feature1': [1.0, 2.0, 3.0],
    >>>     'feature2': [2.0, 4.0, 6.0],
    >>>     'non_numeric': ['A', 'B', 'C']
    >>> })
    >>> pca_result = compute_1d_pca(df)
    >>> print(pca_result)

    Notes:
    ------
    - This function performs PCA only on numeric columns, ignoring any non-numeric data.
    """
    # Extract numeric columns
    numeric_df = df.select_dtypes(include=[float, int])

    # Perform PCA to reduce to 1 dimension
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(numeric_df)

    # Create a new DataFrame with the PCA result
    result_df = pd.DataFrame(
        pca_result, columns=["sentiment_signal"], index=numeric_df.index
    )

    return result_df


def compute_aaii_bull_bear(data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the AAII Bull-Bear Spread by subtracting the 'AAII_Bear' column from the 'AAII_Bull' column.

    Parameters:
    ----------
    data : pd.DataFrame
        A DataFrame containing the 'AAII_Bull' and 'AAII_Bear' columns.

    Returns:
    -------
    pd.DataFrame
        The input DataFrame with a new column 'AAII_BullBear', which is the difference between 'AAII_Bull' and 'AAII_Bear'.
        The original 'AAII_Bull' and 'AAII_Bear' columns are dropped.

    Steps:
    -----
    1. Subtracts the 'AAII_Bear' column from the 'AAII_Bull' column to create a new column called 'AAII_BullBear'.
    2. Drops the original 'AAII_Bull' and 'AAII_Bear' columns.

    Example:
    --------
    >>> data = pd.DataFrame({
    >>>     'AAII_Bull': [45.0, 50.0, 55.0],
    >>>     'AAII_Bear': [25.0, 20.0, 30.0]
    >>> })
    >>> result = compute_aaii_bull_bear(data)
    >>> print(result)

    Notes:
    ------
    - The input DataFrame must contain columns named 'AAII_Bull' and 'AAII_Bear'.
    - The function modifies the input DataFrame in place and drops the original 'AAII_Bull' and 'AAII_Bear' columns.
    """
    data["AAII_BullBear"] = data["AAII_Bull"] - data["AAII_Bear"]
    data.drop(["AAII_Bull", "AAII_Bear"], axis=1, inplace=True)
    return data
