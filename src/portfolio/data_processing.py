import pandas as pd
import scipy.sparse
import cvxpy as cp
import numpy as np


def trim_data(df: pd.DataFrame, percentile: int = 5) -> pd.DataFrame:
    """
    Trims the input DataFrame by capping values at a lower and upper percentile.

    This function limits extreme values in the DataFrame by clipping the values below
    the specified lower percentile and above the upper percentile. This helps in reducing
    the impact of extreme outliers.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing numeric data to be trimmed.

    percentile : int, optional
        The percentile to use for clipping the data (default is 5).
        - The lower bound is set to the `percentile` percentile.
        - The upper bound is set to `100 - percentile` percentile.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with values trimmed between the specified percentiles.

    Example:
    --------
    >>> data = pd.DataFrame({
    >>>     'col1': [1, 2, 3, 100],
    >>>     'col2': [-50, 5, 2, 10]
    >>> })
    >>> trimmed_data = trim_data(data, percentile=5)
    >>> print(trimmed_data)

    Notes:
    ------
    - The function uses the `quantile` method to find the lower and upper bounds
      and the `clip` method to cap the values within these bounds.
    - The result is useful when you want to reduce the influence of extreme outliers
      while retaining most of the data.
    """
    lower = df.quantile(percentile / 100)
    upper = df.quantile(1 - percentile / 100)
    df = df.clip(lower, upper, axis=1)
    return df


def replace_outliers_with_interpolation(
    df: pd.DataFrame, z_thresh: float = 3
) -> pd.DataFrame:
    """
    Replaces outliers in a DataFrame using linear interpolation based on a z-score threshold.

    This function identifies outliers using the z-score method, where any value with
    a z-score greater than the specified threshold is considered an outlier. The outliers
    are then replaced with linearly interpolated values from surrounding non-outliers.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing numeric data in which outliers will be identified and replaced.

    z_thresh : float, optional
        The z-score threshold above which values are considered outliers (default is 3).
        - Values with absolute z-scores above this threshold are considered outliers.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with outliers replaced by linearly interpolated values.

    Example:
    --------
    >>> data = pd.DataFrame({
    >>>     'col1': [1, 2, 3, 100],
    >>>     'col2': [-50, 5, 2, 10]
    >>> })
    >>> clean_data = replace_outliers_with_interpolation(data, z_thresh=2)
    >>> print(clean_data)

    Notes:
    ------
    - The function calculates the z-score for each value by subtracting the mean and dividing by the standard deviation.
    - The identified outliers are masked (set to NaN) and then replaced by linearly interpolating the missing values.
    - The `limit_direction='both'` ensures that the interpolation can extend in both directions when necessary.
    """
    z_scores = np.abs((df - df.mean()) / df.std())
    outliers = z_scores > z_thresh

    df_replaced = df.mask(outliers)
    df_replaced = df_replaced.interpolate(
        method="linear", axis=0, limit_direction="both"
    )
    return df_replaced


def trend_filter(df: pd.DataFrame, col: str, vlambda: float = 1) -> pd.DataFrame:
    """
    Applies a trend filtering technique to smooth a time series column using L1 norm optimization.

    This function applies a trend filter to a specified column of a DataFrame using
    L1 norm regularization. The result is a smoothed version of the original data, where
    sudden fluctuations are penalized to capture long-term trends.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the column to be trend filtered.

    col : str
        The column name to apply the trend filter to.

    vlambda : float, optional
        The regularization parameter that controls the smoothness of the trend filter
        (default is 1). Higher values make the result smoother.

    Returns:
    -------
    pd.DataFrame
        The DataFrame with a new column `<col>_filtered` containing the smoothed trend.

    Example:
    --------
    >>> data = pd.DataFrame({
    >>>     'time': pd.date_range('2020-01-01', periods=10),
    >>>     'value': [1, 2, 3, 6, 9, 4, 2, 1, 3, 5]
    >>> })
    >>> trend_filtered = trend_filter(data, col='value', vlambda=1)
    >>> print(trend_filtered)

    Notes:
    ------
    - This method solves an optimization problem using the `cvxpy` library to minimize
      the difference between the original data and the smoothed version, penalizing changes in the trend.
    - The regularization parameter `vlambda` controls the amount of smoothing, with larger values
      producing smoother trends.
    - The trend-filtered result is stored in a new column with the suffix `_filtered`.
    """
    y = df[col].to_numpy()
    n = y.size
    e = np.ones((1, n))
    D = scipy.sparse.spdiags(np.vstack((e, -2 * e, e)), range(3), n - 2, n)
    x = cp.Variable(shape=n)
    obj = cp.Minimize(0.3 * cp.sum_squares(y - x) + vlambda * cp.norm(D @ x, 1))
    prob = cp.Problem(obj)
    prob.solve(solver=cp.ECOS, verbose=True)
    df[col + "_filtered"] = pd.Series(x.value, index=df.index)
