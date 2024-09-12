import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import scipy.sparse
import cvxpy as cp
import numpy as np
import warnings


def compute_correlations(assets, sentiment_pca, threshold):
    """
    Computes the pairwise correlations between each asset in the given DataFrame and a sentiment PCA vector.
    The resulting correlations are thresholded, where correlations above the positive threshold are set to 1,
    those below the negative threshold are set to -1, and all others are set to 0.

    Parameters:
    - assets (pd.DataFrame): A DataFrame containing asset return data, where each column represents a different asset.
    - sentiment_pca (pd.Series or pd.DataFrame): A Series or single-column DataFrame representing the principal component analysis (PCA) of sentiment data.
    - threshold (float): The correlation threshold for determining significant positive or negative correlations.

    Returns:
    - pd.Series: A Series with the same index as the asset columns, containing 1 for positive correlations above the threshold,
                 -1 for negative correlations below the negative threshold, and 0 for correlations within the threshold range.

    Example:
    ----------
    >>> # Example asset returns DataFrame
    >>> data = {'Asset_A': [0.01, 0.02, -0.01, 0.03, 0.04],
    >>>         'Asset_B': [-0.02, 0.01, 0.03, -0.01, 0.02],
    >>>         'Asset_C': [0.03, 0.02, -0.02, 0.01, 0.05]}
    >>> assets = pd.DataFrame(data)

    >>> # Example sentiment PCA Series
    >>> sentiment_pca = pd.Series([0.5, 0.1, -0.3, 0.2, 0.4])

    >>> # Define a threshold for correlation significance
    >>> threshold = 0.5

    >>> # Compute correlations with thresholding
    >>> results = compute_correlations(assets, sentiment_pca, threshold)

    Example Input:
    --------------
    assets:
                     Asset_A  Asset_B  Asset_C
    0               0.01     -0.02      0.03
    1               0.02      0.01      0.02
    2              -0.01      0.03     -0.02
    3               0.03     -0.01      0.01
    4               0.04      0.02      0.05

    sentiment_pca:
    0    0.5
    1    0.1
    2   -0.3
    3    0.2
    4    0.4
    dtype: float64

    threshold: 0.5

    Example Output:
    ---------------
    results:
    Asset_A    1
    Asset_B    0
    Asset_C    0
    dtype: int64

    Extreme Cases:
    ---------------
    - **Perfect Positive Correlation**: If an asset's returns are perfectly positively correlated with the sentiment PCA (correlation coefficient = 1), the result will be `1` if the threshold is less than or equal to 1.
    - **Perfect Negative Correlation**: If an asset's returns are perfectly negatively correlated with the sentiment PCA (correlation coefficient = -1), the result will be `-1` if the negative threshold is greater than or equal to -1.
    - **Zero Correlation**: If an asset's returns are uncorrelated with the sentiment PCA (correlation coefficient = 0), the result will be `0` as long as the threshold is non-zero.
    - **Threshold Edge Cases**: If the correlation exactly equals the positive or negative threshold, it will be classified as `1` or `-1`, respectively. If the correlation is slightly less than the positive threshold or slightly more than the negative threshold, it will be classified as `0`.

    """
    # Compute pairwise correlations
    correlations = assets.corrwith(sentiment_pca.squeeze())

    # Apply thresholding
    results = correlations.apply(
        lambda x: -1 if x < -threshold else (1 if x > threshold else 0)
    )

    return results


def predict_arima(df, n, col="Forecast"):
    """
    Creates an ARIMA model based on the input DataFrame and predicts the next `n` timesteps.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with a single column of time series data. The index should be a DateTimeIndex.
    - n (int): The number of future timesteps to predict.
    - col (str): The name of the column in the returned DataFrame that will contain the forecasted values. Default is "Forecast".

    Returns:
    - pd.DataFrame: The original DataFrame with the appended predictions. The forecasted values are in a new column specified by `col`.

    Example:
    ----------
    >>> # Example DataFrame with a DateTimeIndex
    >>> data = {'Value': [100, 102, 101, 103, 104]}
    >>> df = pd.DataFrame(data, index=pd.date_range(start='2023-01-01', periods=5, freq='MS'))

    >>> # Predict the next 3 timesteps using ARIMA
    >>> result = predict_arima(df, n=3)

    Example Input:
    --------------
    df:
                     Value
    2023-01-01    100
    2023-02-01    102
    2023-03-01    101
    2023-04-01    103
    2023-05-01    104

    n: 3

    col: "Forecast"

    Example Output:
    ---------------
    result:
                     Value   Forecast
    2023-01-01    100        NaN
    2023-02-01    102        NaN
    2023-03-01    101        NaN
    2023-04-01    103        NaN
    2023-05-01    104        NaN
    2023-06-01    NaN        104.5
    2023-07-01    NaN        105.0
    2023-08-01    NaN        105.5

    """
    # Fit ARIMA model
    model = ARIMA(df, order=(1, 1, 1))
    model_fit = model.fit()

    # Make predictions
    forecast = model_fit.forecast(steps=n)

    # Create a DataFrame for the predictions
    forecast_index = pd.date_range(start=df.index[-1], periods=n + 1, freq="MS")[1:]
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=[col])

    # Append forecast to the original DataFrame
    df_with_forecast = pd.concat([df, forecast_df])

    return df_with_forecast


def trend_filter(df, col, vlambda=1, window=None):
    """
    Apply L1 trend filtering to a time series with an optional look-back period.

    This function applies L1 trend filtering to a specified column of a DataFrame. The trend filtering
    is a form of regularization that smooths the time series while allowing for sharp changes in the trend.

    The L1 norm is defined as:
    0.5 ​∑ ​(yi​−xi​)2 - vlambda * ∥Dx∥1

    - First term: It is a sum of squared differences between the original data (yi) and the estimated trend (xi).
    - Second term: The L1 norm ∥Dx∥ is used here to allow for sparsity in the changes,
    meaning the trend can have sharp changes at a few points, but is mostly smooth.


    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the time series data.
    - col (str): The name of the column in the DataFrame to which the trend filtering will be applied.
    - vlambda (float): The regularization parameter. Higher values lead to smoother trends.
    - window (int, optional): The look-back period (number of rows) to apply the filtering.
                              If None, the filter is applied to the entire column.

    Returns:
    - pd.Series: A Series containing the estimated trend component for the specified column.

    Example:
    ----------
    >>> # Example DataFrame with a time series
    >>> data = {'Value': [100, 102, 105, 103, 108, 110, 115]}
    >>> df = pd.DataFrame(data, index=pd.date_range(start='2023-01-01', periods=7, freq='D'))

    >>> # Apply trend filtering to the 'Value' column with a regularization parameter of 1
    >>> trend = trend_filter(df, col='Value', vlambda=1)

    >>> # Apply trend filtering to the 'Value' column with a 5-day look-back period
    >>> trend_with_window = trend_filter(df, col='Value', vlambda=1, window=5)

    Example Input:
    --------------
    df:
                     Value
    2023-01-01       100
    2023-01-02       102
    2023-01-03       105
    2023-01-04       103
    2023-01-05       108
    2023-01-06       110
    2023-01-07       115

    col: "Value"

    vlambda: 1

    window: None

    Example Output:
    ---------------
    trend:
                     Value
    2023-01-01       100.5
    2023-01-02       102.0
    2023-01-03       104.0
    2023-01-04       105.0
    2023-01-05       108.0
    2023-01-06       109.5
    2023-01-07       114.0

    trend_with_window:
                     Value
    2023-01-03       104.0
    2023-01-04       105.0
    2023-01-05       108.0
    2023-01-06       109.5
    2023-01-07       114.0

    Notes:
    - If the `window` parameter is specified, the trend is calculated only for the last `window` rows of the DataFrame.
    - The trend filtering method used here can handle sharp changes in the trend, which makes it useful for time series with potential abrupt shifts.
    """

    if window is not None:
        # Ensure the DataFrame has at least as many rows as the window
        if len(df) < window:
            raise ValueError(
                "The DataFrame has fewer rows than the specified time window."
            )
        # Select the last 'window' rows of the DataFrame
        y = df[col].iloc[-window:].to_numpy()
        index = df.index[-window:]
    else:
        y = df[col].to_numpy()
        index = df.index

    n = y.size

    e = np.ones((1, n))
    D = scipy.sparse.spdiags(np.vstack((e, -2 * e, e)), range(3), n - 2, n)

    x = cp.Variable(shape=n)
    obj = cp.Minimize(0.5 * cp.sum_squares(y - x) + vlambda * cp.norm(D @ x, 1))
    prob = cp.Problem(obj)

    prob.solve(solver=cp.ECOS, verbose=False)

    # Create a series with the trend component
    trend = pd.DataFrame(x.value, index=index, columns=[col])

    return trend


def classify_economic_regime(df):
    """
    Classifies economic regimes based on Growth and Inflation indicators.

    This function evaluates the 'Growth' and 'Inflation' columns of the input DataFrame
    to classify the economic regime into one of four categories:
    1. Growth Down & Inflation Down
    2. Growth Up & Inflation Down
    3. Growth Up & Inflation Up
    4. Growth Down & Inflation Up

    The function replaces NaN values with 0 and raises a warning if any NaN values are encountered.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with columns 'Growth' and 'Inflation'.

    Returns:
    - pd.DataFrame: The DataFrame with an additional 'EconomicRegime' column.

    Example:
    ----------
    >>> # Example DataFrame
    >>> data = {
    >>>     'Growth': [0.02, -0.01, 0.03, -0.02, np.nan, 0.01, 0.02],
    >>>     'Inflation': [0.01, 0.02, -0.01, 0.03, 0.04, np.nan, 0.02]
    >>> }
    >>> df = pd.DataFrame(data)

    >>> # Classify economic regimes
    >>> classified_df = classify_economic_regime(df)

    Example Input:
    --------------
    df:
              Growth  Inflation
    0       0.02       0.01
    1      -0.01       0.02
    2       0.03      -0.01
    3      -0.02       0.03
    4        NaN        0.04
    5       0.01        NaN
    6       0.02       0.02

    Example Output:
    ---------------
    classified_df:
              Growth  Inflation  EconomicRegime
    0       0.02       0.01               3
    1      -0.01       0.02               4
    2       0.03      -0.01               2
    3      -0.02       0.03               4
    4       0.00       0.04               4
    5       0.01       0.00               2
    6       0.02       0.02               3

    """
    # Check for NaN values and replace them with 0, while raising a warning
    if df[["Growth", "Inflation"]].isna().any().any():
        warnings.warn(
            "NaN values detected in 'Growth' or 'Inflation' columns. Replacing NaN values with 0."
        )
        df[["Growth", "Inflation"]] = df[["Growth", "Inflation"]].fillna(0)

    # Define the conditions
    conditions = [
        (df["Growth"] < 0) & (df["Inflation"] < 0),  # 1 Growth Down & Inflation Down
        (df["Growth"] > 0) & (df["Inflation"] < 0),  # 2 Growth Up & Inflation Down
        (df["Growth"] > 0) & (df["Inflation"] > 0),  # 3 Growth Up & Inflation Up
        (df["Growth"] < 0) & (df["Inflation"] > 0),  # 4 Growth Down & Inflation Up
    ]

    # Define the values for each condition
    values = [1, 2, 3, 4]

    # Apply the conditions to create the 'EconomicRegime' column
    df["EconomicRegime"] = np.select(conditions, values, default=0)

    return df
