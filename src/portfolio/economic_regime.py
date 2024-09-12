import pandas as pd
from signals.utils import concat_dataframes
from statsmodels.tsa.arima.model import ARIMA
import numpy as np


def prepare_economic_regime_data(regimes, asset):
    """
    Prepares a combined DataFrame of economic regimes and asset data by concatenating,
    cleaning, and formatting the input data.

    This function concatenates the economic regime data with the asset data, removes any
    rows with missing values, and ensures that the index is properly formatted as a DateTimeIndex.

    Parameters:
    - regimes (pd.DataFrame): A DataFrame containing economic regime data. The index should be date-based.
    - asset (pd.DataFrame): A DataFrame containing asset return or price data. The index should be date-based.

    Returns:
    - pd.DataFrame: A DataFrame that combines the economic regimes and asset data,
                    with missing values dropped and the index converted to DateTime.

    Example:
    ----------
    >>> # Example economic regime DataFrame
    >>> regimes = pd.DataFrame({
    >>>     'Regime': [1, 2, 3],
    >>>     'Growth': [0.02, 0.03, 0.01],
    >>>     'Inflation': [0.01, 0.02, 0.03]
    >>> }, index=pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']))

    >>> # Example asset DataFrame
    >>> asset = pd.DataFrame({
    >>>     'Asset_A': [100, 102, 104],
    >>>     'Asset_B': [200, 202, 204]
    >>> }, index=pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']))

    >>> # Prepare the combined economic regime data
    >>> combined_data = prepare_economic_regime_data(regimes, asset)

    Example Input:
    --------------
    regimes:
                     Regime  Growth  Inflation
    2023-01-01       1       0.02    0.01
    2023-02-01       2       0.03    0.02
    2023-03-01       3       0.01    0.03

    asset:
                     Asset_A  Asset_B
    2023-01-01       100      200
    2023-02-01       102      202
    2023-03-01       104      204

    Example Output:
    ---------------
    combined_data:
                     Regime  Growth  Inflation  Asset_A  Asset_B
    2023-01-01       1       0.02    0.01       100      200
    2023-02-01       2       0.03    0.02       102      202
    2023-03-01       3       0.01    0.03       104      204

    Notes:
    - The function assumes that the `regimes` and `asset` DataFrames share the same date-based index.
    - Rows with missing values (NaNs) in any of the columns will be removed.
    - The index of the resulting DataFrame is converted to a DateTimeIndex to ensure proper time series alignment.
    """
    economic_regime = concat_dataframes([regimes, asset])
    economic_regime = economic_regime.dropna()
    economic_regime.index = pd.to_datetime(economic_regime.index)
    return economic_regime


def prepare_current_data(economic_regime, current_date):
    """
    Prepares the current economic regime data up to a specified date.

    This function filters the economic regime data up to the given current date, ensures the index is in monthly periods,
    replaces any infinite values with NaN, and interpolates missing values using linear interpolation.

    Parameters:
    - economic_regime (pd.DataFrame): A DataFrame containing economic regime data with a DateTime index.
    - current_date (pd.Timestamp): The date up to which the data should be filtered.

    Returns:
    - pd.DataFrame: A DataFrame containing the filtered and processed economic regime data.

    Example:
    ----------
    >>> # Example economic regime DataFrame
    >>> data = {
    >>>     'Regime': [1, 2, 3, 4],
    >>>     'Growth': [0.02, np.inf, 0.03, -0.01],
    >>>     'Inflation': [0.01, 0.02, -0.02, np.nan]
    >>> }
    >>> economic_regime = pd.DataFrame(data, index=pd.date_range(start='2023-01-01', periods=4, freq='M'))

    >>> # Prepare current data up to a specific date
    >>> current_date = pd.Timestamp('2023-03-01')
    >>> prepared_data = prepare_current_data(economic_regime, current_date)

    Example Input:
    --------------
    economic_regime:
                     Regime  Growth  Inflation
    2023-01-31       1       0.02    0.01
    2023-02-28       2       inf     0.02
    2023-03-31       3       0.03   -0.02
    2023-04-30       4      -0.01    NaN

    current_date: Timestamp('2023-03-01')

    Example Output:
    ---------------
    prepared_data:
                     Regime  Growth  Inflation
    2023-01-31       1       0.02    0.01
    2023-02-28       2       0.025   0.02
    2023-03-31       3       0.03   -0.02

    Notes:
    - The function filters the `economic_regime` DataFrame to include only data up to the specified `current_date`.
    - The index is converted to monthly periods and then back to the end of each month to ensure consistency.
    - Infinite values are replaced with NaN, which are then linearly interpolated.
    """
    # Filter data up to the current date
    tt = economic_regime[economic_regime.index <= current_date].copy()

    # Ensure the index is in monthly periods and aligned to the end of each month
    tt.index = pd.to_datetime(tt.index)
    tt.index = tt.index.to_period("M").to_timestamp("M")

    # Replace infinite values with NaN
    tt.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Interpolate missing values linearly
    tt = tt.interpolate(method="linear")

    return tt


def determine_economic_regime(Growth_direction, Inflation_direction):
    """
    Determines the economic regime based on the direction of Growth and Inflation indicators.

    This function evaluates the directions of Growth and Inflation to classify the economic regime
    into one of four categories: Recession, Early-cycle, Mid-cycle, or Late-cycle.

    Parameters:
    - Growth_direction (float): The direction of the Growth trend, calculated as the difference
                                between the most recent and previous Growth values.
    - Inflation_direction (float): The direction of the Inflation trend, calculated as the difference
                                   between the most recent and previous Inflation values.

    Returns:
    - int: An integer representing the economic regime:
           1 = Recession (Growth Down, Inflation Down)
           2 = Early-cycle (Growth Up, Inflation Down)
           3 = Mid-cycle (Growth Up, Inflation Up)
           4 = Late-cycle (Growth Down, Inflation Up)
           0 = Undefined (if none of the conditions are met)

    Example:
    ----------
    >>> # Example directions for Growth and Inflation
    >>> Growth_direction = 0.005
    >>> Inflation_direction = -0.003

    >>> # Determine the economic regime
    >>> regime = determine_economic_regime(Growth_direction, Inflation_direction)

    Example Input:
    --------------
    Growth_direction: 0.005
    Inflation_direction: -0.003

    Example Output:
    ---------------
    regime: 2  # Early-cycle (Growth Up, Inflation Down)

    Notes:
    - The function uses a set of predefined conditions to determine the economic regime.
    - The conditions compare the directions of Growth and Inflation to classify the regime.
    - If none of the conditions are met, the function returns 0, indicating an undefined regime.
    """
    # Define conditions for each economic regime based on Growth and Inflation directions
    conditions = [
        (Growth_direction < 0) & (Inflation_direction < 0),  # 1 Recession
        (Growth_direction > 0) & (Inflation_direction < 0),  # 2 Early-cycle
        (Growth_direction > 0) & (Inflation_direction > 0),  # 3 Mid-cycle
        (Growth_direction < 0) & (Inflation_direction > 0),  # 4 Late-cycle
    ]

    # Define corresponding regime values
    values = [1, 2, 3, 4]

    # Use np.select to determine the economic regime based on conditions
    return np.select(conditions, values, default=0)


def forecast_economic_signals(tt, arima_order=(1, 1, 1)):
    """
    Forecasts the next time step for the first four columns of a DataFrame using ARIMA models.

    This function applies an ARIMA(1,1,1) model to each of the first four columns in the input DataFrame `tt` and forecasts
    the next time step for each. The predicted values are returned in a list.

    Parameters:
    - tt (pd.DataFrame): A DataFrame containing time series data. The function will forecast the first four columns.

    Returns:
    - list: A list containing the forecasted values for the next time step for each of the first four columns.

    Example:
    ----------
    >>> # Example DataFrame with time series data
    >>> data = {
    >>>     'Growth': [0.02, 0.03, 0.01, 0.04, 0.05],
    >>>     'Inflation': [0.01, 0.02, 0.03, 0.04, 0.05],
    >>>     'Unemployment': [0.05, 0.06, 0.07, 0.06, 0.05],
    >>>     'Interest_Rate': [0.03, 0.025, 0.02, 0.015, 0.01],
    >>>     'Other': [1.2, 1.3, 1.4, 1.5, 1.6]
    >>> }
    >>> tt = pd.DataFrame(data, index=pd.date_range(start='2023-01-01', periods=5, freq='M'))

    >>> # Forecast the next time step for the first four columns
    >>> forecasts = forecast_economic_signals(tt)

    Example Input:
    --------------
    tt:
                     Growth  Inflation  Unemployment  Interest_Rate  Other
    2023-01-31       0.02    0.01       0.05          0.03           1.2
    2023-02-28       0.03    0.02       0.06          0.025          1.3
    2023-03-31       0.01    0.03       0.07          0.02           1.4
    2023-04-30       0.04    0.04       0.06          0.015          1.5
    2023-05-31       0.05    0.05       0.05          0.01           1.6

    Example Output:
    ---------------
    forecasts:
    [0.049, 0.051, 0.052, 0.009]  # Forecasted values for 'Growth', 'Inflation', 'Unemployment', 'Interest_Rate'

    Notes:
    - The function applies an ARIMA(1,1,1) model to each of the first four columns in `tt`.
    - It forecasts the next time step for each of these columns and returns the predicted values in a list.
    - Only the first four columns are used for forecasting. Ensure the DataFrame has at least four columns.
    """
    ariam_forecasting = []

    for col in tt.columns[0:4]:
        # Fit ARIMA(1,1,1) model to the column data
        model = ARIMA(tt[col], order=arima_order).fit()

        # Forecast the next time step and extract the predicted value
        pred = model.get_forecast(steps=1).predicted_mean.iloc[0]

        # Append the forecasted value to the list
        ariam_forecasting.append(pred)

    return ariam_forecasting
