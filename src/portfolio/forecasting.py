import pandas as pd
from portfolio.data_processing import trend_filter


def calculate_growth_inflation(ariam_forecasting):
    """
    Calculates the Growth and Inflation indicators based on ARIMA forecasting results.

    This function computes the Growth and Inflation indicators by averaging specific pairs of forecasted values.
    It assumes that the input list `ariam_forecasting` contains forecasts for the first four economic signals.

    Parameters:
    - ariam_forecasting (list): A list of forecasted values for economic indicators.
                                The list should contain at least four values, where:
                                - The first value corresponds to Growth-related data.
                                - The second value corresponds to additional Growth-related data.
                                - The third value corresponds to Inflation-related data.
                                - The fourth value corresponds to additional Inflation-related data.

    Returns:
    - tuple: A tuple containing the calculated Growth and Inflation indicators as floats.

    Example:
    ----------
    >>> # Example ARIMA forecasting results
    >>> ariam_forecasting = [0.05, 0.04, 0.03, 0.02]

    >>> # Calculate Growth and Inflation indicators
    >>> growth, inflation = calculate_growth_inflation(ariam_forecasting)

    Example Input:
    --------------
    ariam_forecasting:
    [0.05, 0.04, 0.03, 0.02]

    Example Output:
    ---------------
    growth: 0.045  # (0.05 + 0.04) / 2
    inflation: 0.025  # (0.03 + 0.02) / 2

    Notes:
    - The function assumes that the first two elements of `ariam_forecasting` are related to Growth,
      and the last two are related to Inflation.
    - Ensure that the input list has at least four elements to avoid index errors.
    """
    Growth = (ariam_forecasting[0] + ariam_forecasting[1]) / 2
    Inflation = (ariam_forecasting[2] + ariam_forecasting[3]) / 2
    return Growth, Inflation


def update_trend_filters(economic_regime, Growth, Inflation, ariam_forecasting):
    """
    Updates the economic regime DataFrame with new Growth and Inflation values,
    and applies trend filtering to the updated DataFrame.

    This function adds a new row to the `tt` DataFrame with the forecasted Growth and Inflation values,
    then applies a trend filter to both the 'Growth' and 'Inflation' columns to smooth the time series data.

    Parameters:
    - tt (pd.DataFrame): A DataFrame containing economic regime data, including columns for 'Growth' and 'Inflation'.
    - Growth (float): The forecasted Growth value to be added to the DataFrame.
    - Inflation (float): The forecasted Inflation value to be added to the DataFrame.

    Returns:
    - pd.DataFrame: The updated DataFrame with the new Growth and Inflation values and trend-filtered columns.

    Example:
    ----------
    >>> # Example economic regime DataFrame
    >>> data = {
    >>>     'Growth': [0.02, 0.03, 0.01],
    >>>     'Inflation': [0.01, 0.02, 0.03],
    >>>     'Other': [1, 2, 3]
    >>> }
    >>> tt = pd.DataFrame(data, index=pd.date_range(start='2023-01-01', periods=3, freq='M'))

    >>> # Forecasted values
    >>> Growth = 0.04
    >>> Inflation = 0.025

    >>> # Update the DataFrame and apply trend filters
    >>> updated_tt = update_trend_filters(tt, Growth, Inflation)

    Example Input:
    --------------
    tt:
                     Growth  Inflation  Other
    2023-01-31       0.02    0.01       1
    2023-02-28       0.03    0.02       2
    2023-03-31       0.01    0.03       3

    Growth: 0.04

    Inflation: 0.025

    Example Output:
    ---------------
    updated_tt:
                     Growth  Inflation  Other  Growth_filtered  Inflation_filtered
    2023-01-31       0.02    0.01       1      0.020            0.010
    2023-02-28       0.03    0.02       2      0.025            0.015
    2023-03-31       0.01    0.03       3      0.030            0.020
    2023-04-30       0.04    0.025      0      0.035            0.0225

    Notes:
    - The function assumes that the `tt` DataFrame contains at least columns named 'Growth' and 'Inflation'.
    - The new row added to the DataFrame contains the forecasted Growth and Inflation values along with zeros for other columns.
    - The trend filtering function `trend_filter` is assumed to be defined elsewhere in the code and applied to the 'Growth' and 'Inflation' columns.

    """
    # Create a new index for the next month
    t1_index = economic_regime.index[-1] + pd.offsets.MonthEnd(1)

    num_placeholders = len(economic_regime.columns) - len(ariam_forecasting) - 2

    # Combine forecasted values with placeholders for other columns
    t1_values = ariam_forecasting + [Growth, Inflation] + [0] * num_placeholders

    # Create a new DataFrame row with the combined values
    t1_row = pd.DataFrame([t1_values], index=[t1_index], columns=economic_regime.columns)

    # Append the new row to the DataFrame
    economic_regime = pd.concat([economic_regime, t1_row])

    # Apply trend filtering to the 'Growth' and 'Inflation' columns
    trend_filter(economic_regime, "Growth")
    trend_filter(economic_regime, "Inflation")

    return economic_regime


def determine_trend_directions(tt):
    """
    Determines the direction of trends for Growth and Inflation indicators.

    This function calculates the direction of the trend for the 'Growth_filtered'
    and 'Inflation_filtered' columns by comparing the most recent values with
    the previous ones in the DataFrame.

    Parameters:
    - tt (pd.DataFrame): A DataFrame containing the trend-filtered time series data
                         for Growth and Inflation. The DataFrame should have at least
                         two rows and should contain 'Growth_filtered' and
                         'Inflation_filtered' columns.

    Returns:
    - tuple: A tuple containing the direction of the Growth trend and the direction
             of the Inflation trend. These are calculated as the difference between
             the last two values in their respective columns.

    Example:
    ----------
    >>> # Example DataFrame with trend-filtered data
    >>> data = {
    >>>     'Growth_filtered': [0.02, 0.025, 0.03],
    >>>     'Inflation_filtered': [0.01, 0.015, 0.012]
    >>> }
    >>> tt = pd.DataFrame(data, index=pd.date_range(start='2023-01-01', periods=3, freq='M'))

    >>> # Determine trend directions
    >>> growth_dir, inflation_dir = determine_trend_directions(tt)

    Example Input:
    --------------
    tt:
                     Growth_filtered  Inflation_filtered
    2023-01-31       0.02             0.01
    2023-02-28       0.025            0.015
    2023-03-31       0.03             0.012

    Example Output:
    ---------------
    growth_dir: 0.005  # (0.03 - 0.025)
    inflation_dir: -0.003  # (0.012 - 0.015)

    Notes:
    - The function assumes that the DataFrame `tt` has at least two rows.
    - The direction is calculated as the difference between the last two values
      in the 'Growth_filtered' and 'Inflation_filtered' columns.
    - A positive value indicates an upward trend, while a negative value indicates
      a downward trend.
    """
    # Calculate the direction of the Growth trend
    Growth_direction = tt["Growth_filtered"].iloc[-1] - tt["Growth_filtered"].iloc[-2]

    # Calculate the direction of the Inflation trend
    Inflation_direction = (
        tt["Inflation_filtered"].iloc[-1] - tt["Inflation_filtered"].iloc[-2]
    )

    return Growth_direction, Inflation_direction
