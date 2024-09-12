import pandas as pd


def apply_technical_signals(
    regime_weights: dict,
    current_date: pd.Timestamp,
    asset: pd.DataFrame,
    value_signal: pd.DataFrame,
    momentum_signal: pd.DataFrame,
    sentiment_signal: pd.DataFrame,
) -> dict:
    """
    Adjusts portfolio weights based on technical signals for value, momentum, and sentiment.

    This function:
    1. Filters the value, momentum, and sentiment signals for the given `current_date`.
    2. Combines the signals into a single DataFrame to apply adjustments.
    3. Calculates an adjustment factor for each asset based on the sum of the signals.
    4. Adjusts the portfolio weights by applying a 30% adjustment for each signal score.

    Parameters:
    ----------
    regime_weights : dict
        The initial portfolio weights based on the current economic regime.

    current_date : pd.Timestamp
        The date for which to apply the technical signals.

    asset : pd.DataFrame
        The DataFrame containing the asset data (usually prices or returns) with assets as columns.

    value_signal : pd.DataFrame
        A DataFrame containing the value signals for each asset, with dates as the index.

    momentum_signal : pd.DataFrame
        A DataFrame containing the momentum signals for each asset, with dates as the index.

    sentiment_signal : pd.DataFrame
        A DataFrame containing the sentiment signals for each asset, with dates as the index.

    Returns:
    -------
    dict
        A dictionary containing the adjusted portfolio weights for each asset after applying the technical signals.

    Steps:
    -----
    1. For each signal type (value, momentum, sentiment), the signal for `current_date` is extracted.
    2. These signals are combined into a DataFrame where each row corresponds to a different signal type.
    3. For each asset, the signals are summed, and based on the total signal score:
        - Positive signal: Increase the asset's weight by 30%.
        - Negative signal: Decrease the asset's weight by 30%.
        - Zero signal: No adjustment.
    4. The portfolio weights are adjusted accordingly and returned.

    Example:
    --------
    >>> regime_weights = {'Asset_A': 0.4, 'Asset_B': 0.6}
    >>> current_date = pd.Timestamp('2023-01-31')
    >>> value_signal = pd.DataFrame(...)  # Load or calculate value signals
    >>> momentum_signal = pd.DataFrame(...)  # Load or calculate momentum signals
    >>> sentiment_signal = pd.DataFrame(...)  # Load or calculate sentiment signals
    >>> adjusted_weights = apply_technical_signals(regime_weights, current_date, asset, value_signal, momentum_signal, sentiment_signal)
    >>> print(adjusted_weights)

    Notes:
    ------
    - The function assumes that each signal DataFrame has the same columns (representing assets) and index (representing dates).
    - It also assumes that the signals are numeric and are structured such that the value at a given date represents the signal strength for that asset.
    - Signals are weighted by 30%, meaning the final adjustment to portfolio weights is capped at ±30%.
    """
    # Filter the value, momentum, and sentiment signals for the current date
    v1 = value_signal[value_signal.index == current_date]
    v1.columns = asset.columns
    m1 = momentum_signal[momentum_signal.index == current_date]
    m1.columns = asset.columns
    s1 = sentiment_signal[sentiment_signal.index == current_date]
    s1.columns = asset.columns

    # Combine the signals into one DataFrame
    technical_signals = pd.concat([v1, m1, s1], axis=0)
    technical_signals.index = ["value", "momentum", "sentiment"]

    # Create a copy of regime_weights to adjust based on the technical signals
    portfolio_weights = regime_weights.copy()

    # Iterate over each asset to adjust weights based on the summed signal values
    for assets in portfolio_weights.keys():
        signal_sum = technical_signals.sum()[assets]
        original_weight = regime_weights[assets]
        signal_weight = 0.3  # 30% adjustment for each signal score

        if signal_sum > 0:
            # Positive signal: increase weight
            if original_weight < 0:
                adjustment_factor = 1 - abs(signal_sum) * signal_weight
            else:
                adjustment_factor = 1 + abs(signal_sum) * signal_weight
        else:
            # Negative signal: decrease weight
            if original_weight < 0:
                adjustment_factor = 1 + abs(signal_sum) * signal_weight
            else:
                adjustment_factor = 1 - abs(signal_sum) * signal_weight

        # Adjust the asset's portfolio weight
        adjusted_weight = original_weight * adjustment_factor
        portfolio_weights[assets] = adjusted_weight

    return portfolio_weights


def initialize_portfolio(asset, current_date):
    """
    Initialize a portfolio DataFrame with zeros based on the given asset data and current date.

    This function creates a portfolio DataFrame by filtering the asset data to include only the rows
    where the date is greater than the current date. The portfolio is then initialized with zero
    values for all assets, indicating no initial allocation.

    Parameters:
    - asset (pd.DataFrame): A DataFrame containing asset data, with dates as the index and asset prices or returns as columns.
    - current_date (pd.Timestamp): A Timestamp object representing the current date for filtering the asset data.

    Returns:
    - pd.DataFrame: A DataFrame initialized with zeros, containing the same structure as the filtered asset data.

    Example:
    ----------
    >>> # Example asset DataFrame
    >>> data = {'Asset_A': [0.01, 0.02, -0.01, 0.03, 0.04],
    >>>         'Asset_B': [-0.02, 0.01, 0.03, -0.01, 0.02]}
    >>> asset = pd.DataFrame(data, index=pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01']))
    >>> current_date = pd.Timestamp('2023-02-01')

    >>> # Initialize the portfolio
    >>> portfolio = initialize_portfolio(asset, current_date)

    Example Input:
    --------------
    asset:
                     Asset_A  Asset_B
    2023-01-01       0.01      -0.02
    2023-02-01       0.02       0.01
    2023-03-01      -0.01       0.03
    2023-04-01       0.03      -0.01
    2023-05-01       0.04       0.02

    current_date: Timestamp('2023-02-01')

    Example Output:
    ---------------
    portfolio:
                     Asset_A  Asset_B
    2023-03-01       0.0        0.0
    2023-04-01       0.0        0.0
    2023-05-01       0.0        0.0

    """
    portfolio = asset.loc[asset.index > current_date]
    portfolio.iloc[:, :] = 0
    return portfolio


def update_portfolio_weights(portfolio, weights, current_date):
    """
    Updates the portfolio weights for the next month based on the provided weights.

    This function takes the current portfolio DataFrame and updates the weights for the next month's
    allocation using the provided weights dictionary. The function assumes that the 'weights' dictionary
    may be missing certain keys such as 'ISM Manufacturing PMI SA', 'US CPI Urban Consumers YoY NSA',
    and 'Federal Funds Target Rate - Up', which should be handled accordingly.

    Parameters:
    - portfolio (pd.DataFrame): The DataFrame representing the portfolio with datetime index and asset columns.
    - weights (dict): A dictionary where keys are asset names and values are the corresponding weights
                      to be assigned for the next month.
    - current_date (pd.Timestamp): The current date from which the next month's end date will be calculated.

    Returns:
    - pd.DataFrame: The updated portfolio DataFrame with the new weights assigned for the next month's end date.

    Example:
    ----------
    >>> # Example portfolio DataFrame
    >>> data = {
    >>>     'Corporate': [0.1, 0.2],
    >>>     'U.S. Corporate High Yield': [0.15, 0.25],
    >>>     'BBG Commodity TR': [0.05, 0.1]
    >>> }
    >>> dates = pd.date_range(start='2023-01-31', periods=2, freq='M')
    >>> portfolio = pd.DataFrame(data, index=dates)

    >>> # Example weights dictionary
    >>> weights = {
    >>>     'Corporate': 0.25,
    >>>     'U.S. Corporate High Yield': 0.3,
    >>>     'BBG Commodity TR': 0.15
    >>> }

    >>> # Current date
    >>> current_date = pd.Timestamp('2023-01-31')

    >>> # Update the portfolio weights
    >>> updated_portfolio = update_portfolio_weights(portfolio, weights, current_date)

    Example Input:
    --------------
    portfolio:
                     Corporate  U.S. Corporate High Yield  BBG Commodity TR
    2023-01-31       0.10       0.15                       0.05
    2023-02-28       0.20       0.25                       0.10

    weights:
    {'Corporate': 0.25, 'U.S. Corporate High Yield': 0.3, 'BBG Commodity TR': 0.15}

    current_date:
    Timestamp('2023-01-31')

    Example Output:
    ---------------
    updated_portfolio:
                     Corporate  U.S. Corporate High Yield  BBG Commodity TR
    2023-01-31       0.10       0.15                       0.05
    2023-02-28       0.20       0.25                       0.10
    2023-03-31       0.25       0.30                       0.15

    Notes:
    - The function assumes that the weights dictionary may not include certain assets (e.g., 'ISM Manufacturing PMI SA',
      'US CPI Urban Consumers YoY NSA', 'Federal Funds Target Rate - Up'). If these columns are needed, they should be
      handled before calling this function or updated separately.
    - The function updates the portfolio for the next month end date, which is calculated as `current_date + pd.offsets.MonthEnd(1)`.
    """
    # Perform an inner join between the portfolio columns and the weights dictionary keys
    common_assets = portfolio.columns.intersection(weights.keys())

    # Filter the portfolio to only include common assets
    filtered_portfolio = portfolio[common_assets]

    # Update the portfolio with the new weights for the next month's end date
    filtered_portfolio.loc[
        filtered_portfolio.index == (current_date + pd.offsets.MonthEnd(1))
    ] = pd.DataFrame(weights, index=[(current_date + pd.offsets.MonthEnd(1))])

    return filtered_portfolio


def calculate_portfolio_performance(performance, tt, portfolio_weights):
    new_index = pd.date_range(start=tt.index.min(), end=tt.index.max(), freq="MS")
    tt = tt.reindex(new_index)
    common_columns = performance.columns.intersection(portfolio_weights.keys())
    performance_slice = performance.loc[
        performance.index == tt.index[-1], common_columns
    ]
    performance.loc[
        performance.index == tt.index[-1], "portfolio"
    ] = performance_slice.dot(pd.Series(portfolio_weights).loc[common_columns])
    return performance
