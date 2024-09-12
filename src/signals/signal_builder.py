from signals.data_interfaces import *
from signals.utils import *
from signals.statistics import *
from signals.steps import *
import numpy as np


def build_value_signal() -> pd.DataFrame:
    """
    Builds the value signal for assets by calculating 6-month percentage changes and standardizing the results.

    This function:
    1. Reads asset data from an Excel file.
    2. Computes the percentage change over a 6-month period for each asset.
    3. Standardizes the data using z-scores.
    4. Applies a signal:
        - -1 if the z-score is greater than 1 (overvalued).
        - 1 if the z-score is less than -1 (undervalued).
        - 0 otherwise.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the value signals for each asset, where the signal is -1, 1, or 0 based on z-scores.

    Example:
    --------
    >>> value_signal = build_value_signal()
    >>> print(value_signal.head())

    Notes:
    ------
    - The function assumes the existence of `read_asset_data()` and `compute_z_scores()` to read and standardize data.
    - The signal is based on the assumption that z-scores outside the range [-1, 1] indicate extreme valuation.
    """
    value_data = read_asset_data("data/assets.xlsx")

    # Calculate 6-month percentage change and drop NaNs
    value_data = value_data.pct_change(periods=6).dropna()

    # Compute z-scores for value data
    value_zscores = compute_z_scores(value_data)

    # Apply signal logic based on z-scores
    value_signal = value_zscores.applymap(lambda x: -1 if x > 1 else 1 if x < -1 else 0)

    return value_signal


def build_sentiment_signal() -> pd.DataFrame:
    """
    Builds the sentiment signal by processing sentiment data and correlating it with asset returns.

    This function:
    1. Reads sentiment data and computes the AAII Bull-Bear spread.
    2. Standardizes the sentiment data using z-scores.
    3. Performs PCA to reduce the sentiment data to a single dimension.
    4. Computes the percentage change in asset data over a 3-month period.
    5. Calculates the correlation between sentiment signals and asset returns.
    6. Maps the sentiment signal to asset returns based on the calculated correlations.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing sentiment-based signals for each asset.

    Example:
    --------
    >>> sentiment_signal = build_sentiment_signal()
    >>> print(sentiment_signal.head())

    Notes:
    ------
    - The function assumes the existence of `read_sentiment_data()`, `read_asset_data()`, `compute_z_scores()`, `compute_aaii_bull_bear()`, `compute_1d_pca()`, and `compute_correlations()`.
    - The sentiment signal is computed based on the correlation between sentiment and asset returns.
    """
    # Read and process sentiment data
    sentiment = read_sentiment_data(
        "data/sentiment/Final_Right_Merged_Data_with_UMich.csv"
    )
    sentiment = compute_aaii_bull_bear(sentiment)
    sentiment_zscores = compute_z_scores(sentiment)

    # Perform PCA to reduce sentiment data to 1 dimension
    sentiment_pca = compute_1d_pca(sentiment_zscores)

    # Read asset data and compute 3-month percentage change
    assets = read_asset_data("data/assets.xlsx")
    assets = assets.pct_change(3).dropna()

    # Ensure sentiment PCA index is in datetime format
    sentiment_pca.index = pd.to_datetime(sentiment_pca.index)

    # Compute correlations between sentiment signal and asset returns
    correlations = compute_correlations(assets, sentiment_pca, threshold=0.35)

    # Create a DataFrame for mapped assets
    assets_mapped = pd.DataFrame(index=assets.index, columns=assets.columns)
    for col in assets.columns:
        assets_mapped[col] = correlations[col]

    # Merge sentiment signal with asset mapping
    merged = pd.concat([assets_mapped, sentiment_pca], join="inner", axis=1)

    # Calculate the final sentiment-based score for each asset
    for col in merged.columns:
        if col != "sentiment_signal":
            merged[col] = merged.apply(
                lambda row: calculate_score(row[col], row["sentiment_signal"]), axis=1
            )

    # Drop the sentiment signal column from the final result
    merged = merged.drop("sentiment_signal", axis=1)

    return assets_mapped


def build_macro_signal() -> pd.DataFrame:
    """
    Builds the macroeconomic signal by aggregating and standardizing key economic indicators such as GDP and CPI data.

    This function:
    1. Reads data for the Chicago Fed National Activity Index (CFNAI) and GDP forecasts.
    2. Computes the percentage change for the GDP forecast data.
    3. Reads CPI and CPI forecast data.
    4. Standardizes (z-scores) the combined GDP and CPI data.
    5. Computes signals for GDP growth and inflation using average pooling.
    6. Concatenates the results into a single DataFrame for the macroeconomic signal.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the macroeconomic signal, with standardized GDP and CPI data, and corresponding signals for growth and inflation.

    Steps:
    -----
    1. Reads and processes CFNAI, GDP, CPI, and their forecasts.
    2. Computes the percentage change of GDP forecast data.
    3. Computes z-scores for the combined GDP and CPI data.
    4. Aggregates the GDP growth and inflation signals using average pooling.
    5. Concatenates all results into a single DataFrame.

    Example:
    --------
    >>> macro_signal = build_macro_signal()
    >>> print(macro_signal.head())

    Notes:
    ------
    - The function assumes that `read_cfnai()`, `read_gdp_forecast()`, `read_cpi_data()`, `read_cpi_forecast()`, and `concat_dataframes()` are defined elsewhere in your code.
    - Z-scores are computed to standardize the GDP and CPI data. Consider adding warnings if z-scores are above 3 or below -3 for anomaly detection.
    """
    # Read and process data
    df_cfnai = read_cfnai()
    df_gdp_forecast = read_gdp_forecast("NGDP1")
    df_gdp_forecast = df_gdp_forecast.pct_change()

    df_cpi = read_cpi_data()
    df_cpi_forecast = read_cpi_forecast("CPI1")

    # Concatenate and compute z-scores for GDP and CPI data
    gdp_dataframe = concat_dataframes([df_cfnai, df_gdp_forecast])
    cpi_dataframe = concat_dataframes([df_cpi, df_cpi_forecast])

    gdp_dataframe = compute_z_scores(gdp_dataframe)
    cpi_dataframe = compute_z_scores(cpi_dataframe)

    # Compute signals using average pooling
    gdp_signal = average_pooling(gdp_dataframe, col_name="Growth")
    cpi_signal = average_pooling(cpi_dataframe, col_name="Inflation")

    # Concatenate all the signals and data
    df = concat_dataframes([gdp_dataframe, cpi_dataframe, gdp_signal, cpi_signal])

    return df


def build_momentum_signal() -> pd.DataFrame:
    """
    Builds the momentum signal based on asset data by calculating the difference between current values and values from 12 periods ago.

    This function:
    1. Reads asset data from an Excel file.
    2. Calculates the 12-month momentum for each asset.
    3. Applies a signal where:
        - 1 indicates positive momentum (asset price increased).
        - -1 indicates negative momentum (asset price decreased).
        - NaN for unchanged values.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the momentum signals for each asset, with 1 for positive momentum, -1 for negative momentum, and NaN for no change.

    Steps:
    -----
    1. Reads asset data from the Excel file.
    2. Computes the 12-month difference (momentum) for each asset.
    3. Applies the momentum signal logic to each asset's data.

    Example:
    --------
    >>> momentum_signal = build_momentum_signal()
    >>> print(momentum_signal.head())

    Notes:
    ------
    - This function assumes the existence of `read_asset_data()` to read data from "data/assets.xlsx".
    - The function calculates momentum by subtracting values from 12 periods ago. A positive result implies an upward trend, and a negative result implies a downward trend.
    """
    # Read asset data
    value_data = read_asset_data("data/assets.xlsx")

    # Calculate 12-month momentum
    momentum = (value_data - value_data.shift(12)).dropna()

    # Apply momentum signal logic
    momentum_signal = momentum.applymap(
        lambda x: 1 if x > 0 else -1 if x < 0 else np.nan
    )

    return momentum_signal
