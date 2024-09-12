import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np


def read_cfnai(file_path="data/macro_raw/growth_module/cfnai.xlsx"):
    """
    Load and preprocess the CFNAI data from an Excel file.

    Args:
        file_path (str): Path to the CFNAI Excel file.

    Returns:
        pd.DataFrame: Preprocessed CFNAI DataFrame with 'Date' as the index.
    """
    df = pd.read_excel(file_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y:%m")
    df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    df = df.set_index("Date")[["CFNAI"]]
    return df


def read_gdp_forecast(
    gdp_forecast, file_path="data/macro_raw/growth_module/gdp_forecast.xlsx"
):
    """
    Load and preprocess the GDP forecast data from an Excel file.

    Args:
        gdp_forecast (str): Column name for GDP forecast.
        file_path (str): Path to the GDP forecast Excel file.

    Returns:
        pd.DataFrame: Preprocessed GDP forecast DataFrame with 'Date' as the index.
    """
    df = pd.read_excel(file_path)
    df["Date"] = df["YEAR"].astype(str) + " Q" + df["QUARTER"].astype(str)
    quarter_map = {"Q1": "01", "Q2": "04", "Q3": "07", "Q4": "10"}
    df["Date"] = df["Date"].apply(
        lambda x: x.split()[0] + "-" + quarter_map[x.split()[1]]
    )
    df["Date"] = pd.to_datetime(df["Date"])
    date_range = pd.date_range(start=df["Date"].min(), end=df["Date"].max(), freq="MS")
    date_df = pd.DataFrame(date_range, columns=["Date"])
    date_df.set_index("Date", inplace=True)
    merged_df = date_df.join(df.set_index("Date"), how="left")
    forecast_df = merged_df[[gdp_forecast]]
    not_nan = ~np.isnan(forecast_df[gdp_forecast])
    spline = InterpolatedUnivariateSpline(
        forecast_df.index[not_nan].astype(np.int64) // 10**9,
        forecast_df[gdp_forecast][not_nan],
    )
    forecast_df[gdp_forecast] = spline(forecast_df.index.astype(np.int64) // 10**9)
    return forecast_df


def read_cpi_data(file_path="data/macro_raw/inflation_module/cpius_headline.xls"):
    """
    Load and preprocess the CPI data from an Excel file.

    Args:
        file_path (str): Path to the CPI Excel file.

    Returns:
        pd.DataFrame: Preprocessed CPI DataFrame with 'observation_date' as the index.
    """
    df = pd.read_excel(file_path, header=10)
    df.set_index("observation_date", inplace=True)
    df.index = pd.to_datetime(df.index, format="%Y-%m")
    return df


def read_cpi_forecast(
    gdp_forecast, file_path="data/macro_raw/inflation_module/cpi_forecast.xlsx"
):
    """
    Load and preprocess the CPI forecast data from an Excel file.

    Args:
        gdp_forecast (str): Column name for GDP forecast.
        file_path (str): Path to the CPI forecast Excel file.

    Returns:
        pd.DataFrame: Preprocessed CPI forecast DataFrame with 'Date' as the index.
    """
    df = pd.read_excel(file_path)
    df["Date"] = df["YEAR"].astype(str) + " Q" + df["QUARTER"].astype(str)
    quarter_map = {"Q1": "01", "Q2": "04", "Q3": "07", "Q4": "10"}
    df["Date"] = df["Date"].apply(
        lambda x: x.split()[0] + "-" + quarter_map[x.split()[1]]
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df_filtered = df.dropna(subset=[gdp_forecast])
    start_date = df_filtered["Date"].min()
    end_date = df["Date"].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS")
    date_df = pd.DataFrame(date_range, columns=["Date"])
    date_df.set_index("Date", inplace=True)
    merged_df = date_df.join(df.set_index("Date"), how="left")
    forecast_df = merged_df[[gdp_forecast]]
    not_nan = ~np.isnan(forecast_df[gdp_forecast])
    spline = InterpolatedUnivariateSpline(
        forecast_df.index[not_nan].astype(np.int64) // 10**9,
        forecast_df[gdp_forecast][not_nan],
    )
    forecast_df[gdp_forecast] = spline(forecast_df.index.astype(np.int64) // 10**9)
    return forecast_df


def read_sentiment_data(
    file_path="data\sentiment\Final_Right_Merged_Data_with_UMich.csv",
):
    """
    Processes the data from the given CSV file path.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Processed DataFrame.
    """
    # Load the data
    data = pd.read_csv(file_path)

    # Identify the first index where all columns are not NaN
    first_valid_index = data.dropna(how="any").index[0]

    # Filter the DataFrame from that index onward
    filtered_df = data.loc[first_valid_index:]

    # Drop the first column
    filtered_df = filtered_df.iloc[:, 1:]

    # Set 'month_start' as the index and rename it to 'Date'
    filtered_df = filtered_df.set_index("Month_Start")
    filtered_df.index.name = "Date"

    return filtered_df


def read_asset_data(file_path: str) -> pd.DataFrame:
    """
    Reads and processes asset data from an Excel file, resampling the index to match month start dates and filtering out invalid rows.

    Parameters:
    ----------
    file_path : str
        The path to the Excel file containing the asset data. It reads from the "monthly portfolio and weights" sheet in the file.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with the processed asset data, where:
        - The index is resampled to match the month start dates.
        - Rows before the first valid index (where all columns are non-NaN) are removed.
        - The first column is dropped from the dataset.

    Steps:
    -----
    1. Reads the second sheet ("monthly portfolio and weights") from the Excel file, dropping the first 6 rows to clean the data.
    2. Renames the first column as 'Date' and sets it as the index.
    3. Resamples the index to the start of each month.
    4. Filters the DataFrame to start from the first index where all columns have valid (non-NaN) values.
    5. Drops the first column from the DataFrame (assumed unnecessary for analysis).
    6. Ensures the index is in `datetime` format.

    Example:
    --------
    >>> file_path = "data/assets.xlsx"
    >>> asset_data = read_asset_data(file_path)
    >>> print(asset_data.head())

    Notes:
    ------
    - The Excel file is expected to have a sheet named "monthly portfolio and weights" containing the asset data.
    - The function assumes that the relevant data starts after the first 6 rows and needs to be resampled to match month-start periods.
    """
    # Read the second sheet from the Excel file
    sheet2 = pd.read_excel(
        file_path, sheet_name="monthly portfolio and weights", header=3
    )

    # Drop the first 6 rows (0-indexed)
    sheet2 = sheet2.drop([0, 1, 2, 3, 4, 5], axis=0)

    # Rename the first column to 'Date' and set it as the index
    sheet2.rename(columns={sheet2.columns[0]: "Date"}, inplace=True)
    sheet2.set_index("Date", inplace=True)

    # Resample the index to match month-start periods
    sheet2.index = (
        sheet2.index.to_period("M").to_timestamp("M")
        + pd.DateOffset(days=1)
        - pd.DateOffset(months=1)
    )

    # Identify the first index where all columns are not NaN
    first_valid_index = sheet2.dropna(how="any").index[0]

    # Filter the DataFrame from that index onward
    filtered_df = sheet2.loc[first_valid_index:]

    # Drop the first column
    filtered_df = filtered_df.iloc[:, 1:]

    # Ensure the index is in datetime format
    filtered_df.index = pd.to_datetime(filtered_df.index)

    return filtered_df
