import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from signals.data_interfaces import read_asset_data
from logger import logger


# i need to abstract unnecessary imports
from portfolio.utils import initialize_portfolio
from portfolio.economic_regime import (
    prepare_economic_regime_data,
    prepare_current_data,
    forecast_economic_signals,
)

from portfolio.forecasting import (
    calculate_growth_inflation,
    update_trend_filters,
    determine_trend_directions,
)
from portfolio.economic_regime import determine_economic_regime

from signals.statistics import classify_economic_regime
from matplotlib.dates import DateFormatter
from portfolio.simulation import simulate_portfolio_allocation

from signals import *
# maybe changing this afterwards
from utils import calculate_final_metrics, print_performance_metrics
from layout.plots import plot_growth_inflation_line, plot_metrics_by_economic_regime, plot_portfolio_weights, plot_cumulative_returns, plot_signals, plot_economic_regime_pie_chart

st.set_page_config(layout="wide")

# Load asset data from Excel
assets = read_asset_data("data/assets.xlsx")

# growth and inflation need to be within -4, 4 because they are standardized
macro_signal = build_macro_signal()
momentum_signal = build_momentum_signal()
value_signal = build_value_signal()
sentiment_signal = build_sentiment_signal()

# Sidebar date pickers for start and end date
st.sidebar.header("Configuration")
start_date = st.sidebar.date_input("Start Date", pd.Timestamp("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.Timestamp("2023-12-01"))
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)

# ARIMA Order Selection
arima_order = st.sidebar.selectbox(
    "Select ARIMA Order (p, d, q)",
    options=[(1, 1, 1), (2, 2, 2), (3, 3, 3)],
    index=0,
    help="ARIMA is a time series forecasting method. (1, 1, 1) means using 1 lag for AR, "
    "1 degree of differencing for stationarity, and 1 lag for MA.",
)

# Min Weight Bound Slider
min_weight_bound = st.sidebar.slider(
    "Min Portfolio Weight Bound",
    min_value=0.0,
    max_value=0.2,
    step=0.01,
    value=0.0,
    help="The minimum weight that can be assigned to any asset in the portfolio. "
    "For instance, -0.5 allows short positions up to 50%.",
)

# Max Weight Bound Slider
max_weight_bound = st.sidebar.slider(
    "Max Portfolio Weight Bound",
    min_value=0.25,
    max_value=1.0,
    step=0.01,
    value=0.50,
    help="The maximum weight that can be assigned to any asset in the portfolio.",
)

# Max Volatility Slider
max_volatility = st.sidebar.slider(
    "Max Portfolio Volatility",
    min_value=0.05,
    max_value=0.5,
    step=0.01,
    value=0.12,
    help="The maximum allowed volatility (risk) for the portfolio. "
    "A higher value allows more risk in the portfolio.",
)

# Number of Bootstraps Slider
n_bootstraps = st.sidebar.slider(
    "Number of Bootstraps",
    min_value=25,
    max_value=250,
    step=25,
    value=25,
    help="The number of bootstrap samples used in the portfolio optimization process. "
    "A higher number increases robustness but also computational time.",
)

# Rebalance Period Slider
rebalance_period = st.sidebar.slider(
    "Rebalance Period (months)",
    min_value=3,
    max_value=24,
    step=1,
    value=6,
    help="How often the portfolio should be rebalanced. A shorter period adjusts the portfolio more frequently.",
)

# Calculate percentage changes in asset prices (daily returns)
asset = assets.pct_change(1)

# Filter out specific columns that are not needed in performance calculations
performance = asset.drop(
    [
        "Federal Funds Target Rate - Up",
        "US CPI Urban Consumers YoY NSA",
        "ISM Manufacturing PMI SA",
    ],
    axis=1,
).loc[(asset.index >= start_date) & (asset.index <= end_date)]
performance["portfolio"] = 0  # Initialize a 'portfolio' column with zeros

columns_list = list(performance.columns)
columns_list.remove("portfolio")

# Sidebar: Select columns for your portfolio
selected_columns = st.multiselect(
    "Select columns for your portfolio", 
    options=columns_list,  # All available columns
    default=columns_list   # Default: all columns selected
)

regimes = classify_economic_regime(macro_signal)
economic_regime = prepare_economic_regime_data(regimes, asset)

asset.index = pd.to_datetime(asset.index)

def filter_and_concat_last_row(*dfs):
    renamed_dfs = []
    for df_name, df in dfs:
        # Filter the last row
        last_row = df.tail(1)
        # Rename the index to the DataFrame name
        last_row.index = [df_name]
        renamed_dfs.append(last_row)
    
    # Concatenate all DataFrames on index and transpose the result
    concatenated_df = pd.concat(renamed_dfs)
    return concatenated_df.T

# Applying the function to the signals DataFrames
final_df = filter_and_concat_last_row(
    ('Sentiment', sentiment_signal),
    ('Momentum', momentum_signal),
    ('Value', value_signal)
)

# Create tabs
tabs = st.tabs(["Simulation", "Statistics", "Signals"])

def compute_portfolio_performance(weights, performance):
    # Drop the "portfolio" column from the performance DataFrame if it exists
    if "portfolio" in performance.columns:
        performance = performance.drop(columns=["portfolio"])
    
    # Assert that the columns of portfolio_weights match the columns of performance
    assert list(weights.columns) == list(performance.columns), \
        "Columns of portfolio_weights and performance do not match."

    # Assert that the indexes of portfolio_weights and performance match
    assert list(weights.index) == list(performance.index), \
        "Indexes of portfolio_weights and performance do not match."
    
    # Perform element-wise multiplication
    portfolio_performance = weights * performance
    
    # Add a new column 'portfolio' which is the sum of each row (sum of the weighted performance)
    portfolio_performance["portfolio"] = portfolio_performance.sum(axis=1)

    regime_portfolio_cumulative_returns = (1 + portfolio_performance["portfolio"]).cumprod() * 100
    
    # Return the DataFrame with the 'portfolio' column
    return regime_portfolio_cumulative_returns


# Within the "Statistics" tab
with tabs[1]:
    options = ["sharpe", "var", "returns", "variance"]
    selected_option = st.selectbox("Select a metric to plot", options)
    plot_metrics_by_economic_regime(economic_regime, target_variable=selected_option)

with tabs[2]:
    col1, col2 = st.columns([2, 1])
    with col1:
        # Plot the growth and inflation line chart
        plot_growth_inflation_line(economic_regime)

    with col2:
        # Plot the economic regime pie chart
        plot_economic_regime_pie_chart(economic_regime)

    # Dropdown for column selection
    column_options = sentiment_signal.drop(["ISM Manufacturing PMI SA", "Federal Funds Target Rate - Up", "US CPI Urban Consumers YoY NSA"], axis = 1).columns.tolist()
    # need to remove these columns from the signals (source)
    selected_column = st.selectbox("Select Asset", column_options)
    plot_signals(sentiment_signal, momentum_signal, value_signal, selected_column)

with tabs[0]:
    if st.sidebar.button("Run Simulation"):
        # filter performance by the selected columns
        performance = performance[selected_columns]
        asset = asset[selected_columns]
        value_signal = value_signal[selected_columns]
        momentum_signal = momentum_signal[selected_columns]
        sentiment_signal = sentiment_signal[selected_columns]

        # Initialize dataframes for regime portfolio and overall portfolio, filled with zeros
        regime_portfolio = initialize_portfolio(asset, start_date, selected_columns)
        portfolio = regime_portfolio.copy()

        performance, portfolio_weights, tt = simulate_portfolio_allocation(
            economic_regime,  # Preprocessed economic regime data
            regime_portfolio,  # Initialized regime portfolio
            portfolio,  # Portfolio to simulate
            asset,  # Asset data
            value_signal,  # Value signal data
            momentum_signal,  # Momentum signal data
            sentiment_signal,  # Sentiment signal data
            performance,  # Performance DataFrame
            start_date,  # Simulation start date
            end_date,  # Simulation end date
            arima_order=arima_order,  # ARIMA order from sidebar
            min_weight_bound=min_weight_bound,  # Minimum portfolio weight bound
            max_weight_bound=max_weight_bound,  # Maximum portfolio weight bound
            max_volatility=max_volatility,  # Maximum portfolio volatility
            n_bootstraps=n_bootstraps,  # Number of bootstrap samples
            rebalance_period=rebalance_period,  # Rebalancing period (months)
        )
        dmvo_portfolio_weights = portfolio_weights.copy()
        static_portfolio_weights = dmvo_portfolio_weights.iloc[0]


        static_portfolio_weights = pd.DataFrame([static_portfolio_weights.values] * len(portfolio_weights),
                                        index=portfolio_weights.index,
                                        columns=static_portfolio_weights.index)

        print_performance_metrics(performance)

        # Normalize portfolio_weights to assign 1/num_columns to all values in portfolio_weights
        num_columns = portfolio_weights.shape[1]
        portfolio_weights[:] = 1 / num_columns

        equal_portfolio_returns = compute_portfolio_performance(portfolio_weights, performance)
        static_portfolio_returns = compute_portfolio_performance(static_portfolio_weights, performance)

        portfolio_data = performance["portfolio"]

        regime_portfolio_cumulative_returns = (1 + portfolio_data).cumprod() * 100

        # Set up Streamlit app layout
        st.title("Cumulative Portfolio Returns")
        plot_cumulative_returns(regime_portfolio_cumulative_returns, equal_portfolio_returns, static_portfolio_returns)

        plot_portfolio_weights(dmvo_portfolio_weights)
