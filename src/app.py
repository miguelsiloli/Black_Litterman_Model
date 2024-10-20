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
from portfolio.simulation import simulate_portfolio_allocation, PortfolioSimulator

from signals import *
# maybe changing this afterwards
from utils import calculate_final_metrics, print_performance_metrics, filter_and_concat_last_row, compute_portfolio_performance
from layout.plots import plot_growth_inflation_line, plot_metrics_by_economic_regime, plot_portfolio_weights, plot_cumulative_returns, plot_signals, plot_economic_regime_pie_chart
from layout.components.app import render_sidebar
st.set_page_config(layout="wide")

config = render_sidebar()
start_date = config["start_date"]
end_date = config["end_date"]
arima_order = config["arima_order"]
min_weight_bound = config["min_weight_bound"]
max_weight_bound = config["max_weight_bound"]
max_volatility = config["max_volatility"]
n_bootstraps = config["n_bootstraps"]
rebalance_period = config["rebalance_period"]

# Load asset data from Excel
assets = read_asset_data("data/assets.xlsx")
asset = assets.pct_change(1)

# growth and inflation need to be within -4, 4 because they are standardized
macro_signal = build_macro_signal()
momentum_signal = build_momentum_signal()
value_signal = build_value_signal()
sentiment_signal = build_sentiment_signal()

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

# Applying the function to the signals DataFrames
final_df = filter_and_concat_last_row(
    ('Sentiment', sentiment_signal),
    ('Momentum', momentum_signal),
    ('Value', value_signal)
)

# Create tabs
tabs = st.tabs(["Simulation", "Statistics", "Signals"])

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
    plot_signals(sentiment_signal, momentum_signal, value_signal, selected_column, economic_regime["EconomicRegime"])

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

        port = PortfolioSimulator(
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

        performance, portfolio_weights, tt = port.simulate_portfolio_allocation()

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
