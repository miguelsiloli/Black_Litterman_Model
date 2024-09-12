import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from signals.data_interfaces import read_asset_data


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


# move this function out of main app file
def calculate_final_metrics(performance):
    portfolio_data = performance["portfolio"]
    portfolio_return = (1 + portfolio_data.mean()) ** 12 - 1
    portfolio_volatility = portfolio_data.std() * np.sqrt(12)
    portfolio_shape = (portfolio_return - 0.02) / portfolio_volatility
    return portfolio_return, portfolio_volatility, portfolio_shape


def print_performance_metrics(performance):
    portfolio_return, portfolio_volatility, portfolio_shape = calculate_final_metrics(
        performance
    )
    st.write("Return:", portfolio_return)
    st.write("Volatility:", portfolio_volatility)
    st.write("Sharpe:", portfolio_shape)


# Load asset data from Excel
assets = read_asset_data("data/assets.xlsx")

st.set_page_config(layout="wide")

# growth and inflation need to be within -4, 4 because they are standardized
macro_signal = build_macro_signal()
momentum_signal = build_momentum_signal()
value_signal = build_value_signal()
sentiment_signal = build_sentiment_signal()

# Sidebar date pickers for start and end date
st.sidebar.header("Select Date Range")
start_date = st.sidebar.date_input("Start Date", pd.Timestamp("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.Timestamp("2023-12-01"))

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
    min_value=-0.5,
    max_value=0.5,
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
    value=100,
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

start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)

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
).loc[asset.index > start_date]
performance["portfolio"] = 0  # Initialize a 'portfolio' column with zeros

regimes = classify_economic_regime(macro_signal)
economic_regime = prepare_economic_regime_data(regimes, asset)

asset.index = pd.to_datetime(asset.index)

# Initialize dataframes for regime portfolio and overall portfolio, filled with zeros

regime_portfolio = initialize_portfolio(asset, start_date)
portfolio = regime_portfolio.copy()

if st.sidebar.button("Run Simulation"):
    performance = simulate_portfolio_allocation(
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

    portfolio_data = performance["portfolio"]
    regime_portfolio_cumulative_returns = (1 + portfolio_data).cumprod() * 100

    # Set up Streamlit app layout
    st.title("Cumulative Portfolio Returns")

    # Plot cumulative returns using Matplotlib
    fig, ax = plt.subplots()
    ax.plot(regime_portfolio_cumulative_returns, label="Cumulative Returns", color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns")
    ax.set_title("Portfolio Cumulative Returns Over Time")
    ax.legend()

    ax.xaxis.set_major_formatter(DateFormatter("%Y"))

    # Display the plot in Streamlit
    st.pyplot(fig)
