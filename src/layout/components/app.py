import streamlit as st
import pandas as pd

def render_sidebar():
    """
    Renders the sidebar for configuration settings.

    Returns:
        dict: A dictionary containing the sidebar selections for start_date, end_date, arima_order,
              min_weight_bound, max_weight_bound, max_volatility, n_bootstraps, and rebalance_period.
    """
    st.sidebar.header("Configuration")
    
    # Date pickers for start and end date
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
        help="The minimum weight that can be assigned to any asset in the portfolio.",
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
        help="The maximum allowed volatility (risk) for the portfolio.",
    )

    # Number of Bootstraps Slider
    n_bootstraps = st.sidebar.slider(
        "Number of Bootstraps",
        min_value=25,
        max_value=250,
        step=25,
        value=25,
        help="The number of bootstrap samples used in the portfolio optimization process.",
    )

    # Rebalance Period Slider
    rebalance_period = st.sidebar.slider(
        "Rebalance Period (months)",
        min_value=1,
        max_value=24,
        step=1,
        value=6,
        help="How often the portfolio should be rebalanced.",
    )

    # Return all values in a dictionary
    return {
        "start_date": start_date,
        "end_date": end_date,
        "arima_order": arima_order,
        "min_weight_bound": min_weight_bound,
        "max_weight_bound": max_weight_bound,
        "max_volatility": max_volatility,
        "n_bootstraps": n_bootstraps,
        "rebalance_period": rebalance_period
    }