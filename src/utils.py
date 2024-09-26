import streamlit as st
import numpy as np

# move this function out of main app file
def calculate_final_metrics(performance):
    portfolio_data = performance["portfolio"]
    portfolio_return = (1 + portfolio_data.mean()) ** 12 - 1
    portfolio_volatility = portfolio_data.std() * np.sqrt(12)
    portfolio_shape = (portfolio_return - 0.02) / portfolio_volatility
    return portfolio_return, portfolio_volatility, portfolio_shape


def print_performance_metrics(performance):
    # Assuming the function calculate_final_metrics returns (return, volatility, sharpe)
    portfolio_return, portfolio_volatility, portfolio_sharpe = calculate_final_metrics(performance)

    # Convert to percentages and format to two decimal places
    portfolio_return_pct = f"{portfolio_return * 100:.2f}%"
    portfolio_volatility_pct = f"{portfolio_volatility * 100:.2f}%"
    portfolio_sharpe_ratio = f"{portfolio_sharpe:.2f}"

    # Create 3 columns that span the page width
    col1, col2, col3 = st.columns(3)

    # Display metrics side by side
    with col1:
        st.metric(label="Return", value=portfolio_return_pct)
    with col2:
        st.metric(label="Volatility", value=portfolio_volatility_pct)
    with col3:
        st.metric(label="Sharpe Ratio", value=portfolio_sharpe_ratio)