import streamlit as st
import numpy as np
import pandas as pd

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