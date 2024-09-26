import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import plotly.subplots as sp

def plot_signals(sentiment_signal, momentum_signal, value_signal, selected_column):
    """
    This function takes in three signals (sentiment, momentum, value) and creates subplots
    to visualize these signals for a selected asset or column.

    Parameters:
    sentiment_signal (pd.DataFrame): DataFrame containing sentiment signal values over time.
    momentum_signal (pd.DataFrame): DataFrame containing momentum signal values over time.
    value_signal (pd.DataFrame): DataFrame containing value signal values over time.
    selected_column (str): The name of the column/asset to visualize from each signal DataFrame.

    Returns:
    None: Displays the plot in Streamlit.
    """

    # Create subplots for 3 columns
    fig = sp.make_subplots(rows=1, cols=3, subplot_titles=("Sentiment Signal", "Momentum Signal", "Value Signal"))

    # Sentiment signal plot
    fig.add_trace(
        go.Scatter(x=sentiment_signal.index, y=sentiment_signal[selected_column], mode='lines', name='Sentiment'),
        row=1, col=1
    )

    # Momentum signal plot
    fig.add_trace(
        go.Scatter(x=momentum_signal.index, y=momentum_signal[selected_column], mode='lines', name='Momentum', line=dict(color='green')),
        row=1, col=2
    )

    # Value signal plot
    fig.add_trace(
        go.Scatter(x=value_signal.index, y=value_signal[selected_column], mode='lines', name='Value', line=dict(color='red')),
        row=1, col=3
    )

    # Update layout
    fig.update_layout(height=400, width=900, title_text="Signals for Selected Asset", showlegend=False)

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

import plotly.express as px
import streamlit as st

def plot_cumulative_returns(regime_portfolio_cumulative_returns, equal_portfolio_cum_returns, static_mvo_returns):
    """
    This function takes three DataFrames or Series of cumulative returns (regime-based, equal portfolio, and static MVO)
    and plots them over time using Plotly.

    Parameters:
    regime_portfolio_cumulative_returns (pd.Series or pd.DataFrame): A Series or DataFrame where the index 
                                                                     represents dates and the values are 
                                                                     regime portfolio cumulative returns.
    equal_portfolio_cum_returns (pd.Series or pd.DataFrame): A Series or DataFrame where the index 
                                                             represents dates and the values are 
                                                             equal portfolio cumulative returns.
    static_mvo_returns (pd.Series or pd.DataFrame): A Series or DataFrame where the index represents dates and 
                                                    the values are static MVO portfolio cumulative returns.

    Returns:
    None: Displays the plot in Streamlit.
    """
    # Ensure all inputs are DataFrames
    if isinstance(regime_portfolio_cumulative_returns, pd.Series):
        regime_portfolio_cumulative_returns = regime_portfolio_cumulative_returns.to_frame(name="Regime Portfolio")
    
    if isinstance(equal_portfolio_cum_returns, pd.Series):
        equal_portfolio_cum_returns = equal_portfolio_cum_returns.to_frame(name="Equal Portfolio")
    
    if isinstance(static_mvo_returns, pd.Series):
        static_mvo_returns = static_mvo_returns.to_frame(name="Static MVO Portfolio")

    # Combine all DataFrames for plotting
    combined_returns = pd.concat([regime_portfolio_cumulative_returns, 
                                  equal_portfolio_cum_returns, 
                                  static_mvo_returns], axis=1)

    # Create the Plotly line chart
    fig = px.line(combined_returns,
                  x=combined_returns.index,
                  y=combined_returns.columns,
                  title="Portfolio Cumulative Returns Over Time",
                  labels={"x": "Date", "y": "Cumulative Returns"},
                  template="plotly_dark")
    
    # Customize layout if needed
    fig.update_layout(xaxis_title="Date", yaxis_title="Cumulative Returns")

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def plot_portfolio_weights(portfolio_weights):
    """
    This function takes in a DataFrame of portfolio weights and creates a stacked bar chart
    to visualize the weights of each asset over time.

    Parameters:
    portfolio_weights (pd.DataFrame): A DataFrame where columns represent different assets, 
                                      and the index represents dates.

    Returns:
    None: Displays the plot in Streamlit.
    """
    
    # Create a list to store the traces (one trace per asset/column)
    traces = []

    # Get the index (dates) for the X-axis
    x = portfolio_weights.index

    # Loop through each column in the DataFrame to create a trace for each asset
    for column in portfolio_weights.columns:
        traces.append(go.Bar(
            x=x,
            y=portfolio_weights[column],
            name=column
        ))

    # Create the figure with stacked bar mode
    fig = go.Figure(data=traces)

    # Set the layout for the chart
    fig.update_layout(
        barmode='stack',  # Stacks the bars
        title='Stacked Histogram of Portfolio Weights Over Time',
        xaxis_title='Date',
        yaxis_title='Weights',
        xaxis=dict(tickformat='%Y-%m-%d'),  # Format dates on X-axis
        legend_title='Assets',
        template='plotly_white'  # Set the theme to white for better visuals
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def plot_growth_inflation_line(df):
    # Ensure the required columns exist
    required_columns = ["Growth", "Inflation", "EconomicRegime"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"Missing columns: {', '.join(missing_columns)} in the DataFrame.")
        return

    # Create the line plot using Plotly
    fig = go.Figure()

    # Plot Growth
    fig.add_trace(go.Scatter(x=df.index, y=df["Growth"], mode='lines', name='Growth'))

    # Plot Inflation
    fig.add_trace(go.Scatter(x=df.index, y=df["Inflation"], mode='lines', name='Inflation'))

    # Get y-axis limits dynamically
    y_min = min(df["Growth"].min(), df["Inflation"].min())
    y_max = max(df["Growth"].max(), df["Inflation"].max())

    # Assign distinct colors for each unique economic regime
    economic_regimes = df["EconomicRegime"].unique()
    colors = ['rgba(102, 194, 165, 0.8)', 'rgba(252, 141, 98, 0.8)', 'rgba(141, 160, 203, 0.8)',
              'rgba(231, 138, 195, 0.8)', 'rgba(166, 216, 84, 0.8)', 'rgba(255, 217, 47, 0.8)']  # Extended as needed

    # Track which regimes have been added to the legend
    regimes_in_legend = set()

    # Loop through contiguous blocks of each economic regime
    previous_regime = None
    regime_start = None

    for i in range(len(df)):
        current_regime = df["EconomicRegime"].iloc[i]

        # If the regime changes or it's the first entry
        if current_regime != previous_regime:
            if regime_start is not None:
                # We have finished a contiguous block, mark the regime area
                regime_end = df.index[i]  # Now end at the start of the next regime

                # Add colored background for the previous economic regime
                color = colors[economic_regimes.tolist().index(previous_regime) % len(colors)]
                fig.add_shape(
                    type="rect",
                    xref="x", yref="paper",
                    x0=regime_start, x1=regime_end,
                    y0=0, y1=1,  # This covers from bottom (y0=0) to top (y1=1) of the plot
                    fillcolor=color,
                    opacity=0.8,
                    layer="below",  # Ensure that the shaded regions are below the lines
                    line_width=0,
                )

                # Add regime label in the legend only once per regime
                if previous_regime not in regimes_in_legend:
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(size=10, color=color),
                        legendgroup=str(previous_regime),
                        showlegend=True,
                        name=f'Economic Regime: {previous_regime}'
                    ))
                    regimes_in_legend.add(previous_regime)

            # Start a new contiguous block
            regime_start = df.index[i]  # Start the new regime at the current index
            previous_regime = current_regime

    # Handle the last contiguous block
    if regime_start is not None:
        regime_end = df.index[-1]  # End at the last index of the DataFrame
        color = colors[economic_regimes.tolist().index(previous_regime) % len(colors)]
        fig.add_shape(
            type="rect",
            xref="x", yref="paper",
            x0=regime_start, x1=regime_end,
            y0=0, y1=1,
            fillcolor=color,
            opacity=0.8,
            layer="below",
            line_width=0,
        )

        # Add regime label for the last block if it hasn't been added already
        if previous_regime not in regimes_in_legend:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                legendgroup=str(previous_regime),
                showlegend=True,
                name=f'Economic Regime: {previous_regime}'
            ))
            regimes_in_legend.add(previous_regime)

    # Set plot title, axis labels, and inflation-specific y-axis limits
    fig.update_layout(
        title="Growth and Inflation Over Time",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Metrics",
        yaxis=dict(
            title="Growth and Inflation",
            range=[-4, 4],  # Set the y-axis range to [-4, y_max]  # Dynamically set the y-axis range based on data
        )
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def plot_economic_regime_pie_chart(df):
    """
    Create a pie chart of economic regimes based on the percentage of periods they were active.
    
    Args:
    - df (pd.DataFrame): DataFrame containing the "EconomicRegime" column.
    
    Returns:
    - Pie chart showing the percentage of periods for each economic regime.
    """
    # Ensure the required column exists
    if "EconomicRegime" not in df.columns:
        st.error("The DataFrame does not contain the 'EconomicRegime' column.")
        return
    
    # Count the number of occurrences for each economic regime
    regime_counts = df["EconomicRegime"].value_counts()

    # Calculate percentage for each regime
    regime_percentages = (regime_counts / regime_counts.sum()) * 100

    # Create the pie chart using Plotly
    fig = go.Figure(data=[go.Pie(
        labels=regime_percentages.index,  # Economic regime labels
        values=regime_percentages.values,  # Corresponding percentage values
        hoverinfo='label+percent',  # Show label and percentage on hover
        textinfo='label+percent',  # Display label and percentage on the pie chart
        marker=dict(colors=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f'])  # Custom colors
    )])

    # Update the layout of the pie chart
    fig.update_layout(
        title="Economic Regimes by Percentage of Periods",
    )

    # Display the pie chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def plot_metrics_by_economic_regime(df, target_variable):
    # Drop the specified columns
    columns_to_drop = ["CFNAI", "NGDP1", "CPIAUCSL", "CPI1", "US CPI Urban Consumers YoY NSA", 
                       "ISM Manufacturing PMI SA", "Federal Funds Target Rate - Up", "Growth", "Inflation"]
    df = df.drop(columns=columns_to_drop, errors='ignore', axis=1)

    # Ensure 'EconomicRegime' column exists
    if 'EconomicRegime' not in df.columns:
        st.error("The column 'EconomicRegime' is not in the DataFrame.")
        return

    # Determine which metric to plot based on the target_variable
    if target_variable == "returns":
        # Group by 'EconomicRegime' and calculate the mean for each regime (Average Returns)
        mean_returns = df.groupby('EconomicRegime').mean()
        metric_data = mean_returns
        y_label = "Average Return"
        plot_title = "Average Returns by Economic Regime"

    elif target_variable == "var":
        # Group by 'EconomicRegime' and calculate the 5th percentile (Value at Risk)
        var_5_percent = df.groupby('EconomicRegime').quantile(0.05)
        metric_data = var_5_percent
        y_label = "Value at Risk (5% Tail)"
        plot_title = "VaR (5% Tail) by Economic Regime"

    elif target_variable == "sharpe":
        # Calculate Sharpe Ratio (Mean Return / Std Dev)
        mean_returns = df.groupby('EconomicRegime').mean()
        std_devs = df.groupby('EconomicRegime').std()
        sharpe_ratios = mean_returns / std_devs
        metric_data = sharpe_ratios
        y_label = "Sharpe Ratio"
        plot_title = "Sharpe Ratio by Economic Regime"

    elif target_variable == "variance":
        # Group by 'EconomicRegime' and calculate the variance for each regime
        variance = df.groupby('EconomicRegime').var()
        metric_data = variance
        y_label = "Variance"
        plot_title = "Variance by Economic Regime"

    else:
        st.error("Invalid target_variable. Choose from 'sharpe', 'var', 'returns', 'variance'.")
        return

    # Create Streamlit columns based on the number of unique regimes
    regimes = metric_data.index
    cols = st.columns(len(regimes))

    # Iterate over each regime and create a horizontal bar chart for the selected metric
    for i, regime in enumerate(regimes):
        regime_data = metric_data.loc[regime]

        # Create the horizontal bar chart using Plotly
        fig = px.bar(
            regime_data,
            orientation='h',
            labels={'index': 'Metrics', 'value': y_label},
            title=f'{plot_title}: {regime}'
        )

        # Add a non-continuous line (dashed) to represent the mean value
        mean_value = regime_data.mean()
        fig.add_shape(
            type="line",
            x0=mean_value, x1=mean_value,  # The vertical line is based on the mean value
            y0=0, y1=len(regime_data)-17,     # Line spans the entire height of the chart
            line=dict(
                color="Red",
                width=1,
                dash="dash"  # Dashed line for non-continuous effect
            ),
            xref="x",  # Referencing the x-axis (for the mean value)
            yref="paper"  # Referencing the y-axis, spanning the height
        )

        # Add annotation to show the mean value
        fig.add_annotation(
            x=mean_value,
            y=1.1,  # Slightly above the chart
            xref="x",
            yref="paper",
            text=f"Mean: {mean_value:.2f}",
            showarrow=False,
            font=dict(color="Red")
        )

        # Plot each chart in the respective Streamlit column
        cols[i].plotly_chart(fig, use_container_width=True)