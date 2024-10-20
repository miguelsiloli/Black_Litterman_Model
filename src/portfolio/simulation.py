# Economic regime and forecasting
from portfolio.economic_regime import (
    prepare_current_data,
    determine_economic_regime,
    forecast_economic_signals,
)
from portfolio.forecasting import (
    calculate_growth_inflation,
    update_trend_filters,
    determine_trend_directions,
)

# Portfolio optimization and signals
from portfolio.optimization import perform_portfolio_optimization
from portfolio.utils import (
    update_portfolio_weights,
    calculate_portfolio_performance,
    apply_technical_signals,
)

import pandas as pd
import numpy as np
import warnings
from logger import logger
import streamlit as st

def normalize_portfolio_weights(portfolio_weights):
    total_weight = sum(portfolio_weights.values())  # Sum of all values in the dictionary
    normalized_weights = {key: value / total_weight for key, value in portfolio_weights.items()}
    return normalized_weights

def simulate_portfolio_allocation(
    economic_regime: pd.DataFrame,
    regime_portfolio: pd.DataFrame,
    portfolio: pd.DataFrame,
    asset: pd.DataFrame,
    value_signal: pd.DataFrame,
    momentum_signal: pd.DataFrame,
    sentiment_signal: pd.DataFrame,
    performance: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    arima_order: tuple = (1, 1, 1),  # ARIMA model order for forecasting
    min_weight_bound: float = 0,  # Minimum portfolio weight
    max_weight_bound: float = 0.25,  # Maximum portfolio weight
    max_volatility: float = 0.25,  # Maximum volatility for portfolio optimization
    n_bootstraps: int = 100,  # Number of bootstrap samples for portfolio optimization
    rebalance_period: int = 6,  # Rebalance every 6 months
) -> pd.DataFrame:
    """
    Simulates portfolio allocation over time with periodic rebalancing based on economic regimes
    and technical signals.

    Parameters:
    ----------
    economic_regime : pd.DataFrame
        A DataFrame containing the historical economic regime data (growth, inflation, etc.).
        Each row corresponds to a timestamp, and the columns may include different regime indicators.

    regime_portfolio : pd.DataFrame
        A DataFrame used to store portfolio weights at each rebalance step based on the economic regime.

    portfolio : pd.DataFrame
        A DataFrame used to store overall portfolio allocations, including adjustments from technical signals.

    asset : pd.DataFrame
        A DataFrame containing the asset returns or prices used for portfolio allocation and optimization.

    value_signal : pd.DataFrame
        A DataFrame with value signals, used to adjust portfolio allocations based on valuation indicators.

    momentum_signal : pd.DataFrame
        A DataFrame with momentum signals, used to adjust portfolio allocations based on momentum indicators.

    sentiment_signal : pd.DataFrame
        A DataFrame with sentiment signals, used to adjust portfolio allocations based on market sentiment.

    performance : pd.DataFrame
        A DataFrame for recording the performance of the portfolio across time.

    start_date : pd.Timestamp
        The start date for the simulation (beginning of the period for portfolio rebalancing).

    end_date : pd.Timestamp
        The end date for the simulation (end of the period for portfolio rebalancing).

    arima_order : tuple, optional
        The order of the ARIMA model for forecasting economic signals (default is (1, 1, 1)).
        The tuple represents (p, d, q) values for ARIMA.

    min_weight_bound : float, optional
        The minimum allowed weight for portfolio allocation (default is -0.25, allowing up to 25% short positions).

    max_weight_bound : float, optional
        The maximum allowed weight for portfolio allocation (default is 0.25, restricting allocations to a max of 25%).

    max_volatility : float, optional
        The maximum volatility target for portfolio optimization (default is 0.25).
        This controls the risk tolerance in the portfolio.

    n_bootstraps : int, optional
        The number of bootstrap samples for portfolio optimization (default is 100).
        More bootstraps provide better accuracy but increase computational time.

    rebalance_period : int, optional
        The number of months between portfolio rebalancing (default is 6 months).

    Returns:
    -------
    pd.DataFrame
        The updated `performance` DataFrame with portfolio performance metrics after running the simulation.

    Example:
    --------
    >>> # Sample data for assets, signals, and economic regimes
    >>> import pandas as pd
    >>> asset = pd.DataFrame({'Asset_A': [0.01, 0.02, -0.01],
    >>>                       'Asset_B': [-0.02, 0.03, 0.01]})
    >>> economic_regime = pd.DataFrame({'Growth': [0.02, 0.03, 0.01],
    >>>                                 'Inflation': [0.01, 0.02, 0.03]})
    >>> value_signal = pd.DataFrame({'Asset_A': [1, 1, 1], 'Asset_B': [0, 0, 0]})
    >>> momentum_signal = pd.DataFrame({'Asset_A': [1, 1, 1], 'Asset_B': [1, 0, 0]})
    >>> sentiment_signal = pd.DataFrame({'Asset_A': [0.5, 0.3, 0.2], 'Asset_B': [0.6, 0.5, 0.7]})
    >>> regime_portfolio = asset.copy()  # Initialize regime_portfolio with zeros
    >>> regime_portfolio.iloc[:, :] = 0
    >>> portfolio = regime_portfolio.copy()
    >>> performance = asset.copy()
    >>> performance['portfolio'] = 0

    >>> # Run the simulation from Jan 2020 to Dec 2020
    >>> result = simulate_portfolio_allocation(
    >>>     economic_regime=economic_regime,
    >>>     regime_portfolio=regime_portfolio,
    >>>     portfolio=portfolio,
    >>>     asset=asset,
    >>>     value_signal=value_signal,
    >>>     momentum_signal=momentum_signal,
    >>>     sentiment_signal=sentiment_signal,
    >>>     performance=performance,
    >>>     start_date=pd.Timestamp('2020-01-01'),
    >>>     end_date=pd.Timestamp('2020-12-31'),
    >>>     arima_order=(1, 1, 1),
    >>>     min_weight_bound=-0.25,
    >>>     max_weight_bound=0.25,
    >>>     max_volatility=0.25,
    >>>     n_bootstraps=100,
    >>>     rebalance_period=6
    >>> )
    >>> print(result)
    """

    for df_name, df in zip(['economic_regime', 'regime_portfolio', 'portfolio', 
                             'asset', 'value_signal', 'momentum_signal', 
                             'sentiment_signal', 'performance'], 
                            [economic_regime, regime_portfolio, portfolio, 
                             asset, value_signal, momentum_signal, 
                             sentiment_signal, performance]):
        if df.index.equals(performance.index):
            continue
        else:
            warnings.warn(f"{df_name} does not have the same index as performance. Results may be inconsistent.")


    current_date = start_date
    counter = 1
    last_updated_regime_weights: Dict[str, float] = {}
    last_updated_weights: Dict[str, float] = {}

    portfolio_weights_list = []
    date_list = []

    while current_date <= end_date:
        tt = prepare_current_data(economic_regime, current_date)

        if counter % rebalance_period == 1:  # Rebalance every `rebalance_period` months
            # Forecasting raw economic data using ARIMA
            ariam_forecasting = forecast_economic_signals(tt, arima_order=arima_order)

            # Compute growth and inflation signals
            Growth, Inflation = calculate_growth_inflation(ariam_forecasting)

            # Compute L1 trend filter of growth and inflation signals
            tt = update_trend_filters(tt, Growth, Inflation, ariam_forecasting)

            # Determine trend directions for Growth and Inflation
            Growth_direction, Inflation_direction = determine_trend_directions(tt)

            # Determine the economic regime based on trends
            economic_regime_value = determine_economic_regime(
                Growth_direction, Inflation_direction
            )
            tt["EconomicRegime"].iloc[-1] = economic_regime_value

            # Perform portfolio optimization based on the economic regime
            regime_weights = perform_portfolio_optimization(
                tt,
                asset,
                min_weight_bound=min_weight_bound,
                max_weight_bound=max_weight_bound,
                max_volatility=max_volatility,
                n_bootstraps=n_bootstraps,
            )
            last_updated_regime_weights = regime_weights

            # Apply technical signals to adjust portfolio weights
            portfolio_weights = apply_technical_signals(
                regime_weights,
                current_date,
                asset,
                value_signal,
                momentum_signal,
                sentiment_signal,
            )
            last_updated_weights = portfolio_weights

            # Update portfolio weights based on signals
            portfolio = update_portfolio_weights(
                portfolio, portfolio_weights, current_date
            )

        else:
            # Use the last updated weights if not rebalancing
            regime_weights = last_updated_regime_weights
            portfolio_weights = last_updated_weights

        # need to normalize this 
        # bro why is this giving me such a hassle, this should be handle in the engine
        # by adding addition contraint such as weights.sum(axis=1) == 1
        portfolio_weights = normalize_portfolio_weights(portfolio_weights)

        # Calculate portfolio performance for the current month
        performance, returns = calculate_portfolio_performance(
            performance, tt, portfolio_weights, current_date
        )
        logger.info(returns)
        # Append the portfolio weights and the current date to the respective lists
        portfolio_weights_list.append(portfolio_weights)
        date_list.append(current_date)

        # Move to the next month
        current_date += pd.offsets.MonthBegin(1)
        counter += 1

    weights_df = pd.DataFrame(portfolio_weights_list, index=date_list)
    weights_df = weights_df.div(weights_df.sum(axis=1), axis=0)

    return performance, weights_df, tt


class PortfolioSimulator:
    def __init__(self, economic_regime, regime_portfolio, portfolio, asset, 
                 value_signal, momentum_signal, sentiment_signal, performance, 
                 start_date, end_date, arima_order=(1, 1, 1), min_weight_bound=0, 
                 max_weight_bound=0.25, max_volatility=0.25, n_bootstraps=100, 
                 rebalance_period=6):
        """
        Initialize the PortfolioSimulator with required data and configuration.
        
        Parameters:
        - economic_regime: pd.DataFrame
        - regime_portfolio: pd.DataFrame
        - portfolio: pd.DataFrame
        - asset: pd.DataFrame
        - value_signal: pd.DataFrame
        - momentum_signal: pd.DataFrame
        - sentiment_signal: pd.DataFrame
        - performance: pd.DataFrame
        - start_date: pd.Timestamp
        - end_date: pd.Timestamp
        - arima_order: tuple (default: (1, 1, 1))
        - min_weight_bound: float (default: 0)
        - max_weight_bound: float (default: 0.25)
        - max_volatility: float (default: 0.25)
        - n_bootstraps: int (default: 100)
        - rebalance_period: int (default: 6)
        """
        self.economic_regime = economic_regime
        self.regime_portfolio = regime_portfolio
        self.portfolio = portfolio
        self.asset = asset
        self.value_signal = value_signal
        self.momentum_signal = momentum_signal
        self.sentiment_signal = sentiment_signal
        self.performance = performance
        self.start_date = start_date
        self.end_date = end_date
        self.arima_order = arima_order
        self.min_weight_bound = min_weight_bound
        self.max_weight_bound = max_weight_bound
        self.max_volatility = max_volatility
        self.n_bootstraps = n_bootstraps
        self.rebalance_period = rebalance_period

        # Perform input sanitization
        self._sanitize_inputs()

    def _sanitize_inputs(self):
        """
        Sanitize and validate the inputs based on the conditions specified.
        """
        # Ensure min and max weight bounds are valid
        assert 0 <= self.min_weight_bound <= self.max_weight_bound < 1, \
            "min_weight_bound must be > 0 and less than max_weight_bound, which must be < 1"
        
        # Ensure rebalance_period is greater than 1
        assert self.rebalance_period > 1, "rebalance_period must be greater than 1"
        
        # Ensure asset DataFrame has at least 5 rows (for 5 assets)
        assert len(self.asset.columns) > 5, "The asset DataFrame must have more than 5 assets (columns)."

    def _check_internal_consistency(self):
        """
        Check for internal consistency in the input data, ensuring that indexes match between 
        economic_regime, regime_portfolio, portfolio, asset, value_signal, momentum_signal, 
        sentiment_signal, and performance dataframes.
        """
        for df_name, df in zip(['economic_regime', 'regime_portfolio', 'portfolio', 
                                'asset', 'value_signal', 'momentum_signal', 
                                'sentiment_signal', 'performance'], 
                               [self.economic_regime, self.regime_portfolio, self.portfolio, 
                                self.asset, self.value_signal, self.momentum_signal, 
                                self.sentiment_signal, self.performance]):
            if df.index.equals(self.performance.index):
                continue
            else:
                warnings.warn(f"{df_name} does not have the same index as performance. Results may be inconsistent.")

    def simulate_portfolio_allocation(self):
        """
        Simulate portfolio allocation over time with periodic rebalancing based on economic regimes
        and technical signals.
        
        Returns:
        - pd.DataFrame: Updated `performance` DataFrame with portfolio performance metrics after running the simulation.
        """
        # Run internal consistency check before starting the simulation
        self._check_internal_consistency()

        current_date = self.start_date
        counter = 1
        last_updated_regime_weights = {}
        last_updated_weights = {}
        portfolio_weights_list = []
        date_list = []

        while current_date <= self.end_date:
            tt = prepare_current_data(self.economic_regime, current_date)

            if counter % self.rebalance_period == 1:  # Rebalance every `rebalance_period` months
                # Forecasting raw economic data using ARIMA
                ariam_forecasting = forecast_economic_signals(tt, arima_order=self.arima_order)

                # Compute growth and inflation signals
                Growth, Inflation = calculate_growth_inflation(ariam_forecasting)

                # Compute L1 trend filter of growth and inflation signals
                tt = update_trend_filters(tt, Growth, Inflation, ariam_forecasting)

                # Determine trend directions for Growth and Inflation
                Growth_direction, Inflation_direction = determine_trend_directions(tt)

                # Determine the economic regime based on trends
                economic_regime_value = determine_economic_regime(Growth_direction, Inflation_direction)
                tt["EconomicRegime"].iloc[-1] = economic_regime_value

                # Perform portfolio optimization based on the economic regime
                regime_weights = perform_portfolio_optimization(
                    tt, self.asset, 
                    min_weight_bound=self.min_weight_bound,
                    max_weight_bound=self.max_weight_bound, 
                    max_volatility=self.max_volatility, 
                    n_bootstraps=self.n_bootstraps
                )
                last_updated_regime_weights = regime_weights

                # Apply technical signals to adjust portfolio weights
                portfolio_weights = apply_technical_signals(
                    regime_weights, current_date, self.asset, self.value_signal, 
                    self.momentum_signal, self.sentiment_signal
                )
                last_updated_weights = portfolio_weights

                # Update portfolio weights based on signals
                self.portfolio = update_portfolio_weights(self.portfolio, portfolio_weights, current_date)

            else:
                # Use the last updated weights if not rebalancing
                regime_weights = last_updated_regime_weights
                portfolio_weights = last_updated_weights

            # Normalize portfolio weights
            portfolio_weights = normalize_portfolio_weights(portfolio_weights)

            # Calculate portfolio performance for the current month
            self.performance, returns = calculate_portfolio_performance(
                self.performance, tt, portfolio_weights, current_date
            )
            logger.info(returns)

            # Append the portfolio weights and the current date to the respective lists
            portfolio_weights_list.append(portfolio_weights)
            date_list.append(current_date)

            # Move to the next month
            current_date += pd.offsets.MonthBegin(1)
            counter += 1

        # Create DataFrame of portfolio weights over time
        weights_df = pd.DataFrame(portfolio_weights_list, index=date_list)
        weights_df = weights_df.div(weights_df.sum(axis=1), axis=0)

        return self.performance, weights_df, tt


