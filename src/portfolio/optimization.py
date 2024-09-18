import numpy as np
from portfolio.data_processing import trim_data, replace_outliers_with_interpolation
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier


def perform_portfolio_optimization(
    tt,
    asset,
    min_weight_bound=-0.25,
    max_weight_bound=0.25,
    max_volatility=0.12,
    n_bootstraps=100,
):
    """
    Perform portfolio optimization using Mean-Variance Optimization (MVO) with resampling.

    This function optimizes the portfolio weights for a given economic regime. It first
    filters the asset returns based on the current economic regime, then applies a bootstrapping
    method to create multiple samples of returns. For each bootstrap sample, it computes the
    optimal portfolio weights using Efficient Frontier. The final portfolio weights are averaged
    across all bootstrap samples.

    Parameters:
    - tt (pd.DataFrame): A DataFrame containing the asset returns and economic regime data.
    - asset (pd.DataFrame): A DataFrame containing the original asset data used for creating
                            the asset list in the final portfolio.
    - min_weight_bound (float): The minimum allowed weight for any asset in the portfolio. Default is -0.25.
    - max_weight_bound (float): The maximum allowed weight for any asset in the portfolio. Default is 0.25.

    Returns:
    - dict: A dictionary mapping asset names to their optimized portfolio weights.

    Steps:
    1. Filter returns based on the current economic regime.
    2. Drop columns that are not needed for portfolio optimization.
    3. Initialize bootstrapping parameters.
    4. For each bootstrap iteration:
       a. Sample returns with replacement to create a new dataset.
       b. Trim the sample returns to avoid outliers.
       c. Replace outliers with linear interpolation.
       d. Compute mean historical returns (annualized).
       e. Clip the returns to avoid extreme values.
       f. Estimate the covariance matrix using the Ledoit-Wolf shrinkage method.
       g. Perform Mean-Variance Optimization to compute the portfolio weights.
       h. Store the cleaned weights.
    5. Compute the average weights across all bootstrap samples.
    6. Return the average portfolio weights as a dictionary.

    Example:
    ----------
    >>> # Assuming 'tt' is a DataFrame containing historical returns and economic regimes
    >>> # and 'asset' is a DataFrame with asset names and their corresponding returns.
    >>> result = perform_portfolio_optimization(tt, asset)

    Example Input:
    --------------
    tt = pd.DataFrame({
        'Asset_A': [0.01, 0.02, -0.01, 0.03, 0.04],
        'Asset_B': [-0.02, 0.01, 0.03, -0.01, 0.02],
        'EconomicRegime': [1, 1, 2, 2, 1]
    })

    asset = pd.DataFrame({
        'Asset_A': [0.01, 0.02, -0.01, 0.03, 0.04],
        'Asset_B': [-0.02, 0.01, 0.03, -0.01, 0.02]
    })

    Example Output:
    ---------------
    {
        'Asset_A': 0.15,
        'Asset_B': -0.10
    }

    """
    regime_returns = tt[tt["EconomicRegime"] == tt["EconomicRegime"][-1]]
    regime_returns = regime_returns.drop(
        [
            "ISM Manufacturing PMI SA",
            "US CPI Urban Consumers YoY NSA",
            "Federal Funds Target Rate - Up",
            "Growth_filtered",
            "Inflation_filtered",
            "CFNAI",
            "NGDP1",
            "CPIAUCSL",
            "CPI1",
            "Growth",
            "Inflation",
            "EconomicRegime",
        ],
        axis=1,
    )

    n_assets = len(regime_returns.columns)
    bootstrapped_weights = np.zeros((n_bootstraps, n_assets))

    for i in range(n_bootstraps):
        """
        This code snippet creates multiple samples of returns with replacement based on the past performance
        of the current economic regime. The resulting samples are trimmed to avoid extreme values. Extreme values
        are known to skew calculations or create exploding/vanishing values in optimization gradients (when used).

        <<About Covariance Shrinkage>>

        It is subject to estimation error of the kind most likely to perturb a mean-variance optimizer.
        Instead, a matrix can be obtained from the sample covariance matrix through a transformation called shrinkage.
        This tends to pull the most extreme coefficients toward more central values,
        systematically reducing estimation error when it matters most.
        Statistically, the challenge is to know the optimal shrinkage intensity.
        """
        valid_sample = False

        while not valid_sample:
            # Step 1: Bootstrap sample with replacement
            sample_returns = regime_returns.sample(n=len(regime_returns), replace=True)

            # Step 2: Trim the sample returns to avoid extreme values
            trimmed_sample_returns = trim_data(sample_returns)
            trimmed_sample_returns = replace_outliers_with_interpolation(
                trimmed_sample_returns
            )

            # Step 3: Check if variance is non-zero for all columns
            if not (trimmed_sample_returns.var() == 0).any():
                valid_sample = True  # Proceed only if variance is non-zero
            else:
                print(f"Iteration {i}: Zero variance detected, resampling...")

        mu = mean_historical_return(
            trimmed_sample_returns, returns_data=True, frequency=12
        )
        mu = mu.clip(lower=mu.quantile(0.01), upper=mu.quantile(0.99))

        # cov matrix
        Sigma = CovarianceShrinkage(
            trimmed_sample_returns, returns_data=True, frequency=12
        ).ledoit_wolf()

        ef = EfficientFrontier(
            mu, Sigma, solver="SCS", weight_bounds=(min_weight_bound, max_weight_bound)
        )
        ef.add_constraint(lambda w: w.sum() == 1)
        ef.efficient_risk(max_volatility, market_neutral=False)
        cleaned_weights = ef.clean_weights()
        bootstrapped_weights[i, :] = np.array(list(cleaned_weights.values()))
        # total_weight_sum = np.sum(list(cleaned_weights.values()))
        # print("Total weight sum:", total_weight_sum)

    average_weights = np.mean(bootstrapped_weights, axis=0)
    # print(np.sum(average_weights, axis=0))

    return dict(zip(asset.columns, average_weights))
