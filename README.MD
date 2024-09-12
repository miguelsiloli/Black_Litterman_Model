# Directory Structure: BLACK_LITTERMAN_MODEL

**Important**: We need to build a basic UI and data connectors so we can add new assets. Once this is done, we have a fully functional app.

```plaintext
src/
├── data/
│   ├── macro_raw/
│   ├── sentiment/
│   ├── assets.xlsx
├── layout/
│   ├── __init__.py
├── portfolio/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── economic_regime.py
│   ├── forecasting.py
│   ├── optimization.py
│   ├── simulation.py
│   ├── utils.py
├── signals/
│   ├── __init__.py
│   ├── data_interfaces.py
│   ├── signal_builder.py
│   ├── statistics.py
│   ├── steps.py
│   ├── utils.py
├── app.py
├── debug.log
```

- **Data**: It shall contain every data required for the exercise.
- **Layout**: It shall contain all the functions needed for UI
- **Portfolio**: It is the backbone of the project. Contains all the logic and maths behind portfolio optimization, simulation and dynamic regime switching optimization.
- **Signals**: It is the module which computes momentum, sentiment, value, etc signals from raw data.


| **Module**                  | **Description**                                                                 |
|-----------------------------|---------------------------------------------------------------------------------|
| **data_processing.py**       | Contains functions to clean and preprocess raw portfolio data.           |
| **economic_regime.py**       | Handles classification of economic regimes (e.g., recession, expansion). |
| **forecasting.py**           | Contains forecasting functions, possibly for asset returns or economic metrics. |
| **optimization.py**          | Includes portfolio optimization algorithms (e.g., pyopt).       |
| **simulation.py**            | Manages the portfolio simulation over time, applying rebalancing and signals.   |
| **utils.py** (in portfolio)  | Utility functions for the portfolio management modules (helper functions).      |
| **data_interfaces.py**       | Data ingestion from different sources to feed into signal models.|
| **signal_builder.py**        | Builds signals (momentum, value, etc.) for guiding portfolio decisions.         |
| **statistics.py**            | Statistical primitives to calculate metrics for assets, signals, and portfolios. |
| **steps.py**                 | Likely defines steps for processing signals, such as momentum or value steps.   |
| **utils.py** (in signals)    | Utility functions for the signal modules (helper functions for signal processing).| 


# Algorithm

## Signals

There are 4 main signals.

### Macro signal

1. Reads data for the Chicago Fed National Activity Index (CFNAI) and GDP forecasts.
2. Computes the percentage change for the GDP forecast data.
3. Reads CPI and CPI forecast data.
4. Standardizes (z-scores) the combined GDP and CPI data.
5. Computes signals for GDP growth and inflation using average pooling.
6. Concatenates the results into a single DataFrame for the macroeconomic signal.

### Sentiment signal

1. Reads sentiment data and computes the AAII Bull-Bear spread.
2. Standardizes the sentiment data using z-scores.
3. Performs PCA to reduce the sentiment data to a single dimension.
4. Computes the percentage change in asset data over a 3-month period.
5. Calculates the correlation between sentiment signals and asset returns.
6. Maps the sentiment signal to asset returns based on the calculated correlations.

### Value signal

1. Reads asset data from an Excel file.
2. Computes the percentage change over a 6-month period for each asset.
3. Standardizes the data using z-scores.
4. Applies a signal:
  - -1 if the z-score is greater than 1 (overvalued).
  - 1 if the z-score is less than -1 (undervalued).
  - 0 otherwise.

### Momentum signal

1. Reads asset data from an Excel file.
2. Calculates the 12-month momentum for each asset.
3. Applies a signal where:
  - 1 indicates positive momentum (asset price increased).
  - -1 indicates negative momentum (asset price decreased).
  - NaN for unchanged values.


## Dynamic Regime Switching Algorithm

1. Filter returns based on the current economic regime.
2. Initialize bootstrapping parameters.
3. For each bootstrap iteration:
   a. Sample returns with replacement to create a new dataset.
   b. Trim the sample returns to avoid outliers.
   c. Replace outliers with linear interpolation.
   d. Compute mean historical returns (annualized).
   e. Clip the returns to avoid extreme values.
   f. Estimate the covariance matrix using the Ledoit-Wolf shrinkage method.
   g. Perform Mean-Variance Optimization to compute the portfolio weights.
   h. Store the cleaned weights.
4. Compute the average weights across all bootstrap samples.
5. Return the average portfolio weights as a dictionary.


## Considerations about the model

### Assumptions

- Normality of returns and reliance on historical data.
- Risk-aversion and absence of transaction costs.
- Well-conditioned covariance matrix and stability via Ledoit-Wolf shrinkage.
- We assume a static covariance matrix
- Accurate outlier handling through trimming and interpolation. (might introduce bias)


### Possible optimizations

- Compute other metrics along expected returns such as VaR 
- Incorporate a time varying covariance matrix. 
- Incorporate new signals



