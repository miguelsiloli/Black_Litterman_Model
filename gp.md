# L2 Exercise

| Module                    | Sub-components                                                   | Time  | Notes                           |
|---------------------------|------------------------------------------------------------------|-------|---------------------------------|
| **Macro Module**          | Inflation Component, Growth Component, Economic Regime Component | 5-8h    | Develop inflation and growth pipeline to economic regime. Preprocessing, indexing, z scores  |
| **External Signals Module** | Value Signal Component, Momentum Component, Sentiment Component, Merge | 5-8h    | Integrate various signals from different sources     |
| **Portfolio Optimization Module** | -                                                        | 20-25h    | Optimize the portfolio. Aggregate everything together and compute optimal portfolio as well as optimal portfolio performance.          |

**Best case scenario**: 30h ~ 225 - 50 = 175 €
**Worst case scenario**: 41h ~ 307.5 - 50 = 257.5 €
**Most likely**: I would say around 35ish h. The macro module and external signals are clear to me. The optimization isn't so clear, there is a lot of "cheating" he does to adjust his data to achieve the results he wants. These transforms aren't clear in his thesis, so I assume its cheating :P

**Expectations**: The system should be build and you should be able to reproduce the notebook with your own assets.

- Sentiment and macro algorithm will be fixed
- Limited visualizations (the visualizations should be only to help us trouble shoot), we can build better visualizations after
- Data comes from excel sources
- A lot of tweaking in the optimization will be needed: the data is trimmed using quantiles, expected returns from mean historical values, the weight of each technical indicator, etc. These weights are not clear to me and they seem arbitrary.

**Outputs**: Weights of optimal portfolio; return, volatility and sharpe

Once this is done, we can add/change features at your request incrementally.


```dot
digraph G {
    rankdir=TB;  // Top to Bottom layout

    subgraph cluster_0 {
        label = "Growth Module";
        
        node [style=filled];

        // Input nodes
        cfnai [label="CFNAI", shape=box, color=lightgreen];
        gdp_surprise [label="GDP Surprise", shape=box, color=lightgreen];
        gdp_forecast [label="GDP Forecast (FSPF)", shape=box, color=lightgreen];

        // Operation nodes
        z_scores1 [label="Z Scores", shape=ellipse, color=lightgrey];
        z_scores2 [label="Z Scores", shape=ellipse, color=lightgrey];
        average_pool [label="Average Pool (Σ)", shape=circle, color=lightblue];
        arima_growth [label="ARIMA(1,1,1)", shape=ellipse, color=lightgrey];
        l1_trend_filter [label="L1 Trend Filter", shape=ellipse, color=lightgrey];

        // Output nodes
        growth [label="Growth", shape=box, color=lightcoral];

        // Edges
        cfnai -> z_scores1;
        gdp_surprise -> gdp_forecast;
        gdp_forecast -> z_scores2;
        z_scores1 -> average_pool;
        z_scores2 -> average_pool;
        average_pool -> arima_growth;
        arima_growth -> l1_trend_filter;
        l1_trend_filter -> growth;
    }

    subgraph cluster_1 {
        label = "Inflation Module";
        
        node [style=filled];

        // Input nodes
        cpi [label="US Headline CPI (CPI)", shape=box, color=lightgreen];
        cpi_surprise [label="CPI Surprise (CPI FSPF)", shape=box, color=lightgreen];

        // Operation nodes
        z_scores1_2 [label="Z Scores", shape=ellipse, color=lightgrey];
        z_scores2_2 [label="Z Scores", shape=ellipse, color=lightgrey];
        average_pool_2 [label="Average Pool (Σ)", shape=circle, color=lightblue];
        arima_inflation [label="ARIMA(1,1,1)", shape=ellipse, color=lightgrey];
        l1_trend_filter_2 [label="L1 Trend Filter", shape=ellipse, color=lightgrey];

        // Output nodes
        inflation [label="Inflation", shape=box, color=lightcoral];

        // Edges
        cpi -> z_scores1_2;
        cpi_surprise -> z_scores2_2;
        z_scores1_2 -> average_pool_2;
        z_scores2_2 -> average_pool_2;
        average_pool_2 -> arima_inflation;
        arima_inflation -> l1_trend_filter_2;
        l1_trend_filter_2 -> inflation;
    }

    // New module
    subgraph cluster_2 {
        label = "Economic Regime Module";
        
        node [style=filled];

        // Operation nodes
        diff_transform [label="Diff Transform", shape=ellipse, color=lightgrey];
        logical_map [label="Logical Map", shape=ellipse, color=lightgrey];

        // Output nodes
        economic_regime [label="Economic Regime", shape=box, color=lightcoral];

        // Edges
        growth -> diff_transform;
        inflation -> diff_transform;
        diff_transform -> logical_map;
        logical_map -> economic_regime;
    }
}
```

This diagram now uses:
- Light green for input nodes.
- Light coral for output nodes.
- Light grey ellipses for operations.
- A light blue circle for the Average Pool.

```dot
digraph G {
    rankdir=TB;  // Left to Right layout

    subgraph cluster_0 {
        label = "Value Signal Module";
        
        node [style=filled];

        // Input node
        asset_prices [label="Asset Prices", shape=box, color=lightgreen];

        // Operation nodes
        returns [label="Apply Returns", shape=ellipse, color=lightgrey];
        discretization_filter [label="Discretization Filter", shape=ellipse, color=lightgrey];

        // Output node
        value_signal [label="Value Signal: Asset, Date {-1, 1}", shape=box, color=lightcoral];

        // Edges
        asset_prices -> returns;
        returns -> discretization_filter;
        discretization_filter -> value_signal;
    }

    subgraph cluster_1 {
        label = "Momentum Signal Module";
        
        node [style=filled];

        // Input node
        asset_prices_2 [label="Asset Prices", shape=box, color=lightgreen];

        // Operation nodes
        month_diff [label="Compute 12 Month Diff", shape=ellipse, color=lightgrey];
        binarization [label="Binarization (Positive=1, Negative=-1)", shape=ellipse, color=lightgrey];

        // Output node
        momentum_signal [label="Momentum Signal: Asset, Date {-1, 1}", shape=box, color=lightcoral];

        // Edges
        asset_prices_2 -> month_diff;
        month_diff -> binarization;
        binarization -> momentum_signal;
    }

    subgraph cluster_2 {
        label = "Sentiment Signal Module";
        
        node [style=filled];

        // Input node
        sentiment_data [label="Sentiment Data", shape=box, color=lightgreen];

        // Operation nodes
        bull_bear [label="AII Bull-Bear", shape=ellipse, color=lightgrey];
        gold_yen [label="Gold Yen CFTC", shape=ellipse, color=lightgrey];
        filter_dataframe [label="Filter Dataframe", shape=ellipse, color=lightgrey];
        z_scores [label="Apply Z Scores", shape=ellipse, color=lightgrey];
        pca [label="Apply PCA (n_components=1)", shape=ellipse, color=lightgrey];

        // Output node
        sentiment_signal [label="Sentiment Signal", shape=box, color=lightcoral];

        // Edges
        sentiment_data -> bull_bear;
        sentiment_data -> gold_yen;
        bull_bear -> filter_dataframe;
        gold_yen -> filter_dataframe;
        filter_dataframe -> z_scores;
        z_scores -> pca;
        pca -> sentiment_signal;
    }

    // Main merging section
    subgraph cluster_3 {
        label = "Merge Module";

        node [style=filled];

        // Operation node
        merge_signals [label="Merge Value, Momentum, Sentiment with Asset Dataframe on Datetime Index", shape=ellipse, color=lightgrey];

        // Output node
        merged_dataframe [label="Merged Dataframe", shape=box, color=lightcoral];

        // Edges
        value_signal -> merge_signals;
        momentum_signal -> merge_signals;
        sentiment_signal -> merge_signals;
        merge_signals -> merged_dataframe;
    }
}
```