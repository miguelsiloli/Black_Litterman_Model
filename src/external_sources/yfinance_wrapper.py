import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import List, Union, Optional


class StockDataFetcher:
    """
    A simple wrapper for yfinance to fetch stock data for multiple tickers.
    """
    
    def __init__(self):
        """Initialize the StockDataFetcher."""
        self.available_price_types = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    def get_stock_data(
        self,
        tickers: List[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        price_type: str = 'Close',
        interval: str = '1mo'
    ) -> pd.DataFrame:
        """
        Fetch stock data for a list of tickers.
        
        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
        start_date : str or datetime, optional
            Start date for data retrieval (format: 'YYYY-MM-DD' or datetime object)
        end_date : str or datetime, optional
            End date for data retrieval (format: 'YYYY-MM-DD' or datetime object)
        price_type : str, default 'Close'
            Type of price data to return. Options: 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
        interval : str, default '1d'
            Data interval. Options: '1d', '1wk', '1mo', etc.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with dates as index (formatted as MM/DD/YYYY) and tickers as columns
        """
        # Validate price_type
        if price_type not in self.available_price_types:
            raise ValueError(f"Invalid price_type. Must be one of {self.available_price_types}")
        
        # Download data for all tickers
        data = yf.download(
            tickers=' '.join(tickers),
            start=start_date,
            end=end_date,
            interval=interval,
            group_by='ticker',
            auto_adjust=False,
            progress=False
        )
        
        # Handle case when only one ticker is provided
        if len(tickers) == 1:
            ticker = tickers[0]
            # Convert from multi-level columns to single level
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(0)
            # Extract the requested price type
            result = data[price_type].to_frame(name=ticker)
        else:
            # For multiple tickers, extract the requested price type for each ticker
            result = pd.DataFrame()
            for ticker in tickers:
                if ticker in data.columns.levels[0]:
                    result[ticker] = data[ticker][price_type]
        
        # Format the date index to MM/DD/YYYY
        result.index = result.index.strftime('%m/%d/%Y')
        
        return result


# Example usage:
# if __name__ == "__main__":
#     fetcher = StockDataFetcher()
    
#     # Get closing prices for Apple, Microsoft, and Google for the last year
#     df = fetcher.get_stock_data(
#         tickers=['AAPL', 'MSFT', 'GOOGL'],
#         start_date='1990-01-01',
#         end_date='2024-12-31',
#         price_type='Close'
#     )
    
#     print(df.head())
    
#     # Get opening prices for just Apple
#     apple_open = fetcher.get_stock_data(
#         tickers=['AAPL'],
#         start_date='1990-01-01',
#         end_date='2024-12-31',
#         price_type='Open'
#     )
    
#     print(apple_open.head())