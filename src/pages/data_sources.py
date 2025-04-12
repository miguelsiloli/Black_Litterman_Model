# pages/data_sources.py
import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
from external_sources.yfinance_wrapper import StockDataFetcher

def show_data_sources_page():
    st.title("External Data Sources")
    
    # Initialize the fetcher
    fetcher = StockDataFetcher()
    
    # Check for existing data file
    existing_tickers = []
    if os.path.exists('data/assets_yfinance.xlsx'):
        try:
            existing_df = pd.read_excel('data/assets_yfinance.xlsx', index_col=0)
            existing_tickers = existing_df.columns.tolist()
            st.info(f"Found existing external data with tickers: {', '.join(existing_tickers)}")
        except Exception as e:
            st.error(f"Error reading existing file: {e}")
    
    # UI for configuring data source
    with st.form("data_source_form"):
        # Ticker input
        ticker_input = st.text_input("Enter ticker symbols (comma-separated)", 
                                     value=','.join(existing_tickers) if existing_tickers else "")
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime(1990, 1, 1))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        # Price type
        price_type = st.selectbox("Price Type", options=fetcher.available_price_types)
        
        # Interval
        # interval = st.selectbox("Interval", options=['1d', '1wk', '1mo'])
        interval = '1mo'
        
        # Submit button
        submitted = st.form_submit_button("Fetch Data")
    
    if submitted:
        tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
        
        if not tickers:
            st.warning("Please enter at least one ticker symbol")
        else:
            try:
                with st.spinner(f"Fetching data for {len(tickers)} ticker(s)..."):
                    # Get data
                    result = fetcher.get_stock_data(
                        tickers=tickers,
                        start_date=start_date,
                        end_date=end_date,
                        price_type=price_type,
                        interval=interval
                    )
                    
                    # Preview data
                    st.subheader("Data Preview")
                    st.dataframe(result.head(10))
                    
                    # Automatically save the data
                    os.makedirs('data', exist_ok=True)
                    # Convert index from MM/DD/YYYY to datetime for compatibility
                    result.index = pd.to_datetime(result.index)
                    result.to_excel('data/assets_yfinance.xlsx')
                    st.success("Data saved successfully to data/assets_yfinance.xlsx!")
                    
                    # Add info about integration
                    st.info("This data will be automatically combined with your existing asset data when you run the simulation.")
                    
            except Exception as e:
                st.error(f"Error fetching or saving data: {e}")

if __name__ == "__main__":
    show_data_sources_page()