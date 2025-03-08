import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import glob
import re
import traceback

# File paths
DATA_DIR = "data"
SECTORS_FILE = os.path.join(DATA_DIR, "stock_sectors.csv")
DAILY_DATA_DIR = os.path.join(DATA_DIR, "daily_data")
MARKET_CAP_FILE = os.path.join(DATA_DIR, "market_caps.csv")

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DAILY_DATA_DIR, exist_ok=True)

def process_historical_data(filepath):
    """Process multi-day historical data file and properly handle date formats"""
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Convert date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Drop rows with invalid dates
        invalid_dates = df['Date'].isna().sum()
        if invalid_dates > 0:
            st.warning(f"Removed {invalid_dates} rows with invalid dates")
            df = df.dropna(subset=['Date'])
        
        # Get unique symbols and dates
        unique_symbols = df['symbol'].nunique()
        unique_dates = df['Date'].nunique()
        
        st.info(f"Historical data contains {len(df)} rows for {unique_symbols} stocks over {unique_dates} dates")
        
        # Handle symbol as string
        df['symbol'] = df['symbol'].astype(str)
        
        return df
    except Exception as e:
        st.error(f"Error processing historical data: {str(e)}")
        st.code(traceback.format_exc())
        return None

# Function to load sectors data
def load_sectors_data():
    """Load the stock sectors data directly from the included file"""
    try:
        # Simply load from the included file
        sectors_file = os.path.join(DATA_DIR, "stock_sectors.csv")
        df = pd.read_csv(sectors_file)
        # Remove any rows with missing essential data
        df = df.dropna(subset=['symbol', 'sector', 'industry'])
        return df
    except Exception as e:
        st.error(f"Error loading sectors data: {str(e)}")
        return None

# Function to load market caps
def load_market_caps():
    """Load the latest market cap data and handle scientific notation properly"""
    try:
        if os.path.exists(MARKET_CAP_FILE):
            df = pd.read_csv(MARKET_CAP_FILE)
            
            # Handle scientific notation in market_cap column
            if 'market_cap' in df.columns:
                # Ensure market_cap is numeric (handling any string formatting issues)
                df['market_cap'] = pd.to_numeric(df['market_cap'], errors='coerce')
                
            # Report number of stocks with market cap data
            st.info(f"Loaded market cap data for {len(df)} stocks")
            return df
        else:
            st.warning("Market cap data file not found. Will use market cap data from sectors file.")
            return None
    except Exception as e:
        st.error(f"Error loading market cap data: {str(e)}")
        st.code(traceback.format_exc())
        return None

# Get available dates from historical data files
def get_available_dates():
    """Get list of dates with available data files"""
    try:
        files = glob.glob(os.path.join(DAILY_DATA_DIR, "historical_data_*.csv"))
        dates = []
        
        for file in files:
            # Extract date from filename
            match = re.search(r'historical_data_(\d{4}-\d{2}-\d{2})\.csv', file)
            if match:
                dates.append(match.group(1))
                
        return sorted(dates, reverse=True)
    except Exception as e:
        st.error(f"Error getting available dates: {str(e)}")
        return []

def calculate_money_flow_from_multiday(historical_data, sectors_df, market_caps, start_date, end_date, period_name):
    """
    Calculate money flow using multi-day historical data and handle market cap data correctly
    """
    try:
        # Convert dates to datetime objects
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Get available dates in the data
        available_dates = sorted(historical_data['Date'].unique())
        
        if len(available_dates) < 2:
            st.error("Not enough dates in historical data for comparison")
            return None, None, None
        
        # Find closest date to start_date
        start_actual_date = min(available_dates, key=lambda x: abs(x - start_date))
        
        # Find closest date to end_date
        end_actual_date = min(available_dates, key=lambda x: abs(x - end_date))
        
        # If start and end are the same, try to find different dates
        if start_actual_date == end_actual_date and len(available_dates) > 1:
            available_dates.remove(end_actual_date)
            start_actual_date = min(available_dates, key=lambda x: abs(x - start_date))
        
        # Show the actual dates used
        st.info(f"Using data from {start_actual_date.strftime('%Y-%m-%d')} to {end_actual_date.strftime('%Y-%m-%d')}")
        
        # Get data for start and end dates
        start_data = historical_data[historical_data['Date'] == start_actual_date]
        end_data = historical_data[historical_data['Date'] == end_actual_date]
        
        # Handle empty data
        if start_data.empty or end_data.empty:
            st.error(f"No data found for selected dates: {start_actual_date} or {end_actual_date}")
            return None, None, None
        
        # Create dataframes with just symbol and price
        start_prices = start_data[['symbol', 'Close']].rename(columns={'Close': 'start_price'})
        end_prices = end_data[['symbol', 'Close']].rename(columns={'Close': 'end_price'})
        
        # Ensure symbol column is string type for both to match sectors_df
        start_prices['symbol'] = start_prices['symbol'].astype(str)
        end_prices['symbol'] = end_prices['symbol'].astype(str)
        
        # Merge to get both prices for each symbol
        price_data = pd.merge(start_prices, end_prices, on='symbol', how='inner')
        
        # Display count of matched symbols
        st.info(f"Found matching start and end prices for {len(price_data)} stocks")
        
        # Ensure sectors_df symbol is string type for consistency
        sectors_df['symbol'] = sectors_df['symbol'].astype(str)
        
        # Merge with sectors data
        merged_data = pd.merge(sectors_df, price_data, on='symbol', how='inner')
        
        # Calculate price change percentage
        merged_data['price_change_pct'] = ((merged_data['end_price'] - merged_data['start_price']) / merged_data['start_price']) * 100
        
        # Process market caps - ensure all numeric data is handled properly
        if market_caps is not None:
            # Convert symbol to string in market_caps
            market_caps['symbol'] = market_caps['symbol'].astype(str)
            
            # Ensure numeric columns are properly typed
            if 'market_cap' in market_caps.columns:
                market_caps['market_cap'] = pd.to_numeric(market_caps['market_cap'], errors='coerce')
            
            # Get essential columns
            market_caps_slim = market_caps[['symbol', 'market_cap']].copy()
            
            # Merge with our data
            merged_data = pd.merge(merged_data, market_caps_slim, on='symbol', how='left')
            
            # Show how many stocks had market cap data
            with_cap = merged_data['market_cap'].notna().sum()
            st.info(f"Found market cap data for {with_cap} of {len(merged_data)} matched stocks")
        else:
            # Use market cap from sectors file if available
            if 'market_cap_B' in merged_data.columns:
                merged_data['market_cap'] = merged_data['market_cap_B'] * 1e9
                st.info("Using market_cap_B from sectors file (converted to dollars)")
        
        # Filter to stocks with valid market cap
        valid_data = merged_data[merged_data['market_cap'].notna()].copy()
        
        # If we still don't have market caps, try to estimate
        if len(valid_data) == 0 and 'market_cap_B' in merged_data.columns:
            merged_data['market_cap'] = merged_data['market_cap_B'] * 1e9
            valid_data = merged_data.copy()
            st.warning("Using estimated market caps from sectors file")
        
        if len(valid_data) == 0:
            st.error("No valid market cap data found for any stock")
            return None, None, None
        
        # Calculate market cap changes
        valid_data['end_market_cap'] = valid_data['market_cap']
        valid_data['start_market_cap'] = valid_data['end_market_cap'] / (1 + (valid_data['price_change_pct'] / 100))
        valid_data['market_cap_change'] = valid_data['end_market_cap'] - valid_data['start_market_cap']
        
        # Add period information
        valid_data['period'] = period_name
        valid_data['start_date'] = start_actual_date
        valid_data['end_date'] = end_actual_date
        
        # Report how many stocks we have data for
        st.success(f"Analyzed {len(valid_data)} stocks with valid market cap and price data")
        
        # Aggregate by industry
        industry_flow = valid_data.groupby('industry').agg({
            'market_cap_change': 'sum',
            'start_market_cap': 'sum',
            'sector': 'first',
            'symbol': 'count',
            'price_change_pct': 'mean',
            'period': 'first',
            'start_date': 'first',
            'end_date': 'first'
        }).reset_index()
        
        # Calculate percentage change
        industry_flow['percent_change'] = (industry_flow['market_cap_change'] / industry_flow['start_market_cap']) * 100
        
        # Aggregate by sector
        sector_flow = valid_data.groupby('sector').agg({
            'market_cap_change': 'sum',
            'start_market_cap': 'sum',
            'symbol': 'count',
            'price_change_pct': 'mean',
            'period': 'first',
            'start_date': 'first',
            'end_date': 'first'
        }).reset_index()
        
        sector_flow['percent_change'] = (sector_flow['market_cap_change'] / sector_flow['start_market_cap']) * 100
        
        # Report success
        st.success(f"Successfully calculated money flow for {len(industry_flow)} industries and {len(sector_flow)} sectors")
        
        return industry_flow, sector_flow, valid_data
        
    except Exception as e:
        st.error(f"Error calculating money flow: {str(e)}")
        st.code(traceback.format_exc())
        return None, None, None

# Main function to run the app
def main():
    # App title and description
    st.title("Industry Money Flow Tracker")
    st.markdown("Visualize money flow across market sectors and industries over time")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        
        # Option to upload a custom sectors file
        uploaded_file = st.file_uploader("Upload custom stock_sectors.csv file", type=['csv'])
        if uploaded_file is not None:
            # Save the uploaded file
            os.makedirs(os.path.dirname(SECTORS_FILE), exist_ok=True)
            with open(SECTORS_FILE, 'wb') as f:
                f.write(uploaded_file.getvalue())
            st.success("File uploaded successfully!")
        
        # Load sectors data
        sectors_df = load_sectors_data()
        
        # Load market cap data
        market_caps = load_market_caps()
        
        if sectors_df is not None:
            # Display dataset statistics
            st.subheader("Dataset Information")
            st.info(f"Total stocks: {len(sectors_df)}")
            st.info(f"Total sectors: {sectors_df['sector'].nunique()}")
            st.info(f"Total industries: {sectors_df['industry'].nunique()}")
            
            # Get available dates
            available_dates = get_available_dates()
            
            if available_dates:
                st.subheader("Available Data")
                st.info(f"Historical data available for {len(available_dates)} days")
                st.info(f"Latest data: {available_dates[0]}")
                
                # Time period selection
                st.subheader("Time Period")
                period_type = st.radio(
                    "Select Period Type",
                    ["Custom", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
                )
                
                # Default to the latest date
                latest_date = pd.to_datetime(available_dates[0])
                
                if period_type == "Custom":
                    # Custom date range
                    min_date = pd.to_datetime(available_dates[-1])
                    max_date = latest_date
                    
                    start_date = st.date_input(
                        "Start Date",
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date
                    )
                    
                    end_date = st.date_input(
                        "End Date",
                        value=max_date,
                        min_value=min_date,
                        max_value=max_date
                    )
                    
                    period_name = f"Custom: {start_date} to {end_date}"
                    
                elif period_type == "Daily":
                    # Just use the most recent day
                    dates_to_show = min(7, len(available_dates))
                    selected_date_idx = st.selectbox(
                        "Select Date", 
                        range(dates_to_show),
                        format_func=lambda x: available_dates[x]
                    )
                    end_date = pd.to_datetime(available_dates[selected_date_idx])
                    
                    # For daily, we use previous trading day as start
                    if selected_date_idx < len(available_dates) - 1:
                        start_date = pd.to_datetime(available_dates[selected_date_idx + 1])
                    else:
                        # If we're at the oldest date, just use the same day
                        start_date = end_date - timedelta(days=1)
                    
                    period_name = f"Daily: {end_date.strftime('%Y-%m-%d')}"
                    
                else:
                    # Calculate start date based on period
                    if period_type == "Weekly":
                        start_date = latest_date - timedelta(days=7)
                        period_name = f"Weekly: {start_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}"
                    elif period_type == "Monthly":
                        start_date = latest_date - timedelta(days=30)
                        period_name = f"Monthly: {start_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}"
                    elif period_type == "Quarterly":
                        start_date = latest_date - timedelta(days=90)
                        period_name = f"Quarterly: {start_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}"
                    elif period_type == "Yearly":
                        start_date = latest_date - timedelta(days=365)
                        period_name = f"Yearly: {start_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}"
                    
                    end_date = latest_date
            else:
                st.warning("No historical data files found. Run the data collector script first.")
                
                # Create buttons to show instructions
                if st.button("Show Data Collection Instructions"):
                    st.info("""
                    # How to Collect Data
                    
                    1. Run the `stock-data-collector.py` script to collect historical data
                    2. The script will save daily data files to the 'data/daily_data' directory
                    3. Example command:
                       ```
                       python stock-data-collector.py --days 30 --batch-size 5 --batch-pause 1.0
                       ```
                    4. Run this script daily after market close for best results
                    """)
    
    # Main content area - Process and display data
    if 'sectors_df' in locals() and sectors_df is not None and 'available_dates' in locals() and available_dates:
        # Get the latest historical data file
        latest_file = os.path.join(DAILY_DATA_DIR, f"historical_data_{available_dates[0]}.csv")
        
        if os.path.exists(latest_file):
            # Process the historical data file
            historical_data = process_historical_data(latest_file)
            
            if historical_data is not None:
                # Calculate money flow using the multi-day data
                industry_flow, sector_flow, valid_market_cap_df = calculate_money_flow_from_multiday(
                    historical_data,
                    sectors_df,
                    market_caps,
                    start_date,
                    end_date,
                    period_name
                )
                
                # Create visualizations if we have data
                if industry_flow is not None and sector_flow is not None:
                    # Create tabs for different visualizations
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "Industry Money Flow",
                        "Sector Money Flow",
                        "Treemap",
                        "Bubble Chart",
                        "Calculation Details"
                    ])
                    
                    with tab1:
                        st.header(f"Industry Money Flow - {period_name}")
                        
                        # Sort by absolute change
                        industry_sorted = industry_flow.sort_values(by='market_cap_change', ascending=False)
                        
                        # Format for display
                        industry_sorted['market_cap_change_B'] = industry_sorted['market_cap_change'] / 1e9
                        industry_sorted['start_market_cap_B'] = industry_sorted['start_market_cap'] / 1e9
                        
                        # Create bar chart with modifications for showing all labels
                        fig = px.bar(
                            industry_sorted,
                            y='industry',
                            x='market_cap_change_B',
                            color='percent_change',
                            color_continuous_scale='RdBu_r',
                            text=industry_sorted['percent_change'].round(2).astype(str) + '%',
                            hover_data={
                                'start_market_cap_B': ':.2f',
                                'market_cap_change_B': ':.2f',
                                'sector': True,
                                'symbol': True,  # Number of stocks
                                'price_change_pct': ':.2f'
                            },
                            labels={
                                'industry': 'Industry',
                                'market_cap_change_B': 'Market Cap Change ($ Billions)',
                                'percent_change': 'Change (%)',
                                'start_market_cap_B': 'Starting Market Cap ($ Billions)',
                                'sector': 'Sector',
                                'symbol': 'Number of Stocks',
                                'price_change_pct': 'Avg Price Change (%)'
                            },
                            height=max(1200, 25 * len(industry_sorted))  # Dynamic height based on number of industries
                        )
                        
                        # Update layout to ensure all y-axis labels (industries) are shown
                        fig.update_layout(
                            yaxis={
                                'categoryorder': 'total ascending',
                                'showticklabels': True,
                                'tickmode': 'array',
                                'tickvals': list(range(len(industry_sorted))),
                                'ticktext': industry_sorted['industry'].tolist()
                            },
                            margin=dict(l=250)  # Increase left margin to make space for long industry names
                        )
                        
                        # Ensure all ticks are visible
                        fig.update_yaxes(
                            tickfont=dict(size=10),  # Adjust font size if needed
                            automargin=True  # Automatically adjust margins to fit labels
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        st.header(f"Sector Money Flow - {period_name}")
                        
                        # Sort by absolute change
                        sector_sorted = sector_flow.sort_values(by='market_cap_change', ascending=False)
                        
                        # Format for display
                        sector_sorted['market_cap_change_B'] = sector_sorted['market_cap_change'] / 1e9
                        sector_sorted['start_market_cap_B'] = sector_sorted['start_market_cap'] / 1e9
                        
                        # Create bar chart
                        fig = px.bar(
                            sector_sorted,
                            y='sector',
                            x='market_cap_change_B',
                            color='percent_change',
                            color_continuous_scale='RdBu_r',
                            text=sector_sorted['percent_change'].round(2).astype(str) + '%',
                            hover_data={
                                'start_market_cap_B': ':.2f',
                                'market_cap_change_B': ':.2f',
                                'symbol': True,
                                'price_change_pct': ':.2f'
                            },
                            labels={
                                'sector': 'Sector',
                                'market_cap_change_B': 'Market Cap Change ($ Billions)',
                                'percent_change': 'Change (%)',
                                'start_market_cap_B': 'Starting Market Cap ($ Billions)',
                                'symbol': 'Number of Stocks',
                                'price_change_pct': 'Avg Price Change (%)'
                            }
                        )
                        
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        st.header(f"Money Flow Treemap - {period_name}")
                        
                        try:
                            # Create a copy of the industry data for the treemap
                            treemap_data = industry_flow.copy()
                            
                            # Use absolute market cap change as the value for sizing
                            treemap_data['abs_change'] = treemap_data['market_cap_change'].abs()
                            
                            # Make sure there are no zero or negative values for the treemap size
                            treemap_data['abs_change'] = treemap_data['abs_change'].apply(lambda x: max(x, 0.00001))
                            
                            # Create treemap
                            fig = px.treemap(
                                treemap_data,
                                path=[px.Constant("All Industries"), 'sector', 'industry'],
                                values='abs_change',
                                color='percent_change',
                                color_continuous_scale='RdBu_r',
                                hover_data=['start_market_cap', 'market_cap_change', 'symbol', 'price_change_pct'],
                            )
                            
                            fig.update_layout(
                                margin=dict(t=50, l=25, r=25, b=25),
                                height=800
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating treemap visualization: {str(e)}")
                            st.info("The treemap visualization requires non-zero values to function properly.")
                            
                            # Fallback visualization
                            try:
                                st.subheader("Sector Heatmap (Alternative Visualization)")
                                heatmap_data = pd.pivot_table(
                                    valid_market_cap_df,
                                    values='price_change_pct',
                                    index='sector',
                                    aggfunc='mean'
                                ).reset_index()
                                
                                fig = px.imshow(
                                    heatmap_data['price_change_pct'].values.reshape(-1, 1),
                                    y=heatmap_data['sector'],
                                    color_continuous_scale='RdBu_r',
                                    labels=dict(x="", y="Sector", color="Price Change %"),
                                    text_auto='.2f',
                                    aspect="auto",
                                    height=600
                                )
                                
                                fig.update_layout(xaxis_visible=False)
                                st.plotly_chart(fig, use_container_width=True)
                            except:
                                st.warning("Unable to create alternative visualization.")
                    
                    with tab4:
                        st.header(f"Bubble Chart - Industry Money Flow - {period_name}")
                        
                        # Create bubble chart
                        fig = px.scatter(
                            industry_flow,
                            x='percent_change',
                            y='market_cap_change',
                            size='start_market_cap',
                            color='sector',
                            hover_name='industry',
                            log_y=True,
                            size_max=60,
                            hover_data=['symbol', 'price_change_pct'],
                            height=800
                        )
                        
                        fig.update_layout(
                            xaxis_title="Percent Change (%)",
                            yaxis_title="Market Cap Change ($)",
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab5:
                        st.header("Calculation Details")
                        st.write("This tab shows the detailed calculations for a selected industry.")
                        
                        # Create a dropdown to select an industry
                        if not industry_flow.empty:
                            industries = sorted(industry_flow['industry'].unique())
                            selected_industry = st.selectbox("Select an industry to view calculation details:", industries)
                            
                            # Get the data for the selected industry
                            industry_data = valid_market_cap_df[valid_market_cap_df['industry'] == selected_industry].copy()
                            
                            # Calculate totals for verification
                            total_end_cap = industry_data['end_market_cap'].sum()
                            total_start_cap = industry_data['start_market_cap'].sum()
                            total_change = industry_data['market_cap_change'].sum()
                            total_percent = (total_change / total_start_cap) * 100 if total_start_cap > 0 else 0
                            
                            # Display the calculation details
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Industry Summary")
                                st.write(f"Industry: **{selected_industry}**")
                                st.write(f"Total stocks: **{len(industry_data)}**")
                                st.write(f"Period: **{period_name}**")
                                st.write(f"Start date: **{industry_data['start_date'].iloc[0]}**")
                                st.write(f"End date: **{industry_data['end_date'].iloc[0]}**")
                                st.write(f"Total end market cap: **${total_end_cap/1e9:.2f}B**")
                                st.write(f"Total start market cap: **${total_start_cap/1e9:.2f}B**")
                                st.write(f"Total market cap change: **${total_change/1e9:.2f}B**")
                                st.write(f"Percent change: **{total_percent:.2f}%**")
                                
                                # Verify these match the aggregated data
                                st.subheader("Verification")
                                industry_row = industry_flow[industry_flow['industry'] == selected_industry].iloc[0]
                                st.write(f"Aggregated market cap change: **${industry_row['market_cap_change']/1e9:.2f}B**")
                                st.write(f"Aggregated start market cap: **${industry_row['start_market_cap']/1e9:.2f}B**")
                                st.write(f"Aggregated percent change: **{industry_row['percent_change']:.2f}%**")
                                
                                # Calculate the difference to verify accuracy
                                change_diff = abs(total_change - industry_row['market_cap_change']) / max(1, total_change) * 100
                                st.write(f"Verification difference: **{change_diff:.6f}%**")
                            
                            with col2:
                                st.subheader("Calculation Formula")
                                st.write("For each stock:")
                                st.code("""
# Calculate price change percentage
price_change_pct = (end_price - start_price) / start_price * 100

# We have current market cap (end_market_cap)
# Calculate start market cap from price change
start_market_cap = end_market_cap / (1 + price_change_pct/100)

# Calculate market cap change
market_cap_change = end_market_cap - start_market_cap
                                """)
                                st.write("For aggregated industry data:")
                                st.code("""
# Sum up all stocks in the industry
industry_end_market_cap = sum(end_market_cap)
industry_start_market_cap = sum(start_market_cap)
industry_market_cap_change = sum(market_cap_change)

# Calculate percentage change
industry_percent_change = industry_market_cap_change / industry_start_market_cap * 100
                                """)
                            
                            # Display sample stocks from this industry
                            st.subheader("Sample Stock Calculations")
                            st.write("Showing up to 10 stocks from this industry with calculations:")
                            
                            # Format the data for display
                            display_df = industry_data.copy()
                            display_df['start_price'] = display_df['start_price'].round(2)
                            display_df['end_price'] = display_df['end_price'].round(2)
                            display_df['price_change_pct'] = display_df['price_change_pct'].round(2)
                            display_df['end_market_cap'] = (display_df['end_market_cap'] / 1e9).round(3)
                            display_df['start_market_cap'] = (display_df['start_market_cap'] / 1e9).round(3)
                            display_df['market_cap_change'] = (display_df['market_cap_change'] / 1e9).round(3)
                            
                            # Select columns to display
                            columns_to_display = ['symbol', 'start_price', 'end_price', 'price_change_pct', 
                                                'start_market_cap', 'end_market_cap', 'market_cap_change']
                            
                            # Rename columns for better display
                            display_df = display_df[columns_to_display].rename(columns={
                                'start_price': 'Start Price ($)',
                                'end_price': 'End Price ($)',
                                'price_change_pct': 'Price Change (%)',
                                'start_market_cap': 'Start Market Cap ($B)',
                                'end_market_cap': 'End Market Cap ($B)',
                                'market_cap_change': 'Market Cap Change ($B)',
                                'symbol': 'Symbol'
                            })
                            
                            # Display the table
                            st.dataframe(display_df.head(10))
                        else:
                            st.warning("No industry data available to display.")
                else:
                    st.error("Failed to calculate money flow. Please check your data and try again.")
            else:
                st.error("Failed to process historical data file.")
        else:
            st.error(f"Historical data file not found: {latest_file}")
    else:
        st.info("""
        ## Getting Started
        
        1. Upload your stock_sectors.csv file using the uploader in the sidebar
        2. The app will look for historical data files in the data/daily_data directory
        3. If no data is found, run the data collector script first
        
        For more information on how to use this app, see the sidebar.
        """)

if __name__ == "__main__":
    main()