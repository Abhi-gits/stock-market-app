import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Set page config
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Function to load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('tcs-stock-cleaned.csv', parse_dates=['Date'])
        
        # Fill NaN values in numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Replace 0 volume with NaN and then fill with median
        if 'Volume' in df.columns:
            df.loc[df['Volume'] == 0, 'Volume'] = np.nan
            df['Volume'].fillna(df['Volume'].median(), inplace=True)
                
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load the data
df = load_data()

# Dashboard title
st.title("Stock Analysis Dashboard")
st.subheader("TCS Stock Analysis")

if df is None:
    st.error("Failed to load dataset. Please check if 'tcs-stock-cleaned.csv' exists in the current directory.")
else:
    # Sidebar filters
    st.sidebar.header("Filters")

    # Fix date range filter
    if not df.empty:
        # Get first and last dates from the dataset
        first_date = df['Date'].min()
        last_date = df['Date'].max()
        
        # Convert to Python datetime for Streamlit date picker
        min_date = first_date.date()  # Timestamp objects have .date() method
        max_date = last_date.date()
        
        # Default to showing last 6 months of data if enough data is available
        six_months_ago = last_date - pd.Timedelta(days=180)
        default_start = six_months_ago if six_months_ago > first_date else first_date
        default_start_date = default_start.date()
        
        # Create date input widgets with proper defaults and limits
        start_date = st.sidebar.date_input("Start Date", 
                                         value=default_start_date, 
                                         min_value=min_date, 
                                         max_value=max_date)
        end_date = st.sidebar.date_input("End Date", 
                                       value=max_date, 
                                       min_value=min_date,  # Allow selecting any date in dataset range
                                       max_value=max_date)
        
        # Display date range debug info
        st.sidebar.caption(f"Dataset date range: {min_date} to {max_date}")
    else:
        # Fallback for empty dataset
        start_date = datetime.now().date()
        end_date = datetime.now().date()
        st.sidebar.warning("No date data available")

    # Price range filter - handle NaN values
    valid_prices = df['Close'].dropna()
    if not valid_prices.empty:
        min_price = float(valid_prices.min())
        max_price = float(valid_prices.max())
        # Set default range to a reasonable subset if range is large
        default_min = min_price
        default_max = max_price
        if max_price - min_price > 100:  # Arbitrary threshold, adjust as needed
            default_min = max(min_price, max_price - 100)
        price_range = st.sidebar.slider("Price Range (₹)", 
                                       min_value=min_price, 
                                       max_value=max_price, 
                                       value=(default_min, max_price))
    else:
        price_range = (0, 1)  # Fallback if no valid prices
        st.sidebar.warning("No valid price data available")

    # Volume range filter - handle NaN values
    valid_volumes = df['Volume'].dropna()
    if not valid_volumes.empty:
        min_volume = float(valid_volumes.min())
        max_volume = float(valid_volumes.max())
        # Set default to show all volume range
        volume_range = st.sidebar.slider("Volume Range", 
                                        min_value=min_volume,
                                        max_value=max_volume,
                                        value=(min_volume, max_volume))
    else:
        volume_range = (0, 1)  # Fallback if no valid volumes
        st.sidebar.warning("No valid volume data available")

    # Moving average options
    show_ma50 = st.sidebar.checkbox("Show 50-day Moving Average", True)
    show_ma200 = st.sidebar.checkbox("Show 200-day Moving Average", True)

    # Filter the data based on user selection
    try:
        # Convert UI dates to pandas datetime for proper comparison
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # End of day
        
        # Apply filters
        date_mask = (df['Date'] >= start_dt) & (df['Date'] <= end_dt)
        price_mask = (df['Close'] >= price_range[0]) & (df['Close'] <= price_range[1])
        volume_mask = (df['Volume'] >= volume_range[0]) & (df['Volume'] <= volume_range[1])
        
        # Combine masks and handle NaN values
        combined_mask = date_mask & price_mask & volume_mask
        filtered_df = df[combined_mask].copy()
        
        # Log filter information for debugging
        st.sidebar.markdown("---")
        st.sidebar.caption(f"Filtered data: {len(filtered_df)} rows")
        if len(filtered_df) == 0:
            st.sidebar.caption("Date range: " + start_dt.strftime('%Y-%m-%d') + " to " + end_dt.strftime('%Y-%m-%d'))
        
    except Exception as e:
        st.error(f"Error applying filters: {e}")
        st.sidebar.error(f"Filter error: {e}")
        filtered_df = pd.DataFrame()

    # Main content
    if filtered_df.empty:
        st.warning("No data available for the selected filters. Please adjust your selection.")
    else:
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Candlestick Chart", "Price & Volume Trends", "Moving Averages", "Price Analysis"])
        
        with tab1:
            st.subheader("Candlestick Chart")
            
            # Create candlestick chart
            fig = go.Figure()
            
            # Add candlestick trace
            fig.add_trace(
                go.Candlestick(
                    x=filtered_df['Date'],
                    open=filtered_df['Open'],
                    high=filtered_df['High'],
                    low=filtered_df['Low'],
                    close=filtered_df['Close'],
                    name="Candlestick",
                    hoverinfo="text",
                    hovertext=[
                        f"Date: {date.strftime('%Y-%m-%d')}<br>" +
                        f"Open: ₹{open:.2f}<br>" +
                        f"High: ₹{high:.2f}<br>" +
                        f"Low: ₹{low:.2f}<br>" +
                        f"Close: ₹{close:.2f}<br>" +
                        f"Volume: {volume:,}"
                        for date, open, high, low, close, volume in 
                        zip(filtered_df['Date'], filtered_df['Open'], filtered_df['High'], 
                            filtered_df['Low'], filtered_df['Close'], filtered_df['Volume'])
                    ]
                )
            )
            
            # Add MA50 line if data is available and checkbox is selected
            if show_ma50 and 'MA50' in filtered_df.columns and not filtered_df['MA50'].isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['Date'],
                        y=filtered_df['MA50'],
                        mode='lines',
                        name='50-day MA',
                        line=dict(color='orange')
                    )
                )
            
            # Add MA200 line if data is available and checkbox is selected
            if show_ma200 and 'MA200' in filtered_df.columns and not filtered_df['MA200'].isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['Date'],
                        y=filtered_df['MA200'],
                        mode='lines',
                        name='200-day MA',
                        line=dict(color='purple')
                    )
                )
            
            # Update layout
            fig.update_layout(
                title="TCS Stock Price",
                yaxis_title="Price (₹)",
                xaxis_title="Date",
                height=600,
                template="plotly_white",
                xaxis_rangeslider_visible=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Price & Volume Trends")
            
            # Create a figure with 2 y-axes
            fig = go.Figure()
            
            # Add Close price line
            fig.add_trace(
                go.Scatter(
                    x=filtered_df['Date'],
                    y=filtered_df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue')
                )
            )
            
            # Add Volume bars on secondary axis
            fig.add_trace(
                go.Bar(
                    x=filtered_df['Date'],
                    y=filtered_df['Volume'],
                    name='Volume',
                    marker=dict(color='rgba(0, 128, 0, 0.5)'),
                    yaxis='y2'
                )
            )
            
            # Update layout with secondary y-axis
            fig.update_layout(
                title="Price and Volume Trends",
                yaxis=dict(title="Price (₹)", side="left"),
                yaxis2=dict(
                    title="Volume",
                    side="right",
                    overlaying="y",
                    showgrid=False
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=500,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Moving Averages Analysis")
            
            # Handle case where MA columns might be missing
            ma_columns = [col for col in ['MA50', 'MA200'] if col in filtered_df.columns]
            
            if len(ma_columns) > 0:
                valid_ma_data = filtered_df.dropna(subset=ma_columns, how='all')
                
                if not valid_ma_data.empty:
                    # Create figure
                    fig = go.Figure()
                    
                    # Add Close price
                    fig.add_trace(
                        go.Scatter(
                            x=valid_ma_data['Date'],
                            y=valid_ma_data['Close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='blue', width=1)
                        )
                    )
                    
                    # Add MA50 line if data is available
                    if 'MA50' in valid_ma_data.columns and not valid_ma_data['MA50'].isna().all():
                        fig.add_trace(
                            go.Scatter(
                                x=valid_ma_data['Date'],
                                y=valid_ma_data['MA50'],
                                mode='lines',
                                name='50-day MA',
                                line=dict(color='orange', width=2)
                            )
                        )
                    
                    # Add MA200 line if data is available
                    if 'MA200' in valid_ma_data.columns and not valid_ma_data['MA200'].isna().all():
                        fig.add_trace(
                            go.Scatter(
                                x=valid_ma_data['Date'],
                                y=valid_ma_data['MA200'],
                                mode='lines',
                                name='200-day MA',
                                line=dict(color='purple', width=2)
                            )
                        )
                    
                    # Update layout
                    fig.update_layout(
                        title="Moving Average Analysis",
                        yaxis_title="Price (₹)",
                        xaxis_title="Date",
                        height=500,
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation
                    st.info("""
                    **Moving Average Analysis:**
                    - The 50-day moving average (orange line) shows the short-term trend
                    - The 200-day moving average (purple line) shows the long-term trend
                    - When the 50-day MA crosses above the 200-day MA, it's considered a bullish signal (Golden Cross)
                    - When the 50-day MA crosses below the 200-day MA, it's considered a bearish signal (Death Cross)
                    """)
                else:
                    st.warning("Not enough data for moving average analysis with current filter settings.")
            else:
                st.warning("Moving average data not available in the dataset.")
        
        with tab4:
            st.subheader("Price Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily returns calculation (using loc to avoid SettingWithCopyWarning)
                filtered_df.loc[:, 'Daily Return'] = filtered_df['Close'].pct_change() * 100
                
                # Create daily returns chart
                fig = px.histogram(
                    filtered_df.dropna(subset=['Daily Return']), 
                    x='Daily Return',
                    nbins=50,
                    title='Distribution of Daily Returns (%)',
                    color_discrete_sequence=['darkblue']
                )
                
                fig.update_layout(
                    xaxis_title="Daily Return (%)",
                    yaxis_title="Frequency",
                    height=400,
                    template="plotly_white"
                )
                
                # Add a vertical line at x=0
                fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Volatility over time (30-day rolling standard deviation)
                if len(filtered_df) >= 30:
                    # Use loc to avoid SettingWithCopyWarning
                    filtered_df.loc[:, 'Volatility'] = filtered_df['Daily Return'].rolling(window=30).std()
                    
                    # Create volatility chart
                    fig = px.line(
                        filtered_df.dropna(subset=['Volatility']),
                        x='Date',
                        y='Volatility',
                        title='30-Day Rolling Volatility',
                        color_discrete_sequence=['darkred']
                    )
                    
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Volatility (Standard Deviation of Returns)",
                        height=400,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough data points for volatility calculation. Need at least 30 data points.")
        
        # Key stats at the bottom
        st.subheader("Key Statistics")
        
        # Safely calculate metrics
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Average Close Price", 
                    f"₹{filtered_df['Close'].mean():.2f}",
                    f"{((filtered_df['Close'].iloc[-1] / filtered_df['Close'].iloc[0]) - 1) * 100:.2f}%" if len(filtered_df) > 1 else None
                )
            
            with col2:
                st.metric(
                    "Average Daily Volume", 
                    f"{filtered_df['Volume'].mean():.0f}",
                    f"{((filtered_df['Volume'].iloc[-30:].mean() / filtered_df['Volume'].iloc[0:30].mean()) - 1) * 100:.2f}%" if len(filtered_df) > 30 else None
                )
            
            with col3:
                if not filtered_df['High'].empty:
                    highest_price = filtered_df['High'].max()
                    highest_date = filtered_df.loc[filtered_df['High'] == highest_price, 'Date'].iloc[0]
                    st.metric(
                        "Highest Price", 
                        f"₹{highest_price:.2f}",
                        f"on {highest_date.strftime('%Y-%m-%d')}"
                    )
                else:
                    st.metric("Highest Price", "N/A")
            
            with col4:
                if not filtered_df['Low'].empty:
                    lowest_price = filtered_df['Low'].min()
                    lowest_date = filtered_df.loc[filtered_df['Low'] == lowest_price, 'Date'].iloc[0]
                    st.metric(
                        "Lowest Price", 
                        f"₹{lowest_price:.2f}",
                        f"on {lowest_date.strftime('%Y-%m-%d')}"
                    )
                else:
                    st.metric("Lowest Price", "N/A")
        except Exception as e:
            st.error(f"Error computing statistics: {e}")

    # Add footer
    st.markdown("---")
    st.markdown("**TCS Stock Analysis Dashboard** | Data source: tcs-stock-cleaned.csv")
    st.markdown("Made with ❤️ by [@Abhi-gits](https://github.com/Abhi-gits)")