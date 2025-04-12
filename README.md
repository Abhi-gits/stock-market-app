# Stock Analysis Dashboard

An interactive dashboard for analyzing TCS stock data built with Streamlit and Plotly.

## Features

- **Interactive Time Series Visualization**: Candlestick chart with hover details
- **Dynamic Filters**: Customize the view with date range, price, and volume filters
- **Moving Average Analysis**: Visualize 50-day and 200-day moving averages
- **Price Analysis**: View daily returns distribution and volatility over time
- **Key Statistics**: Summary of important stock metrics

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```

## Data

The dashboard uses the `tcs-stock-cleaned.csv` file, which contains historical stock data for Tata Consultancy Services (TCS).

## How to Use

1. **Filters**: Use the sidebar to adjust date ranges, price ranges, and volume ranges
2. **Chart Interaction**: Hover over points on charts to see detailed information
3. **Moving Averages**: Toggle the 50-day and 200-day moving averages on/off using the sidebar checkboxes
4. **Tabs**: Switch between different analysis views using the tabs

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Plotly
- NumPy 