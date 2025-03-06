# Industry Money Flow Tracker

A powerful Streamlit application for visualizing money flow between market sectors and industries. This tool tracks market capitalization changes across 5,700+ stocks, helping you identify where money is moving in the market.

![Industry Money Flow Screenshot](https://i.imgur.com/tNFKEjR.png)

## Features

- **Comprehensive Coverage**: Analyzes 5,742 stocks across 146 industries and 11 sectors
- **Multiple Time Frames**: Daily, weekly, monthly, quarterly, yearly, and custom date ranges
- **Detailed Visualizations**: Interactive charts and graphs powered by Plotly
- **Calculation Transparency**: Detailed breakdown of how money flow is calculated

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/YourUsername/industry-money-flow-tracker.git
   cd industry-money-flow-tracker
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the app:
   ```
   streamlit run money_flow.py
   ```

## Usage

1. Launch the app using the command above
2. Select your desired time period (Daily, Weekly, Monthly, Quarterly, Yearly, or Custom)
3. Explore the various visualization tabs:
   - **Industry Money Flow**: Bar chart showing money flow by industry
   - **Sector Money Flow**: Bar chart showing money flow by sector
   - **Treemap**: Hierarchical visualization of sectors and industries
   - **Bubble Chart**: Multi-dimensional view of money flow dynamics
   - **Calculation Details**: Transparency into how figures are calculated

## Understanding the Visualizations

### Industry/Sector Money Flow
Bar charts display market capitalization changes, with color indicating direction (blue for inflows, red for outflows). The percentage labels show the relative change compared to the starting market cap.

### Treemap
The hierarchical visualization shows sectors containing their component industries. Box size is proportional to the absolute dollar value change, while color intensity indicates percentage change.

### Bubble Chart
This chart plots industries using multiple dimensions:
- X-axis: Percentage change in market cap
- Y-axis: Absolute dollar value change (logarithmic scale)
- Bubble size: Proportional to total market capitalization
- Color: Indicates sector grouping

## Data Sources

- **Stock Classifications**: 5,742 stocks with sector and industry classifications
- **Historical Prices**: End-of-day price data for analysis and comparison
- **Market Caps**: Latest market capitalization data for accurate money flow calculations

## Updating Data

The app currently uses included historical data. For the most current analysis:
1. Run the data collector script:
   ```
   python stock_data_collector.py
   ```
2. This will update the historical data file with the latest market information

## Acknowledgments

- Market data provided by Yahoo Finance
- Visualizations powered by Plotly
- Built with Streamlit