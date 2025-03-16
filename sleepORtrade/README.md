# Sleep or Trade: Stock Trading Strategy Comparison

## Overview
This project implements and compares two stock trading strategies based on short-term momentum:

1. **Strategy A (Daytime Inertia)**: Assumes that if a stock's price moves in one direction overnight, it will continue in that direction during the trading day.
2. **Strategy B (Nighttime Inertia)**: Assumes that if a stock's price moves in one direction during the trading day, it will continue in that direction overnight.

The project evaluates these strategies using historical Apple (AAPL) stock data from 2020 to 2024, analyzing their profitability under different conditions.

## Data Source
- The project fetches historical stock data using the **Yahoo Finance (yfinance)** API.
- Data includes **Open, High, Low, Close** prices for each trading day.
- **Dividends and stock splits** are removed for clean analysis.

## Implementation Details

### Strategy A: Daytime Inertia
1. **Trading Rule**
   - If the opening price is **higher** than the previous day’s close → Go **Long** (Buy).
   - If the opening price is **lower** than the previous day’s close → Go **Short** (Sell Short).
   - Profit/Loss per share is calculated based on daily price movement.

2. **Key Metrics Computed**
   - **Average daily profit**.
   - **Profitability of long vs. short positions**.
   - **Effect of a threshold (only trade if overnight return exceeds a percentage, e.g., 5%)**.

3. **Results Visualization**
   - A **line chart** shows how increasing the trading threshold impacts profitability.

### Strategy B: Nighttime Inertia
1. **Trading Rule**
   - If the stock closes **higher** than it opened → Go **Long** (Buy at Close, Sell at Next Open).
   - If the stock closes **lower** than it opened → Go **Short** (Sell Short at Close, Buy at Next Open).
   - Profit/Loss per share is calculated based on overnight price movement.

2. **Key Metrics Computed**
   - **Average daily profit**.
   - **Profitability of long vs. short positions**.
   - **Impact of a trading threshold on performance**.

3. **Results Visualization**
   - A **line chart** showing profitability trends as the threshold increases.

## Key Questions Answered
- What is the **average daily profit** for each strategy?
- Is **long trading** or **short trading** more profitable?
- How does **introducing a threshold** impact profitability?
- How do these results change **over multiple years**?
- **Should you sleep or trade?**

## Installation & Dependencies
### Prerequisites
Ensure you have Python installed. You will also need the following libraries:

```bash
pip install numpy pandas yfinance matplotlib scipy datetime
```

### Running the Code
1. Clone this repository or download the script.
2. Run the script using:
   ```bash
   python sleep_or_trade.py
   ```

## Conclusion
This project explores the effectiveness of different trading strategies. Through data analysis, we determine whether it is better to trade actively during the day or hold positions overnight.
