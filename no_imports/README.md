# Bitcoin Daily Data Analysis

## Overview
This project analyzes Bitcoin's daily price movements, returns, and volatility using historical OHLC (Open, High, Low, Close) data. The analysis is performed using **pure Python**, without libraries like Pandas, NumPy, or SciPy. The project follows a structured approach to answer key financial questions about Bitcoin price trends.

## Dataset
- The dataset is stored in **Bitcoin_Data.csv**.
- It contains daily price data for Bitcoin since **January 1, 2014**.
- Each row includes **Date, Open, High, Low, Close, and Change %** (daily return).

## Implementation Details

### **Part 1: Basic Analysis**
- Loads the dataset into a list of dictionaries (without Pandas).
- Filters data for **2019** (or another assigned year based on BUID).
- Computes:
  - **Total number of entries**.
  - **Average closing price**.
  - **Average daily return, max/min return, and standard deviation**.

### **Part 2: Positive vs. Negative Days**
- Determines the number of **positive (T⁺) and negative (T⁻) return days**.
- Computes:
  - **Average return, max/min return, and standard deviation** for positive and negative days.
- Identifies the **three highest and three lowest return days**.

### **Part 3: Price Range Analysis**
- Computes the **daily range** as **(High - Low)**.
- Finds:
  - **Max range on a positive day**.
  - **Max range on a negative day**.
  - **Top 3 highest and lowest range days**.

### **Part 4: Volatility Measurement (True Range - TR)**
- Computes **True Range (TR)** for each day:
  \[ TR = max(High - Low, |High - PrevClose|, |Low - PrevClose|) \]
- Saves results into an **output CSV file (data_2019.csv)**.
- Identifies the **three highest TR days**.

### **Question 5: Insights & Learnings**
- Summarizes key findings about Bitcoin's volatility, extreme price movements, and trends.
- Observes the difficulty of handling data without Pandas.

## Running the Code
### Prerequisites
Ensure you have **Python 3.x** installed.

### Instructions
1. **Download the dataset** (`Bitcoin_Data.csv`) and place it in the same directory as the script.
2. **Run the script** using:
   ```bash
   python bitcoin_analysis.py
   ```
3. The script will output key statistics and generate a file **data_2019.csv** with computed **True Range (TR)** values.

## Output Files
- **data_2019.csv** → Contains the filtered 2019 dataset with calculated TR values.
