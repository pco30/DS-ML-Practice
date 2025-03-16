# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:50:33 2025

@author: PRINCELY OSEJI
"""
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np
from datetime import date, datetime, timedelta
import yfinance as yf

# STRATEGY A
aapl_A = yf.Ticker("AAPL").history(start="2020-01-01", end="2024-12-31") # Apple stock from 2020 to 2024
aapl_A = aapl_A.drop(["Dividends", "Stock Splits"], axis = 1)
aapl_A["Return"] = aapl_A["Close"].pct_change() * 100
aapl_A["Return"].fillna(0, inplace = True)
aapl_A["Date"] = aapl_A.index
aapl_A["Date"] = pd.to_datetime(aapl_A["Date"])
aapl_A["Year"] = aapl_A["Date"].dt.year
for col in ["Open", "High", "Low", "Close"]:
    aapl_A[col] = aapl_A[col].round(2)

aapl_A["Decision"] = "-" # Default value
aapl_A.loc[aapl_A["Return"] > 0, "Decision"] = "Long"
aapl_A.loc[aapl_A["Return"] < 0, "Decision"] = "Short"

# Creating P/L per share
aapl_A["P/L per share"] = np.nan # Default value
aapl_A.loc[aapl_A["Decision"] == "Long", "P/L per share"] = (100/aapl_A["Open"])*(aapl_A["Close"] - aapl_A["Open"])
aapl_A.loc[aapl_A["Decision"] == "Short", "P/L per share"] = (100/aapl_A["Open"]) * (aapl_A["Open"] - aapl_A["Close"].shift(1))

# QUESTION 1
avg_A = np.mean(aapl_A["P/L per share"])
print("The average daily profit for strategy A is", round(avg_A, 2))

# Question 2
long_A = np.sum(aapl_A.loc[aapl_A["Decision"] == "Long", "P/L per share"])
short_A = np.sum(aapl_A.loc[aapl_A["Decision"] == "Short", "P/L per share"])
print("The profit or loss from long position of stratgey A is" , round(long_A, 2))
print("The profit or loss from short position of stratgey A is" , round(short_A, 2))
print("Through investigation, the long position of strategy A is more profitable")

# Question 3
# Increasing threshold values
aapl_A[["Decision_1","Decision_2", "Decision_3", "Decision_4","Decision_5","Decision_6","Decision_7","Decision_8","Decision_9","Decision_10"]] = "-" # Default value for each x percentage
aapl_A.loc[np.abs(aapl_A["Return"]) > 1, "Decision_1"] = "Trade"
aapl_A.loc[np.abs(aapl_A["Return"]) > 2, "Decision_2"] = "Trade"
aapl_A.loc[np.abs(aapl_A["Return"]) > 3, "Decision_3"] = "Trade"
aapl_A.loc[np.abs(aapl_A["Return"]) > 4, "Decision_4"] = "Trade"
aapl_A.loc[np.abs(aapl_A["Return"]) > 5, "Decision_5"] = "Trade"
aapl_A.loc[np.abs(aapl_A["Return"]) > 6, "Decision_6"] = "Trade"
aapl_A.loc[np.abs(aapl_A["Return"]) > 7, "Decision_7"] = "Trade"
aapl_A.loc[np.abs(aapl_A["Return"]) > 8, "Decision_8"] = "Trade"
aapl_A.loc[np.abs(aapl_A["Return"]) > 9, "Decision_9"] = "Trade"
aapl_A.loc[np.abs(aapl_A["Return"]) > 10, "Decision_10"] = "Trade"

# Average returns for each threshold
mean_1 = np.mean(aapl_A.loc[aapl_A["Decision_1"] == "Trade", "Return"]) * 100
mean_2 = np.mean(aapl_A.loc[aapl_A["Decision_2"] == "Trade", "Return"]) * 100
mean_3 = np.mean(aapl_A.loc[aapl_A["Decision_3"] == "Trade", "Return"]) * 100
mean_4 = np.mean(aapl_A.loc[aapl_A["Decision_4"] == "Trade", "Return"]) * 100
mean_5 = np.mean(aapl_A.loc[aapl_A["Decision_5"] == "Trade", "Return"]) * 100
mean_6 = np.mean(aapl_A.loc[aapl_A["Decision_6"] == "Trade", "Return"]) * 100
mean_7 = np.mean(aapl_A.loc[aapl_A["Decision_7"] == "Trade", "Return"]) * 100
mean_8 = np.mean(aapl_A.loc[aapl_A["Decision_8"] == "Trade", "Return"]) * 100
mean_9 = np.mean(aapl_A.loc[aapl_A["Decision_9"] == "Trade", "Return"]) * 100
mean_10 = np.mean(aapl_A.loc[aapl_A["Decision_10"] == "Trade", "Return"]) * 100

plt.figure(figsize=(12, 6))
plt.plot(np.array(range(10)), [mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7, mean_8, mean_9, mean_10])
plt.title("Average profit per trade with increasing absolute threshold")
plt.xlabel("Absolute threshold(%)")
plt.ylabel("Average profit")
plt.grid()
plt.show()

print("With increasing absolute threshold, the average profit seems to increase")

# Question 4
temp_A = aapl_A.dropna()
temp_A = temp_A.drop("P/L per share", axis = 1)
temp_A[["Decision_1","Decision_2", "Decision_3", "Decision_4","Decision_5","Decision_6","Decision_7","Decision_8","Decision_9","Decision_10"]] = "-"
temp_A.loc[temp_A["Return"] < -1, "Decision_1"] = "Short"
temp_A.loc[temp_A["Return"] > 1, "Decision_1"] = "Long"
temp_A.loc[temp_A["Return"] < -2, "Decision_2"] = "Short"
temp_A.loc[temp_A["Return"] > 2, "Decision_2"] = "Long"
temp_A.loc[temp_A["Return"] < -3, "Decision_3"] = "Short"
temp_A.loc[temp_A["Return"] > 3, "Decision_3"] = "Long"
temp_A.loc[temp_A["Return"] < -4, "Decision_4"] = "Short"
temp_A.loc[temp_A["Return"] > 4, "Decision_4"] = "Long"
temp_A.loc[temp_A["Return"] < -5, "Decision_5"] = "Short"
temp_A.loc[temp_A["Return"] > 5, "Decision_5"] = "Long"
temp_A.loc[temp_A["Return"] < -6, "Decision_6"] = "Short"
temp_A.loc[temp_A["Return"] > 6, "Decision_6"] = "Long"
temp_A.loc[temp_A["Return"] < -7, "Decision_7"] = "Short"
temp_A.loc[temp_A["Return"] > 7, "Decision_7"] = "Long"
temp_A.loc[temp_A["Return"] < -8, "Decision_8"] = "Short"
temp_A.loc[temp_A["Return"] > 8, "Decision_8"] = "Long"
temp_A.loc[temp_A["Return"] < -9, "Decision_9"] = "Short"
temp_A.loc[temp_A["Return"] > 9, "Decision_9"] = "Long"
temp_A.loc[temp_A["Return"] < -10, "Decision_10"] = "Short"
temp_A.loc[temp_A["Return"] > 10, "Decision_10"] = "Long"

# Creating P/L per share for each decision for strat A
temp_A[["P/L per share 1","P/L per share 2","P/L per share 3","P/L per share 4","P/L per share 5","P/L per share 6","P/L per share 7","P/L per share 8","P/L per share 9","P/L per share 10"]] = np.nan # Default value
temp_A.loc[temp_A["Decision_1"] == "Long", "P/L per share 1"] = (100/temp_A["Open"])*(temp_A["Close"] - temp_A["Open"])
temp_A.loc[temp_A["Decision_1"] == "Short", "P/L per share 1"] = (100/temp_A["Open"]) * (temp_A["Open"] - temp_A["Close"].shift(1))
temp_A.loc[temp_A["Decision_2"] == "Long", "P/L per share 2"] = (100/temp_A["Open"])*(temp_A["Close"] - temp_A["Open"])
temp_A.loc[temp_A["Decision_2"] == "Short", "P/L per share 2"] = (100/temp_A["Open"]) * (temp_A["Open"] - temp_A["Close"].shift(1))
temp_A.loc[temp_A["Decision_3"] == "Long", "P/L per share 3"] = (100/temp_A["Open"])*(temp_A["Close"] - temp_A["Open"])
temp_A.loc[temp_A["Decision_3"] == "Short", "P/L per share 3"] = (100/temp_A["Open"]) * (temp_A["Open"] - temp_A["Close"].shift(1))
temp_A.loc[temp_A["Decision_4"] == "Long", "P/L per share 4"] = (100/temp_A["Open"])*(temp_A["Close"] - temp_A["Open"])
temp_A.loc[temp_A["Decision_4"] == "Short", "P/L per share 4"] = (100/temp_A["Open"]) * (temp_A["Open"] - temp_A["Close"].shift(1))
temp_A.loc[temp_A["Decision_5"] == "Long", "P/L per share 5"] = (100/temp_A["Open"])*(temp_A["Close"] - temp_A["Open"])
temp_A.loc[temp_A["Decision_5"] == "Short", "P/L per share 5"] = (100/temp_A["Open"]) * (temp_A["Open"] - temp_A["Close"].shift(1))
temp_A.loc[temp_A["Decision_6"] == "Long", "P/L per share 6"] = (100/temp_A["Open"])*(temp_A["Close"] - temp_A["Open"])
temp_A.loc[temp_A["Decision_6"] == "Short", "P/L per share 6"] = (100/temp_A["Open"]) * (temp_A["Open"] - temp_A["Close"].shift(1))
temp_A.loc[temp_A["Decision_7"] == "Long", "P/L per share 7"] = (100/temp_A["Open"])*(temp_A["Close"] - temp_A["Open"])
temp_A.loc[temp_A["Decision_7"] == "Short", "P/L per share 7"] = (100/temp_A["Open"]) * (temp_A["Open"] - temp_A["Close"].shift(1))
temp_A.loc[temp_A["Decision_8"] == "Long", "P/L per share 8"] = (100/temp_A["Open"])*(temp_A["Close"] - temp_A["Open"])
temp_A.loc[temp_A["Decision_8"] == "Short", "P/L per share 8"] = (100/temp_A["Open"]) * (temp_A["Open"] - temp_A["Close"].shift(1))
temp_A.loc[temp_A["Decision_9"] == "Long", "P/L per share 9"] = (100/temp_A["Open"])*(temp_A["Close"] - temp_A["Open"])
temp_A.loc[temp_A["Decision_9"] == "Short", "P/L per share 9"] = (100/temp_A["Open"]) * (temp_A["Open"] - temp_A["Close"].shift(1))
temp_A.loc[temp_A["Decision_10"] == "Long", "P/L per share 10"] = (100/temp_A["Open"])*(temp_A["Close"] - temp_A["Open"])
temp_A.loc[temp_A["Decision_10"] == "Short", "P/L per share 10"] = (100/temp_A["Open"]) * (temp_A["Open"] - temp_A["Close"].shift(1))

# Calculate means for P/L per share
mean_1A = np.mean(temp_A["P/L per share 1"]) * 100 #Multiple by 100 for 100 shares bought
mean_2A = np.mean(temp_A["P/L per share 2"]) * 100
mean_3A = np.mean(temp_A["P/L per share 3"]) * 100
mean_4A = np.mean(temp_A["P/L per share 4"]) * 100
mean_5A = np.mean(temp_A["P/L per share 5"]) * 100
mean_6A = np.mean(temp_A["P/L per share 6"]) * 100
mean_7A = np.mean(temp_A["P/L per share 7"]) * 100 
mean_8A = np.mean(temp_A["P/L per share 8"]) * 100
mean_9A = np.mean(temp_A["P/L per share 9"]) * 100
mean_10A = np.mean(temp_A["P/L per share 10"]) * 100

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(np.array(range(10)), [mean_1A, mean_2A, mean_3A, mean_4A, mean_5A, mean_6A, mean_7A, mean_8A, mean_9A, mean_10A])
plt.title("Average profit per trade with increasing threshold for 100 shares bought using strategy A")
plt.xlabel("Absolute threshold(%)")
plt.ylabel("Average profit")
plt.grid()
plt.show()

# Question 5
strat_A = aapl_A.groupby("Year")["Decision"].value_counts().unstack() # Table to store summary stats per year for strategy A
strat_A = strat_A[["Long", "Short"]]
strat_A = strat_A.rename(columns={"Long":"|L|", "Short":"|S|"})

# Compute total profit per year for short and long positions
short_A = aapl_A[aapl_A["Decision"] == "Short"].groupby("Year")["P/L per share"].sum().reset_index()
short_A = short_A.rename(columns={"P/L per share" : "P(S)"})
long_A = aapl_A[aapl_A["Decision"] == "Long"].groupby("Year")["P/L per share"].sum().reset_index()
long_A = long_A.rename(columns={"P/L per share" : "P(L)"})

# Compute average profit per year for short and long positions
short_avg_A = aapl_A[aapl_A["Decision"] == "Short"].groupby("Year")["P/L per share"].mean().reset_index()
short_avg_A = short_avg_A.rename(columns={"P/L per share" : "P(S)/|S|"})
long_avg_A = aapl_A[aapl_A["Decision"] == "Long"].groupby("Year")["P/L per share"].mean().reset_index()
long_avg_A = long_avg_A.rename(columns={"P/L per share" : "P(L)/|L|"})

# Merge all summary stats to the initial table
strat_A = strat_A.merge(short_A, on="Year", how="left")
strat_A = strat_A.merge(long_A, on="Year", how="left")
strat_A = strat_A.merge(short_avg_A, on="Year", how="left")
strat_A = strat_A.merge(long_avg_A, on="Year", how="left")
strat_A.set_index("Year", inplace=True) # Set year as the index
sum_A = strat_A.sum().to_frame().T # Sum all numeric values in each column
sum_A.index = ["2021-2024"]
strat_A = pd.concat([strat_A, sum_A]) # Append sum_A to strat_A

print(strat_A)




# STRATEGY B
aapl_B = yf.Ticker("AAPL").history(start="2020-01-01", end="2024-12-31")
aapl_B = aapl_B.drop(["Dividends", "Stock Splits"], axis = 1)
aapl_B["Return"] = aapl_B["Close"].pct_change() * 100
aapl_B["Return"].fillna(0, inplace = True)
aapl_B["Date"] = aapl_B.index
aapl_B["Date"] = pd.to_datetime(aapl_B["Date"])
aapl_B["Year"] = aapl_B["Date"].dt.year
for col in ["Open", "High", "Low", "Close"]:
    aapl_B[col] = aapl_B[col].round(2)

aapl_B["Decision"] = "-" # Default value
aapl_B.loc[aapl_B["Return"] > 0, "Decision"] = "Long"
aapl_B.loc[aapl_B["Return"] < 0, "Decision"] = "Short"

# Creating P/L per share
aapl_B["P/L per share"] = np.nan # Default value
aapl_B.loc[aapl_B["Decision"] == "Long", "P/L per share"] = (100/aapl_B["Close"])*(100 - aapl_B["Close"])
aapl_B.loc[aapl_B["Decision"] == "Short", "P/L per share"] = (100/aapl_B["Close"]) * (aapl_B["Close"] - aapl_B["Open"].shift(-1))

# QUESTION 1
avg_B = np.mean(aapl_B["P/L per share"])
print("The average daily profit for strategy B is", round(avg_B, 2))

# Question 2
long_B = np.sum(aapl_B.loc[aapl_B["Decision"] == "Long", "P/L per share"])
short_B = np.sum(aapl_B.loc[aapl_B["Decision"] == "Short", "P/L per share"])
print("The profit or loss from long position of stratgey B is" , round(long_B,2))
print("The profit or loss from short position of stratgey B is" , round(short_B,2))
print("Through investigation, the short position of strategy B is more profitable")

# Question 4
temp_B = aapl_B.dropna()
temp_B = temp_B.drop("P/L per share", axis = 1)
temp_B[["Decision_1","Decision_2", "Decision_3", "Decision_4","Decision_5","Decision_6","Decision_7","Decision_8","Decision_9","Decision_10"]] = "-"
temp_B.loc[temp_B["Return"] < -1, "Decision_1"] = "Short"
temp_B.loc[temp_B["Return"] > 1, "Decision_1"] = "Long"
temp_B.loc[temp_B["Return"] < -2, "Decision_2"] = "Short"
temp_B.loc[temp_B["Return"] > 2, "Decision_2"] = "Long"
temp_B.loc[temp_B["Return"] < -3, "Decision_3"] = "Short"
temp_B.loc[temp_B["Return"] > 3, "Decision_3"] = "Long"
temp_B.loc[temp_B["Return"] < -4, "Decision_4"] = "Short"
temp_B.loc[temp_B["Return"] > 4, "Decision_4"] = "Long"
temp_B.loc[temp_B["Return"] < -5, "Decision_5"] = "Short"
temp_B.loc[temp_B["Return"] > 5, "Decision_5"] = "Long"
temp_B.loc[temp_B["Return"] < -6, "Decision_6"] = "Short"
temp_B.loc[temp_B["Return"] > 6, "Decision_6"] = "Long"
temp_B.loc[temp_B["Return"] < -7, "Decision_7"] = "Short"
temp_B.loc[temp_B["Return"] > 7, "Decision_7"] = "Long"
temp_B.loc[temp_B["Return"] < -8, "Decision_8"] = "Short"
temp_B.loc[temp_B["Return"] > 8, "Decision_8"] = "Long"
temp_B.loc[temp_B["Return"] < -9, "Decision_9"] = "Short"
temp_B.loc[temp_B["Return"] > 9, "Decision_9"] = "Long"
temp_B.loc[temp_B["Return"] < -10, "Decision_10"] = "Short"
temp_B.loc[temp_B["Return"] > 10, "Decision_10"] = "Long"

# Creating P/L per share for each decision for strat B
temp_B[["P/L per share 1","P/L per share 2","P/L per share 3","P/L per share 4","P/L per share 5","P/L per share 6","P/L per share 7","P/L per share 8","P/L per share 9","P/L per share 10"]] = np.nan # Default value
temp_B.loc[temp_B["Decision_1"] == "Long", "P/L per share 1"] = (100/aapl_B["Close"])*(100 - aapl_B["Close"])
temp_B.loc[temp_B["Decision_1"] == "Short", "P/L per share 1"] = (100/aapl_B["Close"]) * (aapl_B["Close"] - aapl_B["Open"].shift(-1))
temp_B.loc[temp_B["Decision_2"] == "Long", "P/L per share 2"] = (100/aapl_B["Close"])*(100 - aapl_B["Close"])
temp_B.loc[temp_B["Decision_2"] == "Short", "P/L per share 2"] = (100/aapl_B["Close"]) * (aapl_B["Close"] - aapl_B["Open"].shift(-1))
temp_B.loc[temp_B["Decision_3"] == "Long", "P/L per share 3"] = (100/aapl_B["Close"])*(100 - aapl_B["Close"])
temp_B.loc[temp_B["Decision_3"] == "Short", "P/L per share 3"] = (100/aapl_B["Close"]) * (aapl_B["Close"] - aapl_B["Open"].shift(-1))
temp_B.loc[temp_B["Decision_4"] == "Long", "P/L per share 4"] = (100/aapl_B["Close"])*(100 - aapl_B["Close"])
temp_B.loc[temp_B["Decision_4"] == "Short", "P/L per share 4"] = (100/aapl_B["Close"]) * (aapl_B["Close"] - aapl_B["Open"].shift(-1))
temp_B.loc[temp_B["Decision_5"] == "Long", "P/L per share 5"] = (100/aapl_B["Close"])*(100 - aapl_B["Close"])
temp_B.loc[temp_B["Decision_5"] == "Short", "P/L per share 5"] = (100/aapl_B["Close"]) * (aapl_B["Close"] - aapl_B["Open"].shift(-1))
temp_B.loc[temp_B["Decision_6"] == "Long", "P/L per share 6"] = (100/aapl_B["Close"])*(100 - aapl_B["Close"])
temp_B.loc[temp_B["Decision_6"] == "Short", "P/L per share 6"] = (100/aapl_B["Close"]) * (aapl_B["Close"] - aapl_B["Open"].shift(-1))
temp_B.loc[temp_B["Decision_7"] == "Long", "P/L per share 7"] = (100/aapl_B["Close"])*(100 - aapl_B["Close"])
temp_B.loc[temp_B["Decision_7"] == "Short", "P/L per share 7"] = (100/aapl_B["Close"]) * (aapl_B["Close"] - aapl_B["Open"].shift(-1))
temp_B.loc[temp_B["Decision_8"] == "Long", "P/L per share 8"] = (100/aapl_B["Close"])*(100 - aapl_B["Close"])
temp_B.loc[temp_B["Decision_8"] == "Short", "P/L per share 8"] = (100/aapl_B["Close"]) * (aapl_B["Close"] - aapl_B["Open"].shift(-1))
temp_B.loc[temp_B["Decision_9"] == "Long", "P/L per share 9"] = (100/aapl_B["Close"])*(100 - aapl_B["Close"])
temp_B.loc[temp_B["Decision_9"] == "Short", "P/L per share 9"] = (100/aapl_B["Close"]) * (aapl_B["Close"] - aapl_B["Open"].shift(-1))
temp_B.loc[temp_B["Decision_10"] == "Long", "P/L per share 10"] = (100/aapl_B["Close"])*(100 - aapl_B["Close"])
temp_B.loc[temp_B["Decision_10"] == "Short", "P/L per share 10"] = (100/aapl_B["Close"]) * (aapl_B["Close"] - aapl_B["Open"].shift(-1))

# Calculate means for P/L per share
mean_1B = np.mean(temp_B["P/L per share 1"]) * 100 #Multiple by 100 for 100 shares bought
mean_2B = np.mean(temp_B["P/L per share 2"]) * 100
mean_3B = np.mean(temp_B["P/L per share 3"]) * 100
mean_4B = np.mean(temp_B["P/L per share 4"]) * 100
mean_5B = np.mean(temp_B["P/L per share 5"]) * 100
mean_6B = np.mean(temp_B["P/L per share 6"]) * 100
mean_7B = np.mean(temp_B["P/L per share 7"]) * 100 
mean_8B = np.mean(temp_B["P/L per share 8"]) * 100
mean_9B = np.mean(temp_B["P/L per share 9"]) * 100
mean_10B = np.mean(temp_B["P/L per share 10"]) * 100

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(np.array(range(10)), [mean_1B, mean_2B, mean_3B, mean_4B, mean_5B, mean_6B, mean_7B, mean_8B, mean_9B, mean_10B])
plt.title("Average profit per trade with increasing threshold for 100 shares bought using strategy B")
plt.xlabel("Absolute threshold(%)")
plt.ylabel("Average profit")
plt.grid()
plt.show()

# Question 5
strat_B = aapl_B.groupby("Year")["Decision"].value_counts().unstack() # Table to store summary stats per year for strategy A
strat_B = strat_B[["Long", "Short"]]
strat_B = strat_B.rename(columns={"Long":"|L|", "Short":"|S|"})

# Compute total profit per year for short and long positions
short_B = aapl_B[aapl_B["Decision"] == "Short"].groupby("Year")["P/L per share"].sum().reset_index()
short_B = short_B.rename(columns={"P/L per share" : "P(S)"})
long_B = aapl_B[aapl_B["Decision"] == "Long"].groupby("Year")["P/L per share"].sum().reset_index()
long_B = long_B.rename(columns={"P/L per share" : "P(L)"})

# Compute average profit per year for short and long positions
short_avg_B = aapl_B[aapl_B["Decision"] == "Short"].groupby("Year")["P/L per share"].mean().reset_index()
short_avg_B = short_avg_B.rename(columns={"P/L per share" : "P(S)/|S|"})
long_avg_B = aapl_B[aapl_B["Decision"] == "Long"].groupby("Year")["P/L per share"].mean().reset_index()
long_avg_B = long_avg_B.rename(columns={"P/L per share" : "P(L)/|L|"})

# Merge all summary stats to the initial table
strat_B = strat_B.merge(short_B, on="Year", how="left")
strat_B = strat_B.merge(long_B, on="Year", how="left")
strat_B = strat_B.merge(short_avg_B, on="Year", how="left")
strat_B = strat_B.merge(long_avg_B, on="Year", how="left")
strat_B.set_index("Year", inplace=True) # Set year as the index
sum_B = strat_B.sum().to_frame().T # Sum all numeric values in each column
sum_B.index = ["2021-2024"]
strat_B = pd.concat([strat_B, sum_B]) # Append sum_A to strat_A

print(strat_B)

# Question 6
print("I learnt that putting a threshold can drastically affect returns")

# Question 7
print("You should sleep not trade")


