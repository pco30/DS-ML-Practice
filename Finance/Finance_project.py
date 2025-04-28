# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 21:36:14 2025

@author: PRINCELY OSEJI
"""
# Import relevant libraries
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Define date range
end_date = datetime.today()
start_date = end_date - timedelta(days=6*365)  # Approx 6 years

# Download S&P 500 data
data = yf.download("^GSPC", start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

# Compute daily return (%)
data['Return'] = data['Close'].pct_change() * 100

# Compute 7-day rolling volatility
data['Volatility'] = data['Return'].rolling(window=7).std()

# Label column
data['Label'] = np.where(data['Return'] > 0.35, 'green', 'red')

# Retain relevant columns
data = data[['Return', 'Volatility', 'Volume', 'Label']]

# Shift features by 1 to use previous day's values
data['Prev_Return'] = data['Return'].shift(1)
data['Prev_Volatility'] = data['Volatility'].shift(1)
data['Prev_Volume'] = data['Volume'].shift(1)

# Drop current-day features (keep only shifted ones + label)
data = data[['Prev_Return', 'Prev_Volatility', 'Prev_Volume', 'Label']]

# Drop rows with NaNs (due to shift or rolling)
data.dropna(inplace=True)

# Split into training (first 4 years) and testing (last 2 years)
cutoff_index = int(len(data) * (4/6))
train_data = data.iloc[:cutoff_index]
test_data = data.iloc[cutoff_index:]

# Show previews
print("Training data preview:")
print(train_data.head())

print("\nTest data preview:")
print(test_data.head())

# Prepare feature and label sets
X_train = train_data.drop('Label', axis=1).values
y_train = (train_data['Label'] == 'green').astype(int).values  # Convert to binary

X_test = test_data.drop('Label', axis=1).values
y_test = (test_data['Label'] == 'green').astype(int).values

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network
model = Sequential([
    Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=25, batch_size=32, verbose=0)

# Predict and evaluate
train_preds = (model.predict(X_train_scaled) > 0.5).astype(int)
test_preds = (model.predict(X_test_scaled) > 0.5).astype(int)

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}")

# Define sectors and corresponding popular stock tickers
sector_stocks = {
    "Communication Services": "META",
    "Information Technology": "AAPL",
    "Consumer Discretionary": "AMZN",
    "Financials": "JPM",
    "Utilities": "NEE",
    "Industrials": "BA",
    "Consumer Staples": "PG",
    "Real Estate": "AMT",
    "Energy": "XOM",
    "Health Care": "JNJ",
    "Materials": "SHW"
}

# Date range for past 2 years
end_date = datetime.today()
start_date = end_date - timedelta(days=2*365)

# Store results
results = []

# Neural network model architecture
def create_model(input_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Process each stock
for sector, ticker in sector_stocks.items():
    # Download data
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    
    if data.empty:
        continue

    # Feature engineering
    data['Return'] = data['Close'].pct_change() * 100
    data['Volatility'] = data['Return'].rolling(window=7).std()
    data['Label'] = np.where(data['Return'] > 0.35, 'green', 'red')

    # Shift features
    data['Prev_Return'] = data['Return'].shift(1)
    data['Prev_Volatility'] = data['Volatility'].shift(1)
    data['Prev_Volume'] = data['Volume'].shift(1)

    # Final dataset
    data = data[['Prev_Return', 'Prev_Volatility', 'Prev_Volume', 'Label']].dropna()

    # Prepare features and labels
    X = data.drop('Label', axis=1).values
    y = (data['Label'] == 'green').astype(int).values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = create_model(input_dim=X_scaled.shape[1])
    model.fit(X_scaled, y, epochs=25, batch_size=32, verbose=0)

    # Predict and evaluate
    predictions = (model.predict(X_scaled) > 0.5).astype(int)
    accuracy = accuracy_score(y, predictions)

    # Save result
    results.append({
        "Stock": ticker,
        "Sector": sector,
        "Accuracy": round(accuracy, 4)
    })

# Create results table
results_df = pd.DataFrame(results)
results_df.sort_values(by='Accuracy', ascending=False, inplace=True)
results_df.reset_index(drop=True, inplace=True)

print(results_df)

# Marginal contributions of each feature

# Function to return accuracy of knn with different features
def knn(feature):
    X_train = train_data[feature]
    Y_train = train_data[["Label"]]
    X_test = test_data[feature]
    Y_test = test_data[["Label"]]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, Y_train)
    pred_test = knn.predict(X_test_scaled)
    accuracy = (accuracy_score(Y_test, pred_test))
    return accuracy

accuracy_knn = knn(['Prev_Return', 'Prev_Volatility', 'Prev_Volume']) # Accuracy using all features for knn
accuracy_knn1 = knn(["Prev_Return", "Prev_Volatility"])
accuracy_knn2 = knn(["Prev_Volatility", "Prev_Volume"])
accuracy_knn3 = knn(["Prev_Return", "Prev_Volume"])

# LOGISTIC REGRESSION
# Function to return accuracy of decision tree
def decision(feature):
    X_train = train_data[feature]
    Y_train = train_data[["Label"]]
    X_test = test_data[feature]
    Y_test = test_data[["Label"]]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    tree = DecisionTreeClassifier()
    tree.fit(X_train_scaled, Y_train)
    pred_test = tree.predict(X_test_scaled)
    accuracy = (accuracy_score(Y_test, pred_test))
    return accuracy

accuracy_tree = decision(['Prev_Return', 'Prev_Volatility', 'Prev_Volume']) # Accuracy using all features for logistic regression
accuracy_tree1 = decision(["Prev_Return", "Prev_Volatility"])
accuracy_tree2 = decision(["Prev_Volatility", "Prev_Volume"])
accuracy_tree3 = decision(["Prev_Return", "Prev_Volume"])

# Table to show summary stats of marginal contributions
q = {"Feature": ["Prev_Return", "Prev_volatility", "Prev_Volume"],
     "Marginal Contributions Knn": [accuracy_knn-accuracy_knn2, accuracy_knn-accuracy_knn3, accuracy_knn-accuracy_knn2],
     "Marginal Contributions Decision Tress": [accuracy_tree-accuracy_tree2, accuracy_tree-accuracy_tree3, accuracy_tree-accuracy_tree3]}

table = pd.DataFrame(q) # Table to show summary
