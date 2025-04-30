
# 🧠📈 Market Movement Prediction using Deep Learning

This project explores the application of deep learning models to predict next-day stock market direction based on historical price data. By engineering key financial indicators such as returns, volatility, and trading volume, we train deep neural networks (DNNs) to classify days as either “green” (positive return) or “red” (negative return). The analysis spans the S&P 500 index and leading stocks from various U.S. market sectors.

---

## 📁 Project Structure

```
market-prediction-dnn/
│
├── data/                     # Raw and processed datasets
├── models/                   # Saved model weights and configurations
├── outputs/                  # Charts, logs, and evaluation metrics
├── scripts/
│   ├── sp500_prediction.py   # DNN model for S&P 500 prediction
│   ├── sector_analysis.py    # DNN modeling for sector-specific stocks
│   └── feature_importance.py # ML-based feature analysis using KNN & Decision Tree
├── requirements.txt          # Python packages and dependencies
└── README.md                 # Project overview and instructions
```

---

## 🎯 Objectives

- Predict next-day market movement using DNNs
- Identify which stocks and sectors are more predictable
- Analyze the importance of different features in prediction accuracy
- Understand the strengths and limitations of deep learning in short-term financial forecasting

---

## 🔧 Tools & Technologies

- Python 3.x  
- TensorFlow / Keras  
- Scikit-learn  
- Pandas / NumPy  
- Matplotlib / Seaborn  
- Yahoo Finance (`yfinance`)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/market-prediction-dnn.git
cd market-prediction-dnn
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run a Script
For S&P 500 Prediction:
```bash
python scripts/sp500_prediction.py
```

For Sector-wise Analysis:
```bash
python scripts/sector_analysis.py
```

For Feature Importance:
```bash
python scripts/feature_importance.py
```

---

## 📊 Methodology

1. **Data Collection**: Price data pulled from Yahoo Finance for the S&P 500 and selected sector leaders (e.g., AAPL, XOM, JNJ).

2. **Feature Engineering**:
   - Daily Returns
   - 7-day Rolling Volatility
   - Lagged Features
   - Trading Volume

3. **Labeling**:  
   - 1 (green): Next-day return ≥ 0.35%  
   - 0 (red): Next-day return < 0.35%

4. **Modeling**:
   - Deep Neural Networks (DNNs) with Dropout and ReLU activations
   - EarlyStopping to avoid overfitting
   - Training on first 4 years; testing on final 2 years of data

5. **Evaluation**:
   - Accuracy and classification report
   - Cross-sector comparison
   - Feature importance via Decision Trees and KNN

---

## 📈 Key Results

- **S&P 500 DNN Accuracy**: ~61% on unseen data
- **Strong Predictors**: Previous returns ranked highest in importance
- **High-Performing Stocks**: META, AAPL showed most consistent performance
- **Top Sectors**: Technology and Communications exhibited higher predictability

---

## 🛠 Future Work

- Expand to ensemble models (e.g., XGBoost, Random Forest)
- Introduce technical indicators like RSI, MACD
- Explore multi-class forecasting (large up/down, flat)
- Apply LSTM or Transformer-based architectures for sequence modeling

---

