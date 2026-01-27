import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


def get_stock_data(ticker='AAPL', days=500):
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        data = stock.history(period='2y')
        data.reset_index(inplace=True)
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"Collected {len(data)} records for {ticker}")
        return data
    except:
        print("Using sample data...")
        return generate_sample_data(days)


def generate_sample_data(days=500):
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    price = 150
    prices = [price]
    for _ in range(days - 1):
        price *= (1 + np.random.normal(0, 0.02))
        prices.append(price)
    
    prices = np.array(prices)
    
    return pd.DataFrame({
        'Date': dates,
        'Open': prices * np.random.uniform(0.99, 1.01, days),
        'High': prices * np.random.uniform(1.0, 1.02, days),
        'Low': prices * np.random.uniform(0.98, 1.0, days),
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, days)
    })


def preprocess_data(data):
    df = data.copy()
    
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    
    df.dropna(inplace=True)
    return df


def prepare_lr_data(data, test_size=0.2):
    df = preprocess_data(data)
    
    features = ['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_10']
    X = df[features].values
    y = df['Close'].values
    dates = df['Date'].values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    split = int(len(X) * (1 - test_size))
    
    return {
        'X_train': X_scaled[:split],
        'X_test': X_scaled[split:],
        'y_train': y[:split],
        'y_test': y[split:],
        'dates_test': dates[split:],
        'scaler': scaler
    }


def prepare_lstm_data(data, sequence_length=60, test_size=0.2):
    df = preprocess_data(data)
    prices = df['Close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_prices)):
        X.append(scaled_prices[i-sequence_length:i, 0])
        y.append(scaled_prices[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    split = int(len(X) * (1 - test_size))
    dates = df['Date'].values[sequence_length:]
    
    return {
        'X_train': X[:split],
        'X_test': X[split:],
        'y_train': y[:split],
        'y_test': y[split:],
        'dates_test': dates[split:],
        'scaler': scaler
    }


if __name__ == "__main__":
    data = get_stock_data('AAPL')
    print(data.head())
    
    # Test preprocessing
    lr_data = prepare_lr_data(data)
    print(f"LR Train: {lr_data['X_train'].shape}, Test: {lr_data['X_test'].shape}")
    
    lstm_data = prepare_lstm_data(data)
    print(f"LSTM Train: {lstm_data['X_train'].shape}, Test: {lstm_data['X_test'].shape}")
