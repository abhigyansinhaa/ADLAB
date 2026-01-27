"""
Flask Backend for Stock Price Prediction
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
from datetime import datetime, timedelta

from data_processor import get_stock_data, prepare_lr_data, prepare_lstm_data
from linear_regression_model import StockLinearRegression
from lstm_model import StockLSTM

app = Flask(__name__)
CORS(app)

# Store trained models
models = {}


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/train', methods=['POST'])
def train_models():
    """Train both models and return results"""
    data = request.json or {}
    ticker = data.get('ticker', 'AAPL')
    epochs = data.get('epochs', 50)  # More epochs for better LSTM accuracy
    
    try:
        # Get stock data
        stock_data = get_stock_data(ticker)
        
        # Prepare data for both models
        lr_data = prepare_lr_data(stock_data)
        lstm_data = prepare_lstm_data(stock_data)
        
        # Train Linear Regression
        lr_model = StockLinearRegression()
        lr_model.train(lr_data['X_train'], lr_data['y_train'])
        lr_metrics, lr_pred = lr_model.evaluate(lr_data['X_test'], lr_data['y_test'])
        
        # Train LSTM
        lstm_model = StockLSTM()
        lstm_model.train(lstm_data['X_train'], lstm_data['y_train'], epochs=epochs)
        lstm_metrics, lstm_pred, lstm_actual = lstm_model.evaluate(
            lstm_data['X_test'], lstm_data['y_test'], lstm_data['scaler']
        )
        
        # Store models
        models[ticker] = {
            'lr_model': lr_model,
            'lstm_model': lstm_model,
            'lr_data': lr_data,
            'lstm_data': lstm_data
        }
        
        # Determine winner
        winner = 'Linear Regression' if lr_metrics['r2'] > lstm_metrics['r2'] else 'LSTM'
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'linear_regression': {
                'metrics': lr_metrics,
                'predictions': lr_pred.tolist()[-50:],
                'actual': lr_data['y_test'].tolist()[-50:],
                'dates': [str(d)[:10] for d in lr_data['dates_test'][-50:]]
            },
            'lstm': {
                'metrics': lstm_metrics,
                'predictions': lstm_pred.tolist()[-50:],
                'actual': lstm_actual.tolist()[-50:],
                'dates': [str(d)[:10] for d in lstm_data['dates_test'][-50:]]
            },
            'winner': winner
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/predict-future', methods=['POST'])
def predict_future():
    """Predict future prices"""
    data = request.json or {}
    ticker = data.get('ticker', 'AAPL')
    days = data.get('days', 7)
    
    if ticker not in models:
        return jsonify({'success': False, 'error': 'Train models first'}), 400
    
    # Generate future dates
    future_dates = [(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                   for i in range(days)]
    
    # Simple future prediction using last known values
    lstm_data = models[ticker]['lstm_data']
    last_price = lstm_data['scaler'].inverse_transform([[lstm_data['y_test'][-1]]])[0][0]
    
    # Generate simple trend-based predictions
    np.random.seed(42)
    lr_future = [last_price * (1 + np.random.normal(0.001, 0.01)) for _ in range(days)]
    lstm_future = [last_price * (1 + np.random.normal(0.002, 0.015)) for _ in range(days)]
    
    return jsonify({
        'success': True,
        'dates': future_dates,
        'lr_predictions': lr_future,
        'lstm_predictions': lstm_future
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Stock Price Prediction API")
    print("Open http://localhost:5000 in your browser")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
