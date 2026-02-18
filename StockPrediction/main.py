"""
Stock Price Prediction - Main Script
Run this to train and compare models via command line
"""

from data_processor import get_stock_data, prepare_lr_data, prepare_lstm_data
from linear_regression_model import StockLinearRegression
from lstm_model import StockLSTM


def main(ticker='AAPL', epochs=50):
    print("\n" + "="*50)
    print("   STOCK PRICE PREDICTION")
    print("   Linear Regression vs LSTM")
    print("="*50)
    
    # Step 1: Collect Data
    print("\n[1] Collecting stock data...")
    data = get_stock_data(ticker)
    print(f"    Data shape: {data.shape}")
    
    # Step 2: Prepare Data
    print("\n[2] Preprocessing data...")
    lr_data = prepare_lr_data(data)
    lstm_data = prepare_lstm_data(data)
    print(f"    LR - Train: {lr_data['X_train'].shape}, Test: {lr_data['X_test'].shape}")
    print(f"    LSTM - Train: {lstm_data['X_train'].shape}, Test: {lstm_data['X_test'].shape}")
    
    # Step 3: Train Linear Regression
    print("\n[3] Training Linear Regression...")
    lr_model = StockLinearRegression()
    lr_model.train(lr_data['X_train'], lr_data['y_train'])
    lr_metrics, _ = lr_model.evaluate(lr_data['X_test'], lr_data['y_test'])
    
    # Step 4: Train LSTM
    print("\n[4] Training LSTM...")
    lstm_model = StockLSTM()
    lstm_model.train(lstm_data['X_train'], lstm_data['y_train'], epochs=epochs)
    lstm_metrics, _, _ = lstm_model.evaluate(
        lstm_data['X_test'], 
        lstm_data['y_test'], 
        lstm_data['scaler']
    )
    
    # Step 5: Compare Models
    print("\n" + "="*50)
    print("   MODEL COMPARISON")
    print("="*50)
    print(f"\n{'Metric':<15} {'Linear Reg':<15} {'LSTM':<15}")
    print("-"*45)
    print(f"{'RMSE':<15} ${lr_metrics['rmse']:<14.2f} ${lstm_metrics['rmse']:<14.2f}")
    print(f"{'MAE':<15} ${lr_metrics['mae']:<14.2f} ${lstm_metrics['mae']:<14.2f}")
    print(f"{'R² Score':<15} {lr_metrics['r2']:<14.4f} {lstm_metrics['r2']:<14.4f}")
    
    winner = "Linear Regression" if lr_metrics['r2'] > lstm_metrics['r2'] else "LSTM"
    print(f"\nWinner: {winner}")
    
    print("\n" + "="*50)
    print("   To run web app: python app.py")
    print("   Then open http://localhost:8000")
    print("="*50 + "\n")


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    main(ticker, epochs)
