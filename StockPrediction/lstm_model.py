"""
LSTM Model for Stock Price Prediction
"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class StockLSTM:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        self.model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.1),
            LSTM(100, return_sequences=True),
            Dropout(0.1),
            LSTM(50, return_sequences=False),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        # Use lower learning rate for better convergence
        optimizer = Adam(learning_rate=0.001)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        print("LSTM model built!")
        
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """Train the model"""
        if self.model is None:
            self.build_model((X_train.shape[1], 1))
        
        # Better callbacks
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=10,  # More patience
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
        
        print(f"Training LSTM for {epochs} epochs...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,  # Use 10% for validation
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        print("LSTM training completed!")
        return history
        
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0).flatten()
    
    def evaluate(self, X_test, y_test, scaler):
        """Evaluate model performance"""
        predictions_scaled = self.predict(X_test)
        
        # Inverse transform to get actual prices
        predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        metrics = {
            'mse': mean_squared_error(y_actual, predictions),
            'rmse': np.sqrt(mean_squared_error(y_actual, predictions)),
            'mae': mean_absolute_error(y_actual, predictions),
            'r2': r2_score(y_actual, predictions)
        }
        
        print("\n=== LSTM Results ===")
        print(f"RMSE: ${metrics['rmse']:.2f}")
        print(f"MAE: ${metrics['mae']:.2f}")
        print(f"R² Score: {metrics['r2']:.4f}")
        
        return metrics, predictions, y_actual


if __name__ == "__main__":
    from data_processor import get_stock_data, prepare_lstm_data
    
    # Get and prepare data
    data = get_stock_data('AAPL')
    lstm_data = prepare_lstm_data(data)
    
    # Train and evaluate with more epochs
    model = StockLSTM()
    model.train(lstm_data['X_train'], lstm_data['y_train'], epochs=50)
    metrics, predictions, actual = model.evaluate(
        lstm_data['X_test'], 
        lstm_data['y_test'], 
        lstm_data['scaler']
    )
