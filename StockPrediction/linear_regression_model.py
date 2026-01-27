"""
Linear Regression Model for Stock Price Prediction
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class StockLinearRegression:
    def __init__(self):
        self.model = LinearRegression()
        
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        print("Linear Regression model trained!")
        
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions)
        }
        
        print("\n=== Linear Regression Results ===")
        print(f"RMSE: ${metrics['rmse']:.2f}")
        print(f"MAE: ${metrics['mae']:.2f}")
        print(f"R² Score: {metrics['r2']:.4f}")
        
        return metrics, predictions


if __name__ == "__main__":
    from data_processor import get_stock_data, prepare_lr_data
    
    # Get and prepare data
    data = get_stock_data('AAPL')
    lr_data = prepare_lr_data(data)
    
    # Train and evaluate
    model = StockLinearRegression()
    model.train(lr_data['X_train'], lr_data['y_train'])
    metrics, predictions = model.evaluate(lr_data['X_test'], lr_data['y_test'])
