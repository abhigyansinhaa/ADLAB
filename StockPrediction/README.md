# Stock Price Prediction

A simple stock price prediction project comparing Linear Regression and LSTM models.

## Project Structure

```
StockPrediction/
├── app.py                      # Flask web server
├── main.py                     # Command line runner
├── data_processor.py           # Data collection & preprocessing
├── linear_regression_model.py  # Linear Regression model
├── lstm_model.py               # LSTM model
├── requirements.txt            # Dependencies
└── templates/
    └── index.html              # Web interface
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Command Line
```bash
python main.py AAPL 25
```
- First argument: Stock ticker (default: AAPL)
- Second argument: LSTM epochs (default: 25)

### Option 2: Web Application
```bash
python app.py
```
Open http://localhost:5000 in your browser.

## Features

1. **Data Collection**: Stock data from Yahoo Finance (yfinance)
2. **Preprocessing**: Missing value handling, feature scaling, train/test split
3. **Linear Regression**: Simple prediction using sklearn
4. **LSTM**: Deep learning time-series prediction using TensorFlow
5. **Comparison**: R², RMSE, MAE metrics
6. **Web Interface**: Interactive charts with Chart.js

## Metrics

- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **R²**: Coefficient of determination (higher is better, max 1.0)
