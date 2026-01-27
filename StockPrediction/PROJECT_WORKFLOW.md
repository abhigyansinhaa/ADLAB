# Stock Price Prediction - Project Workflow

## Objective
To predict stock prices using two machine learning approaches:
1. **Linear Regression** - Traditional ML algorithm
2. **LSTM (Long Short-Term Memory)** - Deep Learning approach for time-series

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                    (index.html + Chart.js)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FLASK BACKEND (app.py)                     │
│                    /api/train  /api/predict                     │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
┌─────────────────────┐             ┌─────────────────────┐
│  Linear Regression  │             │       LSTM          │
│      Model          │             │       Model         │
└─────────────────────┘             └─────────────────────┘
            │                                   │
            └─────────────────┬─────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSOR                               │
│         (Collection, Preprocessing, Train/Test Split)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STOCK DATA (yfinance API)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Workflow

### Step 1: Data Collection (`data_processor.py`)

**Source:** Yahoo Finance API (yfinance library)

```python
# Fetches 2 years of historical data
stock = yf.Ticker('AAPL')
data = stock.history(period='2y')
```

**Data Collected:**
| Column | Description |
|--------|-------------|
| Date | Trading date |
| Open | Opening price |
| High | Highest price of the day |
| Low | Lowest price of the day |
| Close | Closing price (TARGET) |
| Volume | Number of shares traded |

**Fallback:** If API fails, synthetic data is generated using random walk simulation.

---

### Step 2: Data Preprocessing

#### 2.1 Handle Missing Values
```python
df.fillna(method='ffill')  # Forward fill
df.fillna(method='bfill')  # Backward fill
```

#### 2.2 Feature Engineering
Add Moving Averages as additional features:
```python
df['MA_5'] = df['Close'].rolling(window=5).mean()   # 5-day average
df['MA_10'] = df['Close'].rolling(window=10).mean() # 10-day average
```

#### 2.3 Feature Scaling
Using MinMaxScaler to normalize data between 0 and 1:
```python
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

#### 2.4 Train/Test Split
- **Training Data:** 80% (used to train models)
- **Testing Data:** 20% (used to evaluate models)

```python
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
```

---

### Step 3: Linear Regression Model (`linear_regression_model.py`)

**Algorithm:** Ordinary Least Squares (OLS)

**How it works:**
1. Finds the best-fit line through data points
2. Minimizes the sum of squared errors
3. Equation: `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`

**Input Features (X):**
- Open, High, Low, Volume, MA_5, MA_10

**Target (y):**
- Close price

```python
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Advantages:**
- Fast training
- Easy to interpret
- Works well for linear relationships

**Limitations:**
- Cannot capture complex patterns
- Doesn't consider time sequence

---

### Step 4: LSTM Model (`lstm_model.py`)

**Algorithm:** Long Short-Term Memory Neural Network

**How it works:**
1. Processes data as sequences (60 days → predict day 61)
2. Has memory cells to remember long-term patterns
3. Uses gates (forget, input, output) to control information flow

**Data Preparation for LSTM:**
```
Sequence Length = 60 days

Input:  [Day1, Day2, Day3, ..., Day60] → Output: Day61
Input:  [Day2, Day3, Day4, ..., Day61] → Output: Day62
...and so on
```

**Model Architecture:**
```
Input Layer (60 time steps)
    │
    ▼
LSTM Layer (50 units) + Dropout (20%)
    │
    ▼
LSTM Layer (50 units) + Dropout (20%)
    │
    ▼
Dense Layer (25 units)
    │
    ▼
Output Layer (1 unit - predicted price)
```

```python
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
```

**Training:**
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Early Stopping: Stops if loss doesn't improve for 5 epochs

**Advantages:**
- Captures temporal patterns
- Handles sequential data well
- Can learn complex relationships

**Limitations:**
- Slower training
- Requires more data
- More complex to tune

---

### Step 5: Model Comparison

**Metrics Used:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MSE** | Mean((y - ŷ)²) | Average squared error |
| **RMSE** | √MSE | Error in same units as price |
| **MAE** | Mean(\|y - ŷ\|) | Average absolute error |
| **R² Score** | 1 - (SS_res/SS_tot) | How well model fits (0-1) |

**Winner Determination:**
```python
winner = 'Linear Regression' if lr_r2 > lstm_r2 else 'LSTM'
```

---

### Step 6: Flask Backend (`app.py`)

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the web interface |
| `/api/train` | POST | Trains both models, returns metrics |
| `/api/predict-future` | POST | Predicts future prices |

**Request/Response Flow:**
```
Frontend                    Backend
   │                           │
   │──POST /api/train─────────▶│
   │   {ticker, epochs}        │
   │                           │── Fetch stock data
   │                           │── Preprocess data
   │                           │── Train Linear Regression
   │                           │── Train LSTM
   │                           │── Evaluate both models
   │◀──JSON Response───────────│
   │   {metrics, predictions}  │
   │                           │
```

---

### Step 7: Frontend Visualization (`index.html`)

**Technologies:**
- HTML/CSS for structure and styling
- JavaScript for interactivity
- Chart.js for graphs

**Features:**
1. Stock ticker selection dropdown
2. LSTM epochs input
3. Train button to start model training
4. Two line charts showing predictions vs actual
5. Metrics comparison table
6. Winner announcement

**Chart Configuration:**
```javascript
new Chart(canvas, {
    type: 'line',
    data: {
        labels: dates,
        datasets: [
            { label: 'Actual', data: actualPrices },
            { label: 'Predicted', data: predictedPrices }
        ]
    }
});
```

---

## Complete Data Flow

```
1. User clicks "Train & Compare"
           │
           ▼
2. Frontend sends POST request to /api/train
           │
           ▼
3. Backend fetches stock data from Yahoo Finance
           │
           ▼
4. Data is preprocessed (missing values, scaling, features)
           │
           ▼
5. Data is split into train (80%) and test (20%)
           │
           ▼
6. Linear Regression model trains on features
           │
           ▼
7. LSTM model trains on price sequences
           │
           ▼
8. Both models predict test set prices
           │
           ▼
9. Metrics (RMSE, MAE, R²) calculated for both
           │
           ▼
10. Results sent back to frontend as JSON
           │
           ▼
11. Charts and metrics table updated
           │
           ▼
12. Winner displayed based on R² score
```

---

## File Summary

| File | Lines | Purpose |
|------|-------|---------|
| `data_processor.py` | ~145 | Data collection & preprocessing |
| `linear_regression_model.py` | ~45 | Linear Regression implementation |
| `lstm_model.py` | ~70 | LSTM implementation |
| `app.py` | ~125 | Flask server & API |
| `templates/index.html` | ~250 | Web interface |
| `main.py` | ~70 | Command-line runner |

---

## How to Run

### Command Line:
```bash
python main.py AAPL 25
```

### Web Application:
```bash
python app.py
# Open http://localhost:5000
```

---

## Expected Output

### Console Output (main.py):
```
==================================================
   STOCK PRICE PREDICTION
   Linear Regression vs LSTM
==================================================

[1] Collecting stock data...
    Data shape: (500, 6)

[2] Preprocessing data...
    LR - Train: (392, 6), Test: (98, 6)
    LSTM - Train: (352, 60, 1), Test: (88, 60, 1)

[3] Training Linear Regression...
=== Linear Regression Results ===
RMSE: $2.45
MAE: $1.89
R² Score: 0.9876

[4] Training LSTM...
=== LSTM Results ===
RMSE: $3.12
MAE: $2.34
R² Score: 0.9654

==================================================
   MODEL COMPARISON
==================================================
Metric          Linear Reg      LSTM           
---------------------------------------------
RMSE            $2.45           $3.12          
MAE             $1.89           $2.34          
R² Score        0.9876          0.9654         

Winner: Linear Regression
```

---

## Conclusion

This project demonstrates:
1. **Data Collection** - Using APIs to fetch real stock data
2. **Preprocessing** - Handling missing data and feature engineering
3. **Traditional ML** - Linear Regression for prediction
4. **Deep Learning** - LSTM for time-series forecasting
5. **Comparison** - Evaluating models using standard metrics
6. **Web Development** - Flask backend + JavaScript frontend
7. **Visualization** - Interactive charts for results
