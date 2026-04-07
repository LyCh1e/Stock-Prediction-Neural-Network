# Stock Price Prediction Neural Network

A multi-file neural network system that fetches stock data from **Yahoo Finance** (no API key required) and predicts OHLCV (Open, High, Low, Close, Volume) prices with scenario analysis, accuracy scoring, and full model persistence. Comes with a complete Tkinter GUI for managing multiple stocks simultaneously.

## Features

- **Multi-Output Neural Network**: Predicts all price points (Open, High, Low, Close, Volume) simultaneously
- **Advanced Scenario Analysis**: Generates Best Case, Average Case, and Worst Case forecasts
- **Yahoo Finance Integration**: Free, no API key needed — data fetched via `yfinance`
- **Adaptive Learning**: Experience-replay update — blends the most recent sequences with a random sample of older ones to adapt without forgetting prior patterns
- **Technical Indicators**: RSI, MACD, SMA (20/50), Bollinger Bands, Volume Ratio, Volatility, Momentum
- **Market Sentiment Analysis**: Bullish/bearish scoring across multiple indicator signals
- **Model Persistence (JSON + CSV)**: Network weights, scaler params, and prediction history saved to `stock_models.json` / `stock_models_history.csv` after every train, predict, and update — model resumes from where it left off on next startup
- **Accuracy Scoring**: Composite 0–100 score with letter grades (A+ → F) measuring price error, directional accuracy, and within-range hits — scored MAPE feeds back into the confidence calculation for each new prediction
- **Complete GUI**: Two-tab interface — Stock Manager and per-symbol Charts
- **Market Status Indicator**: Live color-coded label showing Pre-Market, Open, After-Hours, or Closed (US Eastern time)
- **Interactive Charts**: Actual price history with future forecast band (best/worst/avg); zoom / pan / save toolbar
- **Background Auto-Updates**: Data refreshed at ~1000 calls/hour across all tracked symbols; predictions refreshed every 5 minutes — the UI never freezes
- **Excel Export**: OHLCV history → `stock_data.xlsx`, prediction scenarios + daily scores → `stock_predictions.xlsx`

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch (recommended — checks dependencies first)
python launch.py

# Or launch the GUI directly
python stock_gui.py
```

No API key setup required. Add a symbol and training begins immediately.

## Installation

1. **Clone or download the repository**
2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Required packages: `numpy`, `pandas`, `matplotlib`, `yfinance`, `openpyxl`, `requests`

## System Architecture

### 1. Neural Network Core (`stock_volume_predictor.py`)
- `AdaptiveStockPredictor` — custom two-layer neural network (ReLU hidden, linear output) with a warm-up + step-decay learning rate schedule
- `YahooFinanceAPI` — fetches 365 days of OHLCV history via `yfinance`
- `StockTradingSystem` — orchestrates data fetching, feature engineering, training, prediction, and adaptive updates

### 2. Data & Thread Manager (`stock_store.py`)
- `StockStore` — central registry for all tracked symbols
- Runs training, prediction, and update jobs in background threads
- **JSON persistence**: `save_model_to_xlsx` / `load_model_from_xlsx` save and restore network weights and scaler params to `stock_models.json`; prediction history saved to `stock_models_history.csv`
- Appends OHLCV rows to `stock_data.xlsx` and prediction rows to `stock_predictions.xlsx`

### 3. Accuracy Scorer (`stock_scorer.py`)
- Matches past predictions to actual closing prices
- Computes a composite score (0–100) from three components:

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| MAPE accuracy | 50% | How close the predicted price was |
| Directional accuracy | 30% | Did the model get up/down right? |
| Within-range accuracy | 20% | Did the actual close land inside the best/worst band? |

- Letter grades: A+ (≥93) → F (<20)
- Scores update automatically after every prediction cycle

### 4. GUI (`stock_gui.py`)
- `StockPriceGUI` — two-tab Tkinter interface:
  - **Stock Manager tab**: tracked symbols table (symbol, price, sentiment, confidence, signal, status), market status indicator, action buttons, activity log
  - **Charts tab**: one sub-tab per symbol with actual price history, future forecast band (best/worst/avg), and a zoom/pan/save toolbar

### 5. Launcher (`launch.py`)
- Checks all dependencies before launching
- Prints startup summary and feature list

### 6. Auto-Updater (`stock_auto_updater.py`)
- `StockAutoUpdater` — background service that refreshes stock data on a configurable interval (default 60 min) and persists results to a JSON cache

## Usage

### GUI Workflow

```
1. python launch.py   (or python stock_gui.py)
2. Enter a stock symbol (e.g. AAPL) and click "Add & Train"
3. Watch training progress in the Log
4. Switch to the Charts tab to see the price history and forecast
5. Click "View Score" to evaluate prediction accuracy for a selected stock
6. Click "Predict All" to refresh forecasts
7. Click "Update All" for adaptive incremental learning
8. Click "Update Stock Data" or "Update Predictions" to write xlsx files
```

### GUI Button Reference

| Button | Action |
|--------|--------|
| Add & Train | Add a symbol and train the network (resumes from saved weights if available) |
| Quick Add (SPY/AAPL/MSFT) | Add the three default symbols at once |
| Predict All | Refresh predictions for every tracked stock |
| Update All | Adaptive update for every tracked stock |
| Remove Selected | Remove selected stock(s) from the tracker |
| Update Stock Data | Append new OHLCV rows to `stock_data.xlsx` |
| Update Predictions | Append new prediction rows to `stock_predictions.xlsx` |
| View Score | Open detailed accuracy score window for the selected stock |
| Refresh Charts | Redraw all chart tabs with latest data |

### Programmatic Usage

```python
from stock_volume_predictor import StockTradingSystem

# No API key needed
system = StockTradingSystem(lookback_window=10)

# Train on 180 days of Yahoo Finance data
system.train_model("AAPL", epochs=200)

# Predict next trading day with scenario analysis
prediction = system.predict_next_day("AAPL", include_scenarios=True)

print(f"Current Price:  ${prediction['current_price']:.2f}")
print(f"Sentiment:      {prediction['sentiment_analysis']['sentiment']}")
print(f"Confidence:     {prediction['confidence']*100:.1f}%")
print(f"Signal:         {prediction['recommendation']}")

scenarios = prediction["scenarios"]
print(f"Best close:     ${scenarios['best_case']['close']:.2f}")
print(f"Average close:  ${scenarios['average_case']['close']:.2f}")
print(f"Worst close:    ${scenarios['worst_case']['close']:.2f}")

# Adaptive incremental update with latest data
system.adaptive_update("AAPL")
```

### Accuracy Scoring

```python
from stock_scorer import score_symbol

# pred_history: list of {date, avg, best, worst} dicts
# raw_df:       OHLCV DataFrame from StockTradingSystem
result = score_symbol(pred_history, raw_df, current_price)

print(result.score)               # e.g. 73.4
print(result.letter_grade)        # e.g. "B"
print(result.summary)             # human-readable paragraph
print(result.matched_predictions)
```

## How It Works

### Data Pipeline
```
Yahoo Finance (yfinance) → 365-day OHLCV → Technical Indicators
    → Normalization → Sliding-window sequences → Neural Network
```

### Neural Network Architecture
```
Input Layer  (lookback_window × 12 features)
      ↓
Hidden Layer (30 neurons, ReLU + Monte Carlo Dropout at inference)
      ↓
Output Layer (5 outputs: Open, High, Low, Close, Volume)
```

**12 features per time step:**
1. Open
2. High
3. Low
4. Close
5. Volume
6. SMA-20
7. RSI (14-period)
8. Volume Ratio (vs 20-day average)
9. Volatility (20-day std)
10. MACD
11. MACD Signal
12. Price Momentum (10-day)

### Uncertainty Estimation
Prediction uncertainty is estimated via **Monte Carlo Dropout**: during inference, hidden units are randomly zeroed out across 100 forward passes and the standard deviation of outputs is used as the uncertainty signal. This reflects genuine model uncertainty rather than injected noise.

### Confidence Score
The confidence value shown in the GUI is a weighted combination of:
- **Scored MAPE** (weight 3×) — the model's real historical error rate from the accuracy scorer, when available. This is the primary signal once predictions have been matched to actual prices.
- MC-Dropout uncertainty relative to current price (weight 1.5×)
- Data recency (weight 1×)
- Data quantity (weight 0.5×)
- Sentiment confidence (weight 0.5×)
- Training loss proxy (weight 1×, only when no scored MAPE exists yet)

### Scenario Generation
Three scenarios are derived from the base prediction using the model's directly predicted High and Low values, shaped by:
- Current RSI and MACD readings
- Market sentiment score
- Recent price volatility
- Model confidence

### Model Persistence Flow
```
_train_thread
    → load_model_from_xlsx  (checks saved input_size matches current config;
                             warns and retrains from scratch on shape mismatch)
    → train_model           (full epochs if new; ~20% epochs if resuming)
    → save_model_to_xlsx    (weights + scaler → stock_models.json,
                             pred_history    → stock_models_history.csv)

_predict_thread / _update_thread
    → adaptive_update       (experience replay: recent + random older sequences,
                             5 gradient steps per call)
    → archive_prediction    (push current pred into history)
    → score_symbol          (update accuracy_score from pred_history)
    → predict_next_day      (passes scored_mape into confidence calculation)
    → save_model_to_xlsx
```

### Excel File Layout

| File | Contents |
|------|---------|
| `stock_data.xlsx` | One sheet per symbol — append-only OHLCV history |
| `stock_predictions.xlsx` | One sheet per symbol — scenario rows + daily score rows |

### Persistence File Layout

| File | Contents |
|------|---------|
| `stock_models.json` | Network weights and scaler params per symbol |
| `stock_models_history.csv` | Full prediction log (Symbol, Date, Avg, Best, Worst) |
| `tracked_symbols.json` | Tracked symbols with lookback and epoch settings |

### Market Status Indicator

The status bar shows the current US market session based on Eastern time:

| Status | Hours (ET) | Colour |
|--------|-----------|--------|
| Pre-Market | 4:00 AM – 9:30 AM | Orange |
| Market Open | 9:30 AM – 4:00 PM | Green |
| After-Hours | 4:00 PM – 8:00 PM | Blue |
| Closed / Weekend | All other times | Grey |

Updates every 60 seconds automatically.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback_window` | 10 | Historical days used as input to the network |
| `hidden_size` | 30 | Neurons in the hidden layer |
| `learning_rate` | 0.001 | Initial LR; warms up over first 10 epochs then decays 5% every 20 epochs |
| `epochs` | 200 | Training iterations (full train); ~20% for resume |
| `input_features` | 12 | Features per time step |
| `outputs` | 5 | Predicted values (OHLCV) |

## Files in the Project

| File | Purpose |
|------|---------|
| `stock_volume_predictor.py` | Neural network, Yahoo Finance API, trading system |
| `stock_store.py` | Data manager, background threads, model persistence |
| `stock_scorer.py` | Prediction accuracy scoring and letter grades |
| `stock_gui.py` | Tkinter GUI — stock manager and interactive charts |
| `stock_auto_updater.py` | Background auto-refresh service with JSON cache |
| `launch.py` | Dependency checker and launcher |
| `requirements.txt` | Python package dependencies |
| `tracked_symbols.json` | Auto-generated: saved symbols and their settings |
| `stock_models.json` | Auto-generated: saved network weights + scaler params |
| `stock_models_history.csv` | Auto-generated: full prediction history log |
| `stock_data.xlsx` | Auto-generated: OHLCV history |
| `stock_predictions.xlsx` | Auto-generated: prediction scenarios + scores |

## Technical Indicators

1. **Moving Averages** — SMA-20, SMA-50
2. **RSI** — 14-period Relative Strength Index
3. **Bollinger Bands** — 20-day, ±2 std
4. **MACD** — (12, 26, 9)
5. **Volume Indicators** — SMA, ratio vs average
6. **Volatility** — 20-day standard deviation
7. **Price Momentum** — 10-day
8. **Support & Resistance** levels
9. **Trend Analysis** — short and long term

## Troubleshooting

**Missing packages:**
```bash
pip install numpy pandas matplotlib yfinance openpyxl requests
```

**"No data available" for a symbol:**
- Verify the ticker is valid on Yahoo Finance (e.g. `AAPL`, `MSFT`, `SPY`)
- Check your internet connection
- `yfinance` may be temporarily rate-limited; wait a moment and retry

**GUI not launching:**
```bash
python --version   # requires Python 3.9+
python launch.py   # checks dependencies before opening the GUI
```

**Weights not loading on startup:**
- `stock_models.json` is created after the first successful training run
- If the saved `input_size` doesn't match the current feature count (e.g. after a feature set change), the model logs a clear warning and retrains from scratch — old weights are not silently discarded without notice

## Disclaimer

**IMPORTANT**: This is an educational project for demonstration purposes only.

- **Not Financial Advice**: Predictions are based on historical patterns and must not be used for actual trading decisions
- **Market Risk**: Stock markets are volatile and past performance is not indicative of future results
- **Model Limitations**: Neural networks have inherent uncertainty — treat outputs as one signal among many

Always consult a qualified financial professional before making investment decisions.
