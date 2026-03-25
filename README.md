# Stock Price Prediction Neural Network

A multi-file neural network system that fetches stock data from **Yahoo Finance** (no API key required) and predicts OHLCV (Open, High, Low, Close, Volume) prices with scenario analysis, accuracy scoring, and full model persistence. Comes with a complete Tkinter GUI for managing multiple stocks simultaneously.

## Features

- **Multi-Output Neural Network**: Predicts all price points (Open, High, Low, Close, Volume) simultaneously
- **Advanced Scenario Analysis**: Generates Best Case, Average Case, and Worst Case forecasts
- **Yahoo Finance Integration**: Free, no API key needed — data fetched via `yfinance`
- **Adaptive Learning**: Continues learning from new data on each update cycle
- **Technical Indicators**: RSI, MACD, SMA (20/50), Bollinger Bands, Volume Ratio, Volatility, Momentum
- **Market Sentiment Analysis**: Bullish/bearish scoring across multiple indicator signals
- **Model Persistence (xlsx)**: Network weights, scaler params, and prediction history are saved to `stock_models.xlsx` automatically after every train, predict, and update — so the model resumes from where it left off on next startup
- **Accuracy Scoring**: Composite 0–100 score with letter grades (A+ → F) measuring price error, directional accuracy, and within-range hits
- **Complete GUI**: Two-tab interface — Stock Manager and per-symbol Charts
- **Interactive Charts**: Past predictions overlaid on actual price data; future forecast with best/worst band; zoom / pan / save toolbar
- **Background Processing**: All training and prediction runs in daemon threads — the UI never freezes
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
```
python stock_gui.py
pip install -r requirements.txt
```

Required packages: `numpy`, `pandas`, `matplotlib`, `yfinance`, `openpyxl`, `requests`

## System Architecture

### 1. Neural Network Core (`stock_volume_predictor.py`)
- `AdaptiveStockPredictor` — custom two-layer neural network (ReLU hidden, linear output) with adaptive learning rate
- `YahooFinanceAPI` — fetches 180 days of OHLCV history via `yfinance`
- `StockTradingSystem` — orchestrates data fetching, feature engineering, training, prediction, and adaptive updates

### 2. Data & Thread Manager (`stock_store.py`)
- `StockStore` — central registry for all tracked symbols
- Runs training, prediction, and update jobs in background threads
- **xlsx persistence**: `save_model_to_xlsx` / `load_model_from_xlsx` save and restore network weights, scaler params, and prediction history to `stock_models.xlsx`
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
  - **Stock Manager tab**: tracked symbols table (symbol, price, sentiment, confidence, signal, score, status), action buttons, activity log
  - **Charts tab**: one sub-tab per symbol with actual price history, past-prediction overlay, future forecast band, and a zoom/pan/save toolbar

### 5. Launcher (`launch.py`)
- Checks all dependencies before launching
- Prints startup summary and feature list

### 6. Auto-Updater (`stock_auto_updater.py`)
- `StockAutoUpdater` — background service that refreshes stock data on a configurable interval (default 60 min) and persists results to a JSON cache

## Usage

### GUI Workflow

```
1. python launch.py   (or python stock_gui.py)
2. Enter a stock symbol (e.g. AAPL) and click "Add & Train Stock"
3. Watch training progress in the Activity Log
4. Switch to the Charts tab to see the price history and forecast
5. Use "Score Selected" / "Score All" to evaluate prediction accuracy
6. Click "Predict Selected" or "Predict All" to refresh forecasts
7. Click "Update Selected" or "Update All" for adaptive incremental learning
8. Click "Export Stock Data" or "Export Predictions" to write xlsx files
```

### GUI Button Reference

| Button | Action |
|--------|--------|
| Add & Train Stock | Add a symbol and train the network (resumes from saved weights if available) |
| Predict Selected | Generate a new prediction for the selected stock(s) |
| Update Selected | Adaptive incremental update for selected stock(s) |
| Predict All | Refresh predictions for every tracked stock |
| Update All | Adaptive update for every tracked stock |
| Score Selected | Compute accuracy score for selected stock(s) |
| Score All | Compute accuracy scores for all tracked stocks |
| View Score | Open detailed per-prediction breakdown window |
| Remove Selected | Remove selected stock(s) from the tracker |
| Export Stock Data | Write OHLCV history to `stock_data.xlsx` |
| Export Predictions | Write prediction scenarios + daily scores to `stock_predictions.xlsx` |

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

# Save / load weights manually (pickle)
system.model.save_model("aapl_model.pkl")
system.model.load_model("aapl_model.pkl")

# Adaptive incremental update with latest data
system.adaptive_update("AAPL")
```

### Accuracy Scoring

```python
from stock_scorer import score_symbol

# pred_history: list of {date, avg, best, worst} dicts
# raw_df:       OHLCV DataFrame from StockTradingSystem
result = score_symbol(pred_history, raw_df, current_price)

print(result.score)              # e.g. 73.4
print(result.letter_grade)       # e.g. "B"
print(result.summary)            # human-readable paragraph
print(result.matched_predictions)
```

## How It Works

### Data Pipeline
```
Yahoo Finance (yfinance) → 180-day OHLCV → Technical Indicators
    → Normalization → Sliding-window sequences → Neural Network
```

### Neural Network Architecture
```
Input Layer  (lookback_window × 9 features)
      ↓
Hidden Layer (30 neurons, ReLU)
      ↓
Output Layer (5 outputs: Open, High, Low, Close, Volume)
```

**9 features per time step:**
1. Open
2. High
3. Low
4. Close
5. Volume
6. SMA-20
7. RSI (14-period)
8. Volume Ratio (vs 20-day average)
9. Volatility (20-day std)

### Scenario Generation
Three scenarios are derived from the base prediction by applying volatility multipliers shaped by:
- Current RSI and MACD readings
- Market sentiment score
- Recent price volatility
- Model confidence

### Model Persistence Flow
```
_train_thread
    → load_model_from_xlsx  (if saved weights exist → resume & short retrain)
    → train_model           (full epochs if new; ~20% epochs if resuming)
    → save_model_to_xlsx    (weights + scaler + pred_history)

_predict_thread / _update_thread
    → predict / adaptive_update
    → archive_prediction    (push current pred into history)
    → save_model_to_xlsx
```

### Excel File Layout

| File | Contents |
|------|---------|
| `stock_data.xlsx` | One sheet per symbol — append-only OHLCV history |
| `stock_predictions.xlsx` | One sheet per symbol — scenario rows + daily score rows |
| `stock_models.xlsx` | `{SYMBOL}` sheet: latest network weights (JSON) + scaler params; `{SYMBOL}_history` sheet: full prediction log |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback_window` | 10 | Historical days used as input to the network |
| `hidden_size` | 30 | Neurons in the hidden layer |
| `learning_rate` | 0.001 | Initial learning rate |
| `epochs` | 200 | Training iterations (full train); ~20% for resume |
| `input_features` | 9 | Features per time step |
| `outputs` | 5 | Predicted values (OHLCV) |

## Files in the Project

| File | Purpose |
|------|---------|
| `stock_volume_predictor.py` | Neural network, Yahoo Finance API, trading system |
| `stock_store.py` | Data manager, background threads, xlsx persistence |
| `stock_scorer.py` | Prediction accuracy scoring and letter grades |
| `stock_gui.py` | Tkinter GUI — stock manager and interactive charts |
| `stock_auto_updater.py` | Background auto-refresh service with JSON cache |
| `launch.py` | Dependency checker and launcher |
| `requirements.txt` | Python package dependencies |
| `stock_data.xlsx` | Auto-generated: OHLCV history |
| `stock_predictions.xlsx` | Auto-generated: prediction scenarios + scores |
| `stock_models.xlsx` | Auto-generated: saved network weights + history |

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
python --version   # requires Python 3.7+
python launch.py   # checks dependencies before opening the GUI
```

**`stock_models.xlsx` weights not loading:**
- The file is created after the first successful training run
- If the saved `input_size` doesn't match the current `lookback_window`, the model trains from scratch and overwrites the old weights

## Disclaimer

**IMPORTANT**: This is an educational project for demonstration purposes only.

- **Not Financial Advice**: Predictions are based on historical patterns and must not be used for actual trading decisions
- **Market Risk**: Stock markets are volatile and past performance is not indicative of future results
- **Model Limitations**: Neural networks have inherent uncertainty — treat outputs as one signal among many

Always consult a qualified financial professional before making investment decisions.