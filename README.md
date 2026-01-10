# Stock Price Prediction Neural Network with Scenario Analysis

A comprehensive neural network system that fetches stock data from the Massive API and predicts OHLCV (Open, High, Low, Close, Volume) prices with advanced scenario analysis capabilities. Features a complete GUI for managing multiple stocks with real-time visualization.

## Features

- **Multi-Output Neural Network**: Predicts all price points (Open, High, Low, Close, Volume) simultaneously
- **Advanced Scenario Analysis**: Generates Best Case, Average Case, and Worst Case scenarios
- **Massive API Integration**: Fetches real-time stock data with Bearer token authentication
- **Adaptive Learning**: Continuously learns and adapts to new trading trends
- **Technical Indicators**: Calculates RSI, MACD, Moving Averages, Bollinger Bands, and more
- **Market Sentiment Analysis**: Determines bullish/bearish signals based on multiple factors
- **Model Persistence**: Save and load trained models
- **Complete GUI**: User-friendly interface for managing multiple stocks
- **Multi-Stock Support**: Track and predict prices for unlimited stocks simultaneously
- **Real-time Visualization**: Interactive charts comparing scenario predictions
- **Background Processing**: Train and update models without freezing the interface

## Quick Start

### Option 1: Use the Launcher (Recommended)
```bash
# Run the launcher script
python launch.py
```

### Option 2: Direct GUI Launch
```bash
# Install dependencies first
pip install -r requirements.txt

# Launch the GUI directly
python stock_gui.py
```

## Installation

1. **Clone or download the repository**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

If requirements.txt is not available, install manually:
```bash
pip install numpy pandas requests matplotlib tkinter
```

3. **Get your Massive API key:**
   - Visit: https://www.massiveapi.com/signup
   - Sign up for a free API key
   - Copy your Bearer token

## System Architecture

The system consists of three main components:

### 1. Neural Network Core (`stock_volume_predictor.py`)
- `AdaptiveStockPredictor`: Custom neural network for multi-output prediction
- `MassiveAPI`: Handles API communication with Bearer authentication
- `StockTradingSystem`: Main system orchestrator with scenario analysis

### 2. GUI Interface (`stock_gui.py`)
- `StockPriceGUI`: Complete graphical interface with:
  - Multi-stock management
  - Real-time visualizations
  - Scenario comparison charts
  - Profit potential analysis
  - Activity log and status monitoring

### 3. Launcher (`launch.py`)
- Dependency checking
- Automatic system validation
- User-friendly startup interface

## Usage

### Basic Usage (Command Line)

```python
from stock_volume_predictor import StockTradingSystem

# Initialize with your API key
system = StockTradingSystem(api_key="YOUR_MASSIVE_API_KEY", lookback_window=10)

# Train on a stock symbol
system.train_model("AAPL", epochs=200)

# Predict next day's prices with scenario analysis
prediction = system.predict_next_day("AAPL", include_scenarios=True)

# Access prediction results
print(f"Current Price: ${prediction['current_price']:.2f}")
print(f"Market Sentiment: {prediction['market_sentiment']}")
print(f"Confidence: {prediction['confidence']*100:.1f}%")

# Access different scenarios
scenarios = prediction['scenarios']
print(f"Best Case Buy: ${scenarios['best_case']['buy_price']:.2f}")
print(f"Average Case Profit: {scenarios['average_case']['profit_potential']:.1f}%")

# Save the trained model
system.model.save_model('aapl_model.pkl')

# Load a saved model
system.model.load_model('aapl_model.pkl')
```

### Adaptive Learning

```python
# Update model with latest data (incremental learning)
system.adaptive_update("AAPL")

# This adjusts the model weights based on recent trends
# without forgetting previous patterns
```

### GUI Usage

The GUI provides an intuitive interface with the following features:

#### Main Workflow:
```
1. Launch: python launch.py  or  python stock_gui.py
2. Enter your Massive API key → Click "Save API Key"
3. Add stocks:
   - Symbol: SPY → Add & Train Stock
   - Symbol: AAPL → Add & Train Stock
   - Symbol: MSFT → Add & Train Stock
4. Watch training progress in Activity Log
5. Double-click any stock to see detailed analysis
6. Click "Show Scenario Comparison" for visualizations
7. Click "Predict All" to update forecasts
8. Click "Update All" for adaptive learning
9. Click "Export Data" to save all predictions
```

#### GUI Control Reference:

| Button | Action |
|--------|--------|
| Add & Train Stock | Add new stock and start training neural network |
| Predict Selected | Generate price predictions for selected stock(s) |
| Update Selected | Adaptive learning for selected stock(s) |
| Predict All | Refresh predictions for all tracked stocks |
| Update All | Incremental update for all stock models |
| Remove Selected | Delete stock from tracking list |
| Export Data | Save all predictions to JSON file |
| Show Scenario Comparison | Visual comparison of Best/Average/Worst cases |
| Show Price Trend | Historical trend visualization |
| Show Profit Potential | Compare profit potential across stocks |
| Quick Add (SPY, AAPL, MSFT) | Add popular stocks with one click |

#### Key Features in GUI:
- **Double-click any stock** for detailed scenario analysis
- **Color-coded scenarios**: Green (Best), Blue (Average), Red (Worst)
- **Real-time confidence scores** for each prediction
- **Market sentiment indicators**
- **Trading recommendations** (STRONG BUY, BUY, HOLD, SELL, STRONG SELL)
- **Technical indicators** display (RSI, MACD, Moving Averages)
- **Background training** without UI freezing

## How It Works

### 1. Data Pipeline
```
Massive API → OHLCV Data → Technical Indicators → Normalization → Neural Network
```

### 2. Neural Network Architecture
```
Input Layer (N × 9 features)
    ↓
Hidden Layer (30 neurons, ReLU activation)
    ↓
Output Layer (5 outputs: Open, High, Low, Close, Volume)
```

**Features used per time step:**
1. Open Price
2. High Price
3. Low Price
4. Close Price
5. Volume
6. 20-day SMA
7. RSI (14-period)
8. Volume Ratio (vs 20-day average)
9. Volatility (20-day std)

### 3. Scenario Generation
The system generates three scenarios based on:
- Current market sentiment (bullish/bearish)
- Technical indicator readings (RSI, MACD)
- Historical volatility
- Confidence scores
- Recent price patterns

### 4. Confidence Calculation
Prediction confidence is calculated from:
- Data recency and quality
- Volatility levels
- Technical indicator consistency
- Market sentiment strength
- Model performance history

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback_window` | 10 | Historical days used for prediction |
| `hidden_size` | 30 | Neurons in hidden layer |
| `learning_rate` | 0.001 | Initial learning rate for gradient descent |
| `epochs` | 200 | Training iterations |
| `input_features` | 9 | Features per time step (OHLCV + indicators) |
| `outputs` | 5 | Predicted values (Open, High, Low, Close, Volume) |

## Example Output

```
Training model on AAPL...
Fetched 180 days of data from Massive API
Training on 170 sequences with 90 features
Epoch 0, Loss: 0.234567
Epoch 20, Loss: 0.098765
...
Epoch 180, Loss: 0.012345

Making prediction for next trading day...

STOCK DETAILS: AAPL
============================================================
Current Price:      $175.42
Market Sentiment:   BULLISH
Prediction Date:    2026-01-11
Confidence Level:   84.3%
Last Updated:       2026-01-10T14:30:45

SCENARIO ANALYSIS
============================================================

>>> BEST CASE (Optimistic) <<<
  Buy Price:        $173.25
  Sell Price:       $182.50
  Expected High:    $183.75
  Expected Low:     $172.80
  Profit Potential: 5.3%
  Volume:           58,234,567

>>> AVERAGE CASE (Most Likely) <<<
  Buy Price:        $175.80
  Sell Price:       $179.25
  Expected High:    $180.15
  Expected Low:     $174.90
  Profit Potential: 2.0%
  Volume:           52,345,678

>>> WORST CASE (Conservative) <<<
  Buy Price:        $177.10
  Sell Price:       $176.80
  Expected High:    $178.25
  Expected Low:     $175.50
  Profit Potential: -0.2%
  Volume:           47,856,432

TRADING RECOMMENDATION
============================================================
BUY - Positive profit potential with good confidence
```

## Technical Indicators Calculated

1. **Moving Averages** (SMA 20, SMA 50)
2. **Relative Strength Index** (RSI 14-period)
3. **Bollinger Bands** (20-day, 2 std)
4. **MACD** (12, 26, 9)
5. **Volume Indicators** (SMA, ratios)
6. **Volatility** (20-day standard deviation)
7. **Price Momentum** (10-day)
8. **Support & Resistance Levels**
9. **Trend Analysis** (short/long term)

## Files in the Project

| File | Purpose |
|------|---------|
| `stock_volume_predictor.py` | Core neural network and API implementation |
| `stock_gui.py` | Complete graphical user interface |
| `launch.py` | Dependency checker and launcher |
| `requirements.txt` | Python package dependencies |
| `README.md` | This documentation file |

## Extending the System

You can enhance this system by:

1. **Adding More Features:**
   ```python
   # Add sentiment analysis from news/social media
   # Incorporate options data
   # Add macroeconomic indicators
   ```

2. **Model Improvements:**
   ```python
   # Use LSTM/GRU for better sequence modeling
   # Implement attention mechanisms
   # Add ensemble methods with multiple models
   ```

3. **Advanced Features:**
   ```python
   # Portfolio optimization
   # Risk management modules
   # Backtesting framework
   # Real-time alerts
   ```

4. **Data Sources:**
   ```python
   # Integrate multiple data providers
   # Add fundamental analysis data
   # Include alternative data sources
   ```

## API Information

This project uses the **Massive API** for stock data:
- **Website**: https://www.massiveapi.com
- **Documentation**: https://docs.massiveapi.com
- **Authentication**: Bearer token
- **Free Tier**: Available for educational and personal projects
- **Data Coverage**: Global stock markets with historical OHLCV data

## Troubleshooting

### Common Issues:

1. **"Missing packages" error:**
   ```bash
   pip install numpy pandas requests matplotlib
   ```

2. **"API key not working" error:**
   - Verify your Massive API key is valid
   - Check internet connection
   - Ensure API key is saved in the GUI

3. **"No data available" message:**
   - The system will use simulated data for demonstration
   - Check if the stock symbol is valid
   - Verify API rate limits aren't exceeded

4. **GUI not launching:**
   ```bash
   # Check Python version (requires 3.6+)
   python --version
   
   # Try running directly
   python stock_gui.py
   ```

### Performance Tips:
- Start with 1-2 stocks for initial testing
- Use default parameters for first runs
- Allow training to complete fully before predictions
- Use "Quick Add" for popular stocks with known patterns

## Disclaimer

**IMPORTANT**: This is an educational project for demonstration purposes only.

- **Not Financial Advice**: Predictions are based on historical patterns and should not be used for actual trading decisions
- **Market Risk**: Stock markets are volatile and unpredictable
- **API Limitations**: Free API tiers have rate limits and may not provide real-time data
- **Model Limitations**: Neural networks have inherent uncertainties and should be one of many tools in analysis

Always conduct your own research and consult with qualified financial professionals before making investment decisions. Past performance is not indicative of future results.