# Stock Volume Prediction Neural Network

A neural network system that fetches stock data from an API and predicts trading volumes with adaptive learning and a complete GUI for multi-stock management.

## Features

- **Neural Network**: Custom feedforward network with backpropagation
- **API Integration**: Fetches real-time stock data from Massive API
- **Adaptive Learning**: Continuously learns and adapts to new trading trends
- **Volume Prediction**: Predicts next-day trading volumes based on historical patterns
- **Model Persistence**: Save and load trained models
- **🆕 Graphical Interface**: User-friendly GUI for managing multiple stocks
- **🆕 Multi-Stock Support**: Track and predict volumes for unlimited stocks simultaneously
- **🆕 Real-time Visualization**: Charts comparing predictions vs actual volumes
- **🆕 Background Processing**: Train multiple stocks without freezing the interface

## Quick Start (GUI)

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Launch the GUI:**
```bash
python stock_gui.py
# or use the launcher
python launch.py
```

3. **Get started:**
   - Enter your Massive API key (get free at link in GUI)
   - Add stock symbols (AAPL, MSFT, GOOGL, etc.)
   - Watch training progress in real-time
   - View predictions and visualizations!

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Get a free API key from Massive:
   - Visit: https://www.massiveapi.com/signup
   - Sign up for a free API key (generous rate limits for educational use)

3. Update the API key in the script:
```python
API_KEY = "your_actual_massive_api_key_here"
```

## Usage

### Basic Usage

```python
from stock_volume_predictor import TradingVolumeSystem

# Initialize with your API key
system = TradingVolumeSystem(api_key="YOUR_MASSIVE_API_KEY", lookback_window=10)

# Train on a stock symbol
system.train_model("AAPL", epochs=150)

# Predict next day's volume
prediction = system.predict_next_volume("AAPL")
print(f"Predicted volume: {prediction['predicted_volume']:,}")

# Save the trained model
system.model.save_model('my_model.pkl')
```

### Adaptive Learning

The system can incrementally learn from new data:

```python
# After initial training, update with latest data
system.adaptive_update("AAPL")

# This adjusts the model weights based on recent trends
# without forgetting previous patterns
```

### Loading a Saved Model

```python
system = TradingVolumeSystem(api_key="YOUR_MASSIVE_API_KEY")
system.model.load_model('my_model.pkl')

# Continue making predictions
prediction = system.predict_next_volume("AAPL")
```

## GUI Usage

The GUI provides an intuitive interface for managing multiple stocks:

### Main Features

1. **Multi-Stock Tracking**: Add and monitor multiple stocks simultaneously
2. **Background Training**: Train models without freezing the interface
3. **Real-time Updates**: See training progress and predictions as they happen
4. **Visualization**: Compare predictions vs actual volumes with interactive charts
5. **Export**: Save all predictions to JSON for further analysis

### Quick GUI Workflow

```
1. Launch: python stock_gui.py
2. Enter API key → Click "Save API Key"
3. Add stocks:
   - Symbol: AAPL → Add & Train Stock
   - Symbol: MSFT → Add & Train Stock
   - Symbol: GOOGL → Add & Train Stock
4. Watch training in Activity Log
5. Click "Show Comparison" to visualize
6. Click "Predict All" to update forecasts
7. Click "Update All" for adaptive learning
8. Click "Export Data" to save results
```

### GUI Control Reference

| Button | Action |
|--------|--------|
| Add & Train Stock | Add new stock and start training |
| Predict Selected | Update prediction for selected stock(s) |
| Update Selected | Adaptive learning for selected stock(s) |
| Predict All | Refresh all predictions |
| Update All | Adaptive learning for all stocks |
| Remove Selected | Delete stock from tracking |
| Export Data | Save predictions to JSON |
| Show Comparison | Bar chart of predicted vs actual |
| Show Training Loss | Line chart of training progress |

## How It Works

1. **Data Fetching**: Uses Massive API to get historical daily stock data
2. **Feature Engineering**: Creates sequences of historical volumes (default: 10 days)
3. **Normalization**: Normalizes data for better neural network performance
4. **Training**: Uses gradient descent with adaptive learning rate
5. **Prediction**: Predicts next day's volume based on recent patterns
6. **Adaptation**: Incrementally updates weights when new data arrives

## Neural Network Architecture

```
Input Layer (10 neurons) → Hidden Layer (20 neurons, ReLU) → Output (1 neuron)
```

- **Input**: Last N days of trading volumes (default: 10)
- **Hidden Layer**: 20 neurons with ReLU activation
- **Output**: Predicted next-day volume
- **Learning**: Backpropagation with adaptive learning rate

## Key Parameters

- `lookback_window`: Number of historical days to use (default: 10)
- `hidden_size`: Neurons in hidden layer (default: 20)
- `learning_rate`: Initial learning rate (default: 0.001)
- `epochs`: Training iterations (default: varies by use case)

## Adaptive Learning Features

- **Adaptive Learning Rate**: Automatically reduces if loss oscillates
- **Incremental Updates**: Gently adapts to new trends without catastrophic forgetting
- **Error Tracking**: Monitors prediction errors over time

## Example Output

```
Training model on AAPL...
Fetched 100 days of data from Massive API
Training on 90 sequences
Epoch 0, Loss: 0.234567
Epoch 20, Loss: 0.098765
...
Epoch 140, Loss: 0.012345

Making prediction for next trading day...
Symbol: AAPL
Predicted Volume for 2026-01-11: 52,345,678 shares
Last Actual Volume: 51,234,567 shares
Recent Average Volume: 49,876,543 shares
```

## Limitations

- **API Rate Limits**: Check Massive API's current rate limits for your plan
- **Simple Architecture**: This is a basic network for demonstration
- **Market Complexity**: Real trading involves many more factors
- **No Financial Advice**: This is for educational purposes only

## Extending the System

You can enhance this system by:

1. Adding more features (price, technical indicators, market sentiment)
2. Using LSTM/GRU for better sequence modeling
3. Incorporating multiple stocks for cross-learning
4. Adding ensemble methods
5. Implementing more sophisticated normalization
6. Integrating additional data sources alongside Massive API

## API Information

This project uses the **Massive API** for stock data:
- **Website**: https://www.massiveapi.com
- **Documentation**: https://docs.massiveapi.com
- **Free Tier**: Available for educational and personal projects
- **Data Coverage**: Global stock markets with historical data

## Disclaimer

This is an educational project. Trading decisions should not be based solely on volume predictions. Always consult with financial professionals and do your own research before making investment decisions.