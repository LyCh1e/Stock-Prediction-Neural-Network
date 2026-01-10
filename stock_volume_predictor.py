"""
MASSIVE API Stock Price Predictor
Uses Massive Inc API for comprehensive stock analysis
"""

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import json
import pickle
from typing import Tuple, Optional, Dict, List

class AdaptiveStockPredictor:
    """
    Neural network for predicting stock prices with adaptive learning.
    Predicts multiple price points: buy, sell, high, low, close.
    """
    
    def __init__(self, input_size: int = 10, hidden_size: int = 30, learning_rate: float = 0.001):
        """
        Initialize the neural network for multi-output prediction.
        
        Args:
            input_size: Number of input features (historical data points * features)
            hidden_size: Number of neurons in hidden layer
            learning_rate: Learning rate for gradient descent
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Initialize weights for multi-output prediction (5 outputs: open, high, low, close, volume)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 5) * np.sqrt(2.0 / hidden_size)  # 5 outputs
        self.b2 = np.zeros((1, 5))
        
        # Track performance
        self.losses = []
        self.prediction_errors = []
        
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through the network.
        
        Returns:
            z1: Pre-activation hidden layer
            a1: Activated hidden layer
            output: Final predictions (5 columns)
        """
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        output = np.dot(a1, self.W2) + self.b2
        return z1, a1, output
    
    def backward(self, X: np.ndarray, y: np.ndarray, z1: np.ndarray, 
                 a1: np.ndarray, output: np.ndarray) -> None:
        """
        Backward pass - update weights using gradient descent.
        """
        m = X.shape[0]
        
        # Calculate gradients
        dL_doutput = 2 * (output - y) / m
        dL_dW2 = np.dot(a1.T, dL_doutput)
        dL_db2 = np.sum(dL_doutput, axis=0, keepdims=True)
        
        dL_da1 = np.dot(dL_doutput, self.W2.T)
        dL_dz1 = dL_da1 * self.relu_derivative(z1)
        dL_dW1 = np.dot(X.T, dL_dz1)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
        
        # Update weights with adaptive learning rate
        adaptive_lr = self.learning_rate
        if len(self.losses) > 10:
            recent_losses = self.losses[-10:]
            if np.std(recent_losses) > np.mean(recent_losses) * 0.5:
                adaptive_lr *= 0.5
        
        self.W2 -= adaptive_lr * dL_dW2
        self.b2 -= adaptive_lr * dL_db2
        self.W1 -= adaptive_lr * dL_dW1
        self.b1 -= adaptive_lr * dL_db1
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> None:
        """
        Train the network on data.
        
        Args:
            X: Input features (normalized)
            y: Target values (normalized, 5 columns)
            epochs: Number of training iterations
        """
        for epoch in range(epochs):
            # Forward pass
            z1, a1, output = self.forward(X)
            
            # Calculate loss
            loss = np.mean((output - y) ** 2)
            self.losses.append(loss)
            
            # Backward pass
            self.backward(X, y, z1, a1, output)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        _, _, output = self.forward(X)
        return output
    
    def incremental_update(self, X_new: np.ndarray, y_new: np.ndarray) -> None:
        """
        Update the model with new data.
        """
        z1, a1, output = self.forward(X_new)
        self.backward(X_new, y_new, z1, a1, output)
        
        error = np.mean(np.abs(output - y_new))
        self.prediction_errors.append(error)
    
    def save_model(self, filepath: str) -> None:
        """Save model weights to file"""
        model_data = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'learning_rate': self.learning_rate,
            'losses': self.losses,
            'prediction_errors': self.prediction_errors
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model weights from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.W1 = model_data['W1']
        self.b1 = model_data['b1']
        self.W2 = model_data['W2']
        self.b2 = model_data['b2']
        self.input_size = model_data['input_size']
        self.hidden_size = model_data['hidden_size']
        self.learning_rate = model_data['learning_rate']
        self.losses = model_data.get('losses', [])
        self.prediction_errors = model_data.get('prediction_errors', [])
        print(f"Model loaded from {filepath}")


class MASSIVEStockFetcher:
    """
    Fetch stock data from MASSIVE Inc API.
    Comprehensive financial data including technical indicators.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.massive.com/v3/reference/dividends?apiKey=rftTATcM0oV2iz3MVnj2HObaTOAn0Dwl"
    
    def fetch_stock_data(self, symbol: str, interval: str = 'daily', 
                         period: str = '1month') -> pd.DataFrame:
        """
        Fetch stock data from MASSIVE API.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            interval: Data interval ('daily', 'hourly', 'weekly')
            period: Time period ('1day', '1week', '1month', '3month', '6month', '1year')
        
        Returns:
            DataFrame with OHLCV data and technical indicators
        """
        params = {
            'function': 'TIME_SERIES',
            'symbol': symbol,
            'interval': interval,
            'period': period,
            'apikey': self.api_key,
            'output': 'full',
            'indicators': 'all'  # Get all technical indicators
        }
        
        print(f"Fetching MASSIVE data for {symbol}...")
        
        try:
            response = requests.get(f"{self.base_url}/stocks", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data or 'timeseries' not in data['data']:
                raise ValueError("Invalid response format from MASSIVE API")
            
            # Parse timeseries data
            timeseries = data['data']['timeseries']
            
            # Create DataFrame
            records = []
            for timestamp, values in timeseries.items():
                record = {
                    'timestamp': pd.to_datetime(timestamp),
                    'open': float(values.get('open', 0)),
                    'high': float(values.get('high', 0)),
                    'low': float(values.get('low', 0)),
                    'close': float(values.get('close', 0)),
                    'volume': int(values.get('volume', 0))
                }
                
                # Add technical indicators if available
                indicators = values.get('indicators', {})
                for indicator, value in indicators.items():
                    record[indicator] = float(value)
                
                records.append(record)
            
            df = pd.DataFrame(records)
            df = df.sort_values('timestamp')
            df.set_index('timestamp', inplace=True)
            
            print(f"Retrieved {len(df)} data points with {len(df.columns)} features")
            return df
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"MASSIVE API error: {str(e)}")
        except (KeyError, ValueError) as e:
            raise ValueError(f"Data parsing error: {str(e)}")
    
    def get_market_sentiment(self, symbol: str) -> Dict:
        """
        Get market sentiment and analyst recommendations.
        """
        params = {
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(f"{self.base_url}/sentiment", params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except:
            # Return default sentiment if API fails
            return {
                'sentiment': 'neutral',
                'buy_rating': 2.5,
                'sell_rating': 2.5,
                'hold_rating': 2.5,
                'target_price': 0,
                'target_high': 0,
                'target_low': 0
            }
    
    def get_financial_metrics(self, symbol: str) -> Dict:
        """
        Get company financial metrics.
        """
        params = {
            'symbol': symbol,
            'apikey': self.api_key,
            'metrics': 'all'
        }
        
        try:
            response = requests.get(f"{self.base_url}/financials", params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except:
            return {}


class StockTradingSystem:
    """
    Complete system for stock price prediction with scenario analysis.
    """
    
    def __init__(self, api_key: str, lookback_window: int = 10):
        self.fetcher = MASSIVEStockFetcher(api_key)
        self.model = AdaptiveStockPredictor(input_size=lookback_window * 5, hidden_size=30)
        self.lookback_window = lookback_window
        self.scaler_params = {}
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training with multiple features.
        Uses OHLCV data to predict next day's prices.
        """
        # Select key features for prediction
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        df_features = df[feature_cols].copy()
        
        # Normalize each feature
        normalized_data = []
        self.scaler_params = {}
        
        for i, col in enumerate(feature_cols):
            values = df_features[col].to_numpy().reshape(-1, 1)
            mean_val = np.mean(values)
            std_val = np.std(values) + 1e-8
            
            self.scaler_params[col] = {'mean': mean_val, 'std': std_val}
            normalized = (values - mean_val) / std_val
            normalized_data.append(normalized)
        
        # Combine normalized features
        combined = np.hstack(normalized_data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(combined) - self.lookback_window):
            # Input: lookback_window days of all 5 features
            X_seq = combined[i:i + self.lookback_window].flatten()
            # Output: next day's 5 features
            y_seq = combined[i + self.lookback_window]
            
            X.append(X_seq)
            y.append(y_seq)
        
        return np.array(X), np.array(y)
    
    def denormalize_predictions(self, predictions_norm: np.ndarray) -> np.ndarray:
        """
        Convert normalized predictions back to original scale.
        """
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        predictions = np.zeros_like(predictions_norm)
        
        for i, col in enumerate(feature_cols):
            if col in self.scaler_params:
                mean_val = self.scaler_params[col]['mean']
                std_val = self.scaler_params[col]['std']
                predictions[:, i] = predictions_norm[:, i] * std_val + mean_val
        
        return predictions
    
    def train_model(self, symbol: str, epochs: int = 200) -> None:
        """
        Fetch data and train the model.
        """
        # Fetch data
        df = self.fetcher.fetch_stock_data(symbol, period='6month')
        print(f"Fetched {len(df)} days of data")
        
        # Prepare training data
        X, y = self.prepare_data(df)
        print(f"Training on {len(X)} sequences with {X.shape[1]} features")
        
        # Train model
        self.model.train(X, y, epochs=epochs)
        
        # Evaluate
        predictions_norm = self.model.predict(X)
        predictions = self.denormalize_predictions(predictions_norm)
        y_actual = self.denormalize_predictions(y)
        
        # Calculate errors
        mae_open = np.mean(np.abs(predictions[:, 0] - y_actual[:, 0]))
        mae_close = np.mean(np.abs(predictions[:, 3] - y_actual[:, 3]))
        
        print(f"\nTraining Results for {symbol}:")
        print(f"  Open Price MAE: ${mae_open:.2f}")
        print(f"  Close Price MAE: ${mae_close:.2f}")
    
    def predict_next_day(self, symbol: str, include_scenarios: bool = True) -> Dict:
        """
        Predict next day's stock prices with scenario analysis.
        Returns best case, average case, and worst case predictions.
        """
        # Fetch latest data
        df = self.fetcher.fetch_stock_data(symbol, period='1month')
        
        # Get market sentiment
        sentiment = self.fetcher.get_market_sentiment(symbol)
        
        # Prepare input data
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        recent_data = []
        
        for col in feature_cols:
            values = df[col].values[-self.lookback_window:]
            if col in self.scaler_params:
                mean_val = self.scaler_params[col]['mean']
                std_val = self.scaler_params[col]['std']
                normalized = (values - mean_val) / (std_val + 1e-8)
            else:
                # Fallback normalization
                values_array = np.asarray(values)
                normalized = (values_array - np.mean(values_array)) / (np.std(values_array) + 1e-8)
            recent_data.append(normalized)
        
        # Create input sequence
        X_input = np.hstack(recent_data).flatten().reshape(1, -1)
        
        # Make base prediction
        prediction_norm = self.model.predict(X_input)
        base_prediction = self.denormalize_predictions(prediction_norm)[0]
        
        # Generate scenario predictions
        scenarios = self._generate_scenarios(base_prediction, sentiment)
        
        # Calculate confidence scores based on recent accuracy
        confidence = self._calculate_confidence(symbol, df)
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'prediction_date': (df.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d'),
            'scenarios': scenarios,
            'current_price': float(df['close'].iloc[-1]),
            'market_sentiment': sentiment.get('sentiment', 'neutral'),
            'confidence': confidence,
            'last_updated': datetime.now().isoformat(),
            'technical_indicators': self._get_technical_indicators(df)
        }
    
    def _generate_scenarios(self, base_prediction: np.ndarray, sentiment: Dict) -> Dict:
        """
        Generate best, average, and worst case scenarios.
        """
        open_pred, high_pred, low_pred, close_pred, volume_pred = base_prediction
        
        # Apply sentiment-based adjustments
        sentiment_multiplier = {
            'bullish': 1.05,
            'very_bullish': 1.08,
            'bearish': 0.95,
            'very_bearish': 0.92,
            'neutral': 1.0
        }.get(sentiment.get('sentiment', 'neutral'), 1.0)
        
        # Generate scenarios with variance
        base_variance = 0.02  # 2% variance
        
        # Best case (optimistic)
        best_multiplier = sentiment_multiplier * (1 + base_variance)
        best_scenario = {
            'open': open_pred * best_multiplier,
            'high': high_pred * best_multiplier,
            'low': low_pred * (1 - base_variance/2),  # Less downside
            'close': close_pred * best_multiplier,
            'volume': volume_pred * 1.1,
            'buy_price': low_pred * (1 - base_variance/4),  # Buy near low
            'sell_price': high_pred * best_multiplier,  # Sell near high
            'profit_potential': ((high_pred * best_multiplier) - (low_pred * (1 - base_variance/4))) / (low_pred * (1 - base_variance/4)) * 100
        }
        
        # Average case (most likely)
        avg_scenario = {
            'open': open_pred,
            'high': high_pred,
            'low': low_pred,
            'close': close_pred,
            'volume': volume_pred,
            'buy_price': (open_pred + low_pred) / 2,
            'sell_price': (high_pred + close_pred) / 2,
            'profit_potential': ((high_pred + close_pred) / 2 - (open_pred + low_pred) / 2) / ((open_pred + low_pred) / 2) * 100
        }
        
        # Worst case (pessimistic)
        worst_multiplier = sentiment_multiplier * (1 - base_variance)
        worst_scenario = {
            'open': open_pred * worst_multiplier,
            'high': high_pred * (1 - base_variance/2),  # Less upside
            'low': low_pred * worst_multiplier,
            'close': close_pred * worst_multiplier,
            'volume': volume_pred * 0.9,
            'buy_price': open_pred * worst_multiplier,
            'sell_price': (high_pred * (1 - base_variance/2) + close_pred * worst_multiplier) / 2,
            'profit_potential': ((high_pred * (1 - base_variance/2) + close_pred * worst_multiplier) / 2 - open_pred * worst_multiplier) / (open_pred * worst_multiplier) * 100
        }
        
        return {
            'best_case': {k: float(v) for k, v in best_scenario.items()},
            'average_case': {k: float(v) for k, v in avg_scenario.items()},
            'worst_case': {k: float(v) for k, v in worst_scenario.items()}
        }
    
    def _calculate_confidence(self, symbol: str, df: pd.DataFrame) -> float:
        """
        Calculate prediction confidence based on recent accuracy.
        """
        # Simplified confidence calculation
        if len(df) < 20:
            return 0.6
        
        # Use volatility as confidence indicator (lower volatility = higher confidence)
        recent_volatility = df['close'].pct_change().std() * 100
        
        if recent_volatility < 1.0:
            return 0.85
        elif recent_volatility < 2.0:
            return 0.75
        elif recent_volatility < 5.0:
            return 0.65
        else:
            return 0.55
    
    def _get_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Calculate basic technical indicators.
        """
        if len(df) < 20:
            return {}
        
        close = df['close']
        volume = df['volume']
        
        # Simple Moving Averages
        sma_20 = close.rolling(window=20).mean().iloc[-1]
        sma_50 = close.rolling(window=50).mean().iloc[-1]
        
        # RSI (simplified)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50
        
        # Volume trend
        volume_sma = volume.rolling(window=20).mean().iloc[-1]
        volume_ratio = volume.iloc[-1] / volume_sma if volume_sma > 0 else 1
        
        return {
            'sma_20': float(sma_20),
            'sma_50': float(sma_50),
            'rsi': float(rsi_value),
            'volume_trend': 'above_average' if volume_ratio > 1.2 else 'below_average' if volume_ratio < 0.8 else 'average',
            'trend': 'bullish' if sma_20 > sma_50 else 'bearish'
        }
    
    def adaptive_update(self, symbol: str) -> None:
        """
        Update model with latest data.
        """
        df = self.fetcher.fetch_stock_data(symbol, period='1month')
        X, y = self.prepare_data(df)
        
        if len(X) > 5:
            X_recent = X[-5:]
            y_recent = y[-5:]
            self.model.incremental_update(X_recent, y_recent)
            print(f"Model updated for {symbol} with recent data")
