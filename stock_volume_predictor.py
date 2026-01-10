"""
Stock Price Predictor with Scenario Analysis
Uses MASSIVE API with Bearer token authentication
"""

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import json
import pickle
from typing import Tuple, Optional, Dict, List
import time

class AdaptiveStockPredictor:
    """
    Neural network for predicting stock prices with adaptive learning.
    Predicts multiple price points: open, high, low, close, volume.
    """
    
    def __init__(self, input_size: int = 50, hidden_size: int = 30, learning_rate: float = 0.001):
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
        self.W2 = np.random.randn(hidden_size, 5) * np.sqrt(2.0 / hidden_size)
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


class MassiveAPI:
    """
    Fetch financial data from MASSIVE API with Bearer token authentication.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.massive.com"
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def fetch_stock_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch stock price data from MASSIVE API.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            DataFrame with OHLCV data
        """
        # Set default dates (last 6 months)
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        print(f"Fetching MASSIVE data for {symbol} from {start_date} to {end_date}...")
        
        try:
            # Try to get stock data from different endpoints
            endpoints_to_try = [
                f'/v3/stocks/{symbol}/daily',
                f'/v3/quotes/{symbol}',
                f'/v3/reference/tickers/{symbol}/prices'
            ]
            
            df = None
            for endpoint in endpoints_to_try:
                try:
                    response = requests.get(
                        f"{self.base_url}{endpoint}",
                        headers=self.headers,
                        params={
                            'from': start_date,
                            'to': end_date
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        df = self._parse_stock_data(data, symbol)
                        if df is not None and len(df) > 0:
                            break
                except:
                    continue
            
            # If API calls fail, use simulated data for demonstration
            if df is None or len(df) == 0:
                print(f"API data not available, using simulated data for {symbol}")
                df = self._generate_simulated_data(symbol, start_date, end_date)
            
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            print(f"Retrieved {len(df)} days of data for {symbol}")
            return df
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            # Fallback to simulated data
            return self._generate_simulated_data(symbol, start_date, end_date)
    
    def _parse_stock_data(self, data: Dict, symbol: str) -> Optional[pd.DataFrame]:
        """
        Parse stock data from MASSIVE API response.
        """
        records = []
        
        # Try different response formats
        if isinstance(data, list):
            # List of records
            for item in data:
                record = self._extract_price_data(item)
                if record:
                    records.append(record)
        
        elif isinstance(data, dict):
            # Dictionary with nested structure
            if 'results' in data:
                for item in data['results']:
                    record = self._extract_price_data(item)
                    if record:
                        records.append(record)
            elif 'data' in data:
                for item in data['data']:
                    record = self._extract_price_data(item)
                    if record:
                        records.append(record)
            elif 'ticker' in data and 'results' in data:
                for item in data['results']:
                    record = self._extract_price_data(item)
                    if record:
                        records.append(record)
        
        if not records:
            return None
        
        # Create DataFrame
        df = pd.DataFrame(records)
        df = df.sort_values('date')
        df.set_index('date', inplace=True)
        return df
    
    def _extract_price_data(self, item: Dict) -> Optional[Dict]:
        """
        Extract price data from API response item.
        """
        try:
            # Try different field names
            date_str = item.get('date') or item.get('timestamp') or item.get('time')
            if not date_str:
                return None
            
            # Parse date
            try:
                date = pd.to_datetime(date_str)
            except:
                return None
            
            # Extract prices
            open_price = float(item.get('open', item.get('openPrice', 0)))
            high_price = float(item.get('high', item.get('highPrice', 0)))
            low_price = float(item.get('low', item.get('lowPrice', 0)))
            close_price = float(item.get('close', item.get('closePrice', item.get('price', 0))))
            volume = int(float(item.get('volume', item.get('tradingVolume', 0))))
            
            return {
                'date': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }
        except:
            return None
    
    def _generate_simulated_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate realistic simulated stock data for demonstration.
        """
        # Base prices for popular stocks
        base_prices = {
            'AAPL': 175.0, 'MSFT': 330.0, 'GOOGL': 140.0, 
            'AMZN': 150.0, 'TSLA': 180.0, 'META': 380.0,
            'NVDA': 500.0, 'SPY': 450.0, 'QQQ': 380.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        
        # Generate date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq='B')  # Business days
        
        # Generate price series with trend and randomness
        np.random.seed(hash(symbol) % 10000)
        n_days = len(dates)
        
        # Create price series with realistic patterns
        returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
        
        # Add some trends based on symbol
        if symbol in ['AAPL', 'MSFT', 'GOOGL']:
            returns += 0.0002  # Slight upward bias for tech
        elif symbol == 'TSLA':
            returns = np.random.normal(0.001, 0.04, n_days)  # More volatile
        
        # Calculate prices
        price_series = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC data
        records = []
        for i, date in enumerate(dates):
            close_price = price_series[i]
            
            # Generate realistic OHLC
            daily_volatility = abs(returns[i]) * close_price * 0.5
            open_price = close_price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, close_price) + abs(np.random.normal(0, daily_volatility))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, daily_volatility))
            
            # Ensure high >= low >= 0
            low_price = max(0.01, low_price)
            high_price = max(high_price, low_price * 1.01)
            close_price = np.clip(close_price, low_price, high_price)
            open_price = np.clip(open_price, low_price, high_price)
            
            # Generate volume (correlated with price movement)
            price_change = abs(close_price - open_price) / open_price
            base_volume = 1000000 * (1 + hash(symbol) % 10)  # Vary by symbol
            volume = int(base_volume * (1 + price_change * 10 + np.random.exponential(0.5)))
            
            records.append({
                'date': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(records)
        df.set_index('date', inplace=True)
        return df
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators.
        """
        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Daily returns and volatility
        df['daily_return'] = df['close'].pct_change()
        df['volatility'] = df['daily_return'].rolling(window=20).std()
        
        # Price momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        
        # MACD (simplified)
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        return df
    
    def get_market_sentiment(self, symbol: str) -> Dict:
        """
        Get market sentiment based on technical indicators.
        """
        try:
            # Fetch recent data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            
            df = self.fetch_stock_data(symbol, start_date, end_date)
            
            if len(df) < 20:
                return self._default_sentiment()
            
            # Calculate sentiment based on technical indicators
            current_rsi = df['rsi'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            current_close = df['close'].iloc[-1]
            volume_ratio = df['volume_ratio'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            
            # Determine sentiment
            bullish_signals = 0
            bearish_signals = 0
            
            # RSI analysis
            if current_rsi < 30:
                bullish_signals += 2  # Oversold
            elif current_rsi > 70:
                bearish_signals += 2  # Overbought
            elif 40 <= current_rsi <= 60:
                bullish_signals += 1  # Neutral to slightly bullish
            else:
                bearish_signals += 1
            
            # Moving average analysis
            if sma_20 > sma_50:
                bullish_signals += 2  # Golden cross
            else:
                bearish_signals += 1
            
            # Volume analysis
            if volume_ratio > 1.5:
                bullish_signals += 2  # High volume supporting trend
            elif volume_ratio > 1.2:
                bullish_signals += 1
            elif volume_ratio < 0.8:
                bearish_signals += 1
            
            # MACD analysis
            if macd > macd_signal:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # Price position relative to moving averages
            if current_close > sma_20:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # Determine overall sentiment
            total_signals = bullish_signals + bearish_signals
            sentiment_score = 0
            if total_signals == 0:
                sentiment = 'neutral'
            else:
                sentiment_score = (bullish_signals - bearish_signals) / total_signals
                
                if sentiment_score > 0.4:
                    sentiment = 'very_bullish'
                elif sentiment_score > 0.2:
                    sentiment = 'bullish'
                elif sentiment_score < -0.4:
                    sentiment = 'very_bearish'
                elif sentiment_score < -0.2:
                    sentiment = 'bearish'
                else:
                    sentiment = 'neutral'
            
            # Calculate target prices based on volatility
            volatility = df['volatility'].iloc[-1]
            target_up = current_close * (1 + volatility * 2.5)
            target_down = current_close * (1 - volatility * 2.0)
            
            # Get recommendation
            recommendation = self._get_recommendation(sentiment, current_rsi, macd, macd_signal)
            
            return {
                'sentiment': sentiment,
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'rsi': float(current_rsi),
                'current_price': float(current_close),
                'target_high': float(target_up),
                'target_low': float(target_down),
                'recommendation': recommendation,
                'confidence': min(0.95, max(0.5, 0.7 + sentiment_score * 0.2))
            }
            
        except Exception as e:
            print(f"Error calculating sentiment: {e}")
            return self._default_sentiment()
    
    def _default_sentiment(self) -> Dict:
        """Return default sentiment data"""
        return {
            'sentiment': 'neutral',
            'bullish_signals': 0,
            'bearish_signals': 0,
            'rsi': 50,
            'current_price': 0,
            'target_high': 0,
            'target_low': 0,
            'recommendation': 'HOLD',
            'confidence': 0.7
        }
    
    def _get_recommendation(self, sentiment: str, rsi: float, macd: float, macd_signal: float) -> str:
        """Get trading recommendation based on multiple factors"""
        # Weighted decision
        score = 0
        
        # Sentiment contribution
        sentiment_weights = {
            'very_bullish': 2,
            'bullish': 1,
            'neutral': 0,
            'bearish': -1,
            'very_bearish': -2
        }
        score += sentiment_weights.get(sentiment, 0)
        
        # RSI contribution
        if rsi < 30:
            score += 1.5  # Oversold - bullish signal
        elif rsi > 70:
            score -= 1.5  # Overbought - bearish signal
        elif 40 <= rsi <= 60:
            score += 0.5  # Neutral range - slightly bullish
        
        # MACD contribution
        if macd > macd_signal:
            score += 1  # Bullish crossover
        else:
            score -= 0.5
        
        # Determine recommendation
        if score >= 3:
            return 'STRONG_BUY'
        elif score >= 1.5:
            return 'BUY'
        elif score >= 0:
            return 'HOLD'
        elif score >= -1.5:
            return 'SELL'
        else:
            return 'STRONG_SELL'


class StockTradingSystem:
    """
    Complete system for stock price prediction with scenario analysis.
    """
    
    def __init__(self, api_key: str, lookback_window: int = 10):
        self.api = MassiveAPI(api_key)
        self.model = AdaptiveStockPredictor(input_size=lookback_window * 9, hidden_size=30)
        self.lookback_window = lookback_window
        self.scaler_params = {}
        self.symbol = None
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training with multiple features.
        Uses OHLCV data + technical indicators to predict next day's prices.
        """
        # Select features for prediction
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'sma_20', 'rsi', 'volume_ratio', 'volatility']
        
        # Ensure all features exist
        for col in feature_cols:
            if col not in df.columns:
                if col in ['sma_20', 'rsi', 'volume_ratio', 'volatility']:
                    # Calculate missing indicators
                    df = self.api._calculate_technical_indicators(df)
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        df_features = df[feature_cols].copy()
        
        # Drop any rows with NaN values (from indicator calculations)
        df_features = df_features.dropna()
        
        if len(df_features) < self.lookback_window + 1:
            raise ValueError(f"Insufficient data after cleaning. Need at least {self.lookback_window + 1} days, got {len(df_features)}")
        
        # Normalize each feature
        normalized_data = []
        self.scaler_params = {}
        
        for i, col in enumerate(feature_cols):
            values = np.asarray(df_features[col].values).reshape(-1, 1)
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
            # Input: lookback_window days of all features
            X_seq = combined[i:i + self.lookback_window].flatten()
            # Output: next day's 5 main features (open, high, low, close, volume)
            y_seq = combined[i + self.lookback_window, :5]
            
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
    
    def train_model(self, symbol: str, epochs: int = 200) -> int:
        """
        Fetch data and train the model.
        """
        self.symbol = symbol
        
        # Fetch data (last 6 months)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        df = self.api.fetch_stock_data(symbol, start_date, end_date)
        print(f"Retrieved {len(df)} days of data for {symbol}")
        
        # Prepare training data
        X, y = self.prepare_data(df)
        print(f"Training on {len(X)} sequences with {X.shape[1]} features")
        
        # Train model
        print(f"Training model for {symbol}...")
        self.model.train(X, y, epochs=epochs)
        
        # Evaluate
        predictions_norm = self.model.predict(X)
        predictions = self.denormalize_predictions(predictions_norm)
        y_actual = self.denormalize_predictions(y)
        
        # Calculate errors
        mae_open = np.mean(np.abs(predictions[:, 0] - y_actual[:, 0]))
        mae_close = np.mean(np.abs(predictions[:, 3] - y_actual[:, 3]))
        accuracy = np.mean(np.abs(predictions[:, 3] - y_actual[:, 3]) / y_actual[:, 3])
        
        print(f"\nTraining Results for {symbol}:")
        print(f"  Open Price MAE: ${mae_open:.2f}")
        print(f"  Close Price MAE: ${mae_close:.2f}")
        print(f"  Accuracy: {(1 - accuracy) * 100:.1f}%")
        print(f"  Final Loss: {self.model.losses[-1]:.6f}")
        
        return len(df)
    
    def predict_next_day(self, symbol: str, include_scenarios: bool = True) -> Dict:
        """
        Predict next day's stock prices with scenario analysis.
        Returns best case, average case, and worst case predictions.
        """
        self.symbol = symbol
        
        # Fetch latest data (last 2 months for prediction)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        df = self.api.fetch_stock_data(symbol, start_date, end_date)
        
        if len(df) < self.lookback_window:
            raise ValueError(f"Insufficient data for {symbol}. Need at least {self.lookback_window} days, got {len(df)}")
        
        # Get market sentiment
        sentiment = self.api.get_market_sentiment(symbol)
        
        # Prepare input data
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'sma_20', 'rsi', 'volume_ratio', 'volatility']
        
        # Ensure dataframe has required columns
        for col in feature_cols:
            if col not in df.columns:
                df = self.api._calculate_technical_indicators(df)
                break
        
        df_features = df[feature_cols].copy().dropna()
        
        if len(df_features) < self.lookback_window:
            raise ValueError(f"Insufficient data after cleaning. Need at least {self.lookback_window} days, got {len(df_features)}")
        
        recent_data = []
        for col in feature_cols:
            values = np.asarray(df_features[col].values[-self.lookback_window:])
            
            if col in self.scaler_params:
                mean_val = self.scaler_params[col]['mean']
                std_val = self.scaler_params[col]['std']
            else:
                # Use current data statistics
                mean_val = np.mean(values)
                std_val = np.std(values) + 1e-8
                self.scaler_params[col] = {'mean': mean_val, 'std': std_val}
            
            normalized = (values - mean_val) / (std_val + 1e-8)
            recent_data.append(normalized)
        
        # Create input sequence
        X_input = np.hstack(recent_data).flatten().reshape(1, -1)
        
        # Make base prediction
        prediction_norm = self.model.predict(X_input)
        base_prediction = self.denormalize_predictions(prediction_norm)[0]
        
        # Generate scenario predictions
        scenarios = self._generate_scenarios(base_prediction, sentiment, df_features)
        
        # Calculate confidence scores
        confidence = self._calculate_confidence(df_features, sentiment)
        
        # Get technical indicators
        tech_indicators = self._get_technical_indicators(df_features)
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'prediction_date': (df.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d'),
            'scenarios': scenarios,
            'current_price': float(df['close'].iloc[-1]),
            'market_sentiment': sentiment.get('sentiment', 'neutral'),
            'sentiment_details': sentiment,
            'confidence': confidence,
            'last_updated': datetime.now().isoformat(),
            'technical_indicators': tech_indicators,
            'data_points': len(df)
        }
    
    def _generate_scenarios(self, base_prediction: np.ndarray, sentiment: Dict, 
                          df: pd.DataFrame) -> Dict:
        """
        Generate best, average, and worst case scenarios.
        """
        open_pred, high_pred, low_pred, close_pred, volume_pred = base_prediction
        
        # Get current values for reference
        current_close = df['close'].iloc[-1]
        current_volatility = df['volatility'].iloc[-1] if 'volatility' in df.columns else 0.02
        current_rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
        sentiment_confidence = sentiment.get('confidence', 0.7)
        
        # Apply sentiment-based adjustments
        sentiment_strength = {
            'very_bullish': 1.10,
            'bullish': 1.05,
            'neutral': 1.00,
            'bearish': 0.95,
            'very_bearish': 0.90
        }.get(sentiment.get('sentiment', 'neutral'), 1.0)
        
        # RSI adjustment
        rsi_adjustment = 1.0
        if current_rsi < 30:  # Oversold - potential bounce
            rsi_adjustment = 1.08
        elif current_rsi > 70:  # Overbought - potential pullback
            rsi_adjustment = 0.92
        elif 40 <= current_rsi <= 60:  # Neutral range - stable
            rsi_adjustment = 1.02
        
        # Volatility-based variance
        base_variance = max(current_volatility, 0.015)  # Minimum 1.5% variance
        
        # Calculate adjusted predictions with confidence weighting
        overall_adjustment = (sentiment_strength * 0.6 + rsi_adjustment * 0.4) * sentiment_confidence
        
        # Best case (optimistic)
        best_multiplier = overall_adjustment * (1 + base_variance * 1.5)
        best_scenario = {
            'open': open_pred * best_multiplier,
            'high': max(high_pred * best_multiplier * 1.02, open_pred * best_multiplier),
            'low': min(low_pred * (1 - base_variance/2), open_pred * best_multiplier * 0.98),
            'close': close_pred * best_multiplier,
            'volume': volume_pred * 1.3,
            'buy_price': low_pred * (1 - base_variance/3),
            'sell_price': high_pred * best_multiplier * 0.99,  # Sell slightly below high
            'profit_potential': ((high_pred * best_multiplier * 0.99) - (low_pred * (1 - base_variance/3))) / (low_pred * (1 - base_variance/3)) * 100
        }
        
        # Average case (most likely)
        avg_multiplier = overall_adjustment
        avg_scenario = {
            'open': open_pred,
            'high': high_pred,
            'low': low_pred,
            'close': close_pred,
            'volume': volume_pred,
            'buy_price': (open_pred * 0.99 + low_pred) / 2,  # Buy near open/low
            'sell_price': (high_pred + close_pred * 1.01) / 2,  # Sell near high/close
            'profit_potential': (((high_pred + close_pred * 1.01) / 2) - ((open_pred * 0.99 + low_pred) / 2)) / ((open_pred * 0.99 + low_pred) / 2) * 100
        }
        
        # Worst case (pessimistic)
        worst_multiplier = overall_adjustment * (1 - base_variance * 1.5)
        worst_scenario = {
            'open': open_pred * worst_multiplier,
            'high': high_pred * (1 - base_variance/2),
            'low': low_pred * worst_multiplier,
            'close': close_pred * worst_multiplier,
            'volume': volume_pred * 0.7,
            'buy_price': open_pred * worst_multiplier * 1.01,  # Buy slightly above open
            'sell_price': (high_pred * (1 - base_variance/2) * 0.97 + close_pred * worst_multiplier) / 2,
            'profit_potential': (((high_pred * (1 - base_variance/2) * 0.97 + close_pred * worst_multiplier) / 2) - (open_pred * worst_multiplier * 1.01)) / (open_pred * worst_multiplier * 1.01) * 100
        }
        
        # Ensure logical consistency
        for scenario in [best_scenario, avg_scenario, worst_scenario]:
            scenario['high'] = max(scenario['high'], scenario['open'], scenario['low'], scenario['close'])
            scenario['low'] = min(scenario['low'], scenario['open'], scenario['high'], scenario['close'])
            scenario['sell_price'] = max(scenario['sell_price'], scenario['buy_price'])
            
            # Ensure profit potential is reasonable
            if scenario['profit_potential'] > 100:  # Cap at 100%
                scenario['profit_potential'] = 100
            elif scenario['profit_potential'] < -50:  # Floor at -50%
                scenario['profit_potential'] = -50
        
        return {
            'best_case': {k: float(v) for k, v in best_scenario.items()},
            'average_case': {k: float(v) for k, v in avg_scenario.items()},
            'worst_case': {k: float(v) for k, v in worst_scenario.items()}
        }
    
    def _calculate_confidence(self, df: pd.DataFrame, sentiment: Dict) -> float:
        """
        Calculate prediction confidence based on multiple factors.
        """
        if len(df) < 20:
            return 0.6
        
        confidence_factors = []
        
        # 1. Data quality (more recent = higher confidence)
        days_since_last = (datetime.now().date() - df.index[-1].date()).days
        if days_since_last <= 1:
            confidence_factors.append(0.95)
        elif days_since_last <= 3:
            confidence_factors.append(0.85)
        elif days_since_last <= 7:
            confidence_factors.append(0.75)
        else:
            confidence_factors.append(0.65)
        
        # 2. Volatility (lower volatility = higher confidence)
        if 'volatility' in df.columns:
            vol = df['volatility'].iloc[-1]
            if vol < 0.01:
                confidence_factors.append(0.90)
            elif vol < 0.02:
                confidence_factors.append(0.80)
            elif vol < 0.03:
                confidence_factors.append(0.70)
            else:
                confidence_factors.append(0.60)
        
        # 3. Data quantity
        data_points = len(df)
        if data_points > 100:
            confidence_factors.append(0.90)
        elif data_points > 50:
            confidence_factors.append(0.80)
        elif data_points > 30:
            confidence_factors.append(0.70)
        else:
            confidence_factors.append(0.60)
        
        # 4. Technical indicator consistency
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            if 40 <= rsi <= 60:
                confidence_factors.append(0.85)  # Neutral RSI = more predictable
            elif 30 <= rsi <= 70:
                confidence_factors.append(0.75)
            else:
                confidence_factors.append(0.65)
        
        # 5. Volume consistency
        if 'volume_ratio' in df.columns:
            vol_ratio = df['volume_ratio'].iloc[-1]
            if 0.8 <= vol_ratio <= 1.2:
                confidence_factors.append(0.80)  # Normal volume = stable
            else:
                confidence_factors.append(0.70)  # Abnormal volume = less predictable
        
        # 6. Sentiment confidence
        if 'confidence' in sentiment:
            confidence_factors.append(sentiment['confidence'])
        
        # Calculate weighted average
        if confidence_factors:
            # Give more weight to data quality and sentiment
            weights = [1.2, 1.0, 1.0, 0.8, 0.8, 1.2]
            weighted_sum = sum(f * w for f, w in zip(confidence_factors[:len(weights)], weights[:len(confidence_factors)]))
            total_weight = sum(weights[:len(confidence_factors)])
            confidence = weighted_sum / total_weight
        else:
            confidence = 0.7
        
        # Apply bounds
        confidence = max(0.5, min(0.95, confidence))
        
        return confidence
    
    def _get_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Calculate technical indicators for display.
        """
        if len(df) < 20:
            return {}
        
        indicators = {}
        
        # Moving Averages
        if 'sma_20' in df.columns:
            indicators['sma_20'] = float(df['sma_20'].iloc[-1])
        if 'sma_50' in df.columns:
            indicators['sma_50'] = float(df['sma_50'].iloc[-1])
        
        # RSI
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            indicators['rsi'] = float(rsi)
            if rsi < 30:
                indicators['rsi_status'] = 'OVERSOLD'
            elif rsi > 70:
                indicators['rsi_status'] = 'OVERBOUGHT'
            else:
                indicators['rsi_status'] = 'NEUTRAL'
        
        # Volume
        if 'volume_ratio' in df.columns:
            vol_ratio = df['volume_ratio'].iloc[-1]
            indicators['volume_ratio'] = float(vol_ratio)
            if vol_ratio > 1.5:
                indicators['volume_status'] = 'VERY HIGH'
            elif vol_ratio > 1.2:
                indicators['volume_status'] = 'HIGH'
            elif vol_ratio < 0.8:
                indicators['volume_status'] = 'LOW'
            else:
                indicators['volume_status'] = 'NORMAL'
        
        # Volatility
        if 'volatility' in df.columns:
            vol = df['volatility'].iloc[-1]
            indicators['volatility'] = float(vol * 100)  # as percentage
            if vol < 0.01:
                indicators['volatility_status'] = 'LOW'
            elif vol < 0.03:
                indicators['volatility_status'] = 'MODERATE'
            else:
                indicators['volatility_status'] = 'HIGH'
        
        # Trend
        if 'close' in df.columns:
            current_close = df['close'].iloc[-1]
            if 'sma_20' in df.columns:
                sma_20 = df['sma_20'].iloc[-1]
                if current_close > sma_20:
                    indicators['trend_short'] = 'BULLISH'
                else:
                    indicators['trend_short'] = 'BEARISH'
            
            if 'sma_50' in df.columns:
                sma_50 = df['sma_50'].iloc[-1]
                if current_close > sma_50:
                    indicators['trend_long'] = 'BULLISH'
                else:
                    indicators['trend_long'] = 'BEARISH'
        
        # Support and Resistance
        if len(df) >= 20:
            support = df['low'].rolling(window=20).min().iloc[-1]
            resistance = df['high'].rolling(window=20).max().iloc[-1]
            current_close = df['close'].iloc[-1]
            
            indicators['support'] = float(support)
            indicators['resistance'] = float(resistance)
            indicators['distance_to_support'] = float((current_close - support) / current_close * 100)
            indicators['distance_to_resistance'] = float((resistance - current_close) / current_close * 100)
        
        # MACD
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            indicators['macd'] = float(macd)
            indicators['macd_signal'] = float(macd_signal)
            if macd > macd_signal:
                indicators['macd_signal'] = 'BULLISH'
            else:
                indicators['macd_signal'] = 'BEARISH'
        
        return indicators
    
    def adaptive_update(self, symbol: str) -> None:
        """
        Update model with latest data.
        """
        self.symbol = symbol
        
        # Fetch recent data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        df = self.api.fetch_stock_data(symbol, start_date, end_date)
        
        X, y = self.prepare_data(df)
        
        if len(X) > 3:
            X_recent = X[-3:]
            y_recent = y[-3:]
            self.model.incremental_update(X_recent, y_recent)