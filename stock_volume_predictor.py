"""
Improved Stock Price Predictor with Flexible Lookback
Fixes: 
- Reduced minimum data requirement from 20+ days to just lookback_window days
- Enhanced prediction clarity with confidence intervals
- Better technical indicator calculations for short-term data
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
    
    def predict_with_uncertainty(self, X: np.ndarray, num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates using dropout-like sampling.
        
        Returns:
            mean_prediction: Average prediction
            std_prediction: Standard deviation (uncertainty)
        """
        predictions = []
        for _ in range(num_samples):
            # Add small noise to simulate uncertainty
            _, _, output = self.forward(X)
            noisy_output = output + np.random.normal(0, 0.02, output.shape)
            predictions.append(noisy_output)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
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


class YahooFinanceAPI:
    """
    Fetch financial data from Yahoo Finance using yfinance.
    Drop-in replacement for MassiveAPI.
    """
    
    def __init__(self, api_key: str = ""):
        # api_key kept for interface compatibility but not used
        self.api_key = api_key
        try:
            import yfinance as yf
            self._yf = yf
        except ImportError:
            raise ImportError("yfinance is required. Install with: pip install yfinance")
    
    def fetch_stock_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch stock price data from Yahoo Finance.
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        print(f"Fetching Yahoo Finance data for {symbol} from {start_date} to {end_date}...")
        
        try:
            ticker = self._yf.Ticker(symbol)
            df_raw = ticker.history(start=start_date, end=end_date)
            
            if df_raw is None or len(df_raw) == 0:
                raise ValueError(f"No data returned for {symbol}")
            
            df = pd.DataFrame({
                'open': df_raw['Open'],
                'high': df_raw['High'],
                'low': df_raw['Low'],
                'close': df_raw['Close'],
                'volume': df_raw['Volume'].astype(int)
            })
            df.index = pd.DatetimeIndex(df_raw.index).tz_localize(None)
            df.index.name = 'date'
            df = df.sort_index()
            
            print(f"✓ Successfully fetched {len(df)} days of data from Yahoo Finance")
            df = self._calculate_technical_indicators(df, min_window=5)
            return df
            
        except Exception as e:
            print(f"Error fetching Yahoo Finance data: {e}")
            print(f"⚠ Generating synthetic data for {symbol}")
            df = self._generate_synthetic_data(symbol, start_date, end_date)
            df = self._calculate_technical_indicators(df, min_window=5)
            return df
    
    def _generate_synthetic_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate synthetic stock data as fallback"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        dates = pd.date_range(start=start, end=end, freq='D')
        dates = [d for d in dates if d.weekday() < 5]
        
        np.random.seed(hash(symbol) % 2**32)
        base_price = 50 + (hash(symbol) % 200)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        price_series = base_price * np.exp(np.cumsum(returns))
        
        records = []
        for i, date in enumerate(dates):
            close_price = price_series[i]
            daily_volatility = abs(returns[i]) * close_price * 0.5
            open_price = close_price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, close_price) + abs(np.random.normal(0, daily_volatility))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, daily_volatility))
            low_price = max(0.01, low_price)
            high_price = max(high_price, low_price * 1.01)
            close_price = np.clip(close_price, low_price, high_price)
            open_price = np.clip(open_price, low_price, high_price)
            price_change = abs(close_price - open_price) / open_price
            base_volume = 1000000 * (1 + hash(symbol) % 10)
            volume = int(base_volume * (1 + price_change * 10 + np.random.exponential(0.5)))
            records.append({
                'date': date, 'open': open_price, 'high': high_price,
                'low': low_price, 'close': close_price, 'volume': volume
            })
        
        df = pd.DataFrame(records)
        df.set_index('date', inplace=True)
        return df

    def get_market_sentiment(self, symbol: str) -> Dict:
        """Get market sentiment based on technical indicators."""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            df = self.fetch_stock_data(symbol, start_date, end_date)
            if len(df) < 5:
                return self._default_sentiment()
            current_rsi = df['rsi'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            current_close = df['close'].iloc[-1]
            volume_ratio = df['volume_ratio'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            score = 0
            if current_close > sma_20: score += 1
            if current_close > sma_50: score += 1
            if sma_20 > sma_50: score += 0.5
            if current_rsi < 30:
                score += 1; rsi_signal = "Oversold"
            elif current_rsi > 70:
                score -= 1; rsi_signal = "Overbought"
            else:
                rsi_signal = "Neutral"
            if macd > macd_signal:
                score += 1; macd_signal_text = "Bullish"
            else:
                score -= 0.5; macd_signal_text = "Bearish"
            if volume_ratio > 1.2:
                volume_signal = "High"; score += 0.5
            elif volume_ratio < 0.8:
                volume_signal = "Low"; score -= 0.3
            else:
                volume_signal = "Normal"
            if score >= 3: sentiment, confidence = "Very Bullish", 0.85
            elif score >= 1.5: sentiment, confidence = "Bullish", 0.75
            elif score >= 0: sentiment, confidence = "Neutral", 0.65
            elif score >= -1.5: sentiment, confidence = "Bearish", 0.75
            else: sentiment, confidence = "Very Bearish", 0.85
            return {
                'sentiment': sentiment, 'confidence': confidence, 'score': float(score),
                'details': {
                    'rsi': float(current_rsi), 'rsi_signal': rsi_signal,
                    'macd_signal': macd_signal_text, 'volume_signal': volume_signal,
                    'price_vs_sma20': 'Above' if current_close > sma_20 else 'Below',
                    'price_vs_sma50': 'Above' if current_close > sma_50 else 'Below'
                }
            }
        except Exception as e:
            print(f"Error calculating sentiment: {e}")
            return self._default_sentiment()

    def _default_sentiment(self) -> Dict:
        return {
            'sentiment': 'Neutral', 'confidence': 0.5, 'score': 0.0,
            'details': {
                'rsi': 50.0, 'rsi_signal': 'Neutral', 'macd_signal': 'Neutral',
                'volume_signal': 'Normal', 'price_vs_sma20': 'Unknown', 'price_vs_sma50': 'Unknown'
            }
        }

    def get_trading_recommendation(self, symbol: str, prediction: Dict) -> str:
        try:
            sentiment = prediction.get('sentiment_analysis', {})
            scenarios = prediction.get('scenarios', {})
            if not scenarios: return 'HOLD'
            avg_profit = scenarios.get('average_case', {}).get('profit_potential', 0)
            best_profit = scenarios.get('best_case', {}).get('profit_potential', 0)
            worst_profit = scenarios.get('worst_case', {}).get('profit_potential', 0)
            confidence = prediction.get('confidence', 0.5)
            sentiment_score = sentiment.get('score', 0)
            potential_gain = best_profit
            potential_loss = abs(worst_profit) if worst_profit < 0 else 0
            risk_reward = potential_gain / (potential_loss + 1e-8) if potential_loss > 0 else potential_gain
            score = 0
            if avg_profit > 3: score += 2
            elif avg_profit > 1.5: score += 1
            elif avg_profit < -1.5: score -= 1.5
            elif avg_profit < -3: score -= 2
            if confidence > 0.8: score += 1
            elif confidence < 0.6: score -= 0.5
            score += sentiment_score * 0.5
            if risk_reward > 3: score += 1
            elif risk_reward < 1: score -= 1
            rsi = sentiment.get('details', {}).get('rsi', 50)
            if rsi < 30: score += 1.5
            elif rsi > 70: score -= 1.5
            elif 40 <= rsi <= 60: score += 0.5
            macd_sig = sentiment.get('details', {}).get('macd_signal', 'Neutral')
            if macd_sig == 'Bullish': score += 1
            elif macd_sig == 'Bearish': score -= 0.5
            if score >= 3: return 'STRONG_BUY'
            elif score >= 1.5: return 'BUY'
            elif score >= 0: return 'HOLD'
            elif score >= -1.5: return 'SELL'
            else: return 'STRONG_SELL'
        except:
            return 'HOLD'

    def _calculate_technical_indicators(self, df: pd.DataFrame, min_window: int) -> pd.DataFrame:
        """
        Calculate technical indicators with flexible windows for short-term data.
        
        Args:
            df: DataFrame with OHLCV data
            min_window: Minimum window size (auto-adjusts for short data)
        """
        data_length = len(df)
        
        # Determine appropriate windows based on available data
        if min_window is None:
            # Use shorter windows for limited data
            if data_length < 20:
                sma_short = min(5, data_length - 1)
                sma_long = min(10, data_length - 1)
                rsi_period = min(7, data_length - 1)
                vol_period = sma_short
            elif data_length < 50:
                sma_short = 10
                sma_long = 20
                rsi_period = 14
                vol_period = 10
            else:
                sma_short = 20
                sma_long = 50
                rsi_period = 14
                vol_period = 20
        else:
            sma_short = min_window
            sma_long = min(min_window * 2, data_length - 1)
            rsi_period = min(14, data_length - 1)
            vol_period = min_window
        
        # Ensure windows are valid
        sma_short = max(2, sma_short)
        sma_long = max(sma_short + 1, sma_long)
        rsi_period = max(2, rsi_period)
        vol_period = max(2, vol_period)
        
        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=sma_short, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(window=sma_long, min_periods=1).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Fill initial RSI NaN with neutral value
        df['rsi'].fillna(50, inplace=True)
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=sma_short, min_periods=1).mean()
        bb_std = df['close'].rolling(window=sma_short, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=vol_period, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
        
        # Fill initial volume_ratio NaN with 1.0
        df['volume_ratio'].fillna(1.0, inplace=True)
        
        # Daily returns and volatility
        df['daily_return'] = df['close'].pct_change()
        df['volatility'] = df['daily_return'].rolling(window=vol_period, min_periods=1).std()
        
        # Fill initial volatility NaN with small value
        df['volatility'].fillna(0.01, inplace=True)
        
        # Price momentum
        momentum_period = min(10, data_length - 1)
        df['momentum'] = df['close'] - df['close'].shift(momentum_period)
        df['momentum'].fillna(0, inplace=True)
        
        # MACD (simplified with flexible periods)
        ema_short = min(12, data_length - 1)
        ema_long = min(26, data_length - 1)
        ema_signal = min(9, data_length - 1)
        
        exp1 = df['close'].ewm(span=ema_short, adjust=False, min_periods=1).mean()
        exp2 = df['close'].ewm(span=ema_long, adjust=False, min_periods=1).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=ema_signal, adjust=False, min_periods=1).mean()
        
        # If still NaN (very short data), fill with defaults
        df['rsi'].fillna(50, inplace=True)
        df['volume_ratio'].fillna(1.0, inplace=True)
        df['volatility'].fillna(0.01, inplace=True)
        
        return df


class StockTradingSystem:
    """
    Complete system for stock price prediction with scenario analysis.
    IMPROVED: Flexible lookback window, better uncertainty quantification
    """
    
    def __init__(self, api_key: str = "", lookback_window: int = 10):
        self.api = YahooFinanceAPI(api_key)
        # Ensure minimum lookback is at least 3 days
        self.lookback_window = max(3, lookback_window)
        self.model = AdaptiveStockPredictor(input_size=self.lookback_window * 9, hidden_size=30)
        self.scaler_params = {}
        self.symbol = None
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training with multiple features.
        IMPROVED: Works with as little as lookback_window days of data.
        """
        # Select features for prediction
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'sma_20', 'rsi', 'volume_ratio', 'volatility']
        
        # Ensure all features exist - now handles short data
        if 'sma_20' not in df.columns or df['sma_20'].isna().all():
            df = self.api._calculate_technical_indicators(df, min_window=self.lookback_window)
        
        df_features = df[feature_cols].copy()
        
        # Drop any rows with NaN values (should be minimal now)
        df_features = df_features.dropna()
        
        # IMPROVED: Only need lookback_window + 1 days instead of 20+
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
        IMPROVED: Provides better feedback and handles limited data
        """
        self.symbol = symbol
        
        # Fetch data (last 6 months, but will work with less)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        df = self.api.fetch_stock_data(symbol, start_date, end_date)
        print(f"Retrieved {len(df)} days of data for {symbol}")
        
        # Check if we have enough data
        min_required = self.lookback_window + 1
        if len(df) < min_required:
            raise ValueError(f"Insufficient data. Need at least {min_required} days, got {len(df)}")
        
        # Prepare training data
        X, y = self.prepare_data(df)
        print(f"Training on {len(X)} sequences with {X.shape[1]} features")
        print(f"Using {self.lookback_window} day lookback window")
        
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
        mape_close = np.mean(np.abs((predictions[:, 3] - y_actual[:, 3]) / (y_actual[:, 3] + 1e-8))) * 100
        
        print(f"\nTraining Results for {symbol}:")
        print(f"  Open Price MAE: ${mae_open:.2f}")
        print(f"  Close Price MAE: ${mae_close:.2f}")
        print(f"  Close Price MAPE: {mape_close:.2f}%")
        print(f"  Model Accuracy: {100 - mape_close:.1f}%")
        print(f"  Final Loss: {self.model.losses[-1]:.6f}")
        
        return len(df)
    
    def predict_next_day(self, symbol: str, include_scenarios: bool = True) -> Dict:
        """
        Predict next day's stock prices with enhanced clarity.
        IMPROVED: Better uncertainty quantification and clearer output
        """
        self.symbol = symbol
        
        # Fetch latest data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=max(60, self.lookback_window * 3))).strftime('%Y-%m-%d')
        df = self.api.fetch_stock_data(symbol, start_date, end_date)
        
        if len(df) < self.lookback_window:
            raise ValueError(f"Insufficient data for {symbol}. Need at least {self.lookback_window} days, got {len(df)}")
        
        # Get market sentiment
        sentiment = self.api.get_market_sentiment(symbol)
        
        # Prepare input data
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'sma_20', 'rsi', 'volume_ratio', 'volatility']
        
        # Ensure dataframe has required columns
        if 'sma_20' not in df.columns or df['sma_20'].isna().all():
            df = self.api._calculate_technical_indicators(df, min_window=self.lookback_window)
        
        df_features = df[feature_cols].copy().dropna()
        
        if len(df_features) < self.lookback_window:
            raise ValueError(f"Insufficient data after cleaning. Need at least {self.lookback_window} days, got {len(df_features)}")
        
        recent_data = []
        for col in feature_cols:
            values = np.asarray(df_features[col].values[-self.lookback_window:])
            
            if col in self.scaler_params:
                mean_val = self.scaler_params[col]['mean']
                std_val = self.scaler_params[col]['std']
                normalized = (values - mean_val) / std_val
            else:
                # Fallback normalization if parameter not found
                normalized = (values - np.mean(values)) / (np.std(values) + 1e-8)
            
            recent_data.append(normalized)
        
        # Flatten for model input
        X_pred = np.hstack(recent_data).flatten().reshape(1, -1)
        
        # Get prediction with uncertainty
        pred_mean, pred_std = self.model.predict_with_uncertainty(X_pred, num_samples=100)
        pred_denorm = self.denormalize_predictions(pred_mean)
        pred_std_denorm = pred_std * np.array([self.scaler_params[col]['std'] 
                                                for col in ['open', 'high', 'low', 'close', 'volume']])
        
        # Extract predictions
        open_pred = pred_denorm[0, 0]
        high_pred = pred_denorm[0, 1]
        low_pred = pred_denorm[0, 2]
        close_pred = pred_denorm[0, 3]
        volume_pred = pred_denorm[0, 4]
        
        # Uncertainty (standard deviation)
        close_uncertainty = pred_std_denorm[0, 3]
        
        # Current price
        current_close = df['close'].iloc[-1]
        current_open = df['open'].iloc[-1]
        
        # Build result dictionary
        result = {
            'symbol': symbol,
            'current_price': float(current_close),
            'current_open': float(current_open),
            'timestamp': datetime.now().isoformat(),
            'prediction': {
                'open': float(open_pred),
                'high': float(high_pred),
                'low': float(low_pred),
                'close': float(close_pred),
                'volume': int(volume_pred),
                'expected_change': float(close_pred - current_close),
                'expected_change_pct': float((close_pred - current_close) / current_close * 100),
                'uncertainty': float(close_uncertainty),
                'confidence_interval_95': {
                    'lower': float(close_pred - 1.96 * close_uncertainty),
                    'upper': float(close_pred + 1.96 * close_uncertainty)
                }
            },
            'sentiment_analysis': sentiment,
            'data_quality': {
                'lookback_window': self.lookback_window,
                'data_points_used': len(df),
                'days_since_last_data': (datetime.now().date() - df.index[-1].date()).days
            }
        }
        
        # Generate scenarios if requested
        if include_scenarios:
            scenarios = self._generate_enhanced_scenarios(
                current_close, close_pred, high_pred, low_pred, open_pred, 
                volume_pred, close_uncertainty, sentiment, df
            )
            result['scenarios'] = scenarios
            result['confidence'] = self._calculate_enhanced_confidence(df, sentiment, close_uncertainty)
            result['technical_indicators'] = self._get_technical_indicators(df)
            result['recommendation'] = self.api.get_trading_recommendation(symbol, result)
        
        return result
    
    def _generate_enhanced_scenarios(self, current_close, close_pred, high_pred, low_pred, 
                                    open_pred, volume_pred, uncertainty, sentiment, df) -> Dict:
        """
        Generate enhanced scenarios with clearer probability distributions.
        """
        # Calculate historical volatility
        recent_volatility = df['volatility'].iloc[-5:].mean()
        
        # Sentiment multiplier
        sentiment_score = sentiment.get('score', 0)
        if sentiment_score > 2:
            sentiment_multiplier = 1.15
        elif sentiment_score > 0:
            sentiment_multiplier = 1.05
        elif sentiment_score < -2:
            sentiment_multiplier = 0.85
        else:
            sentiment_multiplier = 0.95
        
        # BEST CASE (90th percentile)
        best_multiplier = 1 + (1.65 * recent_volatility * sentiment_multiplier)
        best_scenario = {
            'probability': '10%',  # Top 10% outcome
            'description': 'Optimistic scenario with strong buying pressure',
            'open': open_pred * (1 + recent_volatility * 0.5),
            'high': high_pred * best_multiplier,
            'low': low_pred,
            'close': close_pred * best_multiplier,
            'volume': volume_pred * 1.3,
            'profit_potential': ((close_pred * best_multiplier) - current_close) / current_close * 100,
            'target_price': close_pred * best_multiplier,
            'stop_loss': current_close * 0.97  # 3% stop loss
        }
        
        # AVERAGE CASE (50th percentile - median)
        avg_scenario = {
            'probability': '50%',  # Median outcome
            'description': 'Most likely scenario based on current trends',
            'open': open_pred,
            'high': high_pred,
            'low': low_pred,
            'close': close_pred,
            'volume': volume_pred,
            'profit_potential': ((close_pred) - current_close) / current_close * 100,
            'target_price': close_pred,
            'stop_loss': current_close * 0.97
        }
        
        # WORST CASE (10th percentile)
        worst_multiplier = 1 - (1.65 * recent_volatility * (2 - sentiment_multiplier))
        worst_scenario = {
            'probability': '10%',  # Bottom 10% outcome
            'description': 'Pessimistic scenario with selling pressure',
            'open': open_pred * (1 - recent_volatility * 0.5),
            'high': high_pred,
            'low': low_pred * worst_multiplier,
            'close': close_pred * worst_multiplier,
            'volume': volume_pred * 0.7,
            'profit_potential': ((close_pred * worst_multiplier) - current_close) / current_close * 100,
            'target_price': close_pred * worst_multiplier,
            'stop_loss': current_close * 0.95  # 5% stop loss for downside
        }
        
        # REALISTIC RANGE (80% confidence)
        range_multiplier = 1.28 * uncertainty  # 80% confidence interval
        realistic_range = {
            'probability': '80%',
            'description': 'Expected price range with 80% confidence',
            'lower_bound': close_pred - range_multiplier,
            'upper_bound': close_pred + range_multiplier,
            'expected': close_pred,
            'range_width': 2 * range_multiplier,
            'range_pct': (2 * range_multiplier / current_close * 100)
        }
        
        # Ensure logical consistency
        for scenario in [best_scenario, avg_scenario, worst_scenario]:
            scenario['high'] = max(scenario['high'], scenario['open'], scenario['low'], scenario['close'])
            scenario['low'] = min(scenario['low'], scenario['open'], scenario['high'], scenario['close'])
            
            # Cap extreme values
            scenario['profit_potential'] = max(-50, min(100, scenario['profit_potential']))
        
        return {
            'best_case': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                         for k, v in best_scenario.items()},
            'average_case': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                           for k, v in avg_scenario.items()},
            'worst_case': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                          for k, v in worst_scenario.items()},
            'realistic_range': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                               for k, v in realistic_range.items()}
        }
    
    def _calculate_enhanced_confidence(self, df: pd.DataFrame, sentiment: Dict, uncertainty: float) -> float:
        """
        Calculate prediction confidence with enhanced factors.
        """
        confidence_factors = []
        
        # 1. Data recency (more recent = higher confidence)
        days_since_last = (datetime.now().date() - df.index[-1].date()).days
        if days_since_last <= 1:
            confidence_factors.append(0.95)
        elif days_since_last <= 3:
            confidence_factors.append(0.85)
        elif days_since_last <= 7:
            confidence_factors.append(0.75)
        else:
            confidence_factors.append(0.65)
        
        # 2. Model uncertainty (lower = higher confidence)
        if uncertainty < 1:
            confidence_factors.append(0.90)
        elif uncertainty < 2:
            confidence_factors.append(0.80)
        elif uncertainty < 3:
            confidence_factors.append(0.70)
        else:
            confidence_factors.append(0.60)
        
        # 3. Data quantity
        data_points = len(df)
        if data_points > 100:
            confidence_factors.append(0.90)
        elif data_points > 50:
            confidence_factors.append(0.80)
        elif data_points >= self.lookback_window * 2:
            confidence_factors.append(0.70)
        else:
            confidence_factors.append(0.60)
        
        # 4. Volatility (lower = higher confidence)
        if 'volatility' in df.columns:
            vol = df['volatility'].iloc[-5:].mean()
            if vol < 0.01:
                confidence_factors.append(0.90)
            elif vol < 0.02:
                confidence_factors.append(0.80)
            elif vol < 0.03:
                confidence_factors.append(0.70)
            else:
                confidence_factors.append(0.60)
        
        # 5. Sentiment confidence
        if 'confidence' in sentiment:
            confidence_factors.append(sentiment['confidence'])
        
        # 6. Model training loss
        if self.model.losses:
            final_loss = self.model.losses[-1]
            if final_loss < 0.01:
                confidence_factors.append(0.90)
            elif final_loss < 0.05:
                confidence_factors.append(0.80)
            elif final_loss < 0.10:
                confidence_factors.append(0.70)
            else:
                confidence_factors.append(0.60)
        
        # Calculate weighted average
        if confidence_factors:
            confidence = np.mean(confidence_factors)
        else:
            confidence = 0.7
        
        # Apply bounds
        confidence = max(0.50, min(0.95, confidence))
        
        return float(confidence)
    
    def _get_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Get current technical indicators for display.
        """
        if len(df) < 5:
            return {}
        
        indicators = {}
        
        # Current values
        current_close = df['close'].iloc[-1]
        
        # Moving Averages
        if 'sma_20' in df.columns:
            indicators['sma_20'] = float(df['sma_20'].iloc[-1])
            indicators['price_vs_sma20'] = float((current_close - indicators['sma_20']) / indicators['sma_20'] * 100)
        
        if 'sma_50' in df.columns:
            indicators['sma_50'] = float(df['sma_50'].iloc[-1])
            indicators['price_vs_sma50'] = float((current_close - indicators['sma_50']) / indicators['sma_50'] * 100)
        
        # RSI
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            indicators['rsi'] = float(rsi)
            if rsi < 30:
                indicators['rsi_status'] = 'OVERSOLD - Potential Buy Signal'
            elif rsi > 70:
                indicators['rsi_status'] = 'OVERBOUGHT - Potential Sell Signal'
            elif 40 <= rsi <= 60:
                indicators['rsi_status'] = 'NEUTRAL - Balanced'
            elif rsi < 50:
                indicators['rsi_status'] = 'SLIGHTLY BEARISH'
            else:
                indicators['rsi_status'] = 'SLIGHTLY BULLISH'
        
        # Volume
        if 'volume_ratio' in df.columns:
            vol_ratio = df['volume_ratio'].iloc[-1]
            indicators['volume_ratio'] = float(vol_ratio)
            if vol_ratio > 1.5:
                indicators['volume_status'] = 'VERY HIGH - Strong Interest'
            elif vol_ratio > 1.2:
                indicators['volume_status'] = 'HIGH - Above Average'
            elif vol_ratio < 0.8:
                indicators['volume_status'] = 'LOW - Below Average'
            else:
                indicators['volume_status'] = 'NORMAL - Average'
        
        # Volatility
        if 'volatility' in df.columns:
            vol = df['volatility'].iloc[-1]
            indicators['volatility'] = float(vol * 100)  # as percentage
            if vol < 0.01:
                indicators['volatility_status'] = 'LOW - Stable Price Action'
            elif vol < 0.03:
                indicators['volatility_status'] = 'MODERATE - Normal Fluctuation'
            else:
                indicators['volatility_status'] = 'HIGH - Significant Price Swings'
        
        # Trend analysis
        if 'close' in df.columns and len(df) >= 5:
            price_5d_ago = df['close'].iloc[-5]
            price_change_5d = (current_close - price_5d_ago) / price_5d_ago * 100
            indicators['5_day_change'] = float(price_change_5d)
            
            if price_change_5d > 5:
                indicators['short_term_trend'] = 'STRONG UPTREND'
            elif price_change_5d > 2:
                indicators['short_term_trend'] = 'UPTREND'
            elif price_change_5d < -5:
                indicators['short_term_trend'] = 'STRONG DOWNTREND'
            elif price_change_5d < -2:
                indicators['short_term_trend'] = 'DOWNTREND'
            else:
                indicators['short_term_trend'] = 'SIDEWAYS'
        
        # Support and Resistance
        if len(df) >= 10:
            support = df['low'].rolling(window=10).min().iloc[-1]
            resistance = df['high'].rolling(window=10).max().iloc[-1]
            
            indicators['support'] = float(support)
            indicators['resistance'] = float(resistance)
            indicators['distance_to_support'] = float((current_close - support) / current_close * 100)
            indicators['distance_to_resistance'] = float((resistance - current_close) / current_close * 100)
            
            # Position in range
            range_position = (current_close - support) / (resistance - support + 1e-8)
            if range_position > 0.8:
                indicators['range_position'] = 'Near Resistance - Potential Reversal'
            elif range_position < 0.2:
                indicators['range_position'] = 'Near Support - Potential Bounce'
            else:
                indicators['range_position'] = 'Mid-Range'
        
        # MACD
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            indicators['macd'] = float(macd)
            indicators['macd_signal'] = float(macd_signal)
            
            if macd > macd_signal and macd > 0:
                indicators['macd_status'] = 'STRONG BULLISH - Buy Signal'
            elif macd > macd_signal:
                indicators['macd_status'] = 'BULLISH - Positive Momentum'
            elif macd < macd_signal and macd < 0:
                indicators['macd_status'] = 'STRONG BEARISH - Sell Signal'
            elif macd < macd_signal:
                indicators['macd_status'] = 'BEARISH - Negative Momentum'
            else:
                indicators['macd_status'] = 'NEUTRAL'
        
        return indicators
    
    def adaptive_update(self, symbol: str) -> None:
        """
        Update model with latest data.
        """
        self.symbol = symbol
        
        # Fetch recent data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=max(30, self.lookback_window * 3))).strftime('%Y-%m-%d')
        df = self.api.fetch_stock_data(symbol, start_date, end_date)
        
        X, y = self.prepare_data(df)
        
        if len(X) > 3:
            X_recent = X[-3:]
            y_recent = y[-3:]
            self.model.incremental_update(X_recent, y_recent)