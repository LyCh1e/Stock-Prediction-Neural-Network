"""
src/data/fetcher.py
~~~~~~~~~~~~~~~~~~~
Concrete IDataFetcher implementation using Yahoo Finance (yfinance).

Responsibilities (Single Responsibility):
  - Fetch OHLCV data from Yahoo Finance
  - Generate synthetic fallback data when the API is unavailable
  - Compute market sentiment from technical indicators
  - Compute trading recommendations from prediction output

Depends on TechnicalIndicators (not on any ML or storage module).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.interfaces import IDataFetcher
from data.indicators import TechnicalIndicators


class YahooFinanceFetcher(IDataFetcher):
    """Fetches stock data from Yahoo Finance with a synthetic-data fallback."""

    def __init__(self) -> None:
        try:
            import yfinance as yf
            self._yf = yf
        except ImportError:
            raise ImportError("yfinance is required. Install with: pip install yfinance")

    # ------------------------------------------------------------------ #
    #  IDataFetcher implementation                                        #
    # ------------------------------------------------------------------ #

    def fetch_stock_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        if end_date is None:
            end_date   = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

        print(f"Fetching Yahoo Finance data for {symbol} from {start_date} to {end_date}...")

        try:
            ticker = self._yf.Ticker(symbol)
            df_raw = ticker.history(start=start_date, end=end_date)

            if df_raw is None or len(df_raw) == 0:
                raise ValueError(f"No data returned for {symbol}")

            df = pd.DataFrame({
                "open":   df_raw["Open"],
                "high":   df_raw["High"],
                "low":    df_raw["Low"],
                "close":  df_raw["Close"],
                "volume": df_raw["Volume"].astype(int),
            })
            df.index = pd.DatetimeIndex(df_raw.index).tz_localize(None)
            df.index.name = "date"
            df = df.sort_index()

            print(f"Successfully fetched {len(df)} days of data for {symbol}")
            return TechnicalIndicators.calculate(df, min_window=5)

        except Exception as exc:
            print(f"Error fetching data for {symbol}: {exc}")
            print(f"Generating synthetic data for {symbol}")
            df = self._generate_synthetic_data(symbol, start_date, end_date)
            return TechnicalIndicators.calculate(df, min_window=5)

    def get_market_sentiment(self, symbol: str) -> Dict:
        try:
            end_date   = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
            df = self.fetch_stock_data(symbol, start_date, end_date)
            if len(df) < 5:
                return self._default_sentiment()

            current_close = df["close"].iloc[-1]
            sma_20        = df["sma_20"].iloc[-1]
            sma_50        = df["sma_50"].iloc[-1]
            current_rsi   = df["rsi"].iloc[-1]
            volume_ratio  = df["volume_ratio"].iloc[-1]
            macd          = df["macd"].iloc[-1]
            macd_signal   = df["macd_signal"].iloc[-1]

            score = 0.0
            if current_close > sma_20:    score += 1
            if current_close > sma_50:    score += 1
            if sma_20 > sma_50:           score += 0.5
            if current_rsi < 30:
                score += 1;  rsi_signal = "Oversold"
            elif current_rsi > 70:
                score -= 1;  rsi_signal = "Overbought"
            else:
                rsi_signal = "Neutral"
            if macd > macd_signal:
                score += 1;  macd_signal_text = "Bullish"
            else:
                score -= 0.5; macd_signal_text = "Bearish"
            if volume_ratio > 1.2:
                volume_signal = "High";   score += 0.5
            elif volume_ratio < 0.8:
                volume_signal = "Low";    score -= 0.3
            else:
                volume_signal = "Normal"

            if   score >= 3:    sentiment, confidence = "Very Bullish", 0.85
            elif score >= 1.5:  sentiment, confidence = "Bullish",      0.75
            elif score >= 0:    sentiment, confidence = "Neutral",       0.65
            elif score >= -1.5: sentiment, confidence = "Bearish",      0.75
            else:               sentiment, confidence = "Very Bearish", 0.85

            return {
                "sentiment":  sentiment,
                "confidence": confidence,
                "score":      float(score),
                "details": {
                    "rsi":           float(current_rsi),
                    "rsi_signal":    rsi_signal,
                    "macd_signal":   macd_signal_text,
                    "volume_signal": volume_signal,
                    "price_vs_sma20": "Above" if current_close > sma_20 else "Below",
                    "price_vs_sma50": "Above" if current_close > sma_50 else "Below",
                },
            }
        except Exception as exc:
            print(f"Error calculating sentiment for {symbol}: {exc}")
            return self._default_sentiment()

    def get_trading_recommendation(self, symbol: str, prediction: Dict) -> str:
        try:
            sentiment    = prediction.get("sentiment_analysis", {})
            scenarios    = prediction.get("scenarios", {})
            if not scenarios:
                return "HOLD"

            avg_profit   = scenarios.get("average_case", {}).get("profit_potential", 0)
            best_profit  = scenarios.get("best_case",    {}).get("profit_potential", 0)
            worst_profit = scenarios.get("worst_case",   {}).get("profit_potential", 0)
            confidence   = prediction.get("confidence", 0.5)
            sentiment_score = sentiment.get("score", 0)

            potential_gain = best_profit
            potential_loss = abs(worst_profit) if worst_profit < 0 else 0
            risk_reward    = potential_gain / (potential_loss + 1e-8) if potential_loss > 0 else potential_gain

            score = 0.0
            if   avg_profit >  3:   score += 2
            elif avg_profit >  1.5: score += 1
            elif avg_profit < -1.5: score -= 1.5
            elif avg_profit < -3:   score -= 2
            if   confidence > 0.8:  score += 1
            elif confidence < 0.6:  score -= 0.5
            score += sentiment_score * 0.5
            if risk_reward > 3:     score += 1
            elif risk_reward < 1:   score -= 1

            rsi = sentiment.get("details", {}).get("rsi", 50)
            if   rsi < 30:          score += 1.5
            elif rsi > 70:          score -= 1.5
            elif 40 <= rsi <= 60:   score += 0.5

            macd_sig = sentiment.get("details", {}).get("macd_signal", "Neutral")
            if   macd_sig == "Bullish": score += 1
            elif macd_sig == "Bearish": score -= 0.5

            if   score >= 3:    return "STRONG_BUY"
            elif score >= 1.5:  return "BUY"
            elif score >= 0:    return "HOLD"
            elif score >= -1.5: return "SELL"
            else:               return "STRONG_SELL"
        except Exception:
            return "HOLD"

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _generate_synthetic_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        start  = datetime.strptime(start_date, "%Y-%m-%d")
        end    = datetime.strptime(end_date,   "%Y-%m-%d")
        dates  = [d for d in pd.date_range(start=start, end=end, freq="D") if d.weekday() < 5]

        np.random.seed(hash(symbol) % 2**32)
        base_price   = 50 + (hash(symbol) % 200)
        returns      = np.random.normal(0.0005, 0.02, len(dates))
        price_series = base_price * np.exp(np.cumsum(returns))

        records = []
        for i, date in enumerate(dates):
            close_price      = price_series[i]
            daily_volatility = abs(returns[i]) * close_price * 0.5
            open_price       = close_price * (1 + np.random.normal(0, 0.005))
            high_price       = max(open_price, close_price) + abs(np.random.normal(0, daily_volatility))
            low_price        = max(0.01, min(open_price, close_price) - abs(np.random.normal(0, daily_volatility)))
            high_price       = max(high_price, low_price * 1.01)
            close_price      = np.clip(close_price, low_price, high_price)
            open_price       = np.clip(open_price,  low_price, high_price)
            price_change     = abs(close_price - open_price) / open_price
            base_volume      = 1_000_000 * (1 + hash(symbol) % 10)
            volume           = int(base_volume * (1 + price_change * 10 + np.random.exponential(0.5)))
            records.append({
                "date": date, "open": open_price, "high": high_price,
                "low":  low_price, "close": close_price, "volume": volume,
            })

        df = pd.DataFrame(records)
        df.set_index("date", inplace=True)
        return df

    @staticmethod
    def _default_sentiment() -> Dict:
        return {
            "sentiment":  "Neutral",
            "confidence": 0.5,
            "score":      0.0,
            "details": {
                "rsi":            50.0,
                "rsi_signal":     "Neutral",
                "macd_signal":    "Neutral",
                "volume_signal":  "Normal",
                "price_vs_sma20": "Unknown",
                "price_vs_sma50": "Unknown",
            },
        }
