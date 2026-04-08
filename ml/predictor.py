# Generates predictions, scenarios, confidence scores, and indicator summaries from a trained network.
# Inference and result construction only — no data fetching, training, or persistence.

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from data.indicators import TechnicalIndicators
from ml.network import NeuralNetwork
from ml.trainer import ModelTrainer, _FEATURE_COLS


# Stateless predictor — turns a trained NeuralNetwork + DataFrame into a structured result dict.
class StockPredictor:

    def __init__(self, trainer: ModelTrainer) -> None:
        self._trainer = trainer

    # Run Monte Carlo inference and build the full prediction result dict including scenarios.
    def predict_next_day(
        self,
        symbol: str,
        df: pd.DataFrame,
        network: NeuralNetwork,
        scaler_params: Dict,
        sentiment: Dict,
        lookback_window: int,
        include_scenarios: bool = True,
    ) -> Dict:
        if "sma_20" not in df.columns or df["sma_20"].isna().all():
            df = TechnicalIndicators.calculate(df.copy(), min_window=lookback_window)

        df_feat = df[_FEATURE_COLS].copy().dropna()

        if len(df_feat) < lookback_window:
            raise ValueError(
                f"Insufficient data for {symbol}. Need {lookback_window} days, "
                f"got {len(df_feat)}"
            )

        X_pred = self._build_input(df_feat, scaler_params, lookback_window)
        pred_mean, pred_std = network.predict_with_uncertainty(X_pred, num_samples=100)

        pred_denorm     = self._trainer.denormalize(pred_mean, scaler_params)
        std_scale       = np.array([scaler_params[c]["std"] for c in ["open", "high", "low", "close", "volume"]])
        pred_std_denorm = pred_std * std_scale

        open_pred     = pred_denorm[0, 0]
        high_pred     = pred_denorm[0, 1]
        low_pred      = pred_denorm[0, 2]
        close_pred    = pred_denorm[0, 3]
        volume_pred   = pred_denorm[0, 4]
        close_uncert  = pred_std_denorm[0, 3]

        current_close = float(df["close"].iloc[-1])
        current_open  = float(df["open"].iloc[-1])

        result: Dict = {
            "symbol":        symbol,
            "current_price": current_close,
            "current_open":  current_open,
            "timestamp":     datetime.now().isoformat(),
            "prediction": {
                "open":   float(open_pred),
                "high":   float(high_pred),
                "low":    float(low_pred),
                "close":  float(close_pred),
                "volume": int(volume_pred),
                "expected_change":     float(close_pred - current_close),
                "expected_change_pct": float((close_pred - current_close) / current_close * 100),
                "uncertainty":         float(close_uncert),
                "confidence_interval_95": {
                    "lower": float(close_pred - 1.96 * close_uncert),
                    "upper": float(close_pred + 1.96 * close_uncert),
                },
            },
            "sentiment_analysis": sentiment,
            "data_quality": {
                "lookback_window":    lookback_window,
                "data_points_used":   len(df),
                "days_since_last_data": (datetime.now().date() - df.index[-1].date()).days,
            },
        }

        if include_scenarios:
            result["scenarios"]             = self._generate_scenarios(
                current_close, close_pred, high_pred, low_pred,
                open_pred, volume_pred, close_uncert, sentiment, df
            )
            result["confidence"]            = self._calculate_confidence(
                df, sentiment, close_uncert, lookback_window, network
            )
            result["technical_indicators"]  = self._get_technical_indicators(df)

        return result

    # ------------------------------------------------------------------ #
    #  Private: scenario generation                                       #
    # ------------------------------------------------------------------ #

    # Build best/average/worst/realistic-range scenario dicts using volatility and sentiment.
    def _generate_scenarios(
        self,
        current_close: float,
        close_pred: float,
        high_pred: float,
        low_pred: float,
        open_pred: float,
        volume_pred: float,
        uncertainty: float,
        sentiment: Dict,
        df: pd.DataFrame,
    ) -> Dict:
        recent_volatility = float(df["volatility"].iloc[-5:].mean())
        sentiment_score   = sentiment.get("score", 0)

        if   sentiment_score > 2:   sentiment_mult = 1.15
        elif sentiment_score > 0:   sentiment_mult = 1.05
        elif sentiment_score < -2:  sentiment_mult = 0.85
        else:                       sentiment_mult = 0.95

        best_mult  = 1 + (1.65 * recent_volatility * sentiment_mult)
        worst_mult = 1 - (1.65 * recent_volatility * (2 - sentiment_mult))
        range_mult = 1.28 * uncertainty

        best  = self._scenario(
            open_pred * (1 + recent_volatility * 0.5),
            high_pred * best_mult, low_pred,
            close_pred * best_mult, volume_pred * 1.3,
            current_close, "10%", "Optimistic scenario with strong buying pressure",
            current_close * 0.97,
        )
        avg   = self._scenario(
            open_pred, high_pred, low_pred,
            close_pred, volume_pred,
            current_close, "50%", "Most likely scenario based on current trends",
            current_close * 0.97,
        )
        worst = self._scenario(
            open_pred * (1 - recent_volatility * 0.5),
            high_pred, low_pred * worst_mult,
            close_pred * worst_mult, volume_pred * 0.7,
            current_close, "10%", "Pessimistic scenario with selling pressure",
            current_close * 0.95,
        )

        for sc in (best, avg, worst):
            sc["high"] = max(sc["high"], sc["open"], sc["low"], sc["close"])
            sc["low"]  = min(sc["low"],  sc["open"], sc["high"], sc["close"])
            sc["profit_potential"] = max(-50, min(100, sc["profit_potential"]))

        realistic_range = {
            "probability":  "80%",
            "description":  "Expected price range with 80% confidence",
            "lower_bound":  float(close_pred - range_mult),
            "upper_bound":  float(close_pred + range_mult),
            "expected":     float(close_pred),
            "range_width":  float(2 * range_mult),
            "range_pct":    float(2 * range_mult / current_close * 100),
        }

        return {
            "best_case":      best,
            "average_case":   avg,
            "worst_case":     worst,
            "realistic_range": realistic_range,
        }

    # Assemble a single scenario dict from OHLCV values, profit %, target, and stop-loss.
    @staticmethod
    def _scenario(
        open_: float, high: float, low: float, close: float, volume: float,
        current_close: float, probability: str, description: str, stop_loss: float,
    ) -> Dict:
        return {
            "probability":    probability,
            "description":    description,
            "open":           float(open_),
            "high":           float(high),
            "low":            float(low),
            "close":          float(close),
            "volume":         float(volume),
            "profit_potential": float((close - current_close) / current_close * 100),
            "target_price":   float(close),
            "stop_loss":      float(stop_loss),
        }

    # ------------------------------------------------------------------ #
    #  Private: confidence calculation                                    #
    # ------------------------------------------------------------------ #

    # Average several confidence factors (data freshness, uncertainty, volatility, loss) into one score.
    def _calculate_confidence(
        self,
        df: pd.DataFrame,
        sentiment: Dict,
        uncertainty: float,
        lookback_window: int,
        network: NeuralNetwork,
    ) -> float:
        factors: List[float] = []

        days_since = (datetime.now().date() - df.index[-1].date()).days
        factors.append(0.95 if days_since <= 1 else 0.85 if days_since <= 3 else 0.75 if days_since <= 7 else 0.65)
        factors.append(0.90 if uncertainty < 1 else 0.80 if uncertainty < 2 else 0.70 if uncertainty < 3 else 0.60)

        n = len(df)
        factors.append(0.90 if n > 100 else 0.80 if n > 50 else 0.70 if n >= lookback_window * 2 else 0.60)

        if "volatility" in df.columns:
            vol = float(df["volatility"].iloc[-5:].mean())
            factors.append(0.90 if vol < 0.01 else 0.80 if vol < 0.02 else 0.70 if vol < 0.03 else 0.60)

        if "confidence" in sentiment:
            factors.append(float(sentiment["confidence"]))

        if network.losses:
            loss = network.losses[-1]
            factors.append(0.90 if loss < 0.01 else 0.80 if loss < 0.05 else 0.70 if loss < 0.10 else 0.60)

        confidence = float(np.mean(factors)) if factors else 0.7
        return max(0.50, min(0.95, confidence))

    # ------------------------------------------------------------------ #
    #  Private: technical indicator summary                               #
    # ------------------------------------------------------------------ #

    # Extract the latest indicator values from df and attach human-readable status labels.
    @staticmethod
    def _get_technical_indicators(df: pd.DataFrame) -> Dict:
        if len(df) < 5:
            return {}

        ind: Dict = {}
        current_close = float(df["close"].iloc[-1])

        if "sma_20" in df.columns:
            ind["sma_20"]         = float(df["sma_20"].iloc[-1])
            ind["price_vs_sma20"] = float((current_close - ind["sma_20"]) / ind["sma_20"] * 100)
        if "sma_50" in df.columns:
            ind["sma_50"]         = float(df["sma_50"].iloc[-1])
            ind["price_vs_sma50"] = float((current_close - ind["sma_50"]) / ind["sma_50"] * 100)
        if "rsi" in df.columns:
            rsi = float(df["rsi"].iloc[-1])
            ind["rsi"] = rsi
            if   rsi < 30:         ind["rsi_status"] = "OVERSOLD - Potential Buy Signal"
            elif rsi > 70:         ind["rsi_status"] = "OVERBOUGHT - Potential Sell Signal"
            elif 40 <= rsi <= 60:  ind["rsi_status"] = "NEUTRAL - Balanced"
            elif rsi < 50:         ind["rsi_status"] = "SLIGHTLY BEARISH"
            else:                  ind["rsi_status"] = "SLIGHTLY BULLISH"
        if "volume_ratio" in df.columns:
            vr = float(df["volume_ratio"].iloc[-1])
            ind["volume_ratio"] = vr
            if   vr > 1.5: ind["volume_status"] = "VERY HIGH - Strong Interest"
            elif vr > 1.2: ind["volume_status"] = "HIGH - Above Average"
            elif vr < 0.8: ind["volume_status"] = "LOW - Below Average"
            else:          ind["volume_status"] = "NORMAL - Average"
        if "volatility" in df.columns:
            vol = float(df["volatility"].iloc[-1])
            ind["volatility"] = vol * 100
            if   vol < 0.01: ind["volatility_status"] = "LOW - Stable Price Action"
            elif vol < 0.03: ind["volatility_status"] = "MODERATE - Normal Fluctuation"
            else:            ind["volatility_status"] = "HIGH - Significant Price Swings"
        if len(df) >= 5:
            p5 = float(df["close"].iloc[-5])
            chg5 = (current_close - p5) / p5 * 100
            ind["5_day_change"] = chg5
            if   chg5 >  5: ind["short_term_trend"] = "STRONG UPTREND"
            elif chg5 >  2: ind["short_term_trend"] = "UPTREND"
            elif chg5 < -5: ind["short_term_trend"] = "STRONG DOWNTREND"
            elif chg5 < -2: ind["short_term_trend"] = "DOWNTREND"
            else:           ind["short_term_trend"] = "SIDEWAYS"
        if len(df) >= 10:
            support    = float(df["low"].rolling(10).min().iloc[-1])
            resistance = float(df["high"].rolling(10).max().iloc[-1])
            ind["support"]    = support
            ind["resistance"] = resistance
            ind["distance_to_support"]    = (current_close - support) / current_close * 100
            ind["distance_to_resistance"] = (resistance - current_close) / current_close * 100
            rng = (current_close - support) / (resistance - support + 1e-8)
            if   rng > 0.8: ind["range_position"] = "Near Resistance - Potential Reversal"
            elif rng < 0.2: ind["range_position"] = "Near Support - Potential Bounce"
            else:           ind["range_position"] = "Mid-Range"
        if "macd" in df.columns:
            macd  = float(df["macd"].iloc[-1])
            msig  = float(df["macd_signal"].iloc[-1])
            ind["macd"] = macd; ind["macd_signal"] = msig
            if   macd > msig and macd > 0: ind["macd_status"] = "STRONG BULLISH - Buy Signal"
            elif macd > msig:              ind["macd_status"] = "BULLISH - Positive Momentum"
            elif macd < msig and macd < 0: ind["macd_status"] = "STRONG BEARISH - Sell Signal"
            elif macd < msig:              ind["macd_status"] = "BEARISH - Negative Momentum"
            else:                          ind["macd_status"] = "NEUTRAL"
        return ind

    # ------------------------------------------------------------------ #
    #  Input construction                                                 #
    # ------------------------------------------------------------------ #

    # Normalise the last lookback_window rows of each feature and flatten into a (1, N) input array.
    @staticmethod
    def _build_input(
        df_feat: pd.DataFrame, scaler_params: Dict, lookback_window: int
    ) -> np.ndarray:
        parts = []
        for col in _FEATURE_COLS:
            values = np.asarray(df_feat[col].values[-lookback_window:])
            if col in scaler_params:
                mean = scaler_params[col]["mean"]
                std  = scaler_params[col]["std"]
                parts.append((values - mean) / std)
            else:
                parts.append((values - np.mean(values)) / (np.std(values) + 1e-8))
        return np.hstack(parts).flatten().reshape(1, -1)
