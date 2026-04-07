"""
src/storage/model_repository.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Responsible for JSON persistence of trained model weights.

Single Responsibility: save/load weights to/from a single JSON file.
Does not know about training, predictions, or the GUI.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Dict, Optional

import numpy as np

from src.core.interfaces import IModelRepository
from src.ml.network import NeuralNetwork

_LOCK = threading.Lock()   # module-level lock shared by all instances


class JsonModelRepository(IModelRepository):
    """Persists NeuralNetwork weights and scaler parameters as JSON."""

    def __init__(self, filepath: str = "stock_models.json") -> None:
        self._filepath = filepath

    # ------------------------------------------------------------------ #
    #  IModelRepository implementation                                    #
    # ------------------------------------------------------------------ #

    def save(self, symbol: str, model: NeuralNetwork, scaler_params: Dict) -> None:
        entry = {
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_size":   int(model.input_size),
            "hidden_size":  int(model.hidden_size),
            "lr":           float(model.learning_rate),
            "final_loss":   float(model.losses[-1]) if model.losses else None,
            "scaler_params": scaler_params,
            "W1": model.W1.tolist(),
            "b1": model.b1.tolist(),
            "W2": model.W2.tolist(),
            "b2": model.b2.tolist(),
        }
        tmp = self._filepath + ".tmp"
        try:
            with _LOCK:
                all_models = self._read_all()
                all_models[symbol] = entry
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(all_models, fh, indent=2)
                os.replace(tmp, self._filepath)
        except Exception as exc:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise exc

    def load(self, symbol: str) -> Optional[Dict]:
        if not os.path.exists(self._filepath):
            return None
        try:
            all_models = self._read_all()
            return all_models.get(symbol)
        except Exception:
            return None

    def restore_weights(self, symbol: str, model: NeuralNetwork) -> Optional[Dict]:
        """
        Restore weights from the JSON file into *model*.

        Returns the scaler_params dict if successful, None if no saved data
        or if the saved input_size does not match the current model shape.
        """
        entry = self.load(symbol)
        if entry is None:
            return None

        saved_input_size = entry.get("input_size", -1)
        if saved_input_size != model.input_size:
            print(
                f"{symbol}: saved input_size={saved_input_size} != "
                f"current input_size={model.input_size} — retraining from scratch"
            )
            return None

        model.W1 = np.array(entry["W1"])
        model.b1 = np.array(entry["b1"])
        model.W2 = np.array(entry["W2"])
        model.b2 = np.array(entry["b2"])
        return entry.get("scaler_params", {})

    # ------------------------------------------------------------------ #
    #  Private                                                            #
    # ------------------------------------------------------------------ #

    def _read_all(self) -> Dict:
        if not os.path.exists(self._filepath):
            return {}
        with open(self._filepath, "r", encoding="utf-8") as fh:
            return json.load(fh)
