# JSON persistence of trained model weights and scaler params.
# Save/load only — no training, prediction, or GUI knowledge.

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Dict, Optional

import numpy as np

from core.interfaces import IModelRepository
from ml.network import NeuralNetwork

_LOCK = threading.Lock()   # module-level lock shared by all instances


# Persists NeuralNetwork weights and scaler parameters as JSON.
class JsonModelRepository(IModelRepository):

    def __init__(self, filepath: str = "stock_models.json") -> None:
        self._filepath = filepath

    # ------------------------------------------------------------------ #
    #  IModelRepository implementation                                    #
    # ------------------------------------------------------------------ #

    # Serialise model weights and scaler_params to JSON, writing atomically via a temp file.
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

    # Read and return the saved entry for symbol from the JSON file, or None if absent.
    def load(self, symbol: str) -> Optional[Dict]:
        if not os.path.exists(self._filepath):
            return None
        try:
            all_models = self._read_all()
            return all_models.get(symbol)
        except Exception:
            return None

    # Load saved weights into model in-place; return scaler_params or None on shape mismatch/missing.
    def restore_weights(self, symbol: str, model: NeuralNetwork) -> Optional[Dict]:
        entry = self.load(symbol)
        if entry is None:
            return None

        saved_input_size = entry.get("input_size", -1)
        if saved_input_size != model.input_size:
            return None

        model.W1 = np.array(entry["W1"])
        model.b1 = np.array(entry["b1"])
        model.W2 = np.array(entry["W2"])
        model.b2 = np.array(entry["b2"])
        return entry.get("scaler_params", {})

    # ------------------------------------------------------------------ #
    #  Private                                                            #
    # ------------------------------------------------------------------ #

    # Read and return the entire JSON file as a dict, or {} if it doesn't exist.
    def _read_all(self) -> Dict:
        if not os.path.exists(self._filepath):
            return {}
        with open(self._filepath, "r", encoding="utf-8") as fh:
            return json.load(fh)
