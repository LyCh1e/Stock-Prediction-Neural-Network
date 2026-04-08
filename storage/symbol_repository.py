# JSON persistence of the tracked-symbols registry: saves/loads the {symbol: {lookback, epochs}} mapping.

from __future__ import annotations

import json
import os
from typing import Dict

from core.interfaces import ISymbolRepository


# Persists and restores the set of tracked symbols and their settings as JSON.
class JsonSymbolRepository(ISymbolRepository):

    def __init__(self, filepath: str = "tracked_symbols.json") -> None:
        self._filepath = filepath

    # Write only lookback and epochs for each symbol to the JSON file.
    def save(self, symbols: Dict[str, Dict]) -> None:
        data = {
            sym: {"lookback": info["lookback"], "epochs": info["epochs"]}
            for sym, info in symbols.items()
        }
        with open(self._filepath, "w") as fh:
            json.dump(data, fh, indent=2)

    # Read and return the symbols dict from JSON, or {} if the file is absent or corrupt.
    def load(self) -> Dict[str, Dict]:
        if not os.path.exists(self._filepath):
            return {}
        try:
            with open(self._filepath, "r") as fh:
                return json.load(fh)
        except Exception:
            return {}
