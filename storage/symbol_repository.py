"""
src/storage/symbol_repository.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Responsible for JSON persistence of the tracked-symbols registry.

Single Responsibility: save/load the {symbol: {lookback, epochs}} mapping.
"""

from __future__ import annotations

import json
import os
from typing import Dict

from core.interfaces import ISymbolRepository


class JsonSymbolRepository(ISymbolRepository):
    """Persists and restores the set of tracked symbols and their settings."""

    def __init__(self, filepath: str = "tracked_symbols.json") -> None:
        self._filepath = filepath

    def save(self, symbols: Dict[str, Dict]) -> None:
        data = {
            sym: {"lookback": info["lookback"], "epochs": info["epochs"]}
            for sym, info in symbols.items()
        }
        with open(self._filepath, "w") as fh:
            json.dump(data, fh, indent=2)

    def load(self) -> Dict[str, Dict]:
        if not os.path.exists(self._filepath):
            return {}
        try:
            with open(self._filepath, "r") as fh:
                return json.load(fh)
        except Exception:
            return {}
