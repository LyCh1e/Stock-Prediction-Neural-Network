# Entry point: shows a splash window while checking dependencies and data files,
# then delegates to ui.app.main().

import json
import os
import sys
import time

_PACKAGES = ["numpy", "pandas", "requests", "matplotlib", "tkinter", "yfinance", "openpyxl"]

_DATA_FILES = [
    ("prediction_score.xlsx",    "Score Data", "xlsx"),
    ("stock_data.xlsx",          "Sheet1",     "xlsx"),
    ("stock_models_history.csv", None,         "csv"),
    ("stock_models.json",        None,         "json"),
    ("stock_predictions.xlsx",   "Sheet1",     "xlsx"),
    ("tracked_symbols.json",     None,         "json"),
]


class _Splash:
    _ICON = {"pending": "○", "ok": "✓", "new": "✓", "error": "✗"}
    _FG   = {"pending": "#aaa", "ok": "#2e7d32", "new": "#1565c0", "error": "#c62828"}

    def __init__(self) -> None:
        import tkinter as tk
        self._tk = tk
        self.root = tk.Tk()
        self.root.title("Stock Price Predictor")
        self.root.resizable(False, False)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        w, h = 420, 460
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
        self.root.configure(bg="white")

        border = tk.Frame(self.root, bg="#c8c8c8")
        border.pack(fill="both", expand=True, padx=1, pady=1)
        body = tk.Frame(border, bg="white")
        body.pack(fill="both", expand=True, padx=1, pady=1)

        tk.Label(body, text="Stock Price Predictor",
                 font=("Helvetica", 15, "bold"), bg="white", fg="#1a237e").pack(pady=(22, 2))
        tk.Label(body, text="Yahoo Finance  ·  Neural Network",
                 font=("Helvetica", 9), bg="white", fg="#777").pack(pady=(0, 12))

        tk.Frame(body, height=1, bg="#e0e0e0").pack(fill="x", padx=18)

        self._list = tk.Frame(body, bg="white")
        self._list.pack(fill="both", expand=True, padx=24, pady=6)

        tk.Frame(body, height=1, bg="#e0e0e0").pack(fill="x", padx=18)

        self._status_var = tk.StringVar(value="Starting…")
        tk.Label(body, textvariable=self._status_var, font=("Helvetica", 8),
                 bg="white", fg="#888", anchor="w").pack(fill="x", padx=24, pady=(5, 16))

        self._rows: dict = {}
        self.root.update()

    def section(self, text: str) -> None:
        self._tk.Label(self._list, text=text, font=("Helvetica", 8, "bold"),
                       bg="white", fg="#555", anchor="w").pack(fill="x", pady=(8, 2))
        self.root.update()

    def item(self, key: str, label: str, state: str = "pending") -> None:
        row = self._tk.Frame(self._list, bg="white")
        row.pack(fill="x", pady=1)
        icon_lbl = self._tk.Label(row, text=self._ICON[state], font=("Courier", 10),
                                  fg=self._FG[state], bg="white", width=3, anchor="center")
        icon_lbl.pack(side="left")
        self._tk.Label(row, text=label, font=("Helvetica", 9),
                       bg="white", fg="#333", anchor="w").pack(side="left")
        self._rows[key] = icon_lbl
        self.root.update()

    def update_item(self, key: str, state: str) -> None:
        if key in self._rows:
            self._rows[key].config(text=self._ICON[state], fg=self._FG[state])
            self.root.update()

    def status(self, text: str) -> None:
        self._status_var.set(text)
        self.root.update()

    def close(self) -> None:
        self.root.destroy()


def _check_packages(splash: _Splash) -> list:
    splash.section("Dependencies")
    splash.status("Checking dependencies…")
    missing = []
    for pkg in _PACKAGES:
        splash.item(pkg, pkg, "pending")
        try:
            __import__("tkinter" if pkg == "tkinter" else pkg)
            splash.update_item(pkg, "ok")
        except ImportError:
            splash.update_item(pkg, "error")
            missing.append(pkg)
    return missing


def _ensure_files(splash: _Splash) -> None:
    import pandas as pd
    from openpyxl import Workbook

    def _empty_xlsx(path: str, sheet: str) -> None:
        wb = Workbook()
        wb.active.title = sheet  # type: ignore[union-attr]
        wb.save(path)

    splash.section("Data Files")
    splash.status("Checking data files…")
    for path, sheet, kind in _DATA_FILES:
        existed = os.path.exists(path)
        splash.item(path, path, "pending")
        try:
            if not existed:
                if kind == "xlsx":
                    _empty_xlsx(path, sheet or "Sheet1")
                elif kind == "csv":
                    pd.DataFrame(columns=["Symbol", "Date", "Avg", "Best", "Worst"]).to_csv(
                        path, index=False
                    )
                else:
                    with open(path, "w", encoding="utf-8") as fh:
                        json.dump({}, fh)
            splash.update_item(path, "ok" if existed else "new")
        except Exception:
            splash.update_item(path, "error")


def main() -> None:
    try:
        import importlib.util
        if importlib.util.find_spec("tkinter") is None:
            raise ImportError
    except ImportError:
        print("tkinter is not available. Install it and retry.")
        sys.exit(1)

    splash = _Splash()

    missing = _check_packages(splash)
    if missing:
        splash.status(f"Missing: {', '.join(missing)} — install and retry.")
        splash.root.mainloop()
        sys.exit(1)

    _ensure_files(splash)

    splash.status("Launching application…")
    splash.root.update()
    time.sleep(0.4)
    splash.close()

    try:
        from ui.app import main as run_app
        run_app()
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
