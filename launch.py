# Entry point: checks dependencies then delegates to ui.app.main().

import json
import os
import sys


def check_dependencies() -> list:
    required = ["numpy", "pandas", "requests", "matplotlib", "tkinter", "yfinance", "openpyxl"]
    missing  = []
    for pkg in required:
        try:
            __import__("tkinter" if pkg == "tkinter" else pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def ensure_data_files() -> None:
    import pandas as pd
    from openpyxl import Workbook

    def _empty_xlsx(path: str, sheet: str = "Sheet1") -> None:
        wb = Workbook()
        wb.active.title = sheet  # type: ignore[union-attr]
        wb.save(path)

    if not os.path.exists("prediction_score.xlsx"):
        _empty_xlsx("prediction_score.xlsx", "Score Data")

    if not os.path.exists("stock_data.xlsx"):
        _empty_xlsx("stock_data.xlsx")

    if not os.path.exists("stock_models_history.csv"):
        pd.DataFrame(columns=["Symbol", "Date", "Avg", "Best", "Worst"]).to_csv(
            "stock_models_history.csv", index=False
        )

    if not os.path.exists("stock_models.json"):
        with open("stock_models.json", "w", encoding="utf-8") as fh:
            json.dump({}, fh)

    if not os.path.exists("stock_predictions.xlsx"):
        _empty_xlsx("stock_predictions.xlsx")

    if not os.path.exists("tracked_symbols.json"):
        with open("tracked_symbols.json", "w", encoding="utf-8") as fh:
            json.dump({}, fh)


def main() -> None:
    missing = check_dependencies()
    if missing:
        sys.exit(1)

    ensure_data_files()

    try:
        from ui.app import main as run_app
        run_app()
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
