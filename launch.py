#!/usr/bin/env python3
"""
launch.py
~~~~~~~~~
Entry point for the Stock Price Predictor.

Checks dependencies, then delegates to src.ui.app.main().
"""

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


def main() -> None:
    print("=" * 60)
    print("Stock Price Predictor — Neural Network")
    print("=" * 60)
    print()

    print("Checking dependencies...")
    missing = check_dependencies()
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("\nPlease install required packages:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    print("All dependencies installed")
    print()

    print("Launching GUI...")
    try:
        from ui.app import main as run_app
        run_app()
    except Exception as exc:
        print(f"Error launching GUI: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
