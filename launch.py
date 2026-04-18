# Entry point: checks dependencies then delegates to ui.app.main().

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
    missing = check_dependencies()
    if missing:
        sys.exit(1)

    try:
        from ui.app import main as run_app
        run_app()
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
