#!/usr/bin/env python3
"""
Quick launcher for MASSIVE Stock Price Predictor GUI
"""

import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required = ['numpy', 'pandas', 'requests', 'matplotlib', 'tkinter']
    missing = []
    
    for package in required:
        try:
            if package == 'tkinter':
                import tkinter
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def main():
    print("=" * 60)
    print("MASSIVE Stock Price Predictor - Neural Network")
    print("=" * 60)
    print()
    print("Features:")
    print("  • Multi-scenario price predictions (Best/Average/Worst)")
    print("  • Expected High/Low price ranges")
    print("  • Profit potential analysis")
    print("  • Market sentiment integration")
    print("  • Technical indicators")
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print()
        print("Please install required packages:")
        print("  pip install -r requirements.txt")
        print()
        sys.exit(1)
    
    print("✓ All dependencies installed")
    print()
    
    # Check for stock_gui.py
    if not os.path.exists('stock_gui.py'):
        print("stock_gui.py not found in current directory")
        print("Please ensure all files are in the same folder")
        sys.exit(1)
    
    print("✓ GUI file found")
    print()
    print("Launching GUI...")
    print()
    
    # Import and run GUI
    try:
        import stock_gui
        stock_gui.main()
    except Exception as e:
        print(f"Error launching GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()