"""
Stock Data Auto-Updater
Automatically fetches and updates stock data at regular intervals
Saves data to JSON file for persistence
"""

import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from stock_volume_predictor import YahooFinanceAPI as MassiveAPI, StockTradingSystem


class StockAutoUpdater:
    """
    Automatically updates stock data at regular intervals and persists to JSON.
    """
    
    def __init__(self, api_key: str, update_interval_minutes: int = 60, 
                 data_file: str = "stock_data_cache.json"):
        """
        Initialize the auto-updater.
        
        Args:
            api_key: MASSIVE API key for authentication
            update_interval_minutes: How often to update data (default: 60 minutes)
            data_file: Path to JSON file for storing data
        """
        self.api_key = api_key
        self.update_interval = update_interval_minutes * 60  # Convert to seconds
        self.data_file = data_file
        self.api = MassiveAPI(api_key)
        
        # Track stocks and their systems
        self.tracked_stocks = {}  # {symbol: StockTradingSystem}
        self.stock_data_cache = {}  # {symbol: {timestamp, data, prediction}}
        
        # Control flags
        self.is_running = False
        self.update_thread = None
        self.lock = threading.Lock()
        
        # Statistics
        self.update_count = 0
        self.last_update_time = None
        self.errors = []
        
        # Load existing data
        self.load_cache()
    
    def add_stock(self, symbol: str, lookback_days: int = 10, 
                  epochs: int = 200) -> Dict:
        """
        Add a stock to track and update automatically.
        
        Args:
            symbol: Stock ticker symbol
            lookback_days: Days of historical data to use
            epochs: Training epochs for the model
        
        Returns:
            Dict with status and initial prediction
        """
        symbol = symbol.upper()
        
        with self.lock:
            if symbol in self.tracked_stocks:
                return {
                    'status': 'exists',
                    'message': f'{symbol} is already being tracked',
                    'data': self.stock_data_cache.get(symbol)
                }
            
            try:
                # Create trading system for this stock
                system = StockTradingSystem(self.api_key, lookback_window=lookback_days)
                
                # Train the model
                system.train_model(symbol, epochs=epochs)
                
                # Get initial prediction
                prediction = system.predict_next_day(symbol, include_scenarios=True)
                
                # Store system and data
                self.tracked_stocks[symbol] = {
                    'system': system,
                    'lookback_days': lookback_days,
                    'epochs': epochs,
                    'added_at': datetime.now().isoformat()
                }
                
                self.stock_data_cache[symbol] = {
                    'symbol': symbol,
                    'last_updated': datetime.now().isoformat(),
                    'prediction': prediction,
                    'historical_predictions': []
                }
                
                # Save immediately
                self.save_cache()
                
                return {
                    'status': 'success',
                    'message': f'{symbol} added successfully',
                    'data': self.stock_data_cache[symbol]
                }
                
            except Exception as e:
                error_msg = f"Error adding {symbol}: {str(e)}"
                self.errors.append({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'error': error_msg
                })
                return {
                    'status': 'error',
                    'message': error_msg
                }
    
    def remove_stock(self, symbol: str) -> Dict:
        """
        Remove a stock from tracking.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dict with status
        """
        symbol = symbol.upper()
        
        with self.lock:
            if symbol not in self.tracked_stocks:
                return {
                    'status': 'error',
                    'message': f'{symbol} is not being tracked'
                }
            
            # Remove from tracking
            del self.tracked_stocks[symbol]
            if symbol in self.stock_data_cache:
                del self.stock_data_cache[symbol]
            
            # Save updated cache
            self.save_cache()
            
            return {
                'status': 'success',
                'message': f'{symbol} removed from tracking'
            }
    
    def update_stock(self, symbol: str) -> Dict:
        """
        Manually update a specific stock.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dict with updated data
        """
        symbol = symbol.upper()
        
        with self.lock:
            if symbol not in self.tracked_stocks:
                return {
                    'status': 'error',
                    'message': f'{symbol} is not being tracked'
                }
            
            try:
                system_data = self.tracked_stocks[symbol]
                system = system_data['system']
                
                # Update the model with latest data
                system.adaptive_update(symbol)
                
                # Get new prediction
                prediction = system.predict_next_day(symbol, include_scenarios=True)
                
                # Store historical prediction
                if symbol in self.stock_data_cache:
                    old_data = self.stock_data_cache[symbol]
                    if 'prediction' in old_data:
                        if 'historical_predictions' not in old_data:
                            old_data['historical_predictions'] = []
                        
                        old_data['historical_predictions'].append({
                            'timestamp': old_data.get('last_updated'),
                            'prediction': old_data['prediction']
                        })
                        
                        # Keep only last 24 predictions (24 hours if updating hourly)
                        if len(old_data['historical_predictions']) > 24:
                            old_data['historical_predictions'] = old_data['historical_predictions'][-24:]
                
                # Update cache
                self.stock_data_cache[symbol] = {
                    'symbol': symbol,
                    'last_updated': datetime.now().isoformat(),
                    'prediction': prediction,
                    'historical_predictions': self.stock_data_cache.get(symbol, {}).get('historical_predictions', [])
                }
                
                # Save to file
                self.save_cache()
                
                return {
                    'status': 'success',
                    'message': f'{symbol} updated successfully',
                    'data': self.stock_data_cache[symbol]
                }
                
            except Exception as e:
                error_msg = f"Error updating {symbol}: {str(e)}"
                self.errors.append({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'error': error_msg
                })
                return {
                    'status': 'error',
                    'message': error_msg
                }
    
    def update_all_stocks(self) -> Dict:
        """
        Update all tracked stocks.
        
        Returns:
            Dict with update summary
        """
        results = {
            'updated': [],
            'failed': [],
            'timestamp': datetime.now().isoformat()
        }
        
        with self.lock:
            symbols = list(self.tracked_stocks.keys())
        
        for symbol in symbols:
            result = self.update_stock(symbol)
            if result['status'] == 'success':
                results['updated'].append(symbol)
            else:
                results['failed'].append({
                    'symbol': symbol,
                    'error': result['message']
                })
        
        self.update_count += 1
        self.last_update_time = datetime.now().isoformat()
        
        return results
    
    def _auto_update_loop(self):
        """
        Background loop that updates stocks at regular intervals.
        """
        print(f"Auto-updater started. Will update every {self.update_interval/60:.1f} minutes")
        
        while self.is_running:
            try:
                # Update all stocks
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting scheduled update...")
                results = self.update_all_stocks()
                
                print(f"Update complete: {len(results['updated'])} successful, {len(results['failed'])} failed")
                if results['failed']:
                    print(f"Failed stocks: {results['failed']}")
                
            except Exception as e:
                error_msg = f"Error in auto-update loop: {str(e)}"
                print(error_msg)
                self.errors.append({
                    'timestamp': datetime.now().isoformat(),
                    'error': error_msg
                })
            
            # Wait for next interval
            time.sleep(self.update_interval)
    
    def start(self):
        """
        Start the auto-update background process.
        """
        if self.is_running:
            return {
                'status': 'already_running',
                'message': 'Auto-updater is already running'
            }
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._auto_update_loop, daemon=True)
        self.update_thread.start()
        
        return {
            'status': 'success',
            'message': f'Auto-updater started (interval: {self.update_interval/60:.1f} minutes)'
        }
    
    def stop(self):
        """
        Stop the auto-update background process.
        """
        if not self.is_running:
            return {
                'status': 'not_running',
                'message': 'Auto-updater is not running'
            }
        
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        
        return {
            'status': 'success',
            'message': 'Auto-updater stopped'
        }
    
    def get_status(self) -> Dict:
        """
        Get current status of the auto-updater.
        
        Returns:
            Dict with status information
        """
        return {
            'is_running': self.is_running,
            'tracked_stocks': list(self.tracked_stocks.keys()),
            'update_interval_minutes': self.update_interval / 60,
            'update_count': self.update_count,
            'last_update': self.last_update_time,
            'data_file': self.data_file,
            'recent_errors': self.errors[-5:] if self.errors else []
        }
    
    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """
        Get cached data for a specific stock.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Cached data dict or None
        """
        symbol = symbol.upper()
        return self.stock_data_cache.get(symbol)
    
    def get_all_data(self) -> Dict:
        """
        Get all cached stock data.
        
        Returns:
            Dict of all stock data
        """
        with self.lock:
            return self.stock_data_cache.copy()
    
    def save_cache(self):
        """
        Save current data cache to JSON file.
        """
        try:
            # Create a serializable version (remove StockTradingSystem objects)
            save_data = {
                'metadata': {
                    'last_save': datetime.now().isoformat(),
                    'update_count': self.update_count,
                    'last_update': self.last_update_time,
                    'tracked_symbols': list(self.tracked_stocks.keys())
                },
                'stocks': self.stock_data_cache,
                'errors': self.errors[-20:]  # Keep last 20 errors
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            print(f"Data cache saved to {self.data_file}")
            
        except Exception as e:
            print(f"Error saving cache: {str(e)}")
    
    def load_cache(self):
        """
        Load data cache from JSON file.
        """
        if not os.path.exists(self.data_file):
            print(f"No existing cache file found at {self.data_file}")
            return
        
        try:
            with open(self.data_file, 'r') as f:
                save_data = json.load(f)
            
            # Restore data
            self.stock_data_cache = save_data.get('stocks', {})
            self.errors = save_data.get('errors', [])
            
            metadata = save_data.get('metadata', {})
            self.update_count = metadata.get('update_count', 0)
            self.last_update_time = metadata.get('last_update')
            
            print(f"Loaded cache with {len(self.stock_data_cache)} stocks")
            print(f"Last update: {self.last_update_time}")
            
        except Exception as e:
            print(f"Error loading cache: {str(e)}")
    
    def export_historical_data(self, symbol: str = '', filename: str = '') -> str:
        """
        Export historical prediction data to a separate file.
        
        Args:
            symbol: Specific stock symbol (None for all stocks)
            filename: Output filename (auto-generated if None)
        
        Returns:
            Filename of exported data
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if symbol:
                filename = f"historical_{symbol}_{timestamp}.json"
            else:
                filename = f"historical_all_{timestamp}.json"
        
        try:
            if symbol:
                symbol = symbol.upper()
                if symbol not in self.stock_data_cache:
                    raise ValueError(f"{symbol} not found in cache")
                export_data = {symbol: self.stock_data_cache[symbol]}
            else:
                export_data = self.stock_data_cache
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"Historical data exported to {filename}")
            return filename
            
        except Exception as e:
            print(f"Error exporting data: {str(e)}")
            raise


# Standalone functions for command-line usage
def create_updater(api_key: str, interval_minutes: int = 60) -> StockAutoUpdater:
    """
    Create and return a new StockAutoUpdater instance.
    
    Args:
        api_key: MASSIVE API key
        interval_minutes: Update interval in minutes
    
    Returns:
        StockAutoUpdater instance
    """
    return StockAutoUpdater(api_key, interval_minutes)


def run_updater_cli():
    """
    Command-line interface for the auto-updater.
    """
    import sys
    
    print("=" * 60)
    print("Stock Data Auto-Updater")
    print("=" * 60)
    print()
    
    # Get API key
    api_key = input("Enter MASSIVE API Key: ").strip()
    if not api_key:
        print("Error: API key is required")
        sys.exit(1)
    
    # Get update interval
    interval_input = input("Update interval in minutes (default: 60): ").strip()
    interval = int(interval_input) if interval_input else 60
    
    # Create updater
    updater = StockAutoUpdater(api_key, interval)
    
    # Add initial stocks
    print("\nAdd stocks to track (comma-separated, e.g., AAPL,MSFT,SPY):")
    stocks_input = input("Stocks: ").strip()
    
    if stocks_input:
        symbols = [s.strip() for s in stocks_input.split(',')]
        for symbol in symbols:
            if symbol:
                print(f"\nAdding {symbol}...")
                result = updater.add_stock(symbol)
                print(f"  {result['message']}")
    
    # Start auto-updater
    print("\nStarting auto-updater...")
    result = updater.start()
    print(result['message'])
    
    # Keep running
    try:
        print("\nAuto-updater is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(60)
            status = updater.get_status()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status: {len(status['tracked_stocks'])} stocks tracked, "
                  f"{status['update_count']} updates completed")
    except KeyboardInterrupt:
        print("\n\nStopping auto-updater...")
        updater.stop()
        print("Auto-updater stopped. Data has been saved.")


if __name__ == "__main__":
    run_updater_cli()