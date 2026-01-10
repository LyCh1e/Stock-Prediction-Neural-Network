import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import queue
from datetime import datetime
import json
import os
from stock_volume_predictor import StockTradingSystem, MASSIVEStockFetcher
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from matplotlib.patches import Rectangle

class StockPriceGUI:
    """
    GUI application for stock price prediction with scenario analysis.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("MASSIVE Stock Price Predictor - Neural Network")
        self.root.geometry("1400x900")
        
        # Data structures
        self.stocks = {}  # {symbol: {system, prediction, status}}
        self.api_key = ""
        self.message_queue = queue.Queue()
        
        # Load saved configuration
        self.load_config()
        
        # Create UI
        self.create_widgets()
        
        # Start message queue processor
        self.process_queue()
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure([0, 1], weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # === API Key Section ===
        api_frame = ttk.LabelFrame(main_frame, text="MASSIVE API Configuration", padding="5")
        api_frame.grid(row=0, column=0, columnspan=2, sticky=tk.W + tk.E, pady=5)
        
        ttk.Label(api_frame, text="MASSIVE API Key:").grid(row=0, column=0, padx=5)
        self.api_key_var = tk.StringVar(value=self.api_key)
        self._api_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, width=50, show="*")
        self._api_entry.grid(row=0, column=1, padx=5)
        
        ttk.Button(api_frame, text="Save API Key", command=self.save_api_key).grid(row=0, column=2, padx=5)
        ttk.Button(api_frame, text="Show/Hide", command=self.toggle_api_visibility).grid(row=0, column=3, padx=5)
        
        # === Stock Management Section ===
        stock_frame = ttk.LabelFrame(main_frame, text="Add Stock", padding="5")
        stock_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W + tk.E, pady=5)
        
        ttk.Label(stock_frame, text="Stock Symbol:").grid(row=0, column=0, padx=5)
        self.symbol_var = tk.StringVar()
        symbol_entry = ttk.Entry(stock_frame, textvariable=self.symbol_var, width=15)
        symbol_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(stock_frame, text="Lookback Days:").grid(row=0, column=2, padx=5)
        self.lookback_var = tk.IntVar(value=10)
        ttk.Spinbox(stock_frame, from_=5, to=30, textvariable=self.lookback_var, width=10).grid(row=0, column=3, padx=5)
        
        ttk.Label(stock_frame, text="Training Epochs:").grid(row=0, column=4, padx=5)
        self.epochs_var = tk.IntVar(value=200)
        ttk.Spinbox(stock_frame, from_=50, to=500, textvariable=self.epochs_var, width=10).grid(row=0, column=5, padx=5)
        
        ttk.Button(stock_frame, text="Add & Train Stock", command=self.add_stock).grid(row=0, column=6, padx=5)
        ttk.Button(stock_frame, text="Quick Add (SPY, AAPL, MSFT)", command=self.quick_add_stocks).grid(row=0, column=7, padx=5)
        
        # === Stock List and Details Section ===
        results_frame = ttk.Frame(main_frame)
        results_frame.grid(row=2, column=0, sticky=tk.W + tk.E + tk.N + tk.S, padx=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Stock list with treeview
        list_frame = ttk.LabelFrame(results_frame, text="Tracked Stocks", padding="5")
        list_frame.grid(row=0, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Create treeview with more columns
        columns = ('Symbol', 'Current', 'Best Buy', 'Best Sell', 'Avg Profit', 'Confidence', 'Sentiment', 'Status')
        self.tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=12)
        
        # Define headings
        self.tree.heading('Symbol', text='Symbol')
        self.tree.heading('Current', text='Current Price')
        self.tree.heading('Best Buy', text='Best Buy')
        self.tree.heading('Best Sell', text='Best Sell')
        self.tree.heading('Avg Profit', text='Avg Profit %')
        self.tree.heading('Confidence', text='Confidence')
        self.tree.heading('Sentiment', text='Sentiment')
        self.tree.heading('Status', text='Status')
        
        # Define column widths
        self.tree.column('Symbol', width=80)
        self.tree.column('Current', width=90)
        self.tree.column('Best Buy', width=90)
        self.tree.column('Best Sell', width=90)
        self.tree.column('Avg Profit', width=90)
        self.tree.column('Confidence', width=90)
        self.tree.column('Sentiment', width=90)
        self.tree.column('Status', width=100)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
        scrollbar.grid(row=0, column=1, sticky=tk.N + tk.S)
        
        # Double-click to show details
        self.tree.bind('<Double-Button-1>', self.show_stock_details)
        
        # Buttons for stock management
        button_frame = ttk.Frame(list_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        ttk.Button(button_frame, text="Predict Selected", command=self.predict_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Update Selected", command=self.update_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Predict All", command=self.predict_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Update All", command=self.update_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Remove Selected", command=self.remove_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Export Data", command=self.export_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Refresh All", command=self.refresh_all).pack(side=tk.LEFT, padx=2)
        
        # === Detailed Stock Info Frame ===
        detail_frame = ttk.LabelFrame(main_frame, text="Stock Details", padding="5")
        detail_frame.grid(row=2, column=1, sticky="nsew", padx=5)
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)
        
        # Create text widget for detailed info
        self.detail_text = scrolledtext.ScrolledText(detail_frame, height=15, width=60, wrap=tk.WORD, font=('Courier', 9))
        self.detail_text.grid(row=0, column=0, sticky="nsew")
        
        # === Visualization Section ===
        viz_frame = ttk.LabelFrame(main_frame, text="Visualization & Scenarios", padding="5")
        viz_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        viz_frame.columnconfigure(0, weight=1)
        
        # Create matplotlib figure for scenarios
        self.fig = Figure(figsize=(12, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="ew", pady=5)
        
        viz_buttons = ttk.Frame(viz_frame)
        viz_buttons.grid(row=1, column=0, pady=5)
        
        ttk.Button(viz_buttons, text="Show Scenario Comparison", command=self.show_scenario_comparison).pack(side=tk.LEFT, padx=2)
        ttk.Button(viz_buttons, text="Show Price Trend", command=self.show_price_trend).pack(side=tk.LEFT, padx=2)
        ttk.Button(viz_buttons, text="Show Profit Potential", command=self.show_profit_potential).pack(side=tk.LEFT, padx=2)
        ttk.Button(viz_buttons, text="Clear Plot", command=self.clear_plot).pack(side=tk.LEFT, padx=2)
        
        # === Log Section ===
        log_frame = ttk.LabelFrame(main_frame, text="Activity Log", padding="5")
        log_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, width=100, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky="ew")
        
        # === Status Bar ===
        self.status_var = tk.StringVar(value="Ready - Add stocks to begin prediction")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, columnspan=2, sticky="ew")
        
        # Initial plot
        self.show_welcome_plot()
    
    def log(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        print(message)
    
    def toggle_api_visibility(self):
        """Toggle API key visibility"""
        current = self.api_key_var.get()
        if hasattr(self, '_api_entry'):
            if self._api_entry.cget('show') == '*':
                self._api_entry.config(show='')
            else:
                self._api_entry.config(show='*')
    
    def save_api_key(self):
        """Save API key to config"""
        self.api_key = self.api_key_var.get().strip()
        if self.api_key:
            self.save_config()
            self.log("MASSIVE API key saved successfully")
            messagebox.showinfo("Success", "API key saved securely!")
        else:
            messagebox.showwarning("Warning", "Please enter a MASSIVE API key")
    
    def quick_add_stocks(self):
        """Quick add popular stocks"""
        popular_stocks = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
        for symbol in popular_stocks:
            if symbol not in self.stocks:
                self.symbol_var.set(symbol)
                self.add_stock()
    
    def add_stock(self):
        """Add a new stock and train the model"""
        symbol = self.symbol_var.get().strip().upper()
        
        if not symbol:
            messagebox.showwarning("Warning", "Please enter a stock symbol")
            return
        
        if not self.api_key:
            messagebox.showwarning("Warning", "Please enter and save your MASSIVE API key first")
            return
        
        if symbol in self.stocks:
            messagebox.showwarning("Warning", f"{symbol} is already being tracked")
            return
        
        # Add to stocks dictionary
        self.stocks[symbol] = {
            'system': None,
            'prediction': None,
            'status': 'Training...',
            'lookback': self.lookback_var.get(),
            'epochs': self.epochs_var.get(),
            'last_update': datetime.now().isoformat()
        }
        
        # Add to treeview with placeholder values
        self.tree.insert('', tk.END, iid=symbol, values=(
            symbol, '-', '-', '-', '-', '-', '-', 'Training...'
        ))
        
        # Start training in background thread
        thread = threading.Thread(target=self.train_stock_thread, args=(symbol,), daemon=True)
        thread.start()
        
        self.log(f"Started training price prediction model for {symbol}")
        self.symbol_var.set("")
    
    def train_stock_thread(self, symbol):
        """Train stock model in background thread"""
        try:
            stock_data = self.stocks[symbol]
            
            # Create system
            system = StockTradingSystem(
                api_key=self.api_key,
                lookback_window=stock_data['lookback']
            )
            
            # Train
            self.message_queue.put(('status', f"Training {symbol} price model..."))
            system.train_model(symbol, epochs=stock_data['epochs'])
            
            # Save system
            stock_data['system'] = system
            stock_data['status'] = 'Trained'
            
            # Make initial prediction
            prediction = system.predict_next_day(symbol, include_scenarios=True)
            stock_data['prediction'] = prediction
            
            # Update UI
            self.message_queue.put(('update_tree', symbol))
            self.message_queue.put(('log', f"Successfully trained {symbol} price prediction model"))
            self.message_queue.put(('status', f'Ready - {symbol} trained successfully'))
            
        except Exception as e:
            error_msg = str(e)
            self.message_queue.put(('error', f"Error training {symbol}: {error_msg}"))
            self.stocks[symbol]['status'] = f'Error: {error_msg[:30]}...'
            self.message_queue.put(('update_tree', symbol))
    
    def predict_selected(self):
        """Make prediction for selected stock"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a stock")
            return
        
        for symbol in selected:
            thread = threading.Thread(target=self.predict_stock_thread, args=(symbol,), daemon=True)
            thread.start()
    
    def predict_stock_thread(self, symbol):
        """Predict stock prices in background thread"""
        try:
            stock_data = self.stocks[symbol]
            
            if stock_data['system'] is None:
                self.message_queue.put(('error', f"{symbol} model not trained yet"))
                return
            
            self.message_queue.put(('status', f"Predicting prices for {symbol}..."))
            prediction = stock_data['system'].predict_next_day(symbol, include_scenarios=True)
            stock_data['prediction'] = prediction
            stock_data['last_update'] = datetime.now().isoformat()
            
            self.message_queue.put(('update_tree', symbol))
            self.message_queue.put(('log', f"Updated price predictions for {symbol}"))
            self.message_queue.put(('status', 'Ready'))
            
        except Exception as e:
            self.message_queue.put(('error', f"Error predicting {symbol}: {str(e)}"))
    
    def update_selected(self):
        """Update model for selected stock with latest data"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a stock")
            return
        
        for symbol in selected:
            thread = threading.Thread(target=self.update_stock_thread, args=(symbol,), daemon=True)
            thread.start()
    
    def update_stock_thread(self, symbol):
        """Update stock model in background thread"""
        try:
            stock_data = self.stocks[symbol]
            
            if stock_data['system'] is None:
                self.message_queue.put(('error', f"{symbol} model not trained yet"))
                return
            
            self.message_queue.put(('status', f"Updating {symbol} model..."))
            stock_data['system'].adaptive_update(symbol)
            
            # Make new prediction
            prediction = stock_data['system'].predict_next_day(symbol, include_scenarios=True)
            stock_data['prediction'] = prediction
            stock_data['last_update'] = datetime.now().isoformat()
            
            self.message_queue.put(('update_tree', symbol))
            self.message_queue.put(('log', f"Adaptively updated {symbol} price model"))
            self.message_queue.put(('status', 'Ready'))
            
        except Exception as e:
            self.message_queue.put(('error', f"Error updating {symbol}: {str(e)}"))
    
    def predict_all(self):
        """Predict all stocks"""
        if not self.stocks:
            messagebox.showinfo("Info", "No stocks to predict")
            return
        
        for symbol in self.stocks.keys():
            thread = threading.Thread(target=self.predict_stock_thread, args=(symbol,), daemon=True)
            thread.start()
    
    def update_all(self):
        """Update all stock models"""
        if not self.stocks:
            messagebox.showinfo("Info", "No stocks to update")
            return
        
        for symbol in self.stocks.keys():
            thread = threading.Thread(target=self.update_stock_thread, args=(symbol,), daemon=True)
            thread.start()
    
    def refresh_all(self):
        """Refresh predictions for all stocks"""
        self.predict_all()
    
    def remove_selected(self):
        """Remove selected stock"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a stock to remove")
            return
        
        if messagebox.askyesno("Confirm", f"Remove {len(selected)} stock(s)?"):
            for symbol in selected:
                self.tree.delete(symbol)
                del self.stocks[symbol]
                self.log(f"Removed {symbol} from tracking")
    
    def update_tree_item(self, symbol):
        """Update treeview item for a stock"""
        stock_data = self.stocks[symbol]
        prediction = stock_data.get('prediction')
        
        if prediction and 'scenarios' in prediction:
            scenarios = prediction['scenarios']
            avg_case = scenarios['average_case']
            
            current_price = f"${prediction['current_price']:.2f}"
            best_buy = f"${scenarios['best_case']['buy_price']:.2f}"
            best_sell = f"${scenarios['best_case']['sell_price']:.2f}"
            avg_profit = f"{avg_case['profit_potential']:.1f}%"
            confidence = f"{prediction['confidence']*100:.0f}%"
            sentiment = prediction['market_sentiment'].capitalize()
        else:
            current_price = '-'
            best_buy = '-'
            best_sell = '-'
            avg_profit = '-'
            confidence = '-'
            sentiment = '-'
        
        self.tree.item(symbol, values=(
            symbol,
            current_price,
            best_buy,
            best_sell,
            avg_profit,
            confidence,
            sentiment,
            stock_data['status']
        ))
    
    def show_stock_details(self, event=None):
        """Show detailed information for selected stock"""
        selected = self.tree.selection()
        if not selected:
            return
        
        symbol = selected[0]
        stock_data = self.stocks.get(symbol)
        
        if not stock_data or not stock_data.get('prediction'):
            self.detail_text.delete(1.0, tk.END)
            self.detail_text.insert(tk.END, f"No detailed data available for {symbol}")
            return
        
        prediction = stock_data['prediction']
        scenarios = prediction.get('scenarios', {})
        
        # Format detailed information
        details = f"""
{'='*60}
STOCK DETAILS: {symbol}
{'='*60}

Current Price:      ${prediction['current_price']:.2f}
Market Sentiment:   {prediction['market_sentiment'].upper()}
Prediction Date:    {prediction['prediction_date']}
Confidence Level:   {prediction['confidence']*100:.1f}%
Last Updated:       {prediction['last_updated']}

{'='*60}
SCENARIO ANALYSIS
{'='*60}

>>> BEST CASE (Optimistic) <<<
  Buy Price:        ${scenarios['best_case']['buy_price']:.2f}
  Sell Price:       ${scenarios['best_case']['sell_price']:.2f}
  Expected High:    ${scenarios['best_case']['high']:.2f}
  Expected Low:     ${scenarios['best_case']['low']:.2f}
  Profit Potential: {scenarios['best_case']['profit_potential']:.1f}%
  Volume:           {scenarios['best_case']['volume']:,.0f}

>>> AVERAGE CASE (Most Likely) <<<
  Buy Price:        ${scenarios['average_case']['buy_price']:.2f}
  Sell Price:       ${scenarios['average_case']['sell_price']:.2f}
  Expected High:    ${scenarios['average_case']['high']:.2f}
  Expected Low:     ${scenarios['average_case']['low']:.2f}
  Profit Potential: {scenarios['average_case']['profit_potential']:.1f}%
  Volume:           {scenarios['average_case']['volume']:,.0f}

>>> WORST CASE (Conservative) <<<
  Buy Price:        ${scenarios['worst_case']['buy_price']:.2f}
  Sell Price:       ${scenarios['worst_case']['sell_price']:.2f}
  Expected High:    ${scenarios['worst_case']['high']:.2f}
  Expected Low:     ${scenarios['worst_case']['low']:.2f}
  Profit Potential: {scenarios['worst_case']['profit_potential']:.1f}%
  Volume:           {scenarios['worst_case']['volume']:,.0f}

{'='*60}
TRADING RECOMMENDATION
{'='*60}
"""
        
        # Add trading recommendation
        avg_profit = scenarios['average_case']['profit_potential']
        confidence = prediction['confidence']
        
        if avg_profit > 5 and confidence > 0.7:
            recommendation = "STRONG BUY - High profit potential with good confidence"
        elif avg_profit > 2:
            recommendation = "BUY - Positive profit potential"
        elif avg_profit > -1:
            recommendation = "HOLD - Wait for better entry point"
        else:
            recommendation = "AVOID - Negative profit potential"
        
        details += f"\n{recommendation}\n"
        
        # Add technical indicators if available
        if 'technical_indicators' in prediction and prediction['technical_indicators']:
            indicators = prediction['technical_indicators']
            details += f"\n{'='*60}\nTECHNICAL INDICATORS\n{'='*60}\n"
            
            for key, value in indicators.items():
                if isinstance(value, float):
                    details += f"{key.replace('_', ' ').title()}: {value:.2f}\n"
                else:
                    details += f"{key.replace('_', ' ').title()}: {value}\n"
        
        # Update detail text
        self.detail_text.delete(1.0, tk.END)
        self.detail_text.insert(tk.END, details)
        
        # Highlight based on sentiment
        self.detail_text.tag_configure("positive", foreground="green")
        self.detail_text.tag_configure("negative", foreground="red")
        self.detail_text.tag_configure("neutral", foreground="blue")
        
        # Apply tags
        for line in details.split('\n'):
            if "BEST CASE" in line:
                start = self.detail_text.search(line, 1.0)
                if start:
                    end = f"{start}+{len(line)}c"
                    self.detail_text.tag_add("positive", start, end)
            elif "WORST CASE" in line:
                start = self.detail_text.search(line, 1.0)
                if start:
                    end = f"{start}+{len(line)}c"
                    self.detail_text.tag_add("negative", start, end)
    
    def show_scenario_comparison(self):
        """Show comparison chart of all scenarios"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a stock")
            return
        
        symbol = selected[0]
        stock_data = self.stocks[symbol]
        
        if not stock_data or not stock_data.get('prediction'):
            messagebox.showinfo("Info", "No prediction data available")
            return
        
        prediction = stock_data['prediction']
        scenarios = prediction['scenarios']
        
        self.fig.clear()
        
        # Create subplots
        ax1 = self.fig.add_subplot(131)
        ax2 = self.fig.add_subplot(132)
        ax3 = self.fig.add_subplot(133)
        
        scenario_names = ['Best Case', 'Average Case', 'Worst Case']
        scenario_data = [scenarios['best_case'], scenarios['average_case'], scenarios['worst_case']]
        
        colors = ['#4CAF50', '#2196F3', '#F44336']
        
        # Plot 1: Buy/Sell Prices
        for i, (name, data) in enumerate(zip(scenario_names, scenario_data)):
            ax1.bar(i, data['buy_price'], width=0.6, color=colors[i], alpha=0.7, label=f'{name} Buy')
            ax1.bar(i + 0.35, data['sell_price'], width=0.6, color=colors[i], alpha=0.9, label=f'{name} Sell', hatch='//')
        
        ax1.set_title(f'{symbol} - Buy/Sell Prices')
        ax1.set_ylabel('Price ($)')
        ax1.set_xticks([0.175, 1.175, 2.175])
        ax1.set_xticklabels(['Best', 'Average', 'Worst'])
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Profit Potential
        profits = [data['profit_potential'] for data in scenario_data]
        bars = ax2.bar(scenario_names, profits, color=colors)
        ax2.set_title(f'{symbol} - Profit Potential')
        ax2.set_ylabel('Profit (%)')
        
        # Color bars based on profit
        for bar, profit in zip(bars, profits):
            if profit > 0:
                bar.set_color('#4CAF50')
            else:
                bar.set_color('#F44336')
        
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Plot 3: Price Ranges (High-Low)
        for i, data in enumerate(scenario_data):
            ax3.plot([i, i], [data['low'], data['high']], 'k-', linewidth=3)
            ax3.plot(i, data['open'], '^', markersize=10, color='blue', label='Open' if i == 0 else "")
            ax3.plot(i, data['close'], 'v', markersize=10, color='red', label='Close' if i == 0 else "")
        
        ax3.set_title(f'{symbol} - Price Ranges')
        ax3.set_ylabel('Price ($)')
        ax3.set_xticks([0, 1, 2])
        ax3.set_xticklabels(['Best', 'Average', 'Worst'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        self.fig.suptitle(f'{symbol} Price Prediction Scenarios - Confidence: {prediction["confidence"]*100:.0f}%', fontsize=14)
        self.fig.tight_layout()
        self.canvas.draw()
        
        self.log(f"Displayed scenario comparison for {symbol}")
    
    def show_price_trend(self):
        """Show price trend visualization"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a stock")
            return
        
        symbol = selected[0]
        stock_data = self.stocks[symbol]
        
        if not stock_data or not stock_data.get('prediction'):
            messagebox.showinfo("Info", "No prediction data available")
            return
        
        # For demonstration, create a simulated trend
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Simulate price data
        np.random.seed(42)
        days = 30
        base_price = stock_data['prediction']['current_price']
        
        # Create trend with some randomness
        trend = np.linspace(0, 0.1, days)  # Upward trend
        noise = np.random.normal(0, 0.02, days)
        prices = base_price * (1 + trend + noise.cumsum())
        
        # Plot historical trend
        ax.plot(range(days), prices, 'b-', linewidth=2, label='Simulated Trend')
        ax.axhline(y=base_price, color='r', linestyle='--', label='Current Price')
        
        # Add prediction markers
        prediction = stock_data['prediction']
        scenarios = prediction['scenarios']
        
        ax.scatter(days, scenarios['best_case']['close'], color='green', s=100, marker='^', label='Best Case Target')
        ax.scatter(days, scenarios['average_case']['close'], color='blue', s=100, marker='o', label='Avg Case Target')
        ax.scatter(days, scenarios['worst_case']['close'], color='red', s=100, marker='v', label='Worst Case Target')
        
        ax.set_xlabel('Days')
        ax.set_ylabel('Price ($)')
        ax.set_title(f'{symbol} - Price Trend & Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        self.log(f"Displayed price trend for {symbol}")
    
    def show_profit_potential(self):
        """Show profit potential comparison across stocks"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        symbols = []
        best_profits = []
        avg_profits = []
        worst_profits = []
        
        for symbol, data in self.stocks.items():
            if data.get('prediction') and 'scenarios' in data['prediction']:
                symbols.append(symbol)
                scenarios = data['prediction']['scenarios']
                best_profits.append(scenarios['best_case']['profit_potential'])
                avg_profits.append(scenarios['average_case']['profit_potential'])
                worst_profits.append(scenarios['worst_case']['profit_potential'])
        
        if not symbols:
            messagebox.showinfo("Info", "No prediction data available")
            return
        
        x = np.arange(len(symbols))
        width = 0.25
        
        ax.bar(x - width, best_profits, width, label='Best Case', color='green', alpha=0.7)
        ax.bar(x, avg_profits, width, label='Average Case', color='blue', alpha=0.7)
        ax.bar(x + width, worst_profits, width, label='Worst Case', color='red', alpha=0.7)
        
        ax.set_xlabel('Stock Symbol')
        ax.set_ylabel('Profit Potential (%)')
        ax.set_title('Profit Potential Comparison Across Stocks')
        ax.set_xticks(x)
        ax.set_xticklabels(symbols)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at 0%
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        self.log(f"Displayed profit potential for {len(symbols)} stocks")
    
    def show_welcome_plot(self):
        """Show welcome plot"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        welcome_text = """MASSIVE Stock Price Predictor
        
• Add stocks to train prediction models
• Get Best/Average/Worst case scenarios
• View detailed trading recommendations
• Compare profit potential across stocks
        
Double-click any stock for detailed analysis"""
        
        ax.text(0.5, 0.5, welcome_text,
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='gray',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.5))
        ax.axis('off')
        
        self.canvas.draw()
    
    def clear_plot(self):
        """Clear the plot"""
        self.show_welcome_plot()
    
    def export_data(self):
        """Export all stock data to JSON"""
        if not self.stocks:
            messagebox.showinfo("Info", "No data to export")
            return
        
        export_data = {}
        for symbol, data in self.stocks.items():
            if data.get('prediction'):
                export_data[symbol] = data['prediction']
        
        filename = f"stock_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.log(f"Exported data to {filename}")
        messagebox.showinfo("Success", f"Data exported to {filename}")
    
    def save_config(self):
        """Save configuration to file"""
        config = {
            'api_key': self.api_key
        }
        with open('stock_predictor_config.json', 'w') as f:
            json.dump(config, f)
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists('stock_predictor_config.json'):
            try:
                with open('stock_predictor_config.json', 'r') as f:
                    config = json.load(f)
                    self.api_key = config.get('api_key', '')
            except:
                pass
    
    def process_queue(self):
        """Process messages from background threads"""
        try:
            while True:
                msg_type, msg_data = self.message_queue.get_nowait()
                
                if msg_type == 'log':
                    self.log(msg_data)
                elif msg_type == 'status':
                    self.status_var.set(msg_data)
                elif msg_type == 'error':
                    self.log(f"ERROR: {msg_data}")
                    messagebox.showerror("Error", msg_data)
                elif msg_type == 'update_tree':
                    self.update_tree_item(msg_data)
                    
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_queue)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = StockPriceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()