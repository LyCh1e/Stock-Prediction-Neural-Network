import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import queue
from datetime import datetime, timedelta
import json
import os
from stock_volume_predictor import StockTradingSystem

import pandas as pd


class StockPriceGUI:
    """
    GUI for stock price prediction using Yahoo Finance data.
    Stock data and predictions each export to their own Excel file.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Predictor — Yahoo Finance")
        self.root.geometry("900x640")

        self.stocks = {}          # {symbol: {system, prediction, raw_df, status}}
        self.message_queue = queue.Queue()

        self.load_config()
        self.create_widgets()
        self.process_queue()

    # ------------------------------------------------------------------ #
    #  UI CONSTRUCTION                                                     #
    # ------------------------------------------------------------------ #

    def create_widgets(self):
        main = ttk.Frame(self.root, padding="10")
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        # ── Add stock controls ─────────────────────────────────────── #
        add_frame = ttk.LabelFrame(main, text="Add Stock  (Yahoo Finance — no API key required)", padding="6")
        add_frame.grid(row=0, column=0, sticky="ew", pady=4)

        ttk.Label(add_frame, text="Symbol:").grid(row=0, column=0, padx=4)
        self.symbol_var = tk.StringVar()
        ttk.Entry(add_frame, textvariable=self.symbol_var, width=10).grid(row=0, column=1, padx=4)

        ttk.Label(add_frame, text="Lookback Days:").grid(row=0, column=2, padx=4)
        self.lookback_var = tk.IntVar(value=10)
        ttk.Spinbox(add_frame, from_=5, to=30, textvariable=self.lookback_var, width=7).grid(row=0, column=3, padx=4)

        ttk.Label(add_frame, text="Epochs:").grid(row=0, column=4, padx=4)
        self.epochs_var = tk.IntVar(value=200)
        ttk.Spinbox(add_frame, from_=50, to=500, textvariable=self.epochs_var, width=7).grid(row=0, column=5, padx=4)

        ttk.Button(add_frame, text="Add & Train", command=self.add_stock).grid(row=0, column=6, padx=6)
        ttk.Button(add_frame, text="Quick Add (SPY, AAPL, MSFT)", command=self.quick_add).grid(row=0, column=7, padx=6)

        # ── Stock list ────────────────────────────────────────────── #
        list_frame = ttk.LabelFrame(main, text="Tracked Stocks", padding="6")
        list_frame.grid(row=1, column=0, sticky="nsew", pady=4)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        cols = ("Symbol", "Current Price", "Sentiment", "Confidence", "Signal", "Status")
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=14)
        widths = [90, 110, 110, 100, 120, 140]
        for col, w in zip(cols, widths):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center")

        self.tree.tag_configure("ready",    background="#e8f5e9")
        self.tree.tag_configure("training", background="#fff9c4")
        self.tree.tag_configure("error",    background="#ffebee")

        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        # ── Action buttons ─────────────────────────────────────────── #
        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=2, column=0, sticky="w", pady=4)
        for text, cmd in [
            ("Predict Selected",       self.predict_selected),
            ("Predict All",            self.predict_all),
            ("Update Selected",        self.update_selected),
            ("Update All",             self.update_all),
            ("Remove Selected",        self.remove_selected),
            ("Export Stock Data",      self.export_stock_data),
            ("Export Predictions",     self.export_predictions),
        ]:
            ttk.Button(btn_frame, text=text, command=cmd).pack(side="left", padx=3)

        # ── Log ───────────────────────────────────────────────────── #
        log_frame = ttk.LabelFrame(main, text="Activity Log", padding="4")
        log_frame.grid(row=3, column=0, sticky="ew", pady=4)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=7, wrap=tk.WORD, font=("Courier", 9))
        self.log_text.grid(row=0, column=0, sticky="ew")

        # ── Status bar ────────────────────────────────────────────── #
        self.status_var = tk.StringVar(value="Ready — add stocks to begin")
        ttk.Label(main, textvariable=self.status_var, relief="sunken", anchor="w").grid(row=4, column=0, sticky="ew")

    # ------------------------------------------------------------------ #
    #  HELPERS                                                             #
    # ------------------------------------------------------------------ #

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_text.see(tk.END)
        print(msg)

    def _tree_has(self, iid):
        try:
            self.tree.item(iid)
            return True
        except tk.TclError:
            return False

    def _update_tree_row(self, symbol):
        data = self.stocks.get(symbol, {})
        pred = data.get("prediction")
        status = data.get("status", "—")

        if pred:
            current = f"${pred['current_price']:.2f}"
            sentiment = pred.get("sentiment_analysis", {}).get("sentiment", "—")
            conf = f"{pred['confidence'] * 100:.0f}%"
            signal = pred.get("recommendation", "—").replace("_", " ")
            tag = "ready"
        else:
            current = sentiment = conf = signal = "—"
            tag = "error" if "Error" in status else "training"

        vals = (symbol, current, sentiment, conf, signal, status)
        if self._tree_has(symbol):
            self.tree.item(symbol, values=vals, tags=(tag,))
        else:
            self.tree.insert("", "end", iid=symbol, values=vals, tags=(tag,))

    def _selected_symbols(self):
        return list(self.tree.selection())

    # ------------------------------------------------------------------ #
    #  STOCK MANAGEMENT                                                    #
    # ------------------------------------------------------------------ #

    def add_stock(self):
        symbol = self.symbol_var.get().strip().upper()
        if not symbol:
            messagebox.showwarning("Warning", "Enter a stock symbol.")
            return
        if symbol in self.stocks:
            messagebox.showwarning("Warning", f"{symbol} is already being tracked.")
            return
        self.stocks[symbol] = {
            "system": None, "prediction": None, "raw_df": None,
            "status": "Training…",
            "lookback": self.lookback_var.get(),
            "epochs": self.epochs_var.get(),
        }
        self._update_tree_row(symbol)
        threading.Thread(target=self._train_thread, args=(symbol,), daemon=True).start()
        self.log(f"Queued training for {symbol}")
        self.symbol_var.set("")

    def quick_add(self):
        for sym in ["SPY", "AAPL", "MSFT"]:
            if sym not in self.stocks:
                self.symbol_var.set(sym)
                self.add_stock()

    def remove_selected(self):
        syms = self._selected_symbols()
        if not syms:
            messagebox.showwarning("Warning", "Select a stock first.")
            return
        if messagebox.askyesno("Confirm", f"Remove {', '.join(syms)}?"):
            for sym in syms:
                if self._tree_has(sym):
                    self.tree.delete(sym)
                self.stocks.pop(sym, None)
                self.log(f"Removed {sym}")

    # ------------------------------------------------------------------ #
    #  BACKGROUND THREADS                                                  #
    # ------------------------------------------------------------------ #

    def _train_thread(self, symbol):
        try:
            data = self.stocks[symbol]
            system = StockTradingSystem(api_key="", lookback_window=data["lookback"])
            self.message_queue.put(("status", f"Training {symbol}…"))

            # Train and capture the raw dataframe for export
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            raw_df = system.api.fetch_stock_data(symbol, start_date, end_date)
            data["raw_df"] = raw_df

            system.train_model(symbol, epochs=data["epochs"])
            data["system"] = system
            data["status"] = "Trained"

            pred = system.predict_next_day(symbol, include_scenarios=True)
            data["prediction"] = pred
            data["status"] = "Ready"

            self.message_queue.put(("refresh", symbol))
            self.message_queue.put(("log", f"✓ {symbol} trained and predicted"))
            self.message_queue.put(("status", f"Ready — {symbol} complete"))
        except Exception as e:
            err = str(e)
            if symbol in self.stocks:
                self.stocks[symbol]["status"] = f"Error: {err[:40]}"
            self.message_queue.put(("refresh", symbol))
            self.message_queue.put(("log", f"✗ {symbol}: {err}"))

    def _predict_thread(self, symbol):
        try:
            data = self.stocks[symbol]
            if not data["system"]:
                self.message_queue.put(("log", f"{symbol} not yet trained."))
                return
            self.message_queue.put(("status", f"Predicting {symbol}…"))
            pred = data["system"].predict_next_day(symbol, include_scenarios=True)
            data["prediction"] = pred
            data["status"] = "Ready"
            self.message_queue.put(("refresh", symbol))
            self.message_queue.put(("log", f"✓ {symbol} predictions refreshed"))
            self.message_queue.put(("status", "Ready"))
        except Exception as e:
            self.message_queue.put(("log", f"✗ {symbol} predict error: {e}"))

    def _update_thread(self, symbol):
        try:
            data = self.stocks[symbol]
            if not data["system"]:
                self.message_queue.put(("log", f"{symbol} not yet trained."))
                return
            self.message_queue.put(("status", f"Updating {symbol}…"))
            data["system"].adaptive_update(symbol)
            pred = data["system"].predict_next_day(symbol, include_scenarios=True)
            data["prediction"] = pred
            data["status"] = "Ready"
            self.message_queue.put(("refresh", symbol))
            self.message_queue.put(("log", f"✓ {symbol} adaptively updated"))
            self.message_queue.put(("status", "Ready"))
        except Exception as e:
            self.message_queue.put(("log", f"✗ {symbol} update error: {e}"))

    # ------------------------------------------------------------------ #
    #  BUTTON ACTIONS                                                      #
    # ------------------------------------------------------------------ #

    def predict_selected(self):
        syms = self._selected_symbols()
        if not syms:
            messagebox.showwarning("Warning", "Select a stock first.")
            return
        for sym in syms:
            threading.Thread(target=self._predict_thread, args=(sym,), daemon=True).start()

    def predict_all(self):
        for sym in list(self.stocks):
            threading.Thread(target=self._predict_thread, args=(sym,), daemon=True).start()

    def update_selected(self):
        syms = self._selected_symbols()
        if not syms:
            messagebox.showwarning("Warning", "Select a stock first.")
            return
        for sym in syms:
            threading.Thread(target=self._update_thread, args=(sym,), daemon=True).start()

    def update_all(self):
        for sym in list(self.stocks):
            threading.Thread(target=self._update_thread, args=(sym,), daemon=True).start()

    # ------------------------------------------------------------------ #
    #  EXCEL EXPORT — STOCK DATA                                          #
    # ------------------------------------------------------------------ #

    def export_stock_data(self):
        has_data = any(d.get("raw_df") is not None for d in self.stocks.values())
        if not has_data:
            messagebox.showinfo("Info", "No stock data available yet. Train a stock first.")
            return

        filename = f"stock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                for symbol, data in self.stocks.items():
                    df = data.get("raw_df")
                    if df is None:
                        continue

                    df_export = df[['open', 'high', 'low', 'close', 'volume']].copy()
                    df_export.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    df_export.index = df_export.index.tz_localize(None)
                    df_export.index = df_export.index.date
                    df_export.index.name = 'Date'
                    df_export.to_excel(writer, sheet_name=symbol)

            self.log(f"Stock data exported → {filename}")
            messagebox.showinfo("Exported", f"Stock data saved as:\n{os.path.abspath(filename)}")
        except PermissionError:
            messagebox.showerror("Error", f"{filename} is open. Close it and try again.")

    # ------------------------------------------------------------------ #
    #  EXCEL EXPORT — PREDICTIONS                                         #
    # ------------------------------------------------------------------ #

    def export_predictions(self):
        has_preds = any(d.get("prediction") is not None for d in self.stocks.values())
        if not has_preds:
            messagebox.showinfo("Info", "No predictions available yet. Train and predict first.")
            return

        filename = f"stock_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                for symbol, data in self.stocks.items():
                    pred = data.get("prediction")
                    if not pred or "scenarios" not in pred:
                        continue

                    sc = pred["scenarios"]
                    rows = []
                    for label, key in [("Best Case", "best_case"),
                                       ("Average Case", "average_case"),
                                       ("Worst Case", "worst_case")]:
                        s = sc[key]
                        rows.append({
                            "Scenario":         label,
                            "Open":             round(s["open"], 2),
                            "High":             round(s["high"], 2),
                            "Low":              round(s["low"], 2),
                            "Close":            round(s["close"], 2),
                            "Profit %":         round(s["profit_potential"], 2),
                            "Current Price":    round(pred["current_price"], 2),
                            "Confidence":       round(pred["confidence"] * 100, 1),
                            "Signal":           pred.get("recommendation", "").replace("_", " "),
                        })

                    df_pred = pd.DataFrame(rows)
                    df_pred.to_excel(writer, sheet_name=symbol, index=False)

            self.log(f"Predictions exported → {filename}")
            messagebox.showinfo("Exported", f"Predictions saved as:\n{os.path.abspath(filename)}")
        except PermissionError:
            messagebox.showerror("Error", f"{filename} is open. Close it and try again.")

    # ------------------------------------------------------------------ #
    #  CONFIG                                                              #
    # ------------------------------------------------------------------ #

    def save_config(self):
        with open("stock_predictor_config.json", "w") as f:
            json.dump({}, f)

    def load_config(self):
        pass  # No API key needed for Yahoo Finance

    # ------------------------------------------------------------------ #
    #  MESSAGE QUEUE                                                       #
    # ------------------------------------------------------------------ #

    def process_queue(self):
        try:
            while True:
                msg_type, payload = self.message_queue.get_nowait()
                if msg_type == "log":
                    self.log(payload)
                elif msg_type == "status":
                    self.status_var.set(payload)
                elif msg_type == "refresh":
                    self._update_tree_row(payload)
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)


def main():
    root = tk.Tk()
    StockPriceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()