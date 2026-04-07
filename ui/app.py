"""
src/ui/app.py
~~~~~~~~~~~~~
Main application window: composes StockManagerTab + ChartsTab, wires the
message-queue bridge, and hosts the background auto-update threads.

Single Responsibility: root window setup, tab composition, and the
message-queue dispatch loop. All domain work is delegated to StockRegistry.

Dependency Inversion: the app knows about the abstract StockRegistry
interface, not about fetchers, repositories, or ML internals.
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

from services.stock_registry import StockRegistry
from ui.chart_tab import ChartsTab
from ui.stock_tab import StockManagerTab


class StockPriceApp:
    """
    Root application window.

    Composes the two main tabs, owns the message queue, and starts the
    background auto-update threads.
    """

    def __init__(self, root: tk.Tk, registry: StockRegistry) -> None:
        self.root     = root
        self.registry = registry

        self.root.title("Stock Price Predictor — Yahoo Finance")
        self.root.geometry("1100x750")

        self._queue: queue.Queue = queue.Queue()

        self._build_tabs()
        self._process_queue()
        self._tick_market_status()
        self.registry.load_symbols()
        self._start_auto_threads()

    # ------------------------------------------------------------------ #
    #  Tab construction                                                   #
    # ------------------------------------------------------------------ #

    def _build_tabs(self) -> None:
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        # Stock Manager tab
        self._stock_tab = StockManagerTab(
            nb,
            on_add          = self._add_stock,
            on_remove       = self._remove_stocks,
            on_predict_all  = self._predict_all,
            on_update_all   = self._update_all,
            on_update_data  = self._update_data,
            on_update_preds = self._update_preds,
            on_view_history = self._view_history,
        )
        nb.add(self._stock_tab, text="  Stock Manager  ")

        # Charts tab
        self._chart_tab = ChartsTab(nb)
        self._chart_tab.set_store(self.registry)
        nb.add(self._chart_tab, text="  Charts  ")

    # ------------------------------------------------------------------ #
    #  Stock actions                                                      #
    # ------------------------------------------------------------------ #

    def _add_stock(self, symbol: str, lookback: int, epochs: int) -> None:
        if self.registry.has(symbol):
            messagebox.showwarning("Warning", f"{symbol} is already tracked.")
            return
        self.registry.add(symbol, lookback, epochs)
        self._stock_tab.update_row(symbol, self.registry.get(symbol) or {})
        self._chart_tab.ensure_tab(symbol)
        self._stock_tab.log(f"Queued training for {symbol}")

    def _remove_stocks(self, symbols: list) -> None:
        for sym in symbols:
            self._stock_tab.remove_row(sym)
            self.registry.remove(sym)
            self._chart_tab.remove_tab(sym)
            self._stock_tab.log(f"Removed {sym}")

    def _predict_all(self) -> None:
        for sym in self.registry.symbols():
            self.registry.predict(sym)

    def _update_all(self) -> None:
        for sym in self.registry.symbols():
            self.registry.update(sym)

    def _update_data(self) -> None:
        if not any(d.get("raw_df") is not None for d in self.registry.stocks.values()):
            messagebox.showinfo("Info", "No stock data yet. Train a stock first.")
            return
        try:
            path = self.registry.update_stock_data()
            messagebox.showinfo("Updated", f"Stock data updated in:\n{path}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _update_preds(self) -> None:
        if not any(d.get("prediction") is not None for d in self.registry.stocks.values()):
            messagebox.showinfo("Info", "No predictions yet. Train and predict first.")
            return
        try:
            path = self.registry.update_predictions()
            messagebox.showinfo("Updated", f"Predictions updated in:\n{path}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ------------------------------------------------------------------ #
    #  Prediction history viewer                                          #
    # ------------------------------------------------------------------ #

    def _view_history(self, symbol: str) -> None:
        from scoring.scorer import _parse_records, _prev_close
        import pandas as pd

        data = self.registry.get(symbol)
        if data is None:
            return
        pred_history = data.get("pred_history", [])
        if not pred_history:
            from tkinter import messagebox
            messagebox.showinfo("No History", f"{symbol} has no archived predictions yet.")
            return

        raw_df  = data.get("raw_df")
        records = _parse_records(pred_history)

        # Build a date → close lookup from raw_df for direct same-day lookup
        close_by_date: dict = {}
        if raw_df is not None and not raw_df.empty:
            idx = pd.DatetimeIndex(raw_df.index)
            if hasattr(idx, "tz") and idx.tz is not None:
                idx = idx.tz_localize(None)
            close_by_date = dict(zip(idx.normalize(), raw_df["close"].values))

        for r in records:
            ts = pd.Timestamp(r.date).normalize()
            if ts in close_by_date:
                r.actual = float(close_by_date[ts])

        # Most recent first
        records = list(reversed(records))

        win = tk.Toplevel(self.root)
        win.title(f"{symbol} — Prediction History ({len(records)} entries)")
        win.geometry("920x500")
        win.resizable(True, True)

        frm = ttk.Frame(win, padding="8")
        frm.pack(fill="both", expand=True)
        frm.rowconfigure(0, weight=1)
        frm.columnconfigure(0, weight=1)

        cols = ("Prediction Date", "Predicted", "Best Case", "Worst Case",
                "Actual", "Error %", "In Range", "Direction")
        col_widths = (150, 90, 90, 90, 90, 80, 70, 75)

        tree = ttk.Treeview(frm, columns=cols, show="headings", height=18)
        for col, w in zip(cols, col_widths):
            tree.heading(col, text=col)
            tree.column(col, width=w, anchor="center")
        tree.tag_configure("matched",   background="#e8f5e9")
        tree.tag_configure("pending",   background="#fff9c4")
        tree.tag_configure("bad_range", background="#ffebee")

        for r in records:
            date_str = r.date.strftime("%Y-%m-%d %H:%M") if hasattr(r.date, "strftime") else str(r.date)
            if r.actual is not None:
                err_pct   = abs(r.avg - r.actual) / (abs(r.actual) + 1e-8) * 100
                in_range  = min(r.best, r.worst) <= r.actual <= max(r.best, r.worst)
                prev      = _prev_close(r.date, raw_df) if raw_df is not None else None
                direction = "N/A"
                if prev is not None:
                    direction = "✓" if (r.avg >= prev) == (r.actual >= prev) else "✗"
                tag = "matched" if in_range else "bad_range"
                tree.insert("", "end", tags=(tag,), values=(
                    date_str,
                    f"${r.avg:.2f}",
                    f"${r.best:.2f}",
                    f"${r.worst:.2f}",
                    f"${r.actual:.2f}",
                    f"{err_pct:.2f}%",
                    "✓" if in_range else "✗",
                    direction,
                ))
            else:
                tree.insert("", "end", tags=("pending",), values=(
                    date_str,
                    f"${r.avg:.2f}",
                    f"${r.best:.2f}",
                    f"${r.worst:.2f}",
                    "Pending", "—", "—", "—",
                ))

        vsb = ttk.Scrollbar(frm, orient="vertical",   command=tree.yview)
        hsb = ttk.Scrollbar(frm, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        legend = ttk.Frame(win, padding="4 2")
        legend.pack(fill="x", padx=8)
        for colour, label in [("#e8f5e9", "Actual inside range"), ("#ffebee", "Actual outside range"), ("#fff9c4", "Pending (future)")]:
            dot = tk.Label(legend, text="■", foreground=colour, background=colour, font=("Helvetica", 12))
            dot.pack(side="left")
            tk.Label(legend, text=label, font=("Helvetica", 9), foreground="#555").pack(side="left", padx=(0, 12))

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=6)

    # ------------------------------------------------------------------ #
    #  Auto-update background threads                                     #
    # ------------------------------------------------------------------ #

    def _start_auto_threads(self) -> None:
        threading.Thread(target=self._auto_update_loop,  daemon=True).start()
        threading.Thread(target=self._auto_predict_loop, daemon=True).start()

    def _auto_update_loop(self) -> None:
        """Round-robin adaptive updates at ~1000 calls/hour."""
        interval = 3600 / 1000
        idx = 0
        while True:
            time.sleep(interval)
            syms = self.registry.symbols()
            if syms:
                self.registry.update(syms[idx % len(syms)])
                idx += 1

    def _auto_predict_loop(self) -> None:
        """Refresh all predictions every 5 minutes."""
        while True:
            time.sleep(300)
            for sym in self.registry.symbols():
                self.registry.predict(sym)

    # ------------------------------------------------------------------ #
    #  Message queue bridge (background → Tk main thread)                #
    # ------------------------------------------------------------------ #

    def post(self, msg_type: str, payload) -> None:
        """Thread-safe: post a message for the Tk main thread to handle."""
        self._queue.put((msg_type, payload))

    def _process_queue(self) -> None:
        try:
            while True:
                msg_type, payload = self._queue.get_nowait()
                if msg_type == "log":
                    self._stock_tab.log(payload)
                elif msg_type == "status":
                    self._stock_tab.set_status(payload)
                elif msg_type == "refresh":
                    data = self.registry.get(payload)
                    if data:
                        self._stock_tab.update_row(payload, data)
                        self._chart_tab.ensure_tab(payload)
                elif msg_type == "chart":
                    self._chart_tab.draw(payload)
                elif msg_type == "pull_status":
                    status, sym = payload
                    self._stock_tab.set_pull_status(status, sym)
        except queue.Empty:
            pass
        self.root.after(100, self._process_queue)

    # ------------------------------------------------------------------ #
    #  Market status ticker                                               #
    # ------------------------------------------------------------------ #

    def _tick_market_status(self) -> None:
        from datetime import time as dtime
        from zoneinfo import ZoneInfo
        now_et  = datetime.now(ZoneInfo("America/New_York"))
        weekday = now_et.weekday()
        t       = now_et.time()

        if weekday >= 5:
            label, colour = "Closed (Weekend)", "#9e9e9e"
        elif dtime(4, 0) <= t < dtime(9, 30):
            label, colour = "Pre-Market",       "#f57f17"
        elif dtime(9, 30) <= t < dtime(16, 0):
            label, colour = "Market Open",      "#2e7d32"
        elif dtime(16, 0) <= t < dtime(20, 0):
            label, colour = "After-Hours",      "#1565c0"
        else:
            label, colour = "Closed",           "#9e9e9e"

        self._stock_tab.set_market_status(label, colour)
        self.root.after(60_000, self._tick_market_status)


def create_app(root: tk.Tk) -> StockPriceApp:
    """
    Factory function: wire all dependencies and return a ready-to-run app.

    This is the single place where concrete implementations are chosen,
    satisfying the Dependency Inversion Principle.
    """
    from data.fetcher import YahooFinanceFetcher
    from ml.network import NeuralNetwork
    from ml.predictor import StockPredictor
    from ml.trainer import ModelTrainer
    from services.trading_service import StockTradingService
    from storage.excel_exporter import ExcelExporter
    from storage.history_repository import CsvHistoryRepository
    from storage.model_repository import JsonModelRepository
    from storage.symbol_repository import JsonSymbolRepository

    fetcher   = YahooFinanceFetcher()
    trainer   = ModelTrainer()
    predictor = StockPredictor(trainer)
    service   = StockTradingService(fetcher, trainer, predictor)

    model_repo  = JsonModelRepository()
    hist_repo   = CsvHistoryRepository()
    sym_repo    = JsonSymbolRepository()
    exporter    = ExcelExporter()

    # The message callback posts into the app's queue
    app_holder: list = []

    def message_cb(msg_type: str, payload) -> None:
        if app_holder:
            app_holder[0].post(msg_type, payload)

    registry = StockRegistry(
        message_cb      = message_cb,
        trading_service = service,
        model_repo      = model_repo,
        history_repo    = hist_repo,
        symbol_repo     = sym_repo,
        excel_exporter  = exporter,
    )

    app = StockPriceApp(root, registry)
    app_holder.append(app)
    return app


def main() -> None:
    root = tk.Tk()
    create_app(root)
    root.mainloop()


if __name__ == "__main__":
    main()
