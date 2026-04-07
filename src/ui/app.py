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

from src.core.models import ScoreResult
from src.services.stock_registry import StockRegistry
from src.ui.chart_tab import ChartsTab
from src.ui.stock_tab import StockManagerTab


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
            on_export_data  = self._export_data,
            on_update_data  = self._update_data,
            on_export_preds = self._export_preds,
            on_update_preds = self._update_preds,
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

    def _export_data(self) -> None:
        if not any(d.get("raw_df") is not None for d in self.registry.stocks.values()):
            messagebox.showinfo("Info", "No stock data yet. Train a stock first.")
            return
        try:
            path = self.registry.export_stock_data()
            messagebox.showinfo("Exported", f"Saved as:\n{path}")
        except PermissionError:
            messagebox.showerror("Error", "File is open. Close it and retry.")

    def _update_data(self) -> None:
        if not any(d.get("raw_df") is not None for d in self.registry.stocks.values()):
            messagebox.showinfo("Info", "No stock data yet. Train a stock first.")
            return
        try:
            path = self.registry.update_stock_data()
            messagebox.showinfo("Updated", f"Stock data updated in:\n{path}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _export_preds(self) -> None:
        if not any(d.get("prediction") is not None for d in self.registry.stocks.values()):
            messagebox.showinfo("Info", "No predictions yet. Train and predict first.")
            return
        try:
            path = self.registry.export_predictions()
            messagebox.showinfo("Exported", f"Saved as:\n{path}")
        except PermissionError:
            messagebox.showerror("Error", "File is open. Close it and retry.")

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
    #  Score viewer                                                       #
    # ------------------------------------------------------------------ #

    def view_score(self, symbol: str) -> None:
        data   = self.registry.get(symbol)
        if data is None:
            messagebox.showinfo("Info", f"{symbol} is not tracked.")
            return
        result: ScoreResult | None = data.get("accuracy_score")
        if result is None:
            messagebox.showinfo("No Score", f"{symbol} has not been scored yet.")
            return
        self._open_score_window(symbol, result)

    def _open_score_window(self, symbol: str, result: ScoreResult) -> None:
        win = tk.Toplevel(self.root)
        win.title(f"{symbol} — Accuracy Score")
        win.geometry("780x540")
        win.resizable(True, True)

        top = ttk.Frame(win, padding="12")
        top.pack(fill="x")
        colour = "#2e7d32" if result.score >= 80 else "#f57f17" if result.score >= 50 else "#c62828"
        ttk.Label(top, text=f"{result.score:.1f} / 100", font=("Helvetica", 36, "bold"),
                  foreground=colour).pack(side="left", padx=12)
        ttk.Label(top, text=f"Grade: {result.letter_grade}", font=("Helvetica", 28),
                  foreground=colour).pack(side="left", padx=4)

        mid = ttk.LabelFrame(win, text="Score Components", padding="8")
        mid.pack(fill="x", padx=12, pady=4)
        for label, comp_score, detail in [
            ("Price Closeness (50%)",    result.mape_score,        f"avg error {result.mean_abs_error_pct:.2f}%"),
            ("Direction Accuracy (30%)", result.directional_score, f"{result.directional_accuracy * 100:.1f}% correct"),
            ("Band Accuracy (20%)",      result.range_score,       f"{result.within_range_pct * 100:.1f}% inside range"),
        ]:
            row = ttk.Frame(mid); row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=28, anchor="w").pack(side="left")
            ttk.Label(row, text=f"{comp_score:.1f}/100", width=10, anchor="e").pack(side="left")
            ttk.Label(row, text=detail, foreground="#555").pack(side="left", padx=8)

        sf = ttk.LabelFrame(win, text="Summary", padding="6")
        sf.pack(fill="x", padx=12, pady=4)
        ttk.Label(sf, text=result.summary, justify="left", font=("Courier", 9)).pack(anchor="w")

        if result.details:
            df_ = ttk.LabelFrame(win, text="Per-Prediction Breakdown", padding="6")
            df_.pack(fill="both", expand=True, padx=12, pady=4)
            df_.rowconfigure(0, weight=1); df_.columnconfigure(0, weight=1)
            d_cols = list(result.details[0].keys())
            dtree  = ttk.Treeview(df_, columns=d_cols, show="headings", height=8)
            for col, cw in zip(d_cols, [130, 110, 80, 80, 100, 80, 65, 65]):
                dtree.heading(col, text=col); dtree.column(col, width=cw, anchor="center")
            for row in result.details:
                tag = "good" if row["In Range"] == "✓" else "bad"
                dtree.insert("", "end", values=[row[c] for c in d_cols], tags=(tag,))
            dtree.tag_configure("good", background="#e8f5e9")
            dtree.tag_configure("bad",  background="#ffebee")
            vsb = ttk.Scrollbar(df_, orient="vertical", command=dtree.yview)
            dtree.configure(yscrollcommand=vsb.set)
            dtree.grid(row=0, column=0, sticky="nsew"); vsb.grid(row=0, column=1, sticky="ns")
        else:
            ttk.Label(win, text="No matched predictions to display.", foreground="gray").pack(pady=8)

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
    from src.data.fetcher import YahooFinanceFetcher
    from src.ml.network import NeuralNetwork
    from src.ml.predictor import StockPredictor
    from src.ml.trainer import ModelTrainer
    from src.services.trading_service import StockTradingService
    from src.storage.excel_exporter import ExcelExporter
    from src.storage.history_repository import CsvHistoryRepository
    from src.storage.model_repository import JsonModelRepository
    from src.storage.symbol_repository import JsonSymbolRepository

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
