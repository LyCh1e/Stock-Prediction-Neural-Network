# Main application window: composes StockManagerTab + ChartsTab, wires the message-queue bridge,
# and hosts the background auto-update threads. All domain work is delegated to StockRegistry.

from __future__ import annotations

import gc
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

from services.stock_registry import StockRegistry
from ui.chart_tab import ChartsTab
from ui.stock_tab import StockManagerTab


# Root application window — composes the two tabs, owns the message queue, starts background threads.
class StockPriceApp:

    def __init__(self, root: tk.Tk, registry: StockRegistry) -> None:
        self.root     = root
        self.registry = registry

        self.root.title("Stock Price Predictor — Yahoo Finance")
        self.root.geometry("1100x750")

        self._queue:        queue.Queue          = queue.Queue()
        self._running:      bool                 = True
        self._score_window: tk.Toplevel | None   = None

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        gc.disable()  # prevent background threads from triggering GC on wrong thread

        self._build_tabs()
        self._process_queue()
        self._tick_market_status()
        self._run_gc()
        self.registry.load_symbols()
        self._start_auto_threads()

    # ------------------------------------------------------------------ #
    #  Tab construction                                                   #
    # ------------------------------------------------------------------ #

    # Create the Notebook, add the Stock Manager and Charts tabs, and wire their callbacks.
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
            on_view_score   = self._view_score,
        )
        nb.add(self._stock_tab, text="  Stock Manager  ")

        # Charts tab
        self._chart_tab = ChartsTab(nb)
        self._chart_tab.set_store(self.registry)
        nb.add(self._chart_tab, text="  Charts  ")

    # ------------------------------------------------------------------ #
    #  Stock actions                                                      #
    # ------------------------------------------------------------------ #

    # Add symbol to the registry, update the table row, and ensure a chart tab exists.
    def _add_stock(self, symbol: str, lookback: int, epochs: int) -> None:
        if self.registry.has(symbol):
            messagebox.showwarning("Warning", f"{symbol} is already tracked.")
            return
        self.registry.add(symbol, lookback, epochs)
        self._stock_tab.update_row(symbol, self.registry.get(symbol) or {})
        self._chart_tab.ensure_tab(symbol)
        self._stock_tab.log(f"Queued training for {symbol}")

    # Remove each symbol from the table, registry, and chart tabs.
    def _remove_stocks(self, symbols: list) -> None:
        for sym in symbols:
            self._stock_tab.remove_row(sym)
            self.registry.remove(sym)
            self._chart_tab.remove_tab(sym)
            self._stock_tab.log(f"Removed {sym}")

    # Trigger a background prediction for every tracked symbol.
    def _predict_all(self) -> None:
        for sym in self.registry.symbols():
            self.registry.predict(sym)

    # Trigger a background adaptive update for every tracked symbol.
    def _update_all(self) -> None:
        for sym in self.registry.symbols():
            self.registry.update(sym)

    # Write fresh OHLCV data to Excel for all trained stocks, showing the path on success.
    def _update_data(self) -> None:
        if not any(d.get("raw_df") is not None for d in self.registry.stocks.values()):
            messagebox.showinfo("Info", "No stock data yet. Train a stock first.")
            return
        try:
            path = self.registry.update_stock_data()
            messagebox.showinfo("Updated", f"Stock data updated in:\n{path}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # Write latest predictions to Excel for all predicted stocks, showing the path on success.
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
    #  Prediction score viewer                                            #
    # ------------------------------------------------------------------ #

    # Open the score viewer window for symbol; brings existing window to front if already open.
    def _view_score(self, symbol: str) -> None:
        import math
        import pandas as pd
        from scoring.scorer import _parse_records, _prev_close

        data = self.registry.get(symbol)
        if data is None:
            return
        pred_history = self.registry.full_pred_history(symbol)
        if not pred_history:
            messagebox.showinfo("No History", f"{symbol} has no archived predictions yet.")
            return

        raw_df  = data.get("raw_df")
        records = _parse_records(pred_history)

        # Same-day close lookup
        close_by_date: dict = {}
        if raw_df is not None and not raw_df.empty:
            idx = pd.DatetimeIndex(raw_df.index)
            if hasattr(idx, "tz") and idx.tz is not None:
                idx = idx.tz_localize(None)
            close_by_date = dict(zip(idx.normalize(), raw_df["close"].values))

        today = pd.Timestamp(datetime.now().date())
        for r in records:
            ts = pd.Timestamp(r.date).normalize()
            if ts in close_by_date and ts < today:
                r.actual = float(close_by_date[ts])

        matched = [r for r in records if r.actual is not None]

        # ── Compute aggregate score ──────────────────────────────────── #
        if matched:
            abs_errs:     list = []
            dir_correct:  list = []
            in_range_hits: list = []
            for r in matched:
                actual = float(r.actual)  # type: ignore[arg-type]
                abs_errs.append(abs(r.avg - actual) / (abs(actual) + 1e-8) * 100)
                in_range_hits.append(min(r.best, r.worst) <= actual <= max(r.best, r.worst))
                prev = _prev_close(r.date, raw_df) if raw_df is not None else None
                if prev is not None:
                    dir_correct.append((r.avg >= prev) == (actual >= prev))

            mean_ape   = sum(abs_errs) / len(abs_errs)
            mape_score = math.exp(-0.15 * mean_ape) * 100
            dir_acc    = (sum(dir_correct) / len(dir_correct)) if dir_correct else 0.5
            dir_score  = dir_acc * 100
            range_frac  = sum(in_range_hits) / len(in_range_hits)
            range_score = range_frac * 100

            final = round(min(100.0, (0.50 * math.exp(-0.15 * mean_ape)
                                      + 0.30 * dir_acc
                                      + 0.20 * range_frac) * 100), 1)

            def _grade(s: float) -> str:
                for threshold, g in [(93,"A+"),(87,"A"),(80,"A-"),(74,"B+"),(67,"B"),
                                      (60,"B-"),(54,"C+"),(47,"C"),(40,"C-"),(34,"D+"),
                                      (27,"D"),(20,"D-")]:
                    if s >= threshold: return g
                return "F"
            grade = _grade(final)
        else:
            mean_ape = mape_score = dir_score = range_score = final = 0.0
            dir_acc = range_frac = 0.0
            grade = "N/A"

        # ── Window ───────────────────────────────────────────────────── #
        if self._score_window is not None and self._score_window.winfo_exists():
            self._score_window.lift()
            self._score_window.focus_force()
            return

        win = tk.Toplevel(self.root)
        self._score_window = win
        win.protocol("WM_DELETE_WINDOW", lambda: (win.destroy(), setattr(self, "_score_window", None)))
        win.title(f"{symbol} — Prediction Score")
        win.geometry("860x560")
        win.resizable(True, True)

        # Header: score + grade
        hdr = ttk.Frame(win, padding="10 8")
        hdr.pack(fill="x")
        colour = "#2e7d32" if final >= 80 else "#f57f17" if final >= 50 else "#c62828"
        ttk.Label(hdr, text=f"{final:.1f} / 100", font=("Helvetica", 34, "bold"),
                  foreground=colour).pack(side="left", padx=10)
        ttk.Label(hdr, text=f"Grade: {grade}", font=("Helvetica", 26),
                  foreground=colour).pack(side="left")
        ttk.Label(hdr, text=f"  ({len(matched)} of {len(records)} predictions matched)",
                  font=("Helvetica", 10), foreground="#666").pack(side="left", padx=12)

        # Score components
        comp = ttk.LabelFrame(win, text="Score Components", padding="8")
        comp.pack(fill="x", padx=10, pady=4)
        for label, comp_score, detail in [
            ("Price Closeness  (50%)", round(mape_score, 1), f"avg error {mean_ape:.2f}%"),
            ("Direction Accuracy (30%)", round(dir_score, 1), f"{dir_acc*100:.1f}% of moves correct"),
            ("In-Range Accuracy  (20%)", round(range_score, 1), f"{range_frac*100:.1f}% inside best/worst band"),
        ]:
            row = ttk.Frame(comp)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label,             width=28, anchor="w").pack(side="left")
            ttk.Label(row, text=f"{comp_score}/100", width=10, anchor="e").pack(side="left")
            ttk.Label(row, text=detail, foreground="#555").pack(side="left", padx=8)

        # Per-prediction table (newest first)
        tbl_frame = ttk.LabelFrame(win, text="Per-Prediction Breakdown (newest first)", padding="6")
        tbl_frame.pack(fill="both", expand=True, padx=10, pady=4)
        tbl_frame.rowconfigure(0, weight=1)
        tbl_frame.columnconfigure(0, weight=1)

        cols = ("Date", "Predicted", "Best Case", "Worst Case", "Actual Close", "Error %", "In Range", "Direction")
        col_widths = (140, 88, 88, 88, 100, 75, 70, 75)
        tree = ttk.Treeview(tbl_frame, columns=cols, show="headings", height=10)
        for col, w in zip(cols, col_widths):
            tree.heading(col, text=col)
            tree.column(col, width=w, anchor="center")
        tree.tag_configure("good",    background="#e8f5e9")
        tree.tag_configure("bad",     background="#ffebee")
        tree.tag_configure("pending", background="#fff9c4")

        for r in reversed(records):
            date_str = r.date.strftime("%Y-%m-%d %H:%M") if hasattr(r.date, "strftime") else str(r.date)
            if r.actual is not None:
                err_pct  = abs(r.avg - r.actual) / (abs(r.actual) + 1e-8) * 100
                in_range = min(r.best, r.worst) <= r.actual <= max(r.best, r.worst)
                prev     = _prev_close(r.date, raw_df) if raw_df is not None else None
                direction = "N/A"
                if prev is not None:
                    direction = "✓" if (r.avg >= prev) == (r.actual >= prev) else "✗"
                tree.insert("", "end", tags=("good" if in_range else "bad",), values=(
                    date_str, f"${r.avg:.2f}", f"${r.best:.2f}", f"${r.worst:.2f}",
                    f"${r.actual:.2f}", f"{err_pct:.2f}%",
                    "✓" if in_range else "✗", direction,
                ))
            else:
                is_weekend = r.date.weekday() >= 5
                actual_label = "Not Available" if is_weekend else "Pending"
                tree.insert("", "end", tags=("pending",), values=(
                    date_str, f"${r.avg:.2f}", f"${r.best:.2f}", f"${r.worst:.2f}",
                    actual_label, "—", "—", "—",
                ))

        vsb = ttk.Scrollbar(tbl_frame, orient="vertical",   command=tree.yview)
        hsb = ttk.Scrollbar(tbl_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=6)

    # ------------------------------------------------------------------ #
    #  Auto-update background threads                                     #
    # ------------------------------------------------------------------ #

    # Run a manual GC cycle every 5 seconds to avoid background-thread collection issues.
    def _run_gc(self) -> None:
        gc.collect()
        if self._running:
            self.root.after(5000, self._run_gc)

    # Stop all loops and quit the Tk main loop when the window is closed.
    def _on_close(self) -> None:
        self._running = False
        self.root.quit()

    # Launch the background auto-update and auto-predict daemon threads.
    def _start_auto_threads(self) -> None:
        threading.Thread(target=self._auto_update_loop,  daemon=True).start()
        threading.Thread(target=self._auto_predict_loop, daemon=True).start()

    # Round-robin adaptive updates across all symbols at ~1000 calls/hour.
    def _auto_update_loop(self) -> None:
        interval = 3600 / 1000
        idx = 0
        while self._running:
            time.sleep(interval)
            if not self._running:
                break
            syms = self.registry.symbols()
            if syms:
                self.registry.update(syms[idx % len(syms)])
                idx += 1

    # Refresh predictions for all symbols every 5 minutes.
    def _auto_predict_loop(self) -> None:
        while self._running:
            time.sleep(300)
            if not self._running:
                break
            for sym in self.registry.symbols():
                self.registry.predict(sym)

    # ------------------------------------------------------------------ #
    #  Message queue bridge (background → Tk main thread)                #
    # ------------------------------------------------------------------ #

    # Thread-safe: enqueue a (msg_type, payload) message for the Tk main thread to handle.
    def post(self, msg_type: str, payload) -> None:
        if self._running:
            self._queue.put((msg_type, payload))

    # Drain the message queue and dispatch each message to the appropriate UI update; reschedules itself.
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
        if self._running:
            self.root.after(100, self._process_queue)

    # ------------------------------------------------------------------ #
    #  Market status ticker                                               #
    # ------------------------------------------------------------------ #

    # Check current ET time and update the market status indicator; reschedules itself every minute.
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


# Factory: wire all concrete dependencies and return a ready-to-run StockPriceApp.
def create_app(root: tk.Tk) -> StockPriceApp:
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

    # Forward messages from background threads to the app's queue once the app is created.
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


# Create the Tk root, build the app, and enter the main loop.
def main() -> None:
    root = tk.Tk()
    create_app(root)
    root.mainloop()
    root.destroy()


if __name__ == "__main__":
    main()
