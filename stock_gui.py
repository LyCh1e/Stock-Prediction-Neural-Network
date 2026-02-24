"""
stock_gui.py
~~~~~~~~~~~~
Tkinter GUI for the Stock Price Predictor.

All data-management, persistence, and background-thread logic lives in
stock_store.py (StockStore).  This module is responsible purely for the
user interface: widgets, charts, colour helpers, and the Tk message-queue
bridge.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import queue
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates

from stock_store import StockStore, STOCK_DATA_FILE, PREDICTIONS_FILE


class StockPriceGUI:
    """
    Stock price predictor GUI with:
    - Tab per stock showing actual vs predicted close price chart
    - Delegates all CRUD / data operations to StockStore
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Stock Price Predictor — Yahoo Finance")
        self.root.geometry("1100x750")

        self.message_queue: queue.Queue = queue.Queue()
        self.store = StockStore(message_cb=lambda msg_type, payload: self.message_queue.put((msg_type, payload)))

        self.create_widgets()
        self.process_queue()
        self.store.load_symbols()

    # ------------------------------------------------------------------ #
    #  Convenience shortcut                                                #
    # ------------------------------------------------------------------ #

    @property
    def stocks(self):
        """Thin alias so Excel helper guard-checks still work."""
        return self.store.stocks

    # ------------------------------------------------------------------ #
    #  UI CONSTRUCTION                                                     #
    # ------------------------------------------------------------------ #

    def create_widgets(self) -> None:
        root_nb = ttk.Notebook(self.root)
        root_nb.pack(fill="both", expand=True, padx=6, pady=6)

        # -- Tab 1: Stock Manager ---------------------------------------- #
        mgr_frame = ttk.Frame(root_nb)
        root_nb.add(mgr_frame, text="  Stock Manager  ")
        mgr_frame.columnconfigure(0, weight=1)
        mgr_frame.rowconfigure(2, weight=1)

        ctrl = ttk.LabelFrame(mgr_frame, text="Add Stock", padding="6")
        ctrl.grid(row=0, column=0, sticky="ew", padx=8, pady=6)

        ttk.Label(ctrl, text="Symbol:").grid(row=0, column=0, padx=4)
        self.symbol_var = tk.StringVar()
        ttk.Entry(ctrl, textvariable=self.symbol_var, width=10).grid(row=0, column=1, padx=4)

        ttk.Label(ctrl, text="Lookback:").grid(row=0, column=2, padx=4)
        self.lookback_var = tk.IntVar(value=10)
        ttk.Spinbox(ctrl, from_=5, to=30, textvariable=self.lookback_var, width=6).grid(row=0, column=3, padx=4)

        ttk.Label(ctrl, text="Epochs:").grid(row=0, column=4, padx=4)
        self.epochs_var = tk.IntVar(value=200)
        ttk.Spinbox(ctrl, from_=50, to=500, textvariable=self.epochs_var, width=7).grid(row=0, column=5, padx=4)

        ttk.Button(ctrl, text="Add & Train",               command=self.add_stock).grid(row=0, column=6, padx=5)
        ttk.Button(ctrl, text="Quick Add (SPY/AAPL/MSFT)", command=self.quick_add).grid(row=0, column=7, padx=5)

        tbl = ttk.LabelFrame(mgr_frame, text="Tracked Stocks", padding="6")
        tbl.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        tbl.columnconfigure(0, weight=1)
        tbl.rowconfigure(0, weight=1)

        cols = ("Symbol", "Current Price", "Sentiment", "Confidence", "Signal", "Status")
        self.tree = ttk.Treeview(tbl, columns=cols, show="headings", height=10)
        for col, w in zip(cols, [90, 110, 120, 100, 130, 150]):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center")
        self.tree.tag_configure("ready",    background="#e8f5e9")
        self.tree.tag_configure("training", background="#fff9c4")
        self.tree.tag_configure("error",    background="#ffebee")
        vsb = ttk.Scrollbar(tbl, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        btn = ttk.Frame(mgr_frame)
        btn.grid(row=3, column=0, sticky="w", padx=8, pady=4)
        for text, cmd in [
            ("Predict Selected",    self.predict_selected),
            ("Predict All",         self.predict_all),
            ("Update Selected",     self.update_selected),
            ("Update All",          self.update_all),
            ("Remove Selected",     self.remove_selected),
            ("Export Stock Data",   self.export_stock_data),
            ("Update Stock Data",   self.update_stock_data),
            ("Export Predictions",  self.export_predictions),
            ("Update Predictions",  self.update_predictions),
        ]:
            ttk.Button(btn, text=text, command=cmd).pack(side="left", padx=2)

        log_f = ttk.LabelFrame(mgr_frame, text="Log", padding="4")
        log_f.grid(row=4, column=0, sticky="ew", padx=8, pady=4)
        log_f.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(
            log_f, height=6, wrap=tk.WORD, font=("Courier", 9)
        )
        self.log_text.grid(row=0, column=0, sticky="ew")

        self.status_var = tk.StringVar(value="Ready — add stocks to begin")
        ttk.Label(mgr_frame, textvariable=self.status_var, relief="sunken", anchor="w"
                  ).grid(row=5, column=0, sticky="ew", padx=8)

        # -- Tab 2: Charts ----------------------------------------------- #
        chart_outer = ttk.Frame(root_nb)
        root_nb.add(chart_outer, text="  Charts  ")
        chart_outer.columnconfigure(0, weight=1)
        chart_outer.rowconfigure(1, weight=1)

        chart_ctrl = ttk.Frame(chart_outer)
        chart_ctrl.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        ttk.Button(chart_ctrl, text="Refresh Charts", command=self.refresh_all_charts).pack(side="left", padx=4)

        self.chart_nb = ttk.Notebook(chart_outer)
        self.chart_nb.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)

        self._chart_tabs:   dict = {}
        self._chart_canvas: dict = {}

    # ------------------------------------------------------------------ #
    #  LOGGING                                                             #
    # ------------------------------------------------------------------ #

    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_text.see(tk.END)
        print(msg)

    # ------------------------------------------------------------------ #
    #  TREE HELPERS                                                        #
    # ------------------------------------------------------------------ #

    def _tree_has(self, iid: str) -> bool:
        try:
            self.tree.item(iid)
            return True
        except tk.TclError:
            return False

    def _update_tree_row(self, symbol: str) -> None:
        data   = self.store.get(symbol) or {}
        pred   = data.get("prediction")
        status = data.get("status", "—")
        sig_bg = sig_fg = None

        if pred:
            current   = f"${pred['current_price']:.2f}"
            sentiment = pred.get("sentiment_analysis", {}).get("sentiment", "—")
            conf      = f"{pred['confidence'] * 100:.0f}%"
            signal    = pred.get("recommendation", "—").replace("_", " ")
            sig_bg, sig_fg = self._signal_colours(signal)
            tag = "ready"
        else:
            current = sentiment = conf = signal = "—"
            tag = "error" if "Error" in status else "training"

        vals = (symbol, current, sentiment, conf, signal, status)
        if self._tree_has(symbol):
            self.tree.item(symbol, values=vals, tags=(tag,))
        else:
            self.tree.insert("", "end", iid=symbol, values=vals, tags=(tag,))

        if pred and sig_bg and sig_fg:
            row_tag = f"row_{symbol}"
            self.tree.tag_configure(row_tag, background=sig_bg, foreground=sig_fg)
            self.tree.item(symbol, tags=(row_tag,))
            if sentiment in ("Very Bullish", "Bullish"):
                sen_icon = "▲ "
            elif sentiment in ("Very Bearish", "Bearish"):
                sen_icon = "▼ "
            else:
                sen_icon = "— "
            self.tree.item(symbol, values=(symbol, current, sen_icon + sentiment, conf, signal, status))

    def _selected_symbols(self) -> list:
        return list(self.tree.selection())

    # ------------------------------------------------------------------ #
    #  GRADIENT COLOUR HELPERS                                             #
    # ------------------------------------------------------------------ #

    _SIGNAL_MAP     = {"STRONG SELL": 0, "SELL": 1, "HOLD": 2, "BUY": 3, "STRONG BUY": 4}
    _SIGNAL_COLOURS = ["#d32f2f", "#e57373", "#bdbdbd", "#81c784", "#2e7d32"]
    _SIGNAL_FG      = ["#ffffff", "#000000", "#000000", "#000000", "#ffffff"]

    _SENTIMENT_MAP     = {"Very Bearish": 0, "Bearish": 1, "Neutral": 2, "Bullish": 3, "Very Bullish": 4}
    _SENTIMENT_COLOURS = ["#d32f2f", "#e57373", "#bdbdbd", "#81c784", "#2e7d32"]
    _SENTIMENT_FG      = ["#ffffff", "#000000", "#000000", "#000000", "#ffffff"]

    def _signal_colours(self, signal_text: str):
        key = signal_text.strip().upper().replace("_", " ")
        idx = self._SIGNAL_MAP.get(key, -1)
        return (None, None) if idx == -1 else (self._SIGNAL_COLOURS[idx], self._SIGNAL_FG[idx])

    def _sentiment_colours(self, sentiment_text: str):
        key = sentiment_text.strip().title()
        idx = self._SENTIMENT_MAP.get(key, -1)
        return (None, None) if idx == -1 else (self._SENTIMENT_COLOURS[idx], self._SENTIMENT_FG[idx])

    # ------------------------------------------------------------------ #
    #  CHART MANAGEMENT                                                    #
    # ------------------------------------------------------------------ #

    def _ensure_chart_tab(self, symbol: str) -> None:
        if symbol in self._chart_tabs:
            return
        frame = ttk.Frame(self.chart_nb)
        self.chart_nb.add(frame, text=f"  {symbol}  ")
        self._chart_tabs[symbol] = frame

    def _remove_chart_tab(self, symbol: str) -> None:
        if symbol in self._chart_tabs:
            frame = self._chart_tabs.pop(symbol)
            self.chart_nb.forget(self.chart_nb.index(frame))
        self._chart_canvas.pop(symbol, None)

    def _draw_chart(self, symbol: str) -> None:
        """Draw actual vs predicted chart with hover crosshair tooltip."""
        data = self.store.get(symbol)
        if not data:
            return

        self._ensure_chart_tab(symbol)
        frame = self._chart_tabs[symbol]

        if symbol in self._chart_canvas:
            try:
                self._chart_canvas[symbol].get_tk_widget().destroy()
            except Exception:
                pass
        for w in frame.winfo_children():
            w.destroy()

        df           = data.get("raw_df")
        pred         = data.get("prediction")
        pred_history = data.get("pred_history", [])

        fig, (ax_main, ax_pred) = plt.subplots(
            2, 1, figsize=(11, 7),
            gridspec_kw={"height_ratios": [3, 2]},
            sharex=False,
        )
        fig.subplots_adjust(hspace=0.45)

        # TOP: actual close
        ax_main.grid(color="#e0e0e0", linewidth=0.7, linestyle="--")
        actual_dates  = []
        actual_closes = []

        if df is not None and len(df) > 0:
            df_plot = df.copy()
            if hasattr(df_plot.index, "tz") and df_plot.index.tz is not None:
                df_plot.index = df_plot.index.tz_localize(None)
            actual_dates  = list(df_plot.index)
            actual_closes = list(df_plot["close"])
            ax_main.plot(actual_dates, actual_closes,
                         color="#1565c0", linewidth=1.5, label="Actual Close", zorder=3)

        ax_main.set_title(f"{symbol} — Actual Close Price", fontsize=10, pad=8)
        ax_main.set_ylabel("Price (USD)", fontsize=9)
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax_main.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax_main.legend(fontsize=8)

        vline_main   = ax_main.axvline(x=actual_dates[0] if actual_dates else datetime.now(),
                                       color="gray", linewidth=0.8, linestyle=":", visible=False)
        hline_main   = ax_main.axhline(y=0, color="gray", linewidth=0.8, linestyle=":", visible=False)
        dot_main,    = ax_main.plot([], [], "o", color="#1565c0", markersize=6, zorder=6)
        tooltip_main = ax_main.annotate(
            "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#555", alpha=0.9),
            fontsize=8, visible=False)

        # BOTTOM: prediction scenarios
        ax_pred.grid(color="#e0e0e0", linewidth=0.7, linestyle="--")
        all_pts = list(pred_history)
        if pred and "scenarios" in pred:
            next_date = datetime.now() + timedelta(days=1)
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)
            all_pts.append({
                "date":  next_date,
                "best":  pred["scenarios"]["best_case"]["close"],
                "avg":   pred["scenarios"]["average_case"]["close"],
                "worst": pred["scenarios"]["worst_case"]["close"],
            })

        pred_dates = pred_best = pred_avg = pred_worst = []
        if all_pts:
            pred_dates = [p["date"]                          for p in all_pts]
            pred_best  = [p.get("best",  p.get("close", 0)) for p in all_pts]
            pred_avg   = [p.get("avg",   p.get("close", 0)) for p in all_pts]
            pred_worst = [p.get("worst", p.get("close", 0)) for p in all_pts]
            ax_pred.fill_between(pred_dates, pred_worst, pred_best,
                                 color="#ff6b35", alpha=0.15, label="Best–Worst Range")
            ax_pred.plot(pred_dates, pred_best,  color="#2e7d32", linewidth=1.2,
                         linestyle="--", label="Best Case")
            ax_pred.plot(pred_dates, pred_avg,   color="#ff6b35", linewidth=1.5, label="Avg Prediction")
            ax_pred.plot(pred_dates, pred_worst, color="#c62828", linewidth=1.2,
                         linestyle="--", label="Worst Case")
            ax_pred.scatter(pred_dates, pred_avg, color="#ff6b35", s=30, zorder=5)
        else:
            ax_pred.text(0.5, 0.5, "No prediction history yet",
                         ha="center", va="center", color="gray",
                         transform=ax_pred.transAxes, fontsize=10)

        ax_pred.set_title(f"{symbol} — Prediction Scenarios", fontsize=10, pad=8)
        ax_pred.set_ylabel("Price (USD)", fontsize=9)
        ax_pred.set_xlabel("Date", fontsize=9)
        ax_pred.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax_pred.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax_pred.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax_pred.legend(fontsize=8)

        vline_pred   = ax_pred.axvline(x=pred_dates[0] if pred_dates else datetime.now(),
                                       color="gray", linewidth=0.8, linestyle=":", visible=False)
        hline_pred   = ax_pred.axhline(y=0, color="gray", linewidth=0.8, linestyle=":", visible=False)
        dot_pred,    = ax_pred.plot([], [], "o", color="#ff6b35", markersize=6, zorder=6)
        tooltip_pred = ax_pred.annotate(
            "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#555", alpha=0.9),
            fontsize=8, visible=False)

        fig.tight_layout(pad=1.5)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._chart_canvas[symbol] = canvas

        import numpy as np

        def on_move(event):
            if event.inaxes == ax_main and actual_dates:
                mx = event.xdata
                if mx is None:
                    return
                idx = int(np.argmin(np.abs(mdates.date2num(actual_dates) - mx)))
                idx = max(0, min(idx, len(actual_dates) - 1))
                xv, yv = actual_dates[idx], actual_closes[idx]
                vline_main.set_xdata([xv, xv]); vline_main.set_visible(True)
                hline_main.set_ydata([yv, yv]); hline_main.set_visible(True)
                dot_main.set_data([xv], [yv])
                tooltip_main.set_text(
                    f"{xv.strftime('%Y-%m-%d') if hasattr(xv,'strftime') else str(xv)[:10]}\n${yv:.2f}")
                tooltip_main.xy = (xv, yv); tooltip_main.set_visible(True)
            else:
                vline_main.set_visible(False); hline_main.set_visible(False)
                dot_main.set_data([], []); tooltip_main.set_visible(False)

            if event.inaxes == ax_pred and pred_dates:
                mx = event.xdata
                if mx is None:
                    canvas.draw_idle(); return
                idx = int(np.argmin(np.abs(mdates.date2num(pred_dates) - mx)))
                idx = max(0, min(idx, len(pred_dates) - 1))
                xv = pred_dates[idx]
                vline_pred.set_xdata([xv, xv]); vline_pred.set_visible(True)
                hline_pred.set_ydata([pred_avg[idx], pred_avg[idx]]); hline_pred.set_visible(True)
                dot_pred.set_data([xv], [pred_avg[idx]])
                dt = xv.strftime("%Y-%m-%d") if hasattr(xv, "strftime") else str(xv)[:10]
                tooltip_pred.set_text(
                    f"{dt}\nAvg: ${pred_avg[idx]:.2f}\nBest: ${pred_best[idx]:.2f}\nWorst: ${pred_worst[idx]:.2f}")
                tooltip_pred.xy = (xv, pred_avg[idx]); tooltip_pred.set_visible(True)
            else:
                vline_pred.set_visible(False); hline_pred.set_visible(False)
                dot_pred.set_data([], []); tooltip_pred.set_visible(False)

            canvas.draw_idle()

        canvas.mpl_connect("motion_notify_event", on_move)
        plt.close(fig)

    def refresh_all_charts(self) -> None:
        for symbol in self.store.symbols():
            self._draw_chart(symbol)
        self.log("Charts refreshed")

    # ------------------------------------------------------------------ #
    #  STOCK MANAGEMENT ACTIONS                                            #
    # ------------------------------------------------------------------ #

    def add_stock(self) -> None:
        symbol = self.symbol_var.get().strip().upper()
        if not symbol:
            messagebox.showwarning("Warning", "Enter a stock symbol.")
            return
        if self.store.has(symbol):
            messagebox.showwarning("Warning", f"{symbol} is already tracked.")
            return
        self.store.add(symbol, self.lookback_var.get(), self.epochs_var.get())
        self._update_tree_row(symbol)
        self._ensure_chart_tab(symbol)
        self.log(f"Queued training for {symbol}")
        self.symbol_var.set("")

    def quick_add(self) -> None:
        for sym in ["SPY", "AAPL", "MSFT"]:
            if not self.store.has(sym):
                self.symbol_var.set(sym)
                self.add_stock()

    def remove_selected(self) -> None:
        syms = self._selected_symbols()
        if not syms:
            messagebox.showwarning("Warning", "Select a stock first.")
            return
        if messagebox.askyesno("Confirm", f"Remove {', '.join(syms)}?"):
            for sym in syms:
                if self._tree_has(sym):
                    self.tree.delete(sym)
                self.store.remove(sym)
                self._remove_chart_tab(sym)
                self.log(f"Removed {sym}")

    # ------------------------------------------------------------------ #
    #  PREDICT / UPDATE ACTIONS                                            #
    # ------------------------------------------------------------------ #

    def predict_selected(self) -> None:
        syms = self._selected_symbols()
        if not syms:
            messagebox.showwarning("Warning", "Select a stock first.")
            return
        for sym in syms:
            self.store.predict(sym)

    def predict_all(self) -> None:
        for sym in self.store.symbols():
            self.store.predict(sym)

    def update_selected(self) -> None:
        syms = self._selected_symbols()
        if not syms:
            messagebox.showwarning("Warning", "Select a stock first.")
            return
        for sym in syms:
            self.store.update(sym)

    def update_all(self) -> None:
        for sym in self.store.symbols():
            self.store.update(sym)

    # ------------------------------------------------------------------ #
    #  EXCEL ACTIONS                                                       #
    # ------------------------------------------------------------------ #

    def export_stock_data(self) -> None:
        if not any(d.get("raw_df") is not None for d in self.stocks.values()):
            messagebox.showinfo("Info", "No stock data yet. Train a stock first.")
            return
        try:
            path = self.store.export_stock_data()
            self.log(f"Stock data exported → {STOCK_DATA_FILE}")
            messagebox.showinfo("Exported", f"Saved as:\n{path}")
        except PermissionError:
            messagebox.showerror("Error", f"{STOCK_DATA_FILE} is open. Close it and retry.")

    def update_stock_data(self) -> None:
        if not any(d.get("raw_df") is not None for d in self.stocks.values()):
            messagebox.showinfo("Info", "No stock data yet. Train a stock first.")
            return
        try:
            path = self.store.update_stock_data()
            messagebox.showinfo("Updated", f"Stock data updated in:\n{path}")
        except PermissionError:
            messagebox.showerror("Error", f"{STOCK_DATA_FILE} is open. Close it and retry.")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.log(f"Update stock data error: {exc}")

    def export_predictions(self) -> None:
        if not any(d.get("prediction") is not None for d in self.stocks.values()):
            messagebox.showinfo("Info", "No predictions yet. Train and predict first.")
            return
        try:
            path = self.store.export_predictions()
            self.log(f"Predictions exported → {PREDICTIONS_FILE}")
            messagebox.showinfo("Exported", f"Saved as:\n{path}")
        except PermissionError:
            messagebox.showerror("Error", f"{PREDICTIONS_FILE} is open. Close it and retry.")

    def update_predictions(self) -> None:
        if not any(d.get("prediction") is not None for d in self.stocks.values()):
            messagebox.showinfo("Info", "No predictions yet. Train and predict first.")
            return
        try:
            path = self.store.update_predictions()
            messagebox.showinfo("Updated", f"Predictions updated in:\n{path}")
        except PermissionError:
            messagebox.showerror("Error", f"{PREDICTIONS_FILE} is open. Close it and retry.")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.log(f"Update predictions error: {exc}")

    # ------------------------------------------------------------------ #
    #  MESSAGE QUEUE  (background threads → Tk main thread)               #
    # ------------------------------------------------------------------ #

    def process_queue(self) -> None:
        try:
            while True:
                msg_type, payload = self.message_queue.get_nowait()
                if msg_type == "log":
                    self.log(payload)
                elif msg_type == "status":
                    self.status_var.set(payload)
                elif msg_type == "refresh":
                    self._ensure_chart_tab(payload)
                    self._update_tree_row(payload)
                elif msg_type == "chart":
                    self._draw_chart(payload)
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)


def main() -> None:
    root = tk.Tk()
    StockPriceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()