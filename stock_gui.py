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
import threading
import time
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
import matplotlib.dates as mdates

from stock_store import StockStore, STOCK_DATA_FILE, PREDICTIONS_FILE
from stock_scorer import ScoreResult


class StockPriceGUI:
    """
    Stock price predictor GUI with:
    - Tab per stock showing actual vs predicted close price chart
    - Zoom / pan toolbar on every chart
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
        self._start_auto_threads()

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

        pull_bar = ttk.Frame(mgr_frame)
        pull_bar.grid(row=1, column=0, sticky="w", padx=8, pady=(0, 2))
        self.pull_dot = tk.Label(pull_bar, text="●", foreground="#9e9e9e", font=("Helvetica", 11))
        self.pull_dot.pack(side="left")
        self.last_pull_var = tk.StringVar(value="Last updated: —")
        tk.Label(pull_bar, textvariable=self.last_pull_var,
                 font=("Helvetica", 9), foreground="#555").pack(side="left", padx=4)

        tbl = ttk.LabelFrame(mgr_frame, text="Tracked Stocks", padding="6")
        tbl.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        tbl.columnconfigure(0, weight=1)
        tbl.rowconfigure(0, weight=1)

        cols = ("Symbol", "Current Price", "Sentiment", "Confidence", "Signal", "Status")
        self.tree = ttk.Treeview(tbl, columns=cols, show="headings", height=10)
        for col, w in zip(cols, [90, 110, 120, 100, 130, 140]):
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
        btn.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        for text, cmd in [
            ("Predict All",         self.predict_all),
            ("Update All",          self.update_all),
            ("Remove Selected",     self.remove_selected),
            ("Update Stock Data",   self.update_stock_data),
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
        ttk.Label(
            chart_ctrl,
            text="Tip: use the toolbar below each chart to zoom 🔍, pan ✋, or save 💾",
            foreground="#666",
            font=("TkDefaultFont", 8),
        ).pack(side="left", padx=12)

        self.chart_nb = ttk.Notebook(chart_outer)
        self.chart_nb.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)

        self._chart_tabs:    dict = {}
        self._chart_canvas:  dict = {}
        self._chart_toolbar: dict = {}

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
        if pred:
            current   = f"${pred['current_price']:.2f}"
            sentiment = pred.get("sentiment_analysis", {}).get("sentiment", "—")
            conf      = f"{pred['confidence'] * 100:.0f}%"
            signal    = pred.get("recommendation", "—").replace("_", " ")
            tag = "ready"
        else:
            current = sentiment = conf = signal = "—"
            tag = "error" if "Error" in status else "training"

        vals = (symbol, current, sentiment, conf, signal, status)
        if self._tree_has(symbol):
            self.tree.item(symbol, values=vals, tags=(tag,))
        else:
            self.tree.insert("", "end", iid=symbol, values=vals, tags=(tag,))

        if pred:
            if sentiment in ("Very Bullish", "Bullish"):
                sen_icon = "▲ "
            elif sentiment in ("Very Bearish", "Bearish"):
                sen_icon = "▼ "
            else:
                sen_icon = "— "
            self.tree.item(symbol, values=(
                symbol, current, sen_icon + sentiment, conf, signal, status
            ))

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
        self._chart_toolbar.pop(symbol, None)

    def _draw_chart(self, symbol: str) -> None:
        """
        Draw actual close history and prediction scenarios on one shared axis,
        with a matplotlib NavigationToolbar for zoom / pan / save.

        Layout
        ------
        ── Actual close line (solid blue) runs through all historical data.
        ── A vertical dotted line marks "today", dividing history from forecast.
        ── Prediction avg line (orange) bridges from the last actual close into
           the future, flanked by best/worst dashed lines and a shaded band.
        ── A fixed info box pinned to the top-right corner updates on hover.
        ── NavigationToolbar2Tk sits below the canvas for zoom/pan/save.
        """
        import numpy as np

        data = self.store.get(symbol)
        if not data:
            return

        self._ensure_chart_tab(symbol)
        frame = self._chart_tabs[symbol]

        # Destroy old canvas + toolbar if they exist
        if symbol in self._chart_canvas:
            try:
                self._chart_canvas[symbol].get_tk_widget().destroy()
            except Exception:
                pass
        if symbol in self._chart_toolbar:
            try:
                self._chart_toolbar[symbol].destroy()
            except Exception:
                pass
        for w in frame.winfo_children():
            w.destroy()

        df           = data.get("raw_df")
        pred         = data.get("prediction")
        pred_history = data.get("pred_history", [])

        # ── Single figure / axis ──────────────────────────────────── #
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("#fafafa")
        ax.set_facecolor("#fafafa")
        ax.grid(color="#e0e0e0", linewidth=0.6, linestyle="--", zorder=0)

        # ── 1. Historical actual close ────────────────────────────── #
        actual_dates:  list = []
        actual_closes: list = []

        if df is not None and len(df) > 0:
            df_plot = df.copy()
            if hasattr(df_plot.index, "tz") and df_plot.index.tz is not None:
                df_plot.index = df_plot.index.tz_localize(None)
            actual_dates  = list(df_plot.index)
            actual_closes = list(df_plot["close"])
            ax.plot(actual_dates, actual_closes,
                    color="#1565c0", linewidth=1.8,
                    label="Actual Close", zorder=4)

        # ── 2. Build prediction points ────────────────────────────── #
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

        pred_dates: list = []
        pred_best:  list = []
        pred_avg:   list = []
        pred_worst: list = []

        # Split predictions into historical (overlaps actual data) and future
        hist_pred_dates: list = []
        hist_pred_avg:   list = []
        future_pred_dates: list = []
        future_pred_best:  list = []
        future_pred_avg:   list = []
        future_pred_worst: list = []

        today = datetime.now()

        if all_pts:
            pred_dates = [p["date"]                          for p in all_pts]
            pred_best  = [p.get("best",  p.get("close", 0)) for p in all_pts]
            pred_avg   = [p.get("avg",   p.get("close", 0)) for p in all_pts]
            pred_worst = [p.get("worst", p.get("close", 0)) for p in all_pts]

            for d, b, a, w in zip(pred_dates, pred_best, pred_avg, pred_worst):
                pt_dt = d if isinstance(d, datetime) else datetime.combine(d, datetime.min.time())
                if pt_dt.date() < today.date():
                    hist_pred_dates.append(d)
                    hist_pred_avg.append(a)
                else:
                    future_pred_dates.append(d)
                    future_pred_best.append(b)
                    future_pred_avg.append(a)
                    future_pred_worst.append(w)

            # Plot historical predictions overlaid on actual data
            if hist_pred_dates:
                ax.plot(hist_pred_dates, hist_pred_avg, color="#e65100", linewidth=1.4,
                        linestyle="--", label="Past Predictions", zorder=5)
                ax.scatter(hist_pred_dates, hist_pred_avg, color="#e65100", s=30,
                           zorder=6, edgecolors="white", linewidths=0.8)

            # Plot future predictions continuing from last actual close
            if future_pred_dates:
                if actual_dates:
                    bridge_dates = [actual_dates[-1]] + future_pred_dates
                    bridge_avg   = [actual_closes[-1]] + future_pred_avg
                    bridge_best  = [actual_closes[-1]] + future_pred_best
                    bridge_worst = [actual_closes[-1]] + future_pred_worst
                else:
                    bridge_dates  = future_pred_dates
                    bridge_avg    = future_pred_avg
                    bridge_best   = future_pred_best
                    bridge_worst  = future_pred_worst

                ax.fill_between(bridge_dates, bridge_worst, bridge_best,
                                color="#ff6b35", alpha=0.12, label="Predicted Range", zorder=2)
                ax.plot(bridge_dates, bridge_best,  color="#2e7d32", linewidth=1.1,
                        linestyle="--", label="Best Case", zorder=3)
                ax.plot(bridge_dates, bridge_worst, color="#c62828", linewidth=1.1,
                        linestyle="--", label="Worst Case", zorder=3)
                ax.plot(bridge_dates, bridge_avg,   color="#e65100", linewidth=1.8,
                        label="Avg Prediction", zorder=4)
                ax.scatter(future_pred_dates, future_pred_avg, color="#e65100", s=40,
                           zorder=6, edgecolors="white", linewidths=0.8)

        # ── 3. "Today" divider ────────────────────────────────────── #
        ax.axvline(x=float(mdates.date2num(today)), color="#9e9e9e", linewidth=1.2,
                   linestyle=":", label="Today", zorder=3)

        # ── 4. Axis formatting ────────────────────────────────────── #
        ax.set_title(f"{symbol} — Price History & Predictions", fontsize=11, pad=10)
        ax.set_ylabel("Price (USD)", fontsize=9)
        ax.set_xlabel("Date", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d '%y"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.85)

        # ── 5. Crosshair elements ─────────────────────────────────── #
        all_x_dates = actual_dates + pred_dates
        vline = ax.axvline(
            x=float(mdates.date2num(all_x_dates[0])) if all_x_dates else float(mdates.date2num(today)),
            color="#555", linewidth=0.8, linestyle=":", visible=False, zorder=7)
        hline = ax.axhline(
            y=0,
            color="#555", linewidth=0.8, linestyle=":", visible=False, zorder=7)
        dot, = ax.plot([], [], "o", color="#1565c0", markersize=7, zorder=8)

        # ── 6. Fixed info box — top-right corner, never moves ─────── #
        info_box = ax.annotate(
            "",
            xy=(1, 0), xycoords="axes fraction",
            xytext=(-10, -10), textcoords="offset points",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="#aaa",
                      alpha=0.93, linewidth=1),
            fontsize=8, family="monospace",
            visible=False, zorder=9,
        )

        fig.tight_layout(pad=1.8)

        # ── 7. Embed canvas ──────────────────────────────────────── #
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # ── 8. Navigation toolbar (zoom / pan / home / save) ─────── #
        toolbar_frame = tk.Frame(frame, bg="#f0f0f0")
        toolbar_frame.pack(fill="x", side="bottom")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        self._chart_canvas[symbol]  = canvas
        self._chart_toolbar[symbol] = toolbar

        # ── 9. Hover handler ─────────────────────────────────────── #
        def on_move(event):
            if event.inaxes != ax:
                vline.set_visible(False)
                hline.set_visible(False)
                dot.set_data([], [])
                info_box.set_visible(False)
                canvas.draw_idle()
                return

            mx = event.xdata
            if mx is None:
                canvas.draw_idle()
                return

            in_future = bool(future_pred_dates and mx >= mdates.date2num(future_pred_dates[0]))

            if in_future:
                x_num = mdates.date2num(future_pred_dates)
                idx   = int(np.argmin(np.abs(x_num - mx)))
                idx   = max(0, min(idx, len(future_pred_dates) - 1))
                xv    = future_pred_dates[idx]
                yv    = future_pred_avg[idx]
                dt    = xv.strftime("%Y-%m-%d") if hasattr(xv, "strftime") else str(xv)[:10]
                text  = (f"  {dt}  [Forecast]  \n"
                         f"  Avg:   ${yv:.2f}       \n"
                         f"  Best:  ${future_pred_best[idx]:.2f}       \n"
                         f"  Worst: ${future_pred_worst[idx]:.2f}       ")
                dot.set_color("#e65100")
            elif actual_dates:
                x_num = mdates.date2num(actual_dates)
                idx   = int(np.argmin(np.abs(x_num - mx)))
                idx   = max(0, min(idx, len(actual_dates) - 1))
                xv    = actual_dates[idx]
                yv    = actual_closes[idx]
                dt    = xv.strftime("%Y-%m-%d") if hasattr(xv, "strftime") else str(xv)[:10]
                text  = (f"  {dt}  [Actual]  \n"
                         f"  Close: ${yv:.2f}       ")
                dot.set_color("#1565c0")
            else:
                canvas.draw_idle()
                return

            vline.set_xdata([xv, xv]); vline.set_visible(True)
            hline.set_ydata([yv, yv]); hline.set_visible(True)
            dot.set_data([xv], [yv])
            info_box.set_text(text)
            info_box.set_visible(True)
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
    #  ACCURACY SCORING ACTIONS                                            #
    # ------------------------------------------------------------------ #

    def score_selected(self) -> None:
        syms = self._selected_symbols()
        if not syms:
            messagebox.showwarning("Warning", "Select a stock first.")
            return
        for sym in syms:
            threading.Thread(
                target=self._score_thread, args=(sym,), daemon=True
            ).start()

    def score_all(self) -> None:
        for sym in self.store.symbols():
            threading.Thread(
                target=self._score_thread, args=(sym,), daemon=True
            ).start()

    def _score_thread(self, symbol: str) -> None:
        self.store.score_symbol_entry(symbol)

    def view_score(self) -> None:
        """Open a detail window showing the full ScoreResult for the selected stock."""
        syms = self._selected_symbols()
        if not syms:
            messagebox.showwarning("Warning", "Select a stock first.")
            return
        symbol = syms[0]
        data   = self.store.get(symbol)
        if data is None:
            messagebox.showinfo("Info", f"{symbol} is not tracked.")
            return

        result: ScoreResult | None = data.get("accuracy_score")
        if result is None:
            messagebox.showinfo(
                "No Score",
                f"{symbol} has not been scored yet.\n\n"
                "Run 'Predict' or 'Update' at least once to build prediction history,\n"
                "then click 'Score Selected'.",
            )
            return

        # ── Detail window ──────────────────────────────────────── #
        win = tk.Toplevel(self.root)
        win.title(f"{symbol} — Accuracy Score")
        win.geometry("780x540")
        win.resizable(True, True)

        top = ttk.Frame(win, padding="12")
        top.pack(fill="x")

        score_colour = (
            "#2e7d32" if result.score >= 80 else
            "#f57f17" if result.score >= 50 else
            "#c62828"
        )
        ttk.Label(
            top,
            text=f"{result.score:.1f} / 100",
            font=("Helvetica", 36, "bold"),
            foreground=score_colour,
        ).pack(side="left", padx=12)
        ttk.Label(
            top,
            text=f"Grade: {result.letter_grade}",
            font=("Helvetica", 28),
            foreground=score_colour,
        ).pack(side="left", padx=4)

        mid = ttk.LabelFrame(win, text="Score Components", padding="8")
        mid.pack(fill="x", padx=12, pady=4)
        for label, comp_score, detail in [
            ("Price Closeness (50%)",    result.mape_score,        f"avg error {result.mean_abs_error_pct:.2f}%"),
            ("Direction Accuracy (30%)", result.directional_score, f"{result.directional_accuracy * 100:.1f}% correct"),
            ("Band Accuracy (20%)",      result.range_score,       f"{result.within_range_pct * 100:.1f}% inside range"),
        ]:
            row = ttk.Frame(mid)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label,                   width=28, anchor="w").pack(side="left")
            ttk.Label(row, text=f"{comp_score:.1f}/100", width=10, anchor="e").pack(side="left")
            ttk.Label(row, text=detail, foreground="#555").pack(side="left", padx=8)

        summary_f = ttk.LabelFrame(win, text="Summary", padding="6")
        summary_f.pack(fill="x", padx=12, pady=4)
        ttk.Label(summary_f, text=result.summary, justify="left",
                  font=("Courier", 9)).pack(anchor="w")

        if result.details:
            detail_f = ttk.LabelFrame(win, text="Per-Prediction Breakdown", padding="6")
            detail_f.pack(fill="both", expand=True, padx=12, pady=4)
            detail_f.rowconfigure(0, weight=1)
            detail_f.columnconfigure(0, weight=1)

            d_cols = list(result.details[0].keys())
            dtree  = ttk.Treeview(detail_f, columns=d_cols, show="headings", height=8)
            for col, cw in zip(d_cols, [130, 110, 80, 80, 100, 80, 65, 65]):
                dtree.heading(col, text=col)
                dtree.column(col, width=cw, anchor="center")
            for row in result.details:
                tag = "good" if row["In Range"] == "✓" else "bad"
                dtree.insert("", "end", values=[row[c] for c in d_cols], tags=(tag,))
            dtree.tag_configure("good", background="#e8f5e9")
            dtree.tag_configure("bad",  background="#ffebee")
            vsb = ttk.Scrollbar(detail_f, orient="vertical", command=dtree.yview)
            dtree.configure(yscrollcommand=vsb.set)
            dtree.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
        else:
            ttk.Label(win, text="No matched predictions to display.",
                      foreground="gray").pack(pady=8)

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=6)

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
    #  AUTO-UPDATE BACKGROUND THREADS                                      #
    # ------------------------------------------------------------------ #

    def _start_auto_threads(self) -> None:
        threading.Thread(target=self._auto_update_data_loop, daemon=True).start()
        threading.Thread(target=self._auto_predict_loop, daemon=True).start()

    def _auto_update_data_loop(self) -> None:
        """Round-robin adaptive updates across tracked symbols at 1000 calls/hour."""
        interval = 3600 / 1000  # 3.6 s between each call
        idx = 0
        while True:
            time.sleep(interval)
            syms = self.store.symbols()
            if not syms:
                continue
            sym = syms[idx % len(syms)]
            idx += 1
            self.store.update(sym)

    def _auto_predict_loop(self) -> None:
        """Refresh predictions for all tracked symbols every 5 minutes."""
        while True:
            time.sleep(300)
            for sym in self.store.symbols():
                self.store.predict(sym)

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
                elif msg_type == "pull_time":
                    self.last_pull_var.set(payload)
                elif msg_type == "pull_status":
                    status, sym = payload
                    ts = datetime.now().strftime("%H:%M:%S")
                    if status == "ok":
                        self.pull_dot.config(foreground="#2e7d32")
                        self.last_pull_var.set(f"Last updated: {sym} @ {ts}")
                    else:
                        self.pull_dot.config(foreground="#c62828")
                        self.last_pull_var.set(f"Last updated: {sym} @ {ts} — failed")
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)


def main() -> None:
    root = tk.Tk()
    StockPriceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()