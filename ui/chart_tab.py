"""
Charts tab widget: one sub-tab per tracked stock with a Matplotlib chart.

Single Responsibility: chart rendering, hover tooltips, and navigation
toolbar. No data fetching, ML, or registry logic.
"""

from __future__ import annotations

import tkinter as tk
from datetime import datetime, timedelta
from tkinter import ttk
from typing import Dict, List

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ChartsTab(ttk.Frame):
    """Tab containing one sub-tab per tracked stock, each with a price chart."""

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        ctrl = ttk.Frame(self)
        ctrl.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        ttk.Button(ctrl, text="Refresh Charts", command=self.refresh_all).pack(side="left", padx=4)
        ttk.Label(
            ctrl,
            text="Tip: use the toolbar below each chart to zoom, pan, or save",
            foreground="#666",
            font=("TkDefaultFont", 8),
        ).pack(side="left", padx=12)

        self._nb      = ttk.Notebook(self)
        self._nb.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)

        self._tabs:    Dict[str, ttk.Frame]             = {}
        self._canvas:  Dict[str, FigureCanvasTkAgg]     = {}
        self._toolbar: Dict[str, NavigationToolbar2Tk]  = {}
        self._cids:    Dict[str, int]                   = {}
        self._view:    Dict[str, tuple]                 = {}   # saved (xlim, ylim)
        self._store_ref = None   # set by the app after construction

    def set_store(self, store) -> None:
        """Inject the StockRegistry so charts can pull live data."""
        self._store_ref = store

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def ensure_tab(self, symbol: str) -> None:
        if symbol not in self._tabs:
            frame = ttk.Frame(self._nb)
            self._nb.add(frame, text=f"  {symbol}  ")
            self._tabs[symbol] = frame

    def remove_tab(self, symbol: str) -> None:
        if symbol in self._tabs:
            frame = self._tabs.pop(symbol)
            self._nb.forget(self._nb.index(frame))
        if symbol in self._cids and symbol in self._canvas:
            try:
                self._canvas[symbol].mpl_disconnect(self._cids.pop(symbol))
            except Exception:
                pass
        if symbol in self._canvas:
            try:
                plt.close(self._canvas[symbol].figure)
            except Exception:
                pass
        self._canvas.pop(symbol, None)
        self._toolbar.pop(symbol, None)

    def draw(self, symbol: str) -> None:
        if self._store_ref is None:
            return
        data = self._store_ref.get(symbol)
        if not data:
            return
        self.ensure_tab(symbol)
        self._render(symbol, data)

    def refresh_all(self) -> None:
        if self._store_ref is None:
            return
        for sym in self._store_ref.symbols():
            self.draw(sym)

    # ------------------------------------------------------------------ #
    #  Chart rendering                                                    #
    # ------------------------------------------------------------------ #

    def _render(self, symbol: str, data: dict) -> None:
        frame = self._tabs[symbol]

        # Disconnect old event handler to break the on_move ↔ canvas reference
        # cycle before the cyclic GC has a chance to collect it from a bg thread.
        if symbol in self._cids and symbol in self._canvas:
            try:
                self._canvas[symbol].mpl_disconnect(self._cids.pop(symbol))
            except Exception:
                pass

        # Save current view before destroying old canvas
        if symbol in self._canvas:
            try:
                old_fig = self._canvas[symbol].figure
                old_ax  = old_fig.axes[0] if old_fig.axes else None
                if old_ax is not None:
                    self._view[symbol] = (old_ax.get_xlim(), old_ax.get_ylim())
                plt.close(old_fig)
                self._canvas[symbol].get_tk_widget().destroy()
            except Exception:
                pass
        if symbol in self._toolbar:
            try:
                self._toolbar[symbol].destroy()
            except Exception:
                pass
        for w in frame.winfo_children():
            w.destroy()

        df           = data.get("raw_df")
        pred         = data.get("prediction")
        pred_history = data.get("pred_history", [])

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("#fafafa")
        ax.set_facecolor("#fafafa")
        ax.grid(color="#e0e0e0", linewidth=0.6, linestyle="--", zorder=0)

        # ── 1. Historical actual close ─────────────────────────────── #
        actual_dates:  List = []
        actual_closes: List = []

        if df is not None and len(df) > 0:
            df_plot = df.copy()
            if hasattr(df_plot.index, "tz") and df_plot.index.tz is not None:
                df_plot.index = df_plot.index.tz_localize(None)
            actual_dates  = list(df_plot.index)
            actual_closes = list(df_plot["close"])
            ax.plot(actual_dates, actual_closes, color="#1565c0", linewidth=1.8,
                    label="Actual Close", zorder=4)

        # ── 2. Prediction points ───────────────────────────────────── #
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

        future_dates:  List = []
        future_best:   List = []
        future_avg:    List = []
        future_worst:  List = []
        today = datetime.now()

        if all_pts:
            for p in all_pts:
                d = p["date"]
                pt_dt = d if isinstance(d, datetime) else datetime.combine(d, datetime.min.time())
                if pt_dt.date() >= today.date():
                    future_dates.append(d)
                    future_best.append(p.get("best",  p.get("close", 0)))
                    future_avg.append(p.get("avg",    p.get("close", 0)))
                    future_worst.append(p.get("worst", p.get("close", 0)))

            if future_dates:
                if actual_dates:
                    bd = [actual_dates[-1]] + future_dates
                    ba = [actual_closes[-1]] + future_avg
                    bb = [actual_closes[-1]] + future_best
                    bw = [actual_closes[-1]] + future_worst
                else:
                    bd, ba, bb, bw = future_dates, future_avg, future_best, future_worst

                ax.fill_between(bd, bw, bb, color="#ff6b35", alpha=0.12, label="Predicted Range", zorder=2)
                ax.plot(bd, bb, color="#2e7d32", linewidth=1.1, linestyle="--", label="Best Case",  zorder=3)
                ax.plot(bd, bw, color="#c62828", linewidth=1.1, linestyle="--", label="Worst Case", zorder=3)
                ax.plot(bd, ba, color="#e65100", linewidth=1.8, label="Avg Prediction", zorder=4)
                ax.scatter(future_dates, future_avg, color="#e65100", s=40,
                           zorder=6, edgecolors="white", linewidths=0.8)

        # ── 3. Today divider ──────────────────────────────────────── #
        ax.axvline(x=float(mdates.date2num(today)), color="#9e9e9e",
                   linewidth=1.2, linestyle=":", label="Today", zorder=3)

        # ── 4. Axis formatting ────────────────────────────────────── #
        ax.set_title(f"{symbol} — Price History & Predictions", fontsize=11, pad=10)
        ax.set_ylabel("Price (USD)", fontsize=9)
        ax.set_xlabel("Date", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d '%y"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.85)

        # ── 4b. Y-axis limits: restore saved view or default to 6 months ─ #
        if symbol in self._view:
            ax.set_xlim(*self._view[symbol][0])
            ax.set_ylim(*self._view[symbol][1])
        else:
            cutoff = today - timedelta(days=182)
            recent_closes = [
                c for d, c in zip(actual_dates, actual_closes)
                if (d if isinstance(d, datetime) else datetime.combine(d, datetime.min.time())) >= cutoff
            ]
            y_vals = list(recent_closes) + future_avg + future_best + future_worst
            if y_vals:
                y_min, y_max = min(y_vals), max(y_vals)
                pad = (y_max - y_min) * 0.05 or y_min * 0.01
                ax.set_ylim(y_min - pad, y_max + pad)

        # ── 5. Crosshair elements ─────────────────────────────────── #
        all_x = actual_dates + future_dates
        start_x = float(mdates.date2num(all_x[0])) if all_x else float(mdates.date2num(today))
        vline = ax.axvline(x=start_x, color="#555", linewidth=0.8, linestyle=":", visible=False, zorder=7)
        hline = ax.axhline(y=0,       color="#555", linewidth=0.8, linestyle=":", visible=False, zorder=7)
        dot,  = ax.plot([], [], "o", color="#1565c0", markersize=7, zorder=8)

        info_box = ax.annotate(
            "", xy=(1, 0), xycoords="axes fraction",
            xytext=(-10, -10), textcoords="offset points",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="#aaa", alpha=0.93, linewidth=1),
            fontsize=8, family="monospace", visible=False, zorder=9,
        )

        fig.tight_layout(pad=1.8)

        # ── 6. Embed canvas ───────────────────────────────────────── #
        # Pack bottom frames first so canvas expand=True doesn't crowd them out
        tb_frame = tk.Frame(frame, bg="#f0f0f0")
        tb_frame.pack(fill="x", side="bottom")

        btn_frame = tk.Frame(frame, bg="#f0f0f0")
        btn_frame.pack(fill="x", side="bottom")

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(canvas, tb_frame)
        toolbar.update()

        def _zoom(factor: float) -> None:
            xlo, xhi = ax.get_xlim()
            ylo, yhi = ax.get_ylim()
            xmid = (xlo + xhi) / 2
            ymid = (ylo + yhi) / 2
            new_xlim = (xmid - (xmid - xlo) * factor, xmid + (xhi - xmid) * factor)
            new_ylim = (ymid - (ymid - ylo) * factor, ymid + (yhi - ymid) * factor)
            ax.set_xlim(*new_xlim)
            ax.set_ylim(*new_ylim)
            self._view[symbol] = (new_xlim, new_ylim)
            canvas.draw_idle()

        tk.Button(btn_frame, text="  +  ", command=lambda: _zoom(0.8),
                  relief="flat", bg="#e0e0e0", activebackground="#bdbdbd").pack(side="left", padx=2, pady=2)
        tk.Button(btn_frame, text="  −  ", command=lambda: _zoom(1.25),
                  relief="flat", bg="#e0e0e0", activebackground="#bdbdbd").pack(side="left", padx=2, pady=2)
        tk.Label(btn_frame, text="Zoom", bg="#f0f0f0", fg="#666",
                 font=("TkDefaultFont", 8)).pack(side="left", padx=2)

        self._canvas[symbol]  = canvas
        self._toolbar[symbol] = toolbar

        # ── 7. Hover handler ─────────────────────────────────────── #
        def on_move(event):
            if event.inaxes != ax:
                vline.set_visible(False); hline.set_visible(False)
                dot.set_data([], []); info_box.set_visible(False)
                canvas.draw_idle(); return
            mx = event.xdata
            if mx is None:
                canvas.draw_idle(); return

            in_future = bool(future_dates and mx >= mdates.date2num(future_dates[0]))
            if in_future:
                x_num = mdates.date2num(future_dates)
                idx   = int(np.argmin(np.abs(x_num - mx)))
                idx   = max(0, min(idx, len(future_dates) - 1))
                xv    = future_dates[idx]; yv = future_avg[idx]
                dt    = xv.strftime("%Y-%m-%d") if hasattr(xv, "strftime") else str(xv)[:10]
                text  = (f"  {dt}  [Forecast]  \n"
                         f"  Avg:   ${yv:.2f}       \n"
                         f"  Best:  ${future_best[idx]:.2f}       \n"
                         f"  Worst: ${future_worst[idx]:.2f}       ")
                dot.set_color("#e65100")
            elif actual_dates:
                x_num = mdates.date2num(actual_dates)
                idx   = int(np.argmin(np.abs(x_num - mx)))
                idx   = max(0, min(idx, len(actual_dates) - 1))
                xv    = actual_dates[idx]; yv = actual_closes[idx]
                dt    = xv.strftime("%Y-%m-%d") if hasattr(xv, "strftime") else str(xv)[:10]
                text  = f"  {dt}  [Actual]  \n  Close: ${yv:.2f}       "
                dot.set_color("#1565c0")
            else:
                canvas.draw_idle(); return

            vline.set_xdata([xv, xv]); vline.set_visible(True)
            hline.set_ydata([yv, yv]); hline.set_visible(True)
            dot.set_data([xv], [yv])
            info_box.set_text(text); info_box.set_visible(True)
            canvas.draw_idle()

        self._cids[symbol] = canvas.mpl_connect("motion_notify_event", on_move)
