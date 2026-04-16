# Stock Manager tab widget — renders the UI and delegates all data operations via callbacks.

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from datetime import datetime
from typing import Callable, List


# The Stock Manager tab: add/remove stocks, trigger predict/update, show table and log.
class StockManagerTab(ttk.Frame):

    _SIGNAL_MAP     = {"STRONG SELL": 0, "SELL": 1, "HOLD": 2, "BUY": 3, "STRONG BUY": 4}
    _SIGNAL_COLOURS = ["#d32f2f", "#e57373", "#bdbdbd", "#81c784", "#2e7d32"]
    _SENTIMENT_MAP  = {"Very Bearish": 0, "Bearish": 1, "Neutral": 2, "Bullish": 3, "Very Bullish": 4}

    def __init__(
        self,
        parent,
        on_add:           Callable,
        on_remove:        Callable,
        on_predict_all:   Callable,
        on_update_all:    Callable,
        on_update_data:   Callable,
        on_update_preds:  Callable,
        on_update_scores: Callable,
        on_view_score:    Callable,
    ) -> None:
        super().__init__(parent)
        self._on_add           = on_add
        self._on_remove        = on_remove
        self._on_predict_all   = on_predict_all
        self._on_update_all    = on_update_all
        self._on_update_data   = on_update_data
        self._on_update_preds  = on_update_preds
        self._on_update_scores = on_update_scores
        self._on_view_score    = on_view_score

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self._build_controls()
        self._build_status_bar()
        self._build_table()
        self._build_action_buttons()
        self._build_log()
        self._build_status_label()

    # ------------------------------------------------------------------ #
    #  Public API (called by the main app)                                #
    # ------------------------------------------------------------------ #

    # Append a timestamped message to the activity log and scroll to the bottom.
    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_text.see(tk.END)

    # Update the bottom status bar text.
    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    # Update the pull indicator dot colour and last-updated label based on ok/error status.
    def set_pull_status(self, status: str, symbol: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        if status == "ok":
            self.pull_dot.config(foreground="#2e7d32")
            self.last_pull_var.set(f"Last updated: {symbol} @ {ts}")
        else:
            self.pull_dot.config(foreground="#c62828")
            self.last_pull_var.set(f"Last updated: {symbol} @ {ts} — failed")

    # Update the market status label text and indicator dot colour.
    def set_market_status(self, label: str, colour: str) -> None:
        self.market_status_var.set(f"Market: {label}")
        self.market_dot.config(foreground=colour)

    # Upsert a row in the tracked-stocks table with the latest prediction and status data.
    def update_row(self, symbol: str, data: dict) -> None:
        pred   = data.get("prediction")
        status = data.get("status", "—")

        if pred:
            current   = f"${pred['current_price']:.2f}"
            sentiment = pred.get("sentiment_analysis", {}).get("sentiment", "—")
            conf      = f"{pred['confidence'] * 100:.0f}%"
            signal    = pred.get("recommendation", "—").replace("_", " ")
            tag       = "ready"
        else:
            current = sentiment = conf = signal = "—"
            tag = "error" if "Error" in status else "training"

        vals = (symbol, current, sentiment, conf, signal, status)
        if self._tree_has(symbol):
            self.tree.item(symbol, values=vals, tags=(tag,))
        else:
            self.tree.insert("", "end", iid=symbol, values=vals, tags=(tag,))

        if pred and sentiment != "—":
            icon = "▲ " if sentiment in ("Very Bullish", "Bullish") else "▼ " if sentiment in ("Very Bearish", "Bearish") else "— "
            self.tree.item(symbol, values=(symbol, current, icon + sentiment, conf, signal, status))

    # Delete symbol's row from the table if it exists.
    def remove_row(self, symbol: str) -> None:
        if self._tree_has(symbol):
            self.tree.delete(symbol)

    # Return the list of symbol IIDs currently selected in the table.
    def selected_symbols(self) -> List[str]:
        return list(self.tree.selection())

    # ------------------------------------------------------------------ #
    #  Widget construction                                                #
    # ------------------------------------------------------------------ #

    # Build the "Add Stock" control row with symbol entry, lookback/epoch spinboxes, and Add button.
    def _build_controls(self) -> None:
        ctrl = ttk.LabelFrame(self, text="Add Stock", padding="6")
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

        ttk.Button(ctrl, text="Add & Train", command=self._add_stock).grid(row=0, column=6, padx=5)

    # Build the status bar showing last-pull dot/label and market open/closed indicator.
    def _build_status_bar(self) -> None:
        bar = ttk.Frame(self)
        bar.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 2))

        self.pull_dot = tk.Label(bar, text="●", foreground="#9e9e9e", font=("Helvetica", 11))
        self.pull_dot.pack(side="left")
        self.last_pull_var = tk.StringVar(value="Last updated: —")
        tk.Label(bar, textvariable=self.last_pull_var, font=("Helvetica", 9), foreground="#555").pack(side="left", padx=4)

        self.market_dot = tk.Label(bar, text="●", foreground="#9e9e9e", font=("Helvetica", 11))
        self.market_dot.pack(side="right", padx=(0, 4))
        self.market_status_var = tk.StringVar(value="Market: —")
        tk.Label(bar, textvariable=self.market_status_var, font=("Helvetica", 9, "bold"), foreground="#333").pack(side="right", padx=(0, 2))

    # Build the tracked-stocks Treeview table with columns and colour-coded row tags.
    def _build_table(self) -> None:
        tbl = ttk.LabelFrame(self, text="Tracked Stocks", padding="6")
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

    # Build the row of action buttons (Predict All, Update All, Remove, Export, View Score).
    def _build_action_buttons(self) -> None:
        btn = ttk.Frame(self)
        btn.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        for text, cmd in [
            ("Predict All",        self._on_predict_all),
            ("Update All",         self._on_update_all),
            ("Remove Selected",    self._remove_selected),
            ("Update Stock Data",  self._on_update_data),
            ("Update Predictions", self._on_update_preds),
            ("Update Scores",      self._on_update_scores),
            ("View Score",         self._view_score),
        ]:
            ttk.Button(btn, text=text, command=cmd).pack(side="left", padx=2)

    # Build the scrollable activity log text widget.
    def _build_log(self) -> None:
        log_f = ttk.LabelFrame(self, text="Log", padding="4")
        log_f.grid(row=4, column=0, sticky="ew", padx=8, pady=4)
        log_f.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_f, height=6, wrap=tk.WORD, font=("Courier", 9))
        self.log_text.grid(row=0, column=0, sticky="ew")

    # Build the sunken status label at the bottom of the tab.
    def _build_status_label(self) -> None:
        self.status_var = tk.StringVar(value="Ready — add stocks to begin")
        ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w"
                  ).grid(row=5, column=0, sticky="ew", padx=8)

    # ------------------------------------------------------------------ #
    #  Button handlers                                                    #
    # ------------------------------------------------------------------ #

    # Validate the symbol entry and fire the on_add callback; clear the entry on success.
    def _add_stock(self) -> None:
        symbol = self.symbol_var.get().strip().upper()
        if not symbol:
            messagebox.showwarning("Warning", "Enter a stock symbol.")
            return
        self._on_add(symbol, self.lookback_var.get(), self.epochs_var.get())
        self.symbol_var.set("")

    # Confirm with the user then fire on_remove for all selected symbols.
    def _remove_selected(self) -> None:
        syms = self.selected_symbols()
        if not syms:
            messagebox.showwarning("Warning", "Select a stock first.")
            return
        if messagebox.askyesno("Confirm", f"Remove {', '.join(syms)}?"):
            self._on_remove(syms)

    # Validate a single selection and fire the on_view_score callback for it.
    def _view_score(self) -> None:
        syms = self.selected_symbols()
        if not syms:
            messagebox.showwarning("Warning", "Select a stock first.")
            return
        self._on_view_score(syms[0])

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    # Return True if iid exists in the Treeview (suppresses the TclError from missing items).
    def _tree_has(self, iid: str) -> bool:
        try:
            self.tree.item(iid)
            return True
        except tk.TclError:
            return False
