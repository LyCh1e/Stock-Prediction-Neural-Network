# Edit Model tab — lets the user view and change lookback/epoch settings per stock.

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, List


class EditModelTab(ttk.Frame):

    def __init__(self, parent, on_save: Callable[[str, int, int], None]) -> None:
        super().__init__(parent)
        self._on_save = on_save

        self.columnconfigure(0, weight=1)
        self._build_explainer()
        self._build_table()
        self._build_editor()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    # Repopulate the table with the current list of (symbol, lookback, epochs) tuples.
    def load(self, stocks: List[tuple]) -> None:
        self.tree.delete(*self.tree.get_children())
        self._all_iids.clear()
        for symbol, lookback, epochs in stocks:
            self.tree.insert("", "end", iid=symbol, values=(symbol, lookback, epochs))
            self._all_iids.add(symbol)
        self._apply_filter()

    # ------------------------------------------------------------------ #
    #  Widget construction                                                #
    # ------------------------------------------------------------------ #

    def _build_explainer(self) -> None:
        box = ttk.LabelFrame(self, text="What do these settings mean?", padding="10")
        box.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))

        row1 = ttk.Frame(box)
        row1.pack(anchor="w", pady=(0, 4))
        ttk.Label(row1, text="Lookback:", font=("Helvetica", 9, "bold")).pack(side="left")
        ttk.Label(row1, text=" how many past days the model uses to predict (recommended 5–20).",
                  font=("Helvetica", 9)).pack(side="left")

        row2 = ttk.Frame(box)
        row2.pack(anchor="w")
        ttk.Label(row2, text="Epochs:", font=("Helvetica", 9, "bold")).pack(side="left")
        ttk.Label(row2, text=" how many times the model trains over the data (recommended 100–300).",
                  font=("Helvetica", 9)).pack(side="left")

    def _build_table(self) -> None:
        tbl = ttk.LabelFrame(self, text="Tracked Stocks", padding="6")
        tbl.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        tbl.columnconfigure(0, weight=1)
        tbl.rowconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        search_row = ttk.Frame(tbl)
        search_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Label(search_row, text="Filter:").pack(side="left", padx=(0, 4))
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._apply_filter())
        ttk.Entry(search_row, textvariable=self._search_var, width=16).pack(side="left")

        self._all_iids: set = set()

        cols = ("Symbol", "Lookback", "Epochs")
        self.tree = ttk.Treeview(tbl, columns=cols, show="headings", height=8,
                                  selectmode="browse")
        for col, w in zip(cols, [120, 120, 120]):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center")

        vsb = ttk.Scrollbar(tbl, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=1, column=0, sticky="nsew")
        vsb.grid(row=1, column=1, sticky="ns")

        self.tree.bind("<<TreeviewSelect>>", self._on_select)

    def _build_editor(self) -> None:
        edit = ttk.LabelFrame(self, text="Edit Selected Stock", padding="10")
        edit.grid(row=2, column=0, sticky="ew", padx=10, pady=(4, 10))

        ttk.Label(edit, text="Symbol:").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        self._sym_var = tk.StringVar()
        ttk.Entry(edit, textvariable=self._sym_var, state="readonly", width=12).grid(
            row=0, column=1, sticky="w", padx=6)

        ttk.Label(edit, text="Lookback (days):").grid(row=1, column=0, sticky="e", padx=6, pady=4)
        self._lookback_var = tk.IntVar(value=10)
        ttk.Spinbox(edit, from_=3, to=60, textvariable=self._lookback_var, width=8).grid(
            row=1, column=1, sticky="w", padx=6)

        ttk.Label(edit, text="Epochs:").grid(row=2, column=0, sticky="e", padx=6, pady=4)
        self._epochs_var = tk.IntVar(value=200)
        ttk.Spinbox(edit, from_=10, to=1000, textvariable=self._epochs_var, width=8).grid(
            row=2, column=1, sticky="w", padx=6)

        ttk.Button(edit, text="Save Changes", command=self._save).grid(
            row=3, column=0, columnspan=2, pady=(8, 0))

    # ------------------------------------------------------------------ #
    #  Handlers                                                           #
    # ------------------------------------------------------------------ #

    def _apply_filter(self) -> None:
        query    = self._search_var.get().strip().upper()
        attached = set(self.tree.get_children(""))
        for iid in self._all_iids:
            matches = not query or query in iid
            in_tree = iid in attached
            if matches and not in_tree:
                self.tree.reattach(iid, "", "end")
            elif not matches and in_tree:
                self.tree.detach(iid)

    def _on_select(self, _event) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        vals = self.tree.item(sel[0], "values")
        self._sym_var.set(vals[0])
        self._lookback_var.set(int(vals[1]))
        self._epochs_var.set(int(vals[2]))

    def _save(self) -> None:
        symbol = self._sym_var.get()
        if not symbol:
            messagebox.showwarning("Warning", "Select a stock from the table first.")
            return
        lookback = self._lookback_var.get()
        epochs   = self._epochs_var.get()
        self._on_save(symbol, lookback, epochs)
        self.tree.item(symbol, values=(symbol, lookback, epochs))
        messagebox.showinfo("Saved", f"{symbol} updated — Lookback: {lookback}, Epochs: {epochs}.\n"
                                     "Changes apply on the next training run.")
