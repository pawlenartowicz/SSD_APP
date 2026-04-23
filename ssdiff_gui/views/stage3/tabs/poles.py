"""Semantic Poles tab — top words at each end of the SSD direction."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import pair_selector


class PolesTab:
    def __init__(self):
        self._widget: QWidget | None = None
        self._pos_group: QGroupBox | None = None
        self._neg_group: QGroupBox | None = None
        self._pos_desc: QLabel | None = None
        self._neg_desc: QLabel | None = None
        self._pos_table: QTableWidget | None = None
        self._neg_table: QTableWidget | None = None

    def create(self, parent, on_pair_changed, pair_combos, pair_frames) -> QWidget:
        tab = QWidget()
        self._widget = tab
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)

        ctrl = QHBoxLayout()
        ctrl.addWidget(
            pair_selector.make_pair_selector(on_pair_changed, pair_combos, pair_frames)
        )
        ctrl.addStretch()
        outer.addLayout(ctrl)

        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)

        self._pos_group = QGroupBox("Positive End (+\u03b2) \u2014 Higher Outcome")
        pos_layout = QVBoxLayout()
        self._pos_desc = QLabel("Words most aligned with higher outcome values:")
        self._pos_desc.setObjectName("label_muted")
        pos_layout.addWidget(self._pos_desc)
        self._pos_table = self._make_table()
        pos_layout.addWidget(self._pos_table)
        self._pos_group.setLayout(pos_layout)
        layout.addWidget(self._pos_group)

        self._neg_group = QGroupBox("Negative End (\u2212\u03b2) \u2014 Lower Outcome")
        neg_layout = QVBoxLayout()
        self._neg_desc = QLabel("Words most aligned with lower outcome values:")
        self._neg_desc.setObjectName("label_muted")
        neg_layout.addWidget(self._neg_desc)
        self._neg_table = self._make_table()
        neg_layout.addWidget(self._neg_table)
        self._neg_group.setLayout(neg_layout)
        layout.addWidget(self._neg_group)

        outer.addWidget(body, stretch=1)
        return tab

    def load(self, view) -> None:
        pos_rows, neg_rows = self._collect_rows(view)
        self._fill(self._pos_table, pos_rows)
        self._fill(self._neg_table, neg_rows)

    def is_visible_for(self, view) -> bool:
        return True

    @staticmethod
    def _make_table() -> QTableWidget:
        t = QTableWidget()
        t.setColumnCount(3)
        t.setHorizontalHeaderLabels(["Rank", "Word", "cos(\u03b2)"])
        t.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        t.setAlternatingRowColors(True)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        return t

    @staticmethod
    def _collect_rows(view) -> tuple[list[dict], list[dict]]:
        pos, neg = [], []
        try:
            for w in view.working.words:
                row = {"rank": w.rank, "word": w.word, "cos_beta": w.cos_beta}
                if w.side == "pos":
                    pos.append(row)
                elif w.side == "neg":
                    neg.append(row)
        except Exception:
            return [], []
        return pos, neg

    @staticmethod
    def _fill(table: QTableWidget, rows: list[dict]) -> None:
        table.setRowCount(len(rows))
        for i, row_data in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(str(row_data.get("rank", i + 1))))
            table.setItem(i, 1, QTableWidgetItem(str(row_data.get("word", ""))))
            cos = row_data.get("cos_beta")
            table.setItem(
                i, 2,
                QTableWidgetItem("" if cos is None else f"{float(cos):.4f}"),
            )
