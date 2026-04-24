"""Extreme Documents tab — highest/lowest scoring documents.

The `if view.is_group:` row-builder branch here is sanctioned by the design.
This is one of two tabs allowed to branch on view type (the other is ScoresTab).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import pair_selector


class ExtremeDocsTab:
    def __init__(self, get_document_text):
        self._get_document_text = get_document_text
        self._widget: QWidget | None = None
        self._high_table: QTableWidget | None = None
        self._low_table: QTableWidget | None = None

    def create(self, parent, on_pair_changed, pair_combos, pair_frames) -> QWidget:
        """Build and return the Extreme Documents tab widget."""
        tab = QWidget()
        self._widget = tab
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)

        ctrl_frame = QWidget()
        ctrl = QHBoxLayout(ctrl_frame)
        ctrl.setContentsMargins(4, 4, 4, 0)
        ctrl.addWidget(
            pair_selector.make_pair_selector(on_pair_changed, pair_combos, pair_frames)
        )
        ctrl.addStretch()
        outer.addWidget(ctrl_frame)
        pair_frames.append(ctrl_frame)

        splitter = QSplitter(Qt.Vertical)

        high_group = QGroupBox("Highest Scoring")
        high_layout = QVBoxLayout()
        self._high_table = self._make_table()
        high_layout.addWidget(self._high_table)
        high_group.setLayout(high_layout)
        splitter.addWidget(high_group)

        low_group = QGroupBox("Lowest Scoring")
        low_layout = QVBoxLayout()
        self._low_table = self._make_table()
        low_layout.addWidget(self._low_table)
        low_group.setLayout(low_layout)
        splitter.addWidget(low_group)

        outer.addWidget(splitter, stretch=1)
        return tab

    @staticmethod
    def _make_table() -> QTableWidget:
        t = QTableWidget()
        t.setAlternatingRowColors(True)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        t.setSelectionBehavior(QTableWidget.SelectRows)
        return t

    def load(self, view) -> None:
        """Populate the extreme documents tab from view."""
        from ....utils import display_limits
        k = display_limits.EXTREME_DOCS
        if k == 0:
            self._high_table.setRowCount(0)
            self._low_table.setRowCount(0)
            return

        if view.is_group:
            self._load_group(view, k)
        else:
            self._load_continuous(view, k)

    def _load_group(self, view, k: int) -> None:
        from ....utils import display_limits
        try:
            scores = view.working.alignment_scores
            groups = view.source.groups
            labels_map = view.source.group_labels or {}
        except Exception:
            return

        order_desc = np.argsort(-scores)
        order_asc = np.argsort(scores)
        pos_idx = order_desc[:k]
        neg_idx = order_asc[:k]
        cap = display_limits.SNIPPET_PREVIEW

        for table, idx_array in [(self._high_table, pos_idx),
                                  (self._low_table, neg_idx)]:
            headers = ["Rank", "Document Text", "Score", "Group"]
            table.setColumnCount(len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.setRowCount(len(idx_array))
            for row_i, doc_id in enumerate(idx_array):
                text = self._get_document_text(int(doc_id)) or ""
                if cap and len(text) > cap:
                    text = text[:cap] + "..."
                g = groups[doc_id]
                label = labels_map.get(g, g)
                g_display = f"{g} ({label})" if label != g else str(g)
                table.setItem(row_i, 0, QTableWidgetItem(str(row_i + 1)))
                table.setItem(row_i, 1, QTableWidgetItem(text))
                table.setItem(row_i, 2, QTableWidgetItem(f"{scores[doc_id]:.4f}"))
                table.setItem(row_i, 3, QTableWidgetItem(g_display))
            table.resizeColumnsToContents()
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

    def _load_continuous(self, view, k: int) -> None:
        from ....utils import display_limits
        try:
            pos_docs = list(view.working.docs.pos(k=k))
            neg_docs = list(view.working.docs.neg(k=k))
        except Exception:
            return

        cap = display_limits.SNIPPET_PREVIEW
        for table, side_docs in [(self._high_table, pos_docs),
                                  (self._low_table, neg_docs)]:
            headers = ["Rank", "Document Text", "Score", "Actual Y"]
            table.setColumnCount(len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.setRowCount(len(side_docs))
            for row_i, d in enumerate(side_docs):
                text = self._get_document_text(d.doc_id) or ""
                if cap and len(text) > cap:
                    text = text[:cap] + "..."
                table.setItem(row_i, 0, QTableWidgetItem(str(row_i + 1)))
                table.setItem(row_i, 1, QTableWidgetItem(text))
                table.setItem(row_i, 2, QTableWidgetItem(f"{d.alignment_score:.4f}"))
                table.setItem(row_i, 3, QTableWidgetItem(f"{d.y_true:.4f}"))
            table.resizeColumnsToContents()
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

    def is_visible_for(self, view) -> bool:
        return True
