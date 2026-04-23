"""Misdiagnosed Documents tab — over/under-predicted docs (continuous only)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHeaderView,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class MisdiagnosedTab:
    def __init__(self, get_document_text):
        self._get_document_text = get_document_text
        self._widget: QWidget | None = None
        self._over_table: QTableWidget | None = None
        self._under_table: QTableWidget | None = None

    def create(self, parent) -> QWidget:
        tab = QWidget()
        self._widget = tab
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Vertical)

        over_group = QGroupBox("Over-predicted")
        over_layout = QVBoxLayout()
        self._over_table = self._make_table()
        over_layout.addWidget(self._over_table)
        over_group.setLayout(over_layout)
        splitter.addWidget(over_group)

        under_group = QGroupBox("Under-predicted")
        under_layout = QVBoxLayout()
        self._under_table = self._make_table()
        under_layout.addWidget(self._under_table)
        under_group.setLayout(under_layout)
        splitter.addWidget(under_group)

        outer.addWidget(splitter, stretch=1)
        return tab

    def load(self, view) -> None:
        from ....utils.report_settings import (
            get_report_setting, KEY_MISDIAGNOSED, KEY_SNIPPET_PREVIEW,
        )
        k = get_report_setting(KEY_MISDIAGNOSED)
        if k == 0:
            self._over_table.setRowCount(0)
            self._under_table.setRowCount(0)
            return

        try:
            over_docs = list(view.working.docs.misdiagnosed(k=k, direction="over"))
            under_docs = list(view.working.docs.misdiagnosed(k=k, direction="under"))
        except Exception:
            self._over_table.setRowCount(0)
            self._under_table.setRowCount(0)
            return

        cap = get_report_setting(KEY_SNIPPET_PREVIEW)
        for table, side_docs in [(self._over_table, over_docs),
                                  (self._under_table, under_docs)]:
            headers = ["Rank", "Document Text", "Actual Y", "Predicted Y", "Residual"]
            table.setColumnCount(len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.setRowCount(len(side_docs))
            for row_i, d in enumerate(side_docs):
                text = self._get_document_text(d.doc_id) or ""
                if cap and len(text) > cap:
                    text = text[:cap] + "..."
                table.setItem(row_i, 0, QTableWidgetItem(str(row_i + 1)))
                table.setItem(row_i, 1, QTableWidgetItem(text))
                table.setItem(row_i, 2, QTableWidgetItem(f"{d.y_true:.4f}"))
                table.setItem(row_i, 3, QTableWidgetItem(f"{d.y_hat:.4f}"))
                table.setItem(row_i, 4, QTableWidgetItem(f"{d.residual:.4f}"))

            table.resizeColumnsToContents()
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

    def is_visible_for(self, view) -> bool:
        return view.analysis_type != "groups"

    @staticmethod
    def _make_table() -> QTableWidget:
        t = QTableWidget()
        t.setAlternatingRowColors(True)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        t.setSelectionBehavior(QTableWidget.SelectRows)
        return t
