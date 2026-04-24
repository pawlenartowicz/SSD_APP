"""Snippets tab — top-ranked sentences along the SSD direction.

For continuous results the label is "Beta Snippets" and the side combo is
+β / −β. For group results the label becomes "Contrast Snippets" and the
combo entries mirror the two group directions; labels.apply_group_labels
drives the combo reset, this tab only owns the widget.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import pair_selector
from ..html_helpers import show_snippet_detail
from ...widgets.loading_overlay import LoadingOverlay


class SnippetsTab:
    # Tab header text depends on whether the active result is a GroupResult.
    HEADER_CONTINUOUS = "Beta Snippets"
    HEADER_GROUP = "Contrast Snippets"

    def __init__(self):
        self._widget: QWidget | None = None
        self._side_combo: QComboBox | None = None
        self._tab_desc: QLabel | None = None
        self._table: QTableWidget | None = None
        self._detail: QTextEdit | None = None
        self._overlay: LoadingOverlay | None = None

        self._view = None
        self._displayed: list[dict] = []
        self._set_tab_header = None

    def set_tab_header_callback(self, fn) -> None:
        self._set_tab_header = fn

    def create(self, parent, on_pair_changed, pair_combos, pair_frames) -> QWidget:
        tab = QWidget()
        self._widget = tab
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        controls = QHBoxLayout()
        controls.setContentsMargins(4, 4, 4, 0)
        controls.addWidget(
            pair_selector.make_pair_selector(on_pair_changed, pair_combos, pair_frames)
        )
        controls.addWidget(QLabel("Side:"))
        self._side_combo = QComboBox()
        self._side_combo.addItems([
            "Positive (+\u03b2)",
            "Negative (\u2212\u03b2)",
        ])
        self._side_combo.currentIndexChanged.connect(self._refresh)
        controls.addWidget(self._side_combo)
        controls.addStretch()

        self._tab_desc = QLabel(
            "Snippets ranked by alignment with the \u03b2 direction (not clustered)"
        )
        self._tab_desc.setObjectName("label_muted")
        controls.addWidget(self._tab_desc)

        layout.addLayout(controls)

        splitter = QSplitter(Qt.Vertical)

        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels([
            "Seed", "Cosine", "Doc ID", "Snippet",
        ])
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self._table.setAlternatingRowColors(True)
        self._table.setWordWrap(True)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.itemSelectionChanged.connect(self._on_row_selected)
        splitter.addWidget(self._table)

        detail_group = QGroupBox("Selected Snippet Detail")
        detail_layout = QVBoxLayout()
        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        detail_layout.addWidget(self._detail)
        detail_group.setLayout(detail_layout)
        splitter.addWidget(detail_group)

        splitter.setSizes([400, 200])
        layout.addWidget(splitter, stretch=1)

        self._overlay = LoadingOverlay(tab)
        return tab

    def load(self, view) -> None:
        self._view = view
        if self._set_tab_header is not None:
            self._set_tab_header(
                self.HEADER_GROUP if view.is_group else self.HEADER_CONTINUOUS
            )
        self._refresh()

    def is_visible_for(self, view) -> bool:
        return True

    # Widget accessors used by labels.apply_*_labels via Stage3Widget._label_widgets.
    @property
    def side_combo(self) -> QComboBox:
        return self._side_combo

    @property
    def tab_desc(self) -> QLabel:
        return self._tab_desc

    def _refresh(self) -> None:
        if self._view is None:
            self._table.setRowCount(0)
            return

        self._overlay.start()
        QApplication.processEvents()

        rows = self._collect_rows(self._view, self._side_combo.currentIndex())
        self._display(rows)

        self._overlay.stop()

    @staticmethod
    def _collect_rows(view, side_index: int) -> list[dict]:
        want = "pos" if side_index == 0 else "neg"
        rows = []
        try:
            snippets_view = view.working.snippets(top_per_side=200)
            for s in snippets_view:
                if s.side != want:
                    continue
                rows.append({
                    "doc_id": s.doc_id,
                    "side": s.side,
                    "seed": getattr(s, "seed", None),
                    "cosine": getattr(s, "cosine", None),
                    "text_window": getattr(s, "text_window", ""),
                    "text_surface": getattr(s, "text_surface", ""),
                    "text_lemmas": getattr(s, "text_lemmas", ""),
                    "cluster_id": getattr(s, "cluster_id", None),
                })
        except Exception:
            return []
        return rows

    def _display(self, snippets: list[dict]) -> None:
        from ....utils import display_limits
        cap = display_limits.SNIPPET_PREVIEW

        limit = min(len(snippets), 500)
        table = self._table
        vheader = table.verticalHeader()

        # ResizeToContents measures every row on each setItem — O(n²) with
        # word-wrap. Switch to Fixed during fill, restore after so Qt measures
        # once in a single batched pass.
        table.setUpdatesEnabled(False)
        table.blockSignals(True)
        vheader.setSectionResizeMode(QHeaderView.Fixed)
        try:
            table.setRowCount(limit)
            for i in range(limit):
                s = snippets[i]
                table.setItem(i, 0, QTableWidgetItem(str(s.get("seed", ""))))
                table.setItem(i, 1, QTableWidgetItem(f"{s.get('cosine', 0):.4f}"))
                table.setItem(i, 2, QTableWidgetItem(str(s.get("doc_id", ""))))
                anchor = s.get("text_window", "")
                if cap and len(anchor) > cap:
                    anchor = anchor[:cap] + "\u2026"
                table.setItem(i, 3, QTableWidgetItem(anchor))
        finally:
            vheader.setSectionResizeMode(QHeaderView.ResizeToContents)
            table.blockSignals(False)
            table.setUpdatesEnabled(True)

        self._detail.clear()
        self._displayed = snippets[:limit]

    def _on_row_selected(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._displayed):
            return
        show_snippet_detail(self._displayed[row], self._detail)
