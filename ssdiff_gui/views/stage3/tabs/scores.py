"""Document Scores tab — continuous: docs.pos/.neg; groups: alignment_scores.

The `if view.is_group:` row-builder branch here is sanctioned by the design.
This is one of two tabs allowed to branch on view type (the other is ExtremeDocsTab).
"""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
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
from ...widgets.loading_overlay import LoadingOverlay


def _html_palette():
    from ....theme import build_current_palette
    return build_current_palette()


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


class ScoresTab:
    def __init__(self, get_document_text):
        self._get_document_text = get_document_text
        self._widget: QWidget | None = None
        self._table: QTableWidget | None = None
        self._sort_combo: QComboBox | None = None
        self._detail_title: QLabel | None = None
        self._detail: QTextEdit | None = None
        self._overlay: LoadingOverlay | None = None
        self._df: pd.DataFrame | None = None
        self._rendered_df: pd.DataFrame | None = None

    def create(self, parent, on_pair_changed, pair_combos, pair_frames) -> QWidget:
        """Build and return the Document Scores tab widget."""
        tab = QWidget()
        self._widget = tab
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)

        controls = QHBoxLayout()
        controls.setContentsMargins(4, 4, 4, 0)
        controls.addWidget(
            pair_selector.make_pair_selector(on_pair_changed, pair_combos, pair_frames)
        )
        controls.addWidget(QLabel("Sort by:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItems([
            "Document Index",
            "Cosine (High \u2192 Low)",
            "Cosine (Low \u2192 High)",
            "Predicted (High \u2192 Low)",
            "Predicted (Low \u2192 High)",
        ])
        self._sort_combo.currentIndexChanged.connect(self._on_sort)
        controls.addWidget(self._sort_combo)
        controls.addStretch()
        outer.addLayout(controls)

        splitter = QSplitter(Qt.Horizontal)

        self._table = QTableWidget()
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.itemSelectionChanged.connect(self._on_row_selected)
        splitter.addWidget(self._table)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 4, 4, 4)

        self._detail_title = QLabel("Select a row to view its document text")
        self._detail_title.setObjectName("label_title")
        self._detail_title.setWordWrap(True)
        right_layout.addWidget(self._detail_title)

        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        right_layout.addWidget(self._detail, stretch=1)

        splitter.addWidget(right)
        splitter.setSizes([400, 600])
        outer.addWidget(splitter, stretch=1)

        self._overlay = LoadingOverlay(tab)
        return tab

    def load(self, view) -> None:
        """Load scores data from view, updating sort-combo and table."""
        self._detail.clear()
        self._detail_title.setText("Select a row to view its document text")
        self._update_sort_combo(view)
        df = self._rows(view)
        self._df = df
        if df is not None:
            self._render(df)

    def _update_sort_combo(self, view) -> None:
        self._sort_combo.blockSignals(True)
        self._sort_combo.clear()
        if not view.is_group:
            self._sort_combo.addItems([
                "Document Index",
                "Cosine (High \u2192 Low)",
                "Cosine (Low \u2192 High)",
                "Predicted (High \u2192 Low)",
                "Predicted (Low \u2192 High)",
            ])
        self._sort_combo.blockSignals(False)

    def _rows(self, view) -> pd.DataFrame | None:
        if view.is_group:
            return self._rows_group(view)
        return self._rows_continuous(view)

    def _rows_continuous(self, view) -> pd.DataFrame | None:
        try:
            n = view.working.stats.n_kept
            pos = list(view.working.docs.pos(k=n))
            neg = list(view.working.docs.neg(k=n))
            rows = []
            seen = set()
            for d in pos + neg:
                if d.doc_id in seen:
                    continue
                seen.add(d.doc_id)
                rows.append({
                    "idx": d.doc_id,
                    "cos_align": d.alignment_score,
                    "score_std": None,
                    "yhat_raw": d.y_hat,
                })
        except Exception:
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            return None

        if not rows:
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            return None

        df = pd.DataFrame(rows)
        col_labels = {
            "idx": "Doc #",
            "cos_align": "Cosine",
            "score_std": "Score (std)",
            "yhat_raw": "Predicted (raw)",
        }
        display_cols = [col_labels.get(c, c) for c in df.columns]
        self._table.setColumnCount(len(df.columns))
        self._table.setHorizontalHeaderLabels(display_cols)
        header = self._table.horizontalHeader()
        for col_idx, col_name in enumerate(df.columns):
            if col_name in ("score_std", "yhat_raw"):
                header.resizeSection(col_idx, 120)
        return df

    def _rows_group(self, view) -> pd.DataFrame | None:
        try:
            scores = view.working.alignment_scores
            groups = view.source.groups
            labels_map = view.source.group_labels or {}
            rows = []
            for doc_id, (score, g) in enumerate(zip(scores, groups)):
                rows.append({
                    "idx": int(doc_id),
                    "group": labels_map.get(g, g),
                    "cos_align": float(score),
                })
        except Exception:
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            return None

        if not rows:
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            return None

        df = pd.DataFrame(rows)
        col_labels = {"idx": "Doc #", "group": "Group", "cos_align": "Cosine"}
        display_cols = [col_labels.get(c, c) for c in df.columns]
        self._table.setColumnCount(len(df.columns))
        self._table.setHorizontalHeaderLabels(display_cols)
        return df

    def _render(self, df: pd.DataFrame) -> None:
        self._rendered_df = df.reset_index(drop=True)
        table = self._table
        table.setSortingEnabled(False)
        table.setUpdatesEnabled(False)
        table.setRowCount(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            for j, (col, value) in enumerate(row.items()):
                if pd.isna(value):
                    text = "\u2014"
                elif isinstance(value, bool):
                    text = str(value)
                elif isinstance(value, float):
                    text = f"{value:.4f}"
                else:
                    text = str(value)
                table.setItem(i, j, QTableWidgetItem(text))
        table.setUpdatesEnabled(True)
        table.setSortingEnabled(True)
        QApplication.processEvents()

    def _on_sort(self) -> None:
        if self._df is None:
            return

        self._overlay.start()
        QApplication.processEvents()

        idx = self._sort_combo.currentIndex()
        df = self._df.copy()

        try:
            if idx == 0:
                df = df.sort_values("idx")
            elif idx == 1:
                df = df.sort_values("cos_align", ascending=False)
            elif idx == 2:
                df = df.sort_values("cos_align", ascending=True)
            elif idx == 3:
                df = df.sort_values("yhat_raw", ascending=False)
            elif idx == 4:
                df = df.sort_values("yhat_raw", ascending=True)
        except KeyError:
            pass

        self._render(df)
        self._overlay.stop()

    def _on_row_selected(self) -> None:
        row = self._table.currentRow()
        if row < 0 or self._rendered_df is None:
            return
        df = self._rendered_df
        if row >= len(df):
            return

        doc_index = int(df.iloc[row].get("idx", -1))
        if doc_index < 0:
            return

        self._detail_title.setText(f"Document {doc_index}")

        p = _html_palette()
        rec = df.iloc[row]
        doc_text = self._get_document_text(doc_index)

        html_parts = []
        html_parts.append(
            '<table cellspacing="8" style="margin-bottom: 12px;">'
            "<tr>"
        )

        cos_val = rec.get("cos_align")
        if pd.notna(cos_val):
            html_parts.append(
                f'<td style="padding-right: 20px;">'
                f'<span style="color: {p.text_secondary}; font-size: {p.font_size_sm};">COSINE</span><br/>'
                f'<span style="font-size: {p.font_size_lg}; font-weight: 600;">{float(cos_val):.4f}</span>'
                f"</td>"
            )

        yhat = rec.get("yhat_raw")
        if pd.notna(yhat):
            html_parts.append(
                f'<td style="padding-right: 20px;">'
                f'<span style="color: {p.text_secondary}; font-size: {p.font_size_sm};">PREDICTED</span><br/>'
                f'<span style="font-size: {p.font_size_lg}; font-weight: 600;">{float(yhat):.4f}</span>'
                f"</td>"
            )

        score_std = rec.get("score_std")
        if pd.notna(score_std):
            html_parts.append(
                f'<td style="padding-right: 20px;">'
                f'<span style="color: {p.text_secondary}; font-size: {p.font_size_sm};">SCORE (STD)</span><br/>'
                f'<span style="font-size: {p.font_size_lg}; font-weight: 600;">{float(score_std):.4f}</span>'
                f"</td>"
            )

        html_parts.append("</tr></table>")

        html_parts.append(
            f'<div style="border-top: 1px solid {p.border}; padding-top: 12px;">'
            f'<span style="color: {p.text_secondary}; font-size: {p.font_size_sm}; text-transform: uppercase;">Document Text</span>'
            f"</div>"
        )

        if doc_text:
            html_parts.append(
                f'<div style="margin-top: 8px; line-height: 1.5;">{_escape_html(doc_text)}</div>'
            )
        else:
            html_parts.append(
                f'<div style="margin-top: 8px; color: {p.text_muted}; font-style: italic;">'
                f"Document text not available \u2014 CSV may not be loaded"
                f"</div>"
            )

        self._detail.setHtml("".join(html_parts))

    def is_visible_for(self, view) -> bool:
        return True
