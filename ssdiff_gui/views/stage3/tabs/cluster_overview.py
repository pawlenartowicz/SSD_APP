"""Cluster Overview tab — pos/neg cluster tables with lazy per-click snippet fetch."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import pair_selector
from ....utils.report_settings import get_report_setting, KEY_SNIPPET_PREVIEW


class ClusterOverviewTab:
    def __init__(self, get_project, get_document_text):
        self._get_project = get_project
        self._get_document_text = get_document_text
        self._widget: QWidget | None = None

        # Left-panel widgets
        self._ov_pos_group: QGroupBox | None = None
        self._ov_neg_group: QGroupBox | None = None
        self._ov_pos_table: QTableWidget | None = None
        self._ov_neg_table: QTableWidget | None = None
        self._cluster_topn_spin: QSpinBox | None = None
        self._cluster_snippets_spin: QSpinBox | None = None

        # Right-panel widgets
        self._ov_right_panel: QWidget | None = None
        self._snippet_panel_title: QLabel | None = None
        self._snippet_panel_keywords: QLabel | None = None
        self._keywords_toggle_link: QLabel | None = None
        self._ov_snippet_table: QTableWidget | None = None
        self._ov_snippet_detail: QTextEdit | None = None

        # Overlay (injected after create())
        self._overlay_right_panel = None

        # State
        self._keywords_expanded: bool = False
        self._current_member_words: list[str] = []
        self._ov_current_snippets: list[dict] = []

        # Reference to the active view (set in load)
        self._view = None
        self._clusters_members: list[dict] = []
        self._pos_summary: list[dict] = []
        self._neg_summary: list[dict] = []

    # ------------------------------------------------------------------ #
    #  Public interface
    # ------------------------------------------------------------------ #

    def create(self, parent, on_pair_changed, pair_combos, pair_frames) -> QWidget:
        """Build and return the Cluster Overview tab widget."""
        tab = QWidget()
        self._widget = tab
        outer = QHBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        # ---- LEFT: cluster tables ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 0, 4)

        ctrl = QHBoxLayout()
        ctrl.addWidget(
            pair_selector.make_pair_selector(on_pair_changed, pair_combos, pair_frames)
        )
        ctrl.addWidget(QLabel("Top words per cluster:"))
        self._cluster_topn_spin = QSpinBox()
        self._cluster_topn_spin.setRange(3, 30)
        self._cluster_topn_spin.setValue(5)
        self._cluster_topn_spin.valueChanged.connect(self._reload_cluster_tables)
        ctrl.addWidget(self._cluster_topn_spin)
        ctrl.addSpacing(16)
        ctrl.addWidget(QLabel("Snippets per cluster:"))
        self._cluster_snippets_spin = QSpinBox()
        self._cluster_snippets_spin.setRange(5, 200)
        self._cluster_snippets_spin.setValue(20)
        self._cluster_snippets_spin.valueChanged.connect(self._refresh_overview_snippets)
        ctrl.addWidget(self._cluster_snippets_spin)
        ctrl.addStretch()
        left_layout.addLayout(ctrl)

        left_splitter = QSplitter(Qt.Vertical)

        self._ov_pos_group = QGroupBox("Positive Clusters  (+\u03b2  \u2192  higher outcome)")
        pos_layout = QVBoxLayout()
        self._ov_pos_table = self._make_cluster_table()
        self._ov_pos_table.itemSelectionChanged.connect(
            lambda: self._on_cluster_clicked(self._ov_pos_table, "pos")
        )
        pos_layout.addWidget(self._ov_pos_table)
        self._ov_pos_group.setLayout(pos_layout)
        left_splitter.addWidget(self._ov_pos_group)

        self._ov_neg_group = QGroupBox("Negative Clusters  (\u2212\u03b2  \u2192  lower outcome)")
        neg_layout = QVBoxLayout()
        self._ov_neg_table = self._make_cluster_table()
        self._ov_neg_table.itemSelectionChanged.connect(
            lambda: self._on_cluster_clicked(self._ov_neg_table, "neg")
        )
        neg_layout.addWidget(self._ov_neg_table)
        self._ov_neg_group.setLayout(neg_layout)
        left_splitter.addWidget(self._ov_neg_group)

        left_splitter.setSizes([300, 300])
        left_layout.addWidget(left_splitter, stretch=1)

        splitter.addWidget(left)

        # ---- RIGHT: snippet panel ----
        right = QWidget()
        self._ov_right_panel = right
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 4, 4, 4)

        self._snippet_panel_title = QLabel("Select a cluster to view its text snippets")
        self._snippet_panel_title.setObjectName("label_title")
        self._snippet_panel_title.setWordWrap(True)
        right_layout.addWidget(self._snippet_panel_title)

        keywords_layout = QHBoxLayout()
        keywords_layout.setContentsMargins(0, 0, 0, 0)
        keywords_layout.setSpacing(4)

        self._snippet_panel_keywords = QLabel("")
        self._snippet_panel_keywords.setObjectName("label_muted")
        self._snippet_panel_keywords.setWordWrap(False)
        self._snippet_panel_keywords.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        keywords_layout.addWidget(self._snippet_panel_keywords, stretch=1)

        self._keywords_toggle_link = QLabel("")
        self._keywords_toggle_link.setObjectName("label_link")
        self._keywords_toggle_link.setCursor(Qt.PointingHandCursor)
        self._keywords_toggle_link.setVisible(False)
        self._keywords_toggle_link.mousePressEvent = lambda e: self._toggle_keywords_display()
        keywords_layout.addWidget(self._keywords_toggle_link)

        right_layout.addLayout(keywords_layout)

        right_splitter = QSplitter(Qt.Vertical)

        self._ov_snippet_table = QTableWidget()
        self._ov_snippet_table.setColumnCount(4)
        self._ov_snippet_table.setHorizontalHeaderLabels(["Seed", "Cosine", "Doc", "Snippet"])
        self._ov_snippet_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self._ov_snippet_table.setAlternatingRowColors(True)
        self._ov_snippet_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._ov_snippet_table.setWordWrap(True)
        self._ov_snippet_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self._ov_snippet_table.itemSelectionChanged.connect(self._on_snippet_selected)
        right_splitter.addWidget(self._ov_snippet_table)

        detail_group = QGroupBox("Full Document Text")
        detail_layout = QVBoxLayout()
        self._ov_snippet_detail = QTextEdit()
        self._ov_snippet_detail.setReadOnly(True)
        detail_layout.addWidget(self._ov_snippet_detail)
        detail_group.setLayout(detail_layout)
        right_splitter.addWidget(detail_group)

        right_splitter.setSizes([400, 200])
        right_layout.addWidget(right_splitter, stretch=1)

        splitter.addWidget(right)
        splitter.setSizes([500, 500])
        outer.addWidget(splitter)

        return tab

    def load(self, view) -> None:
        """Populate cluster tables from view; clear snippet panel."""
        self._view = view

        cluster_kwargs = self._build_cluster_kwargs()
        pos_clusters_view, neg_clusters_view = self._get_cluster_views(view, cluster_kwargs)

        # Build cluster-summary rows for the tables
        from ..widget import _clusters_to_summary, _clusters_to_members

        self._pos_summary = _clusters_to_summary(pos_clusters_view, "pos")
        self._neg_summary = _clusters_to_summary(neg_clusters_view, "neg")
        self._clusters_members = (
            _clusters_to_members(pos_clusters_view, "pos")
            + _clusters_to_members(neg_clusters_view, "neg")
        )

        self._fill_cluster_table(self._ov_pos_table, self._pos_summary)
        self._fill_cluster_table(self._ov_neg_table, self._neg_summary)

        # Clear snippet panel
        self._snippet_panel_title.setText("Select a cluster to view its text snippets")
        self._current_member_words = []
        self._keywords_expanded = False
        self._update_keywords_display()
        self._ov_snippet_table.setRowCount(0)
        self._ov_snippet_detail.clear()
        self._ov_current_snippets = []

    def is_visible_for(self, view) -> bool:
        return True

    # ------------------------------------------------------------------ #
    #  Cluster-table helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_cluster_table() -> QTableWidget:
        t = QTableWidget()
        t.setColumnCount(5)
        t.setHorizontalHeaderLabels(["#", "Size", "Coherence", "Centroid cos(\u03b2)", "Top Words"])
        t.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        t.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        t.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        t.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        t.setAlternatingRowColors(True)
        t.setSelectionBehavior(QTableWidget.SelectRows)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        return t

    def _fill_cluster_table(self, table: QTableWidget, clusters: list) -> None:
        topn = self._cluster_topn_spin.value()
        table.setRowCount(len(clusters))
        for i, c in enumerate(clusters):
            table.setItem(i, 0, QTableWidgetItem(str(c.get("cluster_rank", ""))))
            table.setItem(i, 1, QTableWidgetItem(str(c.get("size", ""))))
            table.setItem(i, 2, QTableWidgetItem(f"{c.get('coherence', 0):.3f}"))
            table.setItem(i, 3, QTableWidgetItem(f"{c.get('centroid_cos_beta', 0):.3f}"))
            top_words_str = c.get("top_words", "")
            words_list = [w.strip() for w in top_words_str.split(",") if w.strip()]
            table.setItem(i, 4, QTableWidgetItem(", ".join(words_list[:topn])))

    def _reload_cluster_tables(self) -> None:
        """Re-fill cluster tables when topN changes."""
        if self._view is None:
            return
        self.load(self._view)

    def _refresh_overview_snippets(self) -> None:
        """Re-fill the snippet table when the snippets-per-cluster limit changes."""
        if self._ov_current_snippets:
            self._fill_snippet_table(self._ov_current_snippets)

    # ------------------------------------------------------------------ #
    #  Cluster-click — lazy snippet fetch
    # ------------------------------------------------------------------ #

    def _on_cluster_clicked(self, table: QTableWidget, side: str) -> None:
        """On row click: deselect other table, then lazy-fetch this cluster's snippets."""
        other = self._ov_neg_table if side == "pos" else self._ov_pos_table
        other.clearSelection()

        row = table.currentRow()
        if row < 0 or self._view is None:
            return

        # Identify which cluster was clicked from the cached summary
        summary = self._pos_summary if side == "pos" else self._neg_summary
        if row >= len(summary):
            return

        cluster_info = summary[row]
        cluster_rank = cluster_info.get("cluster_rank")
        cluster_id = cluster_info.get("cluster_id")

        # Build member words from the cached members list
        members = [
            m for m in self._clusters_members
            if m.get("cluster_id") == cluster_id and m.get("side") == side
        ]
        member_words = [m.get("word", "") for m in members]

        side_label = "Positive" if side == "pos" else "Negative"
        self._snippet_panel_title.setText(f"{side_label} Cluster {cluster_rank}")

        self._current_member_words = member_words
        self._keywords_expanded = False
        self._update_keywords_display()

        # Lazy fetch: access the cached cluster view for this side, then filter snippets
        # to this cluster_id.
        snippets = []
        cluster_kwargs = self._build_cluster_kwargs()
        pos_clusters_view, neg_clusters_view = self._get_cluster_views(self._view, cluster_kwargs)
        clusters_view_sided = pos_clusters_view if side == "pos" else neg_clusters_view
        if clusters_view_sided is not None:
            snippets_sided = clusters_view_sided.snippets
            filtered = snippets_sided(k=None, cluster_id=cluster_id)
            snippets = self._view_to_rows(filtered)

        self._ov_current_snippets = snippets
        self._fill_snippet_table(snippets)
        self._ov_snippet_detail.clear()

    # ------------------------------------------------------------------ #
    #  Snippet-panel helpers
    # ------------------------------------------------------------------ #

    def _fill_snippet_table(self, snippets: list) -> None:
        cap = get_report_setting(KEY_SNIPPET_PREVIEW)
        limit = min(len(snippets), self._cluster_snippets_spin.value())
        table = self._ov_snippet_table
        vheader = table.verticalHeader()

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

    def _on_snippet_selected(self) -> None:
        """Show full document text for the selected snippet."""
        row = self._ov_snippet_table.currentRow()
        if row < 0 or row >= len(self._ov_current_snippets):
            return
        snip = self._ov_current_snippets[row]
        # Delegate rendering to the parent Stage3Widget via the injected helper
        self._show_snippet_detail(snip)

    def _show_snippet_detail(self, snip: dict) -> None:
        """Render snippet dict into the detail QTextEdit."""
        doc_id = snip.get("doc_id", "N/A")
        seed = snip.get("seed", "N/A")
        cosine = snip.get("cosine", 0)
        anchor = snip.get("text_window", "")
        surface = snip.get("text_surface", "")
        cluster = snip.get("cluster_id")

        import html as _html_mod

        lines = ['<table cellspacing="8" style="margin-bottom: 12px;"><tr>']
        lines.append(
            f'<td style="padding-right: 20px;">'
            f'<span style="font-size: 10px;">DOCUMENT</span><br/>'
            f'<span style="font-size: 14px; font-weight: 600;">{doc_id}</span>'
            f'</td>'
        )
        lines.append(
            f'<td style="padding-right: 20px;">'
            f'<span style="font-size: 10px;">SEED WORD</span><br/>'
            f'<span style="font-size: 14px; font-weight: 600;">{seed}</span>'
            f'</td>'
        )
        lines.append(
            f'<td style="padding-right: 20px;">'
            f'<span style="font-size: 10px;">COSINE</span><br/>'
            f'<span style="font-size: 14px; font-weight: 600;">{cosine:.4f}</span>'
            f'</td>'
        )
        if cluster is not None:
            lines.append(
                f'<td style="padding-right: 20px;">'
                f'<span style="font-size: 10px;">CLUSTER</span><br/>'
                f'<span style="font-size: 14px; font-weight: 600;">{cluster}</span>'
                f'</td>'
            )
        lines.append('</tr></table>')

        if anchor:
            lines.append(
                f'<div style="padding-top: 12px;">'
                f'<span style="font-size: 10px; text-transform: uppercase;">Snippet Context</span>'
                f'</div>'
                f'<div style="margin-top: 8px; line-height: 1.5;">'
                f'{_html_mod.escape(anchor)}</div>'
            )
        if surface:
            lines.append(
                f'<div style="padding-top: 12px; margin-top: 12px;">'
                f'<span style="font-size: 10px; text-transform: uppercase;">Full Document Text</span>'
                f'</div>'
                f'<div style="margin-top: 8px; line-height: 1.5;">'
                f'{_html_mod.escape(surface)}</div>'
            )

        self._ov_snippet_detail.setHtml("".join(lines))

    # ------------------------------------------------------------------ #
    #  Keywords display helpers
    # ------------------------------------------------------------------ #

    def _update_keywords_display(self) -> None:
        words = self._current_member_words
        count = len(words)
        if count == 0:
            self._snippet_panel_keywords.setText("")
            self._keywords_toggle_link.setVisible(False)
            return

        if self._keywords_expanded:
            self._snippet_panel_keywords.setWordWrap(True)
            self._snippet_panel_keywords.setText(f"Members ({count}): {', '.join(words)} ")
            self._keywords_toggle_link.setText("see less")
            self._keywords_toggle_link.setVisible(True)
        else:
            self._snippet_panel_keywords.setWordWrap(False)
            preview_count = min(8, count)
            preview = ", ".join(words[:preview_count])
            remaining = count - preview_count
            if remaining > 0:
                self._snippet_panel_keywords.setText(f"Members ({count}): {preview}... ")
                self._keywords_toggle_link.setText(f"+{remaining} more")
                self._keywords_toggle_link.setVisible(True)
            else:
                self._snippet_panel_keywords.setText(f"Members ({count}): {preview}")
                self._keywords_toggle_link.setVisible(False)

    def _toggle_keywords_display(self) -> None:
        self._keywords_expanded = not self._keywords_expanded
        self._update_keywords_display()

    # ------------------------------------------------------------------ #
    #  Internal: cluster-kwargs + view helpers
    # ------------------------------------------------------------------ #

    def _build_cluster_kwargs(self) -> dict:
        p = self._get_project()
        if p is None:
            return {}
        k = None if p.clustering_k_auto else p.clustering_k_min
        return {
            "topn": p.clustering_topn,
            "k": k,
            "k_min": p.clustering_k_min,
            "k_max": p.clustering_k_max,
        }

    def _get_cluster_views(self, view, cluster_kwargs: dict):
        """Return (pos_clusters_view, neg_clusters_view) applying cluster_kwargs."""
        clusters_index = view.working.clusters

        def _sided(attr: str):
            base = getattr(clusters_index, attr, None)
            if base is None:
                return None
            if cluster_kwargs:
                # _GroupClustersParentShim._clusters_for doesn't accept cluster_kwargs in some
                # ssdiff versions; fall back to the no-arg accessor when the kwargs call fails.
                # Only TypeError — shim raises TypeError on unknown kwargs.
                try:
                    return base(**cluster_kwargs)
                except TypeError:
                    pass
            return base

        return _sided("pos"), _sided("neg")

    @staticmethod
    def _view_to_rows(view) -> list[dict]:
        """Convert a SnippetsViewSided (or any iterable of Snippet) to dicts."""
        rows = []
        for s in view:
            rows.append({
                "doc_id": s.doc_id,
                "post_id": getattr(s, "post_id", None),
                "snippet_id": getattr(s, "snippet_id", None),
                "side": s.side,
                "seed": getattr(s, "seed", None),
                "cosine": getattr(s, "cosine", None),
                "text_window": getattr(s, "text_window", ""),
                "text_surface": getattr(s, "text_surface", ""),
                "text_lemmas": getattr(s, "text_lemmas", ""),
                "cluster_id": getattr(s, "cluster_id", None),
            })
        return rows
