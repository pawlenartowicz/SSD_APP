"""Stage 3: Results view for SSD."""

import re
from typing import Optional, List, Dict, Tuple
from pathlib import Path

import pandas as pd

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QMessageBox,
    QHeaderView,
    QSplitter,
    QFrame,
    QScrollArea,
    QSpinBox,
    QApplication,
    QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QEvent, QTimer
from PySide6.QtGui import QPixmap

import numpy as np

from ssdiff import PLSResult, PCAOLSResult, GroupResult

from ..models.project import Project, Result
from ..utils.settings import app_settings
from .widgets.loading_overlay import LoadingOverlay
from .widgets.info_button import InfoButton
from .widgets.removable_delegate import RemovableItemDelegate


# Characters that are unsafe or platform-dependent in folder names.
_INVALID_FOLDER_CHARS = re.compile(r'[/\\:*?"<>|\x00-\x1f]')


def _sanitize_folder_name(name: str) -> str:
    """Turn a user-provided name into a filesystem-safe folder basename.

    Empty string means "no usable name" — caller should fall back to a timestamp.
    """
    name = name.strip()
    if not name:
        return ""
    name = _INVALID_FOLDER_CHARS.sub("_", name)
    name = re.sub(r"\s+", "_", name)
    name = name.lstrip(".")           # no hidden/dotfile folders
    name = name.rstrip(" .")          # Windows: trailing space/dot illegal
    if len(name) > 100:
        name = name[:100].rstrip("_")
    return name


def _resolve_folder_collision(results_dir: Path, base: str) -> str:
    """Return `base`, or `base_2`, `base_3`… so the folder doesn't exist yet."""
    if not (results_dir / base).exists():
        return base
    i = 2
    while (results_dir / f"{base}_{i}").exists():
        i += 1
    return f"{base}_{i}"


def _result_dropdown_label(result: Result) -> str:
    """Human-readable label for the results dropdown.

    Format: "<display_name> (YYYY-MM-DD HH:MM:SS)" + optional state tag.
    """
    display_name = result.name or result.result_id
    timestamp = result.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    label = f"{display_name} ({timestamp})"

    if result.status == "missing" or result.result_path is None and not result._result:
        # Unsaved in-memory results have result_path=None too — they go
        # through a different branch in _populate_result_selector and
        # never reach this function.
        if result.status == "missing":
            label += " [missing]"
    elif result.load_error or result.status == "error":
        label += " [broken]"
    elif result.is_orphan:
        label += " [orphan]"
    return label


def _clusters_to_summary(clusters_side, side: str) -> list[dict]:
    """Transform a ClustersViewSided into flat summary rows."""
    if not clusters_side:
        return []
    rows = []
    try:
        for rank, c in enumerate(clusters_side, 1):
            top_words_str = ", ".join(
                w.word for w in list(clusters_side.words(c.cluster_id))[:5]
            )
            rows.append({
                "cluster_rank": rank,
                "cluster_id": c.cluster_id,
                "side": side,
                "size": c.size,
                "coherence": c.coherence,
                "centroid_cos_beta": c.centroid_cos_beta,
                "top_words": top_words_str,
            })
    except Exception:
        return []
    return rows


def _clusters_to_members(clusters_side, side: str) -> list[dict]:
    """Transform a ClustersViewSided into flat member rows."""
    if not clusters_side:
        return []
    rows = []
    try:
        for rank, c in enumerate(clusters_side, 1):
            for w in clusters_side.words(c.cluster_id):
                rows.append({
                    "cluster_rank": rank,
                    "cluster_id": c.cluster_id,
                    "side": side,
                    "word": w.word,
                    "cos_centroid": w.cos_centroid,
                    "cos_beta": w.cos_beta,
                })
    except Exception:
        return []
    return rows


class Stage3Widget(QWidget):
    """Stage 3: Results - View and export analysis results."""

    new_run_requested = Signal()
    result_saved = Signal()  # emitted after a result is saved

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Optional[Project] = None
        self.current_result: Optional[Result] = None
        self._current_snippets: List[Dict] = []
        self._scores_df: Optional[pd.DataFrame] = None
        self._current_pair_key: Optional[Tuple[str, str]] = None

        # Pair-selector widgets (one per pair-scoped tab — Cluster Overview,
        # Contrast Snippets, Semantic Poles, Document Scores, Extreme Documents).
        # All combos are kept in sync via _on_pair_changed.
        self._pair_combos: List[QComboBox] = []
        self._pair_frames: List[QWidget] = []

        # Unsaved-result tracking
        self._unsaved_result: Optional[Result] = None
        self._is_viewing_unsaved: bool = False

        self._setup_ui()

    def resizeEvent(self, event):
        """Re-apply fit-to-window zoom when the widget is resized."""
        super().resizeEvent(event)
        if (
            hasattr(self, "_pca_sweep_zoom_pct")
            and self._pca_sweep_zoom_pct == 0
            and self._pca_sweep_pixmap is not None
        ):
            self._pca_sweep_apply_zoom()

    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 16)
        main_layout.setSpacing(12)

        # Header
        header_layout = QHBoxLayout()

        title = QLabel("Results")
        title.setObjectName("label_title")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Analysis type badge
        self._analysis_badge = QLabel("")
        self._analysis_badge.setObjectName("label_badge")
        self._analysis_badge.setStyleSheet(
            "QLabel {"
            "  padding: 3px 10px;"
            "  border-radius: 4px;"
            "  font-weight: 600;"
            "  font-size: 11px;"
            "}"
        )
        self._analysis_badge.hide()
        header_layout.addWidget(self._analysis_badge)

        # Result selector
        header_layout.addWidget(QLabel("Result:"))
        self.result_selector = QComboBox()
        self.result_selector.setMinimumWidth(200)
        self.result_selector.currentIndexChanged.connect(self._on_result_selected)
        self._run_delegate = RemovableItemDelegate(
            self.result_selector,
            self._delete_result_by_index,
            is_removable=lambda row: self.result_selector.itemData(row) is not self._unsaved_result,
            parent=self.result_selector,
        )
        self.result_selector.setItemDelegate(self._run_delegate)
        header_layout.addWidget(self.result_selector)

        main_layout.addLayout(header_layout)

        # Summary stats
        self._create_summary_section(main_layout)

        # Tabs — Cluster Overview is the first and default tab
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_cluster_overview_tab(), "Cluster Overview")  # 0
        self.tabs.addTab(self._create_config_tab(), "Details")                     # 1
        self.tabs.addTab(self._create_pca_sweep_tab(), "PCA Sweep")                # 2
        self.tabs.addTab(self._create_snippets_tab(), "Beta Snippets")             # 3
        self.tabs.addTab(self._create_poles_tab(), "Semantic Poles")               # 4
        self.tabs.addTab(self._create_scores_tab(), "Document Scores")             # 5
        self.tabs.addTab(self._create_extreme_docs_tab(), "Extreme Documents")   # 6
        self.tabs.addTab(self._create_misdiagnosed_tab(), "Misdiagnosed")        # 7

        # Per-tab help tooltips — single overlay info button on the tab bar
        self._tab_tooltips = {
            0: (
                "<b>Cluster Overview</b><br><br>"
                "Semantic clusters found along the SSD dimension, split into "
                "positive and negative sides.<br><br>"
                "Each cluster shows its top representative words. "
                "Click a cluster row to see the text snippets that belong to it. "
                "Adjust <i>Top words</i> and <i>Snippets per cluster</i> above."
            ),
            1: (
                "<b>Details</b><br><br>"
                "Three panels summarising the completed result:<br>"
                "<b>Result Information</b> — dataset stats, sample counts, and "
                "model fit statistics (R², p-value, effect sizes).<br>"
                "<b>Concept Configuration</b> — analysis type, mode, and "
                "the lexicon words used (if applicable).<br>"
                "<b>Model Configuration</b> — hyperparameters, PCA components, "
                "and embedding coverage."
            ),
            2: (
                "<b>PCA Sweep</b><br><br>"
                "Shows how much variance is explained at each number of PCA "
                "components, along with the corresponding R² and p-value.<br><br>"
                "The chosen K (number of components) is highlighted. "
                "Use the zoom controls to inspect the plot in detail."
            ),
            3: (
                "<b>Snippets</b><br><br>"
                "Sentences ranked by their loading on the SSD dimension.<br><br>"
                "Use the <i>Side</i> dropdown to switch between positive and "
                "negative ends. Click a row to see the full source sentence "
                "highlighted in context below."
            ),
            4: (
                "<b>Semantic Poles</b><br><br>"
                "Words at the two ends of the SSD dimension.<br>"
                "<b>Positive end</b> — words most aligned with higher outcome "
                "values (or the first group).<br>"
                "<b>Negative end</b> — words aligned with lower values "
                "(or the second group).<br><br>"
                "The cosine similarity column shows how strongly each word "
                "aligns with the SSD direction."
            ),
            5: (
                "<b>Document Scores</b><br><br>"
                "Each document's cosine similarity to the SSD direction.<br><br>"
                "Sort by index, score, or outcome value. Click a row to see "
                "the full document text on the right."
            ),
            6: (
                "<b>Extreme Documents</b><br><br>"
                "Documents with the highest and lowest cosine alignment "
                "to the SSD dimension.<br><br>"
                "Useful for inspecting what content maps to each end "
                "of the semantic gradient."
            ),
            7: (
                "<b>Misdiagnosed Documents</b><br><br>"
                "Documents where the model's prediction diverges most "
                "from the actual outcome value.<br><br>"
                "<b>Over-predicted</b> — model predicted higher than actual.<br>"
                "<b>Under-predicted</b> — model predicted lower than actual."
            ),
        }

        self._tab_info_btn = InfoButton(
            self._tab_tooltips.get(0, ""), parent=self.tabs,
        )
        self.tabs.installEventFilter(self)
        self.tabs.currentChanged.connect(self._on_tab_info_update)
        QTimer.singleShot(0, self._reposition_tab_info)

        main_layout.addWidget(self.tabs, stretch=1)

        # Actions
        self._create_actions_section(main_layout)

        # Loading overlays (one per target widget)
        self._overlay_right_panel = LoadingOverlay(self._ov_right_panel)
        self._overlay_snippets = LoadingOverlay(self._snippets_tab)
        self._overlay_scores = LoadingOverlay(self._scores_tab)
        self._overlay_tabs = LoadingOverlay(self.tabs)

    # ------------------------------------------------------------------ #
    #  Tab info button (overlay on tab bar)
    # ------------------------------------------------------------------ #

    def eventFilter(self, obj, event):
        if obj is self.tabs:
            if event.type() in (QEvent.Resize, QEvent.LayoutRequest):
                self._reposition_tab_info()
        return super().eventFilter(obj, event)

    def _reposition_tab_info(self):
        """Pin the info button to the far right of the tab widget, aligned with the tab bar."""
        bar = self.tabs.tabBar()
        x = self.tabs.width() - self._tab_info_btn.width() - 2
        y = (bar.height() - self._tab_info_btn.height()) // 2
        self._tab_info_btn.move(x, y)
        self._tab_info_btn.raise_()

    def _on_tab_info_update(self, index: int):
        """Update the info button tooltip for the active tab."""
        self._tab_info_btn._tooltip_html = self._tab_tooltips.get(index, "")

    # ------------------------------------------------------------------ #
    #  Summary stats strip
    # ------------------------------------------------------------------ #

    def _create_summary_section(self, parent_layout):
        """Create the summary statistics section with 7 generic stat cards."""
        self.summary_frame = QFrame()
        self.summary_frame.setFrameStyle(QFrame.Box)
        self.summary_frame.setObjectName("frame_summary")

        layout = QHBoxLayout(self.summary_frame)

        self._stat_cards = []
        default_labels = [
            "R-squared", "Adj. R-squared", "F-statistic",
            "p-value", "Documents Used", "PCA K", "PCA Variance Explained",
        ]
        for label in default_labels:
            card = self._create_stat_card(label, "\u2014")
            self._stat_cards.append(card)
            layout.addWidget(card)

        parent_layout.addWidget(self.summary_frame)

    def _create_stat_card(self, label: str, value: str) -> QWidget:
        """Create a statistics card widget."""
        card = QWidget()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 10, 15, 10)

        value_label = QLabel(value)
        value_label.setObjectName("label_stat_value")
        value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(value_label)

        name_label = QLabel(label)
        name_label.setObjectName("label_stat_name")
        name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(name_label)

        card.value_label = value_label
        card.name_label = name_label
        return card

    def _set_stat_card(self, index: int, label: str, value: str):
        """Update a stat card's label and value by index."""
        if 0 <= index < len(self._stat_cards):
            card = self._stat_cards[index]
            card.name_label.setText(label)
            card.value_label.setText(value)

    def _make_pair_selector(self) -> QWidget:
        """Build one synchronized pair-selector (label + combo) for a pair-scoped tab.

        Each call appends to self._pair_combos / self._pair_frames so all instances
        can be populated and synced together. The returned frame is hidden when the
        result isn't a multi-pair GroupResult.
        """
        frame = QWidget()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Choose pair:"))
        combo = QComboBox()
        combo.setMinimumWidth(250)
        combo.currentIndexChanged.connect(self._on_pair_changed)
        layout.addWidget(combo)
        frame.hide()
        self._pair_combos.append(combo)
        self._pair_frames.append(frame)
        return frame

    # ------------------------------------------------------------------ #
    #  Tab 0 — Cluster Overview  (centrepiece)
    # ------------------------------------------------------------------ #

    def _create_cluster_overview_tab(self) -> QWidget:
        """Create the cluster overview tab with pos/neg tables and snippet panel."""
        tab = QWidget()
        outer = QHBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        # ---- LEFT: cluster tables ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 0, 4)

        # Controls row
        ctrl = QHBoxLayout()
        ctrl.addWidget(self._make_pair_selector())
        ctrl.addWidget(QLabel("Top words per cluster:"))
        self.cluster_topn_spin = QSpinBox()
        self.cluster_topn_spin.setRange(3, 30)
        self.cluster_topn_spin.setValue(5)
        self.cluster_topn_spin.valueChanged.connect(self._reload_cluster_tables)
        ctrl.addWidget(self.cluster_topn_spin)
        ctrl.addSpacing(16)
        ctrl.addWidget(QLabel("Snippets per cluster:"))
        self.cluster_snippets_spin = QSpinBox()
        self.cluster_snippets_spin.setRange(5, 200)
        self.cluster_snippets_spin.setValue(20)
        self.cluster_snippets_spin.valueChanged.connect(self._refresh_overview_snippets)
        ctrl.addWidget(self.cluster_snippets_spin)
        ctrl.addStretch()
        left_layout.addLayout(ctrl)

        # Vertical splitter for positive and negative cluster tables
        left_splitter = QSplitter(Qt.Vertical)

        # Positive clusters
        self._ov_pos_group = QGroupBox("Positive Clusters  (+\u03b2  \u2192  higher outcome)")
        pos_layout = QVBoxLayout()
        self.ov_pos_table = self._make_cluster_table()
        self.ov_pos_table.itemSelectionChanged.connect(
            lambda: self._on_overview_cluster_clicked(self.ov_pos_table, "pos")
        )
        pos_layout.addWidget(self.ov_pos_table)
        self._ov_pos_group.setLayout(pos_layout)
        left_splitter.addWidget(self._ov_pos_group)

        # Negative clusters
        self._ov_neg_group = QGroupBox("Negative Clusters  (\u2212\u03b2  \u2192  lower outcome)")
        neg_layout = QVBoxLayout()
        self.ov_neg_table = self._make_cluster_table()
        self.ov_neg_table.itemSelectionChanged.connect(
            lambda: self._on_overview_cluster_clicked(self.ov_neg_table, "neg")
        )
        neg_layout.addWidget(self.ov_neg_table)
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

        self.snippet_panel_title = QLabel("Select a cluster to view its text snippets")
        self.snippet_panel_title.setObjectName("label_title")
        self.snippet_panel_title.setWordWrap(True)
        right_layout.addWidget(self.snippet_panel_title)

        # Keywords section with expand/collapse
        keywords_layout = QHBoxLayout()
        keywords_layout.setContentsMargins(0, 0, 0, 0)
        keywords_layout.setSpacing(4)

        self.snippet_panel_keywords = QLabel("")
        self.snippet_panel_keywords.setObjectName("label_muted")
        self.snippet_panel_keywords.setWordWrap(False)
        self.snippet_panel_keywords.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        keywords_layout.addWidget(self.snippet_panel_keywords, stretch=1)

        self.keywords_toggle_link = QLabel("")
        self.keywords_toggle_link.setObjectName("label_link")
        self.keywords_toggle_link.setCursor(Qt.PointingHandCursor)
        self.keywords_toggle_link.setVisible(False)
        self.keywords_toggle_link.mousePressEvent = lambda e: self._toggle_keywords_display()
        keywords_layout.addWidget(self.keywords_toggle_link)

        right_layout.addLayout(keywords_layout)

        # Store state for keywords toggle
        self._keywords_expanded = False
        self._current_member_words = []

        # Vertical splitter for snippet table and detail
        right_splitter = QSplitter(Qt.Vertical)

        # Snippet list (table)
        self.ov_snippet_table = QTableWidget()
        self.ov_snippet_table.setColumnCount(4)
        self.ov_snippet_table.setHorizontalHeaderLabels([
            "Seed", "Cosine", "Doc", "Snippet",
        ])
        self.ov_snippet_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Stretch
        )
        self.ov_snippet_table.setAlternatingRowColors(True)
        self.ov_snippet_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.ov_snippet_table.setWordWrap(True)
        self.ov_snippet_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.ov_snippet_table.itemSelectionChanged.connect(
            self._on_overview_snippet_selected
        )
        right_splitter.addWidget(self.ov_snippet_table)

        # Full-text detail
        detail_group = QGroupBox("Full Document Text")
        detail_layout = QVBoxLayout()
        self.ov_snippet_detail = QTextEdit()
        self.ov_snippet_detail.setReadOnly(True)
        detail_layout.addWidget(self.ov_snippet_detail)
        detail_group.setLayout(detail_layout)
        right_splitter.addWidget(detail_group)

        right_splitter.setSizes([400, 200])
        right_layout.addWidget(right_splitter, stretch=1)

        splitter.addWidget(right)

        splitter.setSizes([500, 500])
        outer.addWidget(splitter)

        return tab

    @staticmethod
    def _make_cluster_table() -> QTableWidget:
        """Create a cluster summary table widget."""
        t = QTableWidget()
        t.setColumnCount(5)
        t.setHorizontalHeaderLabels([
            "#", "Size", "Coherence", "Centroid cos(\u03b2)", "Top Words",
        ])
        t.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        t.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        t.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        t.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        t.setAlternatingRowColors(True)
        t.setSelectionBehavior(QTableWidget.SelectRows)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        return t

    # ------------------------------------------------------------------ #
    #  Tab 4 — Semantic Poles
    # ------------------------------------------------------------------ #

    def _create_poles_tab(self) -> QWidget:
        """Create the semantic poles tab."""
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)

        ctrl = QHBoxLayout()
        ctrl.addWidget(self._make_pair_selector())
        ctrl.addStretch()
        outer.addLayout(ctrl)

        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)

        # Positive pole
        self._pos_pole_group = QGroupBox("Positive End (+\u03b2) \u2014 Higher Outcome")
        pos_layout = QVBoxLayout()
        self._pos_pole_desc = QLabel("Words most aligned with higher outcome values:")
        self._pos_pole_desc.setObjectName("label_muted")
        pos_layout.addWidget(self._pos_pole_desc)
        self.pos_table = QTableWidget()
        self.pos_table.setColumnCount(3)
        self.pos_table.setHorizontalHeaderLabels(["Rank", "Word", "cos(\u03b2)"])
        self.pos_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.pos_table.setAlternatingRowColors(True)
        self.pos_table.setEditTriggers(QTableWidget.NoEditTriggers)
        pos_layout.addWidget(self.pos_table)
        self._pos_pole_group.setLayout(pos_layout)
        layout.addWidget(self._pos_pole_group)

        # Negative pole
        self._neg_pole_group = QGroupBox("Negative End (\u2212\u03b2) \u2014 Lower Outcome")
        neg_layout = QVBoxLayout()
        self._neg_pole_desc = QLabel("Words most aligned with lower outcome values:")
        self._neg_pole_desc.setObjectName("label_muted")
        neg_layout.addWidget(self._neg_pole_desc)
        self.neg_table = QTableWidget()
        self.neg_table.setColumnCount(3)
        self.neg_table.setHorizontalHeaderLabels(["Rank", "Word", "cos(\u03b2)"])
        self.neg_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.neg_table.setAlternatingRowColors(True)
        self.neg_table.setEditTriggers(QTableWidget.NoEditTriggers)
        neg_layout.addWidget(self.neg_table)
        self._neg_pole_group.setLayout(neg_layout)
        layout.addWidget(self._neg_pole_group)

        outer.addWidget(body, stretch=1)
        return tab

    # ------------------------------------------------------------------ #
    #  Tab 3 — Text Snippets
    # ------------------------------------------------------------------ #

    def _create_snippets_tab(self) -> QWidget:
        """Create the beta text snippets tab."""
        tab = QWidget()
        self._snippets_tab = tab
        layout = QVBoxLayout(tab)

        # Controls
        controls = QHBoxLayout()
        controls.addWidget(self._make_pair_selector())
        controls.addWidget(QLabel("Side:"))
        self.snippet_side_combo = QComboBox()
        self.snippet_side_combo.addItems([
            "Positive (+\u03b2)",
            "Negative (\u2212\u03b2)",
        ])
        self.snippet_side_combo.currentIndexChanged.connect(self._load_snippets_tab)
        controls.addWidget(self.snippet_side_combo)

        controls.addStretch()

        self._snippet_tab_desc = QLabel(
            "Snippets ranked by alignment with the \u03b2 direction (not clustered)"
        )
        self._snippet_tab_desc.setObjectName("label_muted")
        controls.addWidget(self._snippet_tab_desc)

        layout.addLayout(controls)

        # Vertical splitter for table and detail
        snippets_splitter = QSplitter(Qt.Vertical)

        # Snippets table
        self.snippets_table = QTableWidget()
        self.snippets_table.setColumnCount(4)
        self.snippets_table.setHorizontalHeaderLabels([
            "Seed", "Cosine", "Doc ID", "Snippet",
        ])
        self.snippets_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Stretch
        )
        self.snippets_table.setAlternatingRowColors(True)
        self.snippets_table.setWordWrap(True)
        self.snippets_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.snippets_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.snippets_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.snippets_table.itemSelectionChanged.connect(self._on_snippet_tab_selected)
        snippets_splitter.addWidget(self.snippets_table)

        # Snippet detail
        detail_group = QGroupBox("Selected Snippet Detail")
        detail_layout = QVBoxLayout()
        self.snippet_detail = QTextEdit()
        self.snippet_detail.setReadOnly(True)
        detail_layout.addWidget(self.snippet_detail)
        detail_group.setLayout(detail_layout)
        snippets_splitter.addWidget(detail_group)

        snippets_splitter.setSizes([400, 200])
        layout.addWidget(snippets_splitter, stretch=1)

        return tab

    # ------------------------------------------------------------------ #
    #  Tab 5 — Document Scores
    # ------------------------------------------------------------------ #

    def _create_scores_tab(self) -> QWidget:
        """Create the document scores tab — table on left, text detail on right."""
        tab = QWidget()
        self._scores_tab = tab
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)

        # Controls row
        controls = QHBoxLayout()
        controls.setContentsMargins(4, 4, 4, 0)
        controls.addWidget(self._make_pair_selector())
        controls.addWidget(QLabel("Sort by:"))
        self.scores_sort_combo = QComboBox()
        self.scores_sort_combo.addItems([
            "Document Index",
            "Cosine (High \u2192 Low)",
            "Cosine (Low \u2192 High)",
            "Predicted (High \u2192 Low)",
            "Predicted (Low \u2192 High)",
        ])
        self.scores_sort_combo.currentIndexChanged.connect(self._sort_scores)
        controls.addWidget(self.scores_sort_combo)

        controls.addStretch()

        outer.addLayout(controls)

        # Splitter: table (left) | document text (right)
        splitter = QSplitter(Qt.Horizontal)

        self.scores_table = QTableWidget()
        self.scores_table.setAlternatingRowColors(True)
        self.scores_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.scores_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.scores_table.itemSelectionChanged.connect(self._on_score_row_selected)
        splitter.addWidget(self.scores_table)

        # Right side: document text detail
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 4, 4, 4)

        self.scores_detail_title = QLabel("Select a row to view its document text")
        self.scores_detail_title.setObjectName("label_title")
        self.scores_detail_title.setWordWrap(True)
        right_layout.addWidget(self.scores_detail_title)

        self.scores_detail = QTextEdit()
        self.scores_detail.setReadOnly(True)
        right_layout.addWidget(self.scores_detail, stretch=1)

        splitter.addWidget(right)
        splitter.setSizes([400, 600])

        outer.addWidget(splitter, stretch=1)

        return tab

    # ------------------------------------------------------------------ #
    #  Tab 6 — Extreme Documents
    # ------------------------------------------------------------------ #

    def _create_extreme_docs_tab(self) -> QWidget:
        """Create the extreme documents tab — highest/lowest scoring docs."""
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)

        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(4, 4, 4, 0)
        ctrl.addWidget(self._make_pair_selector())
        ctrl.addStretch()
        outer.addLayout(ctrl)

        splitter = QSplitter(Qt.Vertical)

        # Highest scoring
        high_group = QGroupBox("Highest Scoring")
        high_layout = QVBoxLayout()
        self._extreme_high_table = QTableWidget()
        self._extreme_high_table.setAlternatingRowColors(True)
        self._extreme_high_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._extreme_high_table.setSelectionBehavior(QTableWidget.SelectRows)
        high_layout.addWidget(self._extreme_high_table)
        high_group.setLayout(high_layout)
        splitter.addWidget(high_group)

        # Lowest scoring
        low_group = QGroupBox("Lowest Scoring")
        low_layout = QVBoxLayout()
        self._extreme_low_table = QTableWidget()
        self._extreme_low_table.setAlternatingRowColors(True)
        self._extreme_low_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._extreme_low_table.setSelectionBehavior(QTableWidget.SelectRows)
        low_layout.addWidget(self._extreme_low_table)
        low_group.setLayout(low_layout)
        splitter.addWidget(low_group)

        outer.addWidget(splitter, stretch=1)
        return tab

    # ------------------------------------------------------------------ #
    #  Tab 7 — Misdiagnosed Documents
    # ------------------------------------------------------------------ #

    def _create_misdiagnosed_tab(self) -> QWidget:
        """Create the misdiagnosed documents tab — over/under-predicted docs."""
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Vertical)

        # Over-predicted
        over_group = QGroupBox("Over-predicted")
        over_layout = QVBoxLayout()
        self._misdiag_over_table = QTableWidget()
        self._misdiag_over_table.setAlternatingRowColors(True)
        self._misdiag_over_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._misdiag_over_table.setSelectionBehavior(QTableWidget.SelectRows)
        over_layout.addWidget(self._misdiag_over_table)
        over_group.setLayout(over_layout)
        splitter.addWidget(over_group)

        # Under-predicted
        under_group = QGroupBox("Under-predicted")
        under_layout = QVBoxLayout()
        self._misdiag_under_table = QTableWidget()
        self._misdiag_under_table.setAlternatingRowColors(True)
        self._misdiag_under_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._misdiag_under_table.setSelectionBehavior(QTableWidget.SelectRows)
        under_layout.addWidget(self._misdiag_under_table)
        under_group.setLayout(under_layout)
        splitter.addWidget(under_group)

        outer.addWidget(splitter, stretch=1)
        return tab

    # ------------------------------------------------------------------ #
    #  Tab 1 — Configuration / Details
    # ------------------------------------------------------------------ #

    def _create_config_tab(self) -> QWidget:
        """Create the configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        # Horizontal splitter for the three config sections side by side
        config_splitter = QSplitter(Qt.Horizontal)

        result_group = QGroupBox("Result Information")
        result_layout = QVBoxLayout()
        self.result_info_text = QTextEdit()
        self.result_info_text.setReadOnly(True)
        result_layout.addWidget(self.result_info_text)
        result_group.setLayout(result_layout)
        config_splitter.addWidget(result_group)

        self.concept_group = QGroupBox("Concept Configuration")
        concept_layout = QVBoxLayout()
        self.concept_config_text = QTextEdit()
        self.concept_config_text.setReadOnly(True)
        concept_layout.addWidget(self.concept_config_text)
        self.concept_group.setLayout(concept_layout)
        config_splitter.addWidget(self.concept_group)

        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()
        self.model_config_text = QTextEdit()
        self.model_config_text.setReadOnly(True)
        model_layout.addWidget(self.model_config_text)
        model_group.setLayout(model_layout)
        config_splitter.addWidget(model_group)

        config_splitter.setSizes([250, 350, 400])
        layout.addWidget(config_splitter, stretch=1)

        return tab

    # ------------------------------------------------------------------ #
    #  Tab 2 — PCA Sweep
    # ------------------------------------------------------------------ #

    def _create_pca_sweep_tab(self) -> QWidget:
        """Create the PCA Sweep tab showing the sweep summary figure."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Header row: info label + zoom controls
        header = QHBoxLayout()

        self.pca_sweep_info = QLabel()
        self.pca_sweep_info.setWordWrap(True)
        header.addWidget(self.pca_sweep_info, stretch=1)

        zoom_out_btn = QPushButton("\u2212")  # minus sign
        zoom_out_btn.setFixedSize(28, 28)
        zoom_out_btn.setToolTip("Zoom out")
        zoom_out_btn.clicked.connect(lambda: self._pca_sweep_zoom(-10))
        header.addWidget(zoom_out_btn)

        self._pca_sweep_zoom_label = QLabel("Fit")
        self._pca_sweep_zoom_label.setFixedWidth(44)
        self._pca_sweep_zoom_label.setAlignment(Qt.AlignCenter)
        header.addWidget(self._pca_sweep_zoom_label)

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedSize(28, 28)
        zoom_in_btn.setToolTip("Zoom in")
        zoom_in_btn.clicked.connect(lambda: self._pca_sweep_zoom(10))
        header.addWidget(zoom_in_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.setFixedWidth(48)
        reset_btn.setToolTip("Reset to fit window")
        reset_btn.clicked.connect(self._pca_sweep_zoom_reset)
        header.addWidget(reset_btn)

        layout.addLayout(header)

        # Scrollable image area
        self._pca_sweep_scroll = QScrollArea()
        self._pca_sweep_scroll.setWidgetResizable(True)
        self._pca_sweep_scroll.setFrameShape(QFrame.NoFrame)

        self.pca_sweep_image = QLabel()
        self.pca_sweep_image.setAlignment(Qt.AlignCenter)
        self._pca_sweep_scroll.setWidget(self.pca_sweep_image)
        layout.addWidget(self._pca_sweep_scroll, stretch=1)

        # Internal state
        self._pca_sweep_pixmap: Optional[QPixmap] = None
        self._pca_sweep_zoom_pct: int = 0  # 0 means fit-to-window

        return tab

    # ------------------------------------------------------------------ #
    #  Action bar
    # ------------------------------------------------------------------ #

    def _create_actions_section(self, parent_layout):
        """Create the actions section."""
        actions_layout = QHBoxLayout()

        back_btn = QPushButton("\u2039  Back to Concept")
        back_btn.setObjectName("btn_secondary")
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.clicked.connect(self._go_back)
        actions_layout.addWidget(back_btn)

        new_run_btn = QPushButton("New Analysis Run")
        new_run_btn.setObjectName("btn_ghost")
        new_run_btn.setCursor(Qt.PointingHandCursor)
        new_run_btn.clicked.connect(self._new_run)
        actions_layout.addWidget(new_run_btn)

        actions_layout.addStretch()

        report_settings_btn = QPushButton("Report Settings")
        report_settings_btn.setObjectName("btn_ghost")
        report_settings_btn.setCursor(Qt.PointingHandCursor)
        report_settings_btn.clicked.connect(self._open_report_settings)
        actions_layout.addWidget(report_settings_btn)

        # Save controls (visible for unsaved results)
        self.run_name_input = QLineEdit()
        self.run_name_input.setPlaceholderText("Name (optional)")
        self.run_name_input.setMinimumWidth(160)
        actions_layout.addWidget(self.run_name_input)

        self.save_result_btn = QPushButton("Save Result")
        self.save_result_btn.setMinimumHeight(44)
        self.save_result_btn.setMinimumWidth(200)
        self.save_result_btn.setObjectName("btn_export")
        self.save_result_btn.setCursor(Qt.PointingHandCursor)
        self.save_result_btn.clicked.connect(self._save_result)
        actions_layout.addWidget(self.save_result_btn)

        # Initially hidden
        self.run_name_input.hide()
        self.save_result_btn.hide()

        parent_layout.addLayout(actions_layout)

    # ================================================================== #
    #  DATA LOADING
    # ================================================================== #

    def _populate_result_selector(self):
        """Populate the result selector dropdown.

        Order: unsaved (in-memory only) first, then tracked + orphan
        entries from project.results in their stored order (tracked in
        project.json order, then orphans newest-first — see load_project).
        """
        self.result_selector.blockSignals(True)
        self.result_selector.clear()

        if self._unsaved_result is not None:
            self.result_selector.addItem(
                self._unsaved_label(self._unsaved_result), self._unsaved_result,
            )

        if self.project and self.project.results:
            for result in reversed(self.project.results):
                self.result_selector.addItem(_result_dropdown_label(result), result)

        self.result_selector.blockSignals(False)

        if self.result_selector.count() > 0:
            self.result_selector.setCurrentIndex(0)
            self._on_result_selected(0)

    @staticmethod
    def _unsaved_label(result: Result) -> str:
        """Label for the in-memory unsaved slot (fresh run or imported)."""
        if result.name:
            return f"[Imported: {result.name}]"
        return "[Unsaved Result]"

    def _on_result_selected(self, index: int):
        """Handle result selection change."""
        if index < 0:
            return
        result = self.result_selector.itemData(index)
        if result is None:
            return

        # Unsaved in-memory slot (fresh run or just-imported result)
        if result is self._unsaved_result:
            self._is_viewing_unsaved = True
            self.run_name_input.show()
            self.save_result_btn.setText("Save Result")
            self.save_result_btn.setToolTip("")
            self.save_result_btn.show()
            self.show_result(result)
            return

        self._is_viewing_unsaved = False
        self.run_name_input.hide()

        # Missing folder — nothing to show; only the X button can act on it.
        if result.status == "missing" or result.result_path is None:
            self.save_result_btn.hide()
            self.current_result = result
            QMessageBox.warning(
                self, "Result Missing",
                "This result's folder is no longer on disk.\n\n"
                "Use the X in the dropdown to remove it from the project.",
            )
            return

        # Broken folder — can't render, but let the user delete it.
        if result.load_error or (result.status == "error" and result._result is None):
            self.save_result_btn.hide()
            self.current_result = result
            QMessageBox.warning(
                self, "Result Could Not Be Loaded",
                f"Failed to load this result's data:\n{result.load_error or 'unknown error'}",
            )
            return

        # Orphan — promote to tracked on first open so Save Project persists it.
        if result.is_orphan:
            result.is_orphan = False
            if self.project is not None:
                self.project.mark_dirty()
            self.result_selector.setItemText(index, _result_dropdown_label(result))

        if result._result is not None:
            self.save_result_btn.setText("Overwrite")
            self.save_result_btn.setToolTip(
                "Re-save this result in place (updates report / files)"
            )
            self.save_result_btn.show()
        else:
            self.save_result_btn.hide()

        self.show_result(result)

    # ================================================================== #
    #  SHOW RESULT — main entry point for displaying a result
    # ================================================================== #

    def show_result(self, result: Result):
        """Display a specific result."""
        self.current_result = result
        self._current_pair_key = None

        if not result._result:
            QMessageBox.warning(
                self, "No Results",
                "This result does not have data (may have failed).",
            )
            return

        self._overlay_tabs.start()
        QApplication.processEvents()

        ssd_result = result._result

        # Re-attach shared project embeddings if missing (stripped during save)
        if getattr(ssd_result, "embeddings", None) is None and self.project._emb is not None:
            ssd_result.embeddings = self.project._emb

        is_crossgroup = isinstance(ssd_result, GroupResult)
        is_pca_ols = isinstance(ssd_result, PCAOLSResult)

        # Analysis type badge — theme-aware; pulls semantic colors from the current palette
        p = self._html_palette()
        if isinstance(ssd_result, PLSResult):
            badge_text, fg_color, bg_color = "PLS", p.accent, p.accent_muted
        elif isinstance(ssd_result, PCAOLSResult):
            badge_text, fg_color, bg_color = "PCA+OLS", p.success, p.success_bg
        elif isinstance(ssd_result, GroupResult):
            badge_text, fg_color, bg_color = "Groups", p.warning, p.warning_bg
        else:
            badge_text, fg_color, bg_color = "?", p.text_secondary, p.bg_elevated
        self._analysis_badge.setText(badge_text)
        self._analysis_badge.setStyleSheet(
            f"QLabel {{"
            f"  padding: 3px 10px;"
            f"  border-radius: 4px;"
            f"  font-weight: 600;"
            f"  font-size: 11px;"
            f"  background-color: {bg_color};"
            f"  color: {fg_color};"
            f"  border: 1px solid {fg_color};"
            f"}}"
        )
        self._analysis_badge.show()

        # Configure tabs for analysis type
        self.tabs.setTabVisible(2, is_pca_ols)  # PCA Sweep — only for PCA+OLS
        self.tabs.setTabVisible(5, True)   # Document Scores
        self.tabs.setTabVisible(6, True)   # Extreme Docs
        self.tabs.setTabVisible(7, not is_crossgroup)  # Misdiagnosed — not for groups
        self.tabs.setTabText(3, "Contrast Snippets" if is_crossgroup else "Beta Snippets")

        # Populate per-tab pair selectors (Cluster Overview, Contrast Snippets,
        # Semantic Poles, Document Scores, Extreme Documents). All combos are
        # kept in sync via _on_pair_changed.
        if is_crossgroup and hasattr(ssd_result, "pairs") and ssd_result.pairs:
            labels = ssd_result.group_labels or {}
            items: List[Tuple[str, Tuple[str, str]]] = []
            for pr in ssd_result.pairs:
                key = (pr.g1, pr.g2)
                lbl1 = labels.get(pr.g1, pr.g1)
                lbl2 = labels.get(pr.g2, pr.g2)
                if lbl1 != pr.g1 or lbl2 != pr.g2:
                    text = f"{pr.g1} vs {pr.g2}  \u2014  {lbl1} vs {lbl2}"
                else:
                    text = f"{pr.g1} vs {pr.g2}"
                items.append((text, key))
            for combo in self._pair_combos:
                combo.blockSignals(True)
                combo.clear()
                for text, key in items:
                    combo.addItem(text, userData=key)
                combo.setCurrentIndex(0)
                combo.blockSignals(False)
            self._current_pair_key = (ssd_result.pairs[0].g1, ssd_result.pairs[0].g2)
            show_selectors = len(ssd_result.pairs) > 1
            for frame in self._pair_frames:
                frame.setVisible(show_selectors)
        else:
            for frame in self._pair_frames:
                frame.hide()

        # Update stats strip
        if is_crossgroup:
            self._show_crossgroup_stats(ssd_result)
        else:
            self._show_continuous_stats(ssd_result)

        # Update group-specific labels
        if is_crossgroup:
            self._update_crossgroup_labels()
        else:
            self._reset_continuous_labels()

        # Update scores sort combo for analysis type
        self._update_scores_sort_combo(is_crossgroup)

        # Resolve the display-data object (result itself or active ContrastResult)
        display = self._get_display_data()

        # Populate every tab, pumping events between each so the spinner animates
        self._load_cluster_overview(display)
        QApplication.processEvents()
        self._load_poles_tab(display)
        QApplication.processEvents()
        self._load_snippets_tab()
        QApplication.processEvents()
        self._load_scores_tab(ssd_result)
        QApplication.processEvents()
        self._load_config_tab(result)
        QApplication.processEvents()
        if is_pca_ols:
            self._load_pca_sweep_tab(result)
        self._load_extreme_docs_tab(ssd_result)
        QApplication.processEvents()
        if not is_crossgroup:
            self._load_misdiagnosed_tab(ssd_result)
            QApplication.processEvents()

        self._overlay_tabs.stop()

    # ------------------------------------------------------------------ #
    #  Stats strip population
    # ------------------------------------------------------------------ #

    def _show_continuous_stats(self, results):
        """Populate the stats strip for continuous analysis (PLS or PCA+OLS)."""
        p_str = f"{results.stats.pvalue:.2e}" if results.stats.pvalue < 0.001 else f"{results.stats.pvalue:.4f}"

        if isinstance(results, PCAOLSResult):
            # PCA+OLS: R² | Adj R² | p-value | β‖ | Docs | Selected K | Corr(y,ŷ)
            self._set_stat_card(0, "R\u00b2", f"{results.stats.r2:.4f}")
            self._set_stat_card(1, "Adj. R\u00b2", f"{results.stats.r2_adj:.4f}")
            self._set_stat_card(2, "p-value", p_str)
            self._set_stat_card(3, "\u2016\u03b2\u2016", f"{results.stats.beta_norm:.4f}")
            self._set_stat_card(4, "Documents Used", f"{results.stats.n_kept:,}")
            k_text = str(results.pca_k) if results.pca_k else "\u2014"
            self._set_stat_card(5, "Selected K", k_text)
            self._set_stat_card(6, "Corr(y, \u0177)", f"{results.stats.y_corr_pred:.4f}")
        else:
            # PLS: R² | p-value | β‖ | Δy | Docs | IQR Effect | Corr(y,ŷ)
            self._set_stat_card(0, "R\u00b2", f"{results.stats.r2:.4f}")
            self._set_stat_card(1, "p-value", p_str)
            self._set_stat_card(2, "\u2016\u03b2\u2016", f"{results.stats.beta_norm:.4f}")
            self._set_stat_card(3, "\u0394y / +0.10 cos", f"{results.stats.delta:.4f}")
            self._set_stat_card(4, "Documents Used", f"{results.stats.n_kept:,}")
            self._set_stat_card(5, "IQR Effect", f"{results.stats.iqr_effect:.4f}")
            self._set_stat_card(6, "Corr(y, \u0177)", f"{results.stats.y_corr_pred:.4f}")

    def _pair_for_key(self, result, pair_key):
        """Return the Pair dataclass for a canonical pair tuple, or None."""
        if not pair_key or not hasattr(result, "pairs") or not result.pairs:
            return None
        for p in result.pairs:
            if (p.g1, p.g2) == tuple(pair_key):
                return p
        return None

    def _show_crossgroup_stats(self, results):
        """Populate the stats strip for group comparison analysis."""
        cr = self._pair_for_key(results, self._current_pair_key)
        is_multi = len(results.pairs) > 1

        def _fmt_p(v):
            if v is None:
                return "\u2014"
            return f"{v:.2e}" if v < 0.001 else f"{v:.4f}"

        if cr is None:
            for i in range(7):
                self._set_stat_card(i, "", "")
            return

        if is_multi:
            self._set_stat_card(0, "Omnibus p", _fmt_p(results.test.omnibus_p))
            self._set_stat_card(1, "p (corrected)", _fmt_p(cr.p_corrected))
            self._set_stat_card(2, "Cohen's d", f"{cr.cohens_d:+.3f}")
            self._set_stat_card(3, "\u2016Contrast\u2016", f"{cr.contrast_norm:.4f}")
            self._set_stat_card(4, "Documents Used", f"{results.stats.n_kept:,}")
            self._set_stat_card(5, "n (g1)", f"{cr.n_g1:,}")
            self._set_stat_card(6, "n (g2)", f"{cr.n_g2:,}")
        else:
            self._set_stat_card(0, "p-value", _fmt_p(cr.p_raw))
            self._set_stat_card(1, "Cohen's d", f"{cr.cohens_d:+.3f}")
            self._set_stat_card(2, "\u2016Contrast\u2016", f"{cr.contrast_norm:.4f}")
            self._set_stat_card(3, "Permutations", f"{results.stats.n_perm:,}")
            self._set_stat_card(4, "Documents Used", f"{results.stats.n_kept:,}")
            self._set_stat_card(5, "n (g1)", f"{cr.n_g1:,}")
            self._set_stat_card(6, "n (g2)", f"{cr.n_g2:,}")

    # ------------------------------------------------------------------ #
    #  Crossgroup UI helpers
    # ------------------------------------------------------------------ #

    def _update_crossgroup_labels(self):
        """Update UI labels for the current crossgroup contrast."""
        if not self._current_pair_key:
            return

        result = self.current_result._result if self.current_result else None
        cr = self._pair_for_key(result, self._current_pair_key) if result else None
        if cr is not None:
            labels_map = result.group_labels or {}
            g1_name = labels_map.get(cr.g1, cr.g1)
            g2_name = labels_map.get(cr.g2, cr.g2)
        else:
            g1_name, g2_name = "Group A", "Group B"

        # Cluster overview groupboxes
        self._ov_pos_group.setTitle(f"Positive Direction  \u2192  {g1_name}")
        self._ov_neg_group.setTitle(f"Negative Direction  \u2192  {g2_name}")

        # Semantic poles groupboxes
        self._pos_pole_group.setTitle(f"{g1_name} Direction")
        self._neg_pole_group.setTitle(f"{g2_name} Direction")
        self._pos_pole_desc.setText(f"Words most aligned with {g1_name}:")
        self._neg_pole_desc.setText(f"Words most aligned with {g2_name}:")

        # Snippets tab side combo
        self.snippet_side_combo.blockSignals(True)
        self.snippet_side_combo.clear()
        self.snippet_side_combo.addItems([
            f"{g1_name} direction",
            f"{g2_name} direction",
        ])
        self.snippet_side_combo.blockSignals(False)

        self._snippet_tab_desc.setText(
            "Snippets ranked by alignment with the contrast direction (not clustered)"
        )

    def _reset_continuous_labels(self):
        """Reset UI labels for continuous analysis."""
        self._ov_pos_group.setTitle("Positive Clusters  (+\u03b2  \u2192  higher outcome)")
        self._ov_neg_group.setTitle("Negative Clusters  (\u2212\u03b2  \u2192  lower outcome)")

        self._pos_pole_group.setTitle("Positive End (+\u03b2) \u2014 Higher Outcome")
        self._neg_pole_group.setTitle("Negative End (\u2212\u03b2) \u2014 Lower Outcome")
        self._pos_pole_desc.setText("Words most aligned with higher outcome values:")
        self._neg_pole_desc.setText("Words most aligned with lower outcome values:")

        self.snippet_side_combo.blockSignals(True)
        self.snippet_side_combo.clear()
        self.snippet_side_combo.addItems([
            "Positive (+\u03b2)",
            "Negative (\u2212\u03b2)",
        ])
        self.snippet_side_combo.blockSignals(False)

        self._snippet_tab_desc.setText(
            "Snippets ranked by alignment with the \u03b2 direction (not clustered)"
        )

    def _get_display_data(self):
        """Return a namespace whose fields drive the current display.

        For continuous analyses: data comes from the result directly.
        For groups: data is filtered to the active contrast.

        Memoized on (result identity, contrast key); a change in either
        automatically misses the cache, so no explicit invalidation is needed.
        """
        from types import SimpleNamespace
        result = self.current_result._result if self.current_result else None
        if result is None:
            return None

        cache_key = (id(result), self._current_pair_key)
        cached = getattr(self, "_display_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        is_group = isinstance(result, GroupResult)
        target = None
        if is_group and self._current_pair_key and len(result.pairs) > 1:
            target = tuple(self._current_pair_key)

        words_view = result.words
        clusters_index = result.clusters
        if target is not None:
            try:
                words_view = result.words[target]
                clusters_index = result.clusters[target]
            except (KeyError, TypeError):
                pass

        # Match the runner's params so we hit the same ssdiff cache entries;
        # otherwise cluster_ids on clusters vs. snippets diverge.
        p = self.project
        cluster_kwargs = {}
        snippet_kwargs = {"top_per_side": 200}
        if p is not None:
            k = None if p.clustering_k_auto else p.clustering_k_min
            cluster_kwargs = {
                "topn": p.clustering_topn,
                "k": k,
                "k_min": p.clustering_k_min,
                "k_max": p.clustering_k_max,
            }

        def _sided(attr: str):
            base = getattr(clusters_index, attr, None)
            if base is None:
                return None
            if cluster_kwargs:
                try:
                    return base(**cluster_kwargs)
                except Exception:
                    pass
            return base

        pos_clusters_view = _sided("pos")
        neg_clusters_view = _sided("neg")

        try:
            snippets_view = result.snippets(**snippet_kwargs)
        except Exception:
            snippets_view = result.snippets
        if target is not None:
            try:
                snippets_view = result.snippets[target]
            except (KeyError, TypeError):
                pass

        # -- top words / poles --
        pos_neighbors = []
        neg_neighbors = []
        try:
            for w in words_view:
                if w.side == "pos":
                    pos_neighbors.append({"side": "pos", "word": w.word, "cos_beta": w.cos_beta, "rank": w.rank})
                elif w.side == "neg":
                    neg_neighbors.append({"side": "neg", "word": w.word, "cos_beta": w.cos_beta, "rank": w.rank})
        except Exception:
            pass

        clusters_summary = (
            _clusters_to_summary(pos_clusters_view, "pos")
            + _clusters_to_summary(neg_clusters_view, "neg")
        )
        clusters_members = (
            _clusters_to_members(pos_clusters_view, "pos")
            + _clusters_to_members(neg_clusters_view, "neg")
        )

        # -- snippets --
        snippets_pos = []
        snippets_neg = []
        try:
            for s in snippets_view:
                row = {
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
                }
                if s.side == "pos":
                    snippets_pos.append(row)
                elif s.side == "neg":
                    snippets_neg.append(row)
        except Exception:
            pass

        # -- cluster snippets: centroid-based per-cluster extractor --
        cluster_snippets_pos = []
        cluster_snippets_neg = []

        def _snippets_view_to_rows(view):
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

        def _cluster_rows(clusters_view):
            if clusters_view is None or getattr(clusters_view, "_parent", None) is None:
                return []
            try:
                return _snippets_view_to_rows(clusters_view.snippets(k=None))
            except Exception:
                return []
        cluster_snippets_pos = _cluster_rows(pos_clusters_view)
        cluster_snippets_neg = _cluster_rows(neg_clusters_view)

        display = SimpleNamespace(
            pos_neighbors=pos_neighbors,
            neg_neighbors=neg_neighbors,
            clusters_summary=clusters_summary,
            clusters_members=clusters_members,
            snippets_pos=snippets_pos,
            snippets_neg=snippets_neg,
            cluster_snippets_pos=cluster_snippets_pos,
            cluster_snippets_neg=cluster_snippets_neg,
        )
        self._display_cache = (cache_key, display)
        return display

    def _on_pair_changed(self, index: int):
        """Handle pair selection change from any per-tab combo; sync all combos."""
        if index < 0 or not self.current_result or not self.current_result._result:
            return
        result = self.current_result._result
        if not hasattr(result, "pairs") or not result.pairs:
            return

        sender = self.sender()
        pair_key = sender.itemData(index) if sender is not None else None
        if pair_key is None:
            return
        known = {(p.g1, p.g2) for p in result.pairs}
        if pair_key not in known:
            return
        self._current_pair_key = tuple(pair_key)

        for combo in self._pair_combos:
            if combo is sender:
                continue
            combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)

        self._show_crossgroup_stats(result)
        self._update_crossgroup_labels()

        self._overlay_tabs.start()
        QApplication.processEvents()

        display = self._get_display_data()
        self._load_cluster_overview(display)
        QApplication.processEvents()
        self._load_poles_tab(display)
        QApplication.processEvents()
        self._load_snippets_tab()
        QApplication.processEvents()
        self._load_scores_tab(result)
        QApplication.processEvents()
        self._load_extreme_docs_tab(result)
        QApplication.processEvents()

        self._overlay_tabs.stop()

    def _update_scores_sort_combo(self, is_crossgroup: bool):
        """Update the scores sort combo for the current analysis type."""
        self.scores_sort_combo.blockSignals(True)
        self.scores_sort_combo.clear()
        if not is_crossgroup:
            self.scores_sort_combo.addItems([
                "Document Index",
                "Cosine (High \u2192 Low)",
                "Cosine (Low \u2192 High)",
                "Predicted (High \u2192 Low)",
                "Predicted (Low \u2192 High)",
            ])
        self.scores_sort_combo.blockSignals(False)

    # ------------------------------------------------------------------ #
    #  Cluster Overview helpers
    # ------------------------------------------------------------------ #

    def _load_cluster_overview(self, results):
        """Populate both cluster tables and clear snippet panel."""
        self._fill_cluster_table(
            self.ov_pos_table,
            [c for c in (results.clusters_summary or []) if c.get("side") == "pos"],
        )
        self._fill_cluster_table(
            self.ov_neg_table,
            [c for c in (results.clusters_summary or []) if c.get("side") == "neg"],
        )
        # Clear snippet panel
        self.snippet_panel_title.setText("Select a cluster to view its text snippets")
        self._current_member_words = []
        self._keywords_expanded = False
        self._update_keywords_display()
        self.ov_snippet_table.setRowCount(0)
        self.ov_snippet_detail.clear()

    def _fill_cluster_table(self, table: QTableWidget, clusters: list):
        """Fill a cluster summary QTableWidget from a list of cluster dicts."""
        topn = self.cluster_topn_spin.value()
        table.setRowCount(len(clusters))
        for i, c in enumerate(clusters):
            table.setItem(i, 0, QTableWidgetItem(str(c.get("cluster_rank", ""))))
            table.setItem(i, 1, QTableWidgetItem(str(c.get("size", ""))))
            table.setItem(
                i, 2, QTableWidgetItem(f"{c.get('coherence', 0):.3f}")
            )
            table.setItem(
                i, 3, QTableWidgetItem(f"{c.get('centroid_cos_beta', 0):.3f}")
            )
            # Truncate top_words to requested N
            top_words_str = c.get("top_words", "")
            words_list = [w.strip() for w in top_words_str.split(",") if w.strip()]
            table.setItem(i, 4, QTableWidgetItem(", ".join(words_list[:topn])))

    def _reload_cluster_tables(self):
        """Re-fill cluster tables when topN changes."""
        if not self.current_result or not self.current_result._result:
            return
        display = self._get_display_data()
        if display:
            self._load_cluster_overview(display)

    def _refresh_overview_snippets(self):
        """Re-fill the snippet table when the snippets-per-cluster limit changes."""
        if hasattr(self, "_ov_current_snippets") and self._ov_current_snippets:
            self._fill_overview_snippet_table(self._ov_current_snippets)

    def _on_overview_cluster_clicked(self, table: QTableWidget, side: str):
        """When a row is clicked in one of the two cluster tables, show its snippets."""
        # Deselect the other table
        other = self.ov_neg_table if side == "pos" else self.ov_pos_table
        other.clearSelection()

        row = table.currentRow()
        if row < 0 or not self.current_result or not self.current_result._result:
            return

        self._overlay_right_panel.start()
        QApplication.processEvents()

        display = self._get_display_data()
        if display is None:
            self._overlay_right_panel.stop()
            return

        clusters_for_side = [
            c for c in (display.clusters_summary or []) if c.get("side") == side
        ]
        if row >= len(clusters_for_side):
            self._overlay_right_panel.stop()
            return

        cluster_info = clusters_for_side[row]
        cluster_rank = cluster_info.get("cluster_rank")
        cluster_id = cluster_info.get("cluster_id")

        # Build member words string from members data
        members = [
            m for m in (display.clusters_members or [])
            if m.get("cluster_id") == cluster_id and m.get("side") == side
        ]
        member_words = [m.get("word", "") for m in members]

        side_label = "Positive" if side == "pos" else "Negative"
        self.snippet_panel_title.setText(
            f"{side_label} Cluster {cluster_rank}"
        )

        # Store member words and show collapsed view
        self._current_member_words = member_words
        self._keywords_expanded = False
        self._update_keywords_display()

        # Find matching snippets (filter from cluster-based snippets by cluster_id)
        snippets_pool = (
            display.cluster_snippets_pos if side == "pos"
            else display.cluster_snippets_neg
        ) or []

        snippets = [
            s for s in snippets_pool
            if s.get("cluster_id") == cluster_id
        ]
        self._ov_current_snippets = snippets
        self._fill_overview_snippet_table(snippets)
        self.ov_snippet_detail.clear()

        self._overlay_right_panel.stop()

    def _update_keywords_display(self):
        """Update the keywords label based on expanded/collapsed state."""
        words = self._current_member_words
        count = len(words)

        if count == 0:
            self.snippet_panel_keywords.setText("")
            self.keywords_toggle_link.setVisible(False)
            return

        if self._keywords_expanded:
            # Show all keywords with word wrap
            self.snippet_panel_keywords.setWordWrap(True)
            self.snippet_panel_keywords.setText(
                f"Members ({count}): {', '.join(words)} "
            )
            self.keywords_toggle_link.setText("see less")
            self.keywords_toggle_link.setVisible(True)
        else:
            # Show truncated (first ~8 words or what fits in one line)
            self.snippet_panel_keywords.setWordWrap(False)
            preview_count = min(8, count)
            preview = ", ".join(words[:preview_count])
            remaining = count - preview_count

            if remaining > 0:
                self.snippet_panel_keywords.setText(
                    f"Members ({count}): {preview}... "
                )
                self.keywords_toggle_link.setText(f"+{remaining} more")
                self.keywords_toggle_link.setVisible(True)
            else:
                self.snippet_panel_keywords.setText(
                    f"Members ({count}): {preview}"
                )
                self.keywords_toggle_link.setVisible(False)

    def _toggle_keywords_display(self):
        """Toggle between expanded and collapsed keywords view."""
        self._keywords_expanded = not self._keywords_expanded
        self._update_keywords_display()

    def _fill_overview_snippet_table(self, snippets: list):
        """Populate the overview snippet table."""
        from ..utils.report_settings import get_report_setting, KEY_SNIPPET_PREVIEW
        cap = get_report_setting(KEY_SNIPPET_PREVIEW)

        limit = min(len(snippets), self.cluster_snippets_spin.value())
        table = self.ov_snippet_table
        vheader = table.verticalHeader()

        # See _display_snippets_tab: defer per-row height measurement by
        # switching to Fixed during fill, restore ResizeToContents once at end.
        table.setUpdatesEnabled(False)
        table.blockSignals(True)
        vheader.setSectionResizeMode(QHeaderView.Fixed)
        try:
            table.setRowCount(limit)
            for i in range(limit):
                s = snippets[i]
                table.setItem(i, 0, QTableWidgetItem(str(s.get("seed", ""))))
                table.setItem(
                    i, 1, QTableWidgetItem(f"{s.get('cosine', 0):.4f}")
                )
                table.setItem(i, 2, QTableWidgetItem(str(s.get("doc_id", ""))))
                anchor = s.get("text_window", "")
                if cap and len(anchor) > cap:
                    anchor = anchor[:cap] + "\u2026"
                table.setItem(i, 3, QTableWidgetItem(anchor))
        finally:
            vheader.setSectionResizeMode(QHeaderView.ResizeToContents)
            table.blockSignals(False)
            table.setUpdatesEnabled(True)

    def _on_overview_snippet_selected(self):
        """Show full document text for the selected snippet."""
        row = self.ov_snippet_table.currentRow()
        if row < 0 or not hasattr(self, "_ov_current_snippets"):
            return
        if row >= len(self._ov_current_snippets):
            return

        snip = self._ov_current_snippets[row]
        self._show_snippet_detail(snip, self.ov_snippet_detail)

    # ------------------------------------------------------------------ #
    #  Poles tab helpers
    # ------------------------------------------------------------------ #

    def _load_poles_tab(self, results):
        """Load the poles tab data."""
        self._fill_poles_table(self.pos_table, results.pos_neighbors)
        self._fill_poles_table(self.neg_table, results.neg_neighbors)

    @staticmethod
    def _fill_poles_table(table: QTableWidget, neighbors: list):
        """Fill a poles table from a list of neighbor dicts."""
        table.setRowCount(len(neighbors))
        for i, row_data in enumerate(neighbors):
            table.setItem(i, 0, QTableWidgetItem(str(row_data.get("rank", i + 1))))
            table.setItem(i, 1, QTableWidgetItem(str(row_data.get("word", ""))))
            cos = row_data.get("cos", None)
            table.setItem(
                i, 2,
                QTableWidgetItem("" if cos is None else f"{float(cos):.4f}"),
            )

    # ------------------------------------------------------------------ #
    #  Snippets tab helpers
    # ------------------------------------------------------------------ #

    def _load_snippets_tab(self):
        """Load beta snippets based on current side selection."""
        if not self.current_result or not self.current_result._result:
            self.snippets_table.setRowCount(0)
            return

        self._overlay_snippets.start()
        QApplication.processEvents()

        display = self._get_display_data()
        idx = self.snippet_side_combo.currentIndex()

        if idx == 0:
            snippets = (display.snippets_pos if display else None) or []
        else:
            snippets = (display.snippets_neg if display else None) or []

        self._display_snippets_tab(snippets)

        self._overlay_snippets.stop()

    def _display_snippets_tab(self, snippets: list):
        """Display beta snippets in the snippets-tab table."""
        from ..utils.report_settings import get_report_setting, KEY_SNIPPET_PREVIEW
        cap = get_report_setting(KEY_SNIPPET_PREVIEW)

        limit = min(len(snippets), 500)
        table = self.snippets_table
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
                table.setItem(
                    i, 1, QTableWidgetItem(f"{s.get('cosine', 0):.4f}")
                )
                table.setItem(i, 2, QTableWidgetItem(str(s.get("doc_id", ""))))
                anchor = s.get("text_window", "")
                if cap and len(anchor) > cap:
                    anchor = anchor[:cap] + "\u2026"
                table.setItem(i, 3, QTableWidgetItem(anchor))
        finally:
            vheader.setSectionResizeMode(QHeaderView.ResizeToContents)
            table.blockSignals(False)
            table.setUpdatesEnabled(True)

        self.snippet_detail.clear()
        self._tab_displayed_snippets = snippets[:limit]

    def _on_snippet_tab_selected(self):
        """Handle snippet selection in the snippets tab."""
        row = self.snippets_table.currentRow()
        if row < 0 or not hasattr(self, "_tab_displayed_snippets"):
            return
        if row >= len(self._tab_displayed_snippets):
            return
        snip = self._tab_displayed_snippets[row]
        self._show_snippet_detail(snip, self.snippet_detail)

    # ------------------------------------------------------------------ #
    #  Shared helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _html_palette():
        """Return the current theme palette for HTML content styling."""
        from ..theme import build_current_palette
        return build_current_palette()

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters and preserve line breaks."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )

    def _show_snippet_detail(self, snip: dict, text_edit: QTextEdit):
        """Render a snippet dict into a QTextEdit with rich HTML."""
        p = self._html_palette()
        doc_id = snip.get("doc_id", "N/A")

        html = []
        html.append(
            '<table cellspacing="8" style="margin-bottom: 12px;"><tr>'
        )

        # Stats cards
        html.append(
            f'<td style="padding-right: 20px;">'
            f'<span style="color: {p.text_secondary}; font-size: {p.font_size_sm};">DOCUMENT</span><br/>'
            f'<span style="font-size: {p.font_size_lg}; font-weight: 600;">{doc_id}</span>'
            f'</td>'
        )
        html.append(
            f'<td style="padding-right: 20px;">'
            f'<span style="color: {p.text_secondary}; font-size: {p.font_size_sm};">SEED WORD</span><br/>'
            f'<span style="font-size: {p.font_size_lg}; font-weight: 600;">{snip.get("seed", "N/A")}</span>'
            f'</td>'
        )
        html.append(
            f'<td style="padding-right: 20px;">'
            f'<span style="color: {p.text_secondary}; font-size: {p.font_size_sm};">COSINE</span><br/>'
            f'<span style="font-size: {p.font_size_lg}; font-weight: 600;">{snip.get("cosine", 0):.4f}</span>'
            f'</td>'
        )
        cluster = snip.get("cluster_id")
        if cluster is not None:
            html.append(
                f'<td style="padding-right: 20px;">'
                f'<span style="color: {p.text_secondary}; font-size: {p.font_size_sm};">CLUSTER</span><br/>'
                f'<span style="font-size: {p.font_size_lg}; font-weight: 600;">{cluster}</span>'
                f'</td>'
            )
        html.append('</tr></table>')

        # Anchor snippet section
        anchor = snip.get("text_window", "")
        if anchor:
            html.append(
                f'<div style="border-top: 1px solid {p.border}; padding-top: 12px;">'
                f'<span style="color: {p.text_secondary}; font-size: {p.font_size_sm}; text-transform: uppercase;">Snippet Context</span>'
                f'</div>'
                f'<div style="margin-top: 8px; line-height: 1.5;">{self._escape_html(anchor)}</div>'
            )

        # Full document text section
        surface = snip.get("text_surface", "")
        if surface:
            html.append(
                f'<div style="border-top: 1px solid {p.border}; padding-top: 12px; margin-top: 12px;">'
                f'<span style="color: {p.text_secondary}; font-size: {p.font_size_sm}; text-transform: uppercase;">Full Document Text</span>'
                f'</div>'
                f'<div style="margin-top: 8px; line-height: 1.5;">{self._escape_html(surface)}</div>'
            )

        text_edit.setHtml("".join(html))

    # ------------------------------------------------------------------ #
    #  Scores tab helpers
    # ------------------------------------------------------------------ #

    def _load_scores_tab(self, results):
        """Load the scores tab data."""
        self.scores_detail.clear()
        self.scores_detail_title.setText("Select a row to view its document text")

        if isinstance(results, GroupResult):
            from ._pair_view import resolve_pair_data
            try:
                pair_view = resolve_pair_data(results, self._current_pair_key)
            except Exception:
                self.scores_table.setRowCount(0)
                self.scores_table.setColumnCount(0)
                self._scores_df = None
                return

            scores = pair_view.alignment_scores
            groups = results.groups
            labels_map = results.group_labels or {}
            rows = []
            for doc_id, (score, g) in enumerate(zip(scores, groups)):
                rows.append({
                    "idx": int(doc_id),
                    "group": labels_map.get(g, g),
                    "cos_align": float(score),
                })
            if not rows:
                self.scores_table.setRowCount(0)
                self.scores_table.setColumnCount(0)
                self._scores_df = None
                return

            df = pd.DataFrame(rows)
            self._scores_df = df
            col_labels = {"idx": "Doc #", "group": "Group", "cos_align": "Cosine"}
            display_cols = [col_labels.get(c, c) for c in df.columns]
            self.scores_table.setColumnCount(len(df.columns))
            self.scores_table.setHorizontalHeaderLabels(display_cols)
            self._render_scores_df(df)
            return

        try:
            n = results.stats.n_kept
            pos = list(results.docs.pos(k=n))
            neg = list(results.docs.neg(k=n))
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
            self.scores_table.setRowCount(0)
            self.scores_table.setColumnCount(0)
            self._scores_df = None
            return

        if not rows:
            self.scores_table.setRowCount(0)
            self.scores_table.setColumnCount(0)
            self._scores_df = None
            return

        df = pd.DataFrame(rows)
        self._scores_df = df

        col_labels = {
            "idx": "Doc #",
            "cos_align": "Cosine",
            "score_std": "Score (std)",
            "yhat_raw": "Predicted (raw)",
        }
        display_cols = [col_labels.get(c, c) for c in df.columns]
        self.scores_table.setColumnCount(len(df.columns))
        self.scores_table.setHorizontalHeaderLabels(display_cols)
        header = self.scores_table.horizontalHeader()
        for col_idx, col_name in enumerate(df.columns):
            if col_name in ("score_std", "yhat_raw"):
                header.resizeSection(col_idx, 120)
        self._render_scores_df(df)

    def _render_scores_df(self, df: pd.DataFrame):
        """Render a DataFrame into the scores table."""
        self._scores_rendered_df = df.reset_index(drop=True)
        table = self.scores_table
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

    def _sort_scores(self):
        """Sort the scores table."""
        if self._scores_df is None:
            return

        self._overlay_scores.start()
        QApplication.processEvents()

        idx = self.scores_sort_combo.currentIndex()
        df = self._scores_df.copy()

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

        self._render_scores_df(df)

        self._overlay_scores.stop()

    def _on_score_row_selected(self):
        """Show the document text for the selected score row."""
        row = self.scores_table.currentRow()
        if row < 0 or not hasattr(self, "_scores_rendered_df"):
            return
        df = self._scores_rendered_df
        if row >= len(df):
            return

        doc_index = int(df.iloc[row].get("idx", -1))
        if doc_index < 0:
            return

        # Build formatted title
        self.scores_detail_title.setText(f"Document {doc_index}")

        # Build HTML content with stats and document text
        p = self._html_palette()
        rec = df.iloc[row]
        doc_text = self._get_document_text(doc_index)

        html_parts = []
        html_parts.append(
            '<table cellspacing="8" style="margin-bottom: 12px;">'
            "<tr>"
        )

        # Stats row — continuous results: cosine, predicted, score
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

        # Document text section
        html_parts.append(
            f'<div style="border-top: 1px solid {p.border}; padding-top: 12px;">'
            f'<span style="color: {p.text_secondary}; font-size: {p.font_size_sm}; text-transform: uppercase;">Document Text</span>'
            f"</div>"
        )

        if doc_text:
            html_parts.append(
                f'<div style="margin-top: 8px; line-height: 1.5;">{self._escape_html(doc_text)}</div>'
            )
        else:
            html_parts.append(
                f'<div style="margin-top: 8px; color: {p.text_muted}; font-style: italic;">'
                f"Document text not available \u2014 CSV may not be loaded"
                f"</div>"
            )

        self.scores_detail.setHtml("".join(html_parts))

    def _load_extreme_docs_tab(self, ssd_result):
        """Populate the extreme documents tab."""
        from ..utils.report_settings import (
            get_report_setting, KEY_EXTREME_DOCS, KEY_SNIPPET_PREVIEW,
        )
        k = get_report_setting(KEY_EXTREME_DOCS)
        if k == 0:
            self._extreme_high_table.setRowCount(0)
            self._extreme_low_table.setRowCount(0)
            return

        if isinstance(ssd_result, GroupResult):
            from ._pair_view import resolve_pair_data
            try:
                pair_view = resolve_pair_data(ssd_result, self._current_pair_key)
            except Exception:
                return
            scores = pair_view.alignment_scores
            groups = ssd_result.groups
            labels_map = ssd_result.group_labels or {}
            order_desc = np.argsort(-scores)
            order_asc = np.argsort(scores)
            pos_idx = order_desc[:k]
            neg_idx = order_asc[:k]
            cap = get_report_setting(KEY_SNIPPET_PREVIEW)
            for table, idx_array in [(self._extreme_high_table, pos_idx),
                                      (self._extreme_low_table, neg_idx)]:
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
                table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
                table.resizeColumnsToContents()
                table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            return

        try:
            pos_docs = list(ssd_result.docs.pos(k=k))
            neg_docs = list(ssd_result.docs.neg(k=k))
        except Exception:
            return

        cap = get_report_setting(KEY_SNIPPET_PREVIEW)
        for table, side_docs in [(self._extreme_high_table, pos_docs),
                                  (self._extreme_low_table, neg_docs)]:
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
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            table.resizeColumnsToContents()
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

    def _load_misdiagnosed_tab(self, ssd_result):
        """Populate the misdiagnosed documents tab."""
        from ..utils.report_settings import (
            get_report_setting, KEY_MISDIAGNOSED, KEY_SNIPPET_PREVIEW,
        )
        k = get_report_setting(KEY_MISDIAGNOSED)
        if k == 0:
            self._misdiag_over_table.setRowCount(0)
            self._misdiag_under_table.setRowCount(0)
            return

        try:
            over_docs = list(ssd_result.docs.misdiagnosed(k=k, direction="over"))
            under_docs = list(ssd_result.docs.misdiagnosed(k=k, direction="under"))
        except Exception:
            return

        cap = get_report_setting(KEY_SNIPPET_PREVIEW)
        for table, side_docs in [(self._misdiag_over_table, over_docs),
                                  (self._misdiag_under_table, under_docs)]:
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

            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            table.resizeColumnsToContents()
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

    def _get_document_text(self, doc_index: int) -> Optional[str]:
        """Retrieve the original text for a document by its index."""
        if self.project is None:
            return None

        p = self.project
        if not p.csv_path or not p.text_column:
            return None

        # Use the cached DataFrame if available
        df = p._df
        if df is None:
            # Lazily load the CSV
            csv_path = Path(str(p.csv_path))
            if not csv_path.exists():
                return None
            try:
                df = pd.read_csv(csv_path)
                p._df = df
            except Exception:
                return None

        if p.text_column not in df.columns:
            return None
        if doc_index < 0 or doc_index >= len(df):
            return None

        value = df.iloc[doc_index][p.text_column]
        if pd.isna(value):
            return None
        return str(value)

    # ------------------------------------------------------------------ #
    #  Config tab helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_group_detail_html(gr, section_style: str, label_style: str, value_style: str, p) -> list:
        """Build HTML fragments for the GroupResult-specific Details sections.

        Covers: Omnibus, Group Sizes (from ``gr.groups`` + ``gr.group_labels``),
        and the per-pair table (from ``gr.pairs``). Returns a list of HTML
        strings ready to be joined and fed into a QTextEdit.
        """
        html = []
        n_groups = len(gr.group_labels) if gr.group_labels else gr.G
        test_label = "Omnibus Permutation Test" if n_groups > 2 else "Pairwise Permutation Test"
        html.append(f'<div style="{section_style}">{test_label}</div>')
        html.append('<table cellspacing="6" style="width: 100%;">')
        if gr.test.omnibus_T is not None:
            html.append(
                f'<tr><td style="{label_style}">Test Statistic (T)</td>'
                f'<td style="{value_style}">{gr.test.omnibus_T:.4f}</td></tr>'
            )
        if gr.test.omnibus_p is not None:
            p_str = f"{gr.test.omnibus_p:.2e}" if gr.test.omnibus_p < 0.001 else f"{gr.test.omnibus_p:.4f}"
            html.append(
                f'<tr><td style="{label_style}">p-value</td>'
                f'<td style="{value_style}">{p_str}</td></tr>'
            )
        html.append(
            f'<tr><td style="{label_style}">Permutations</td>'
            f'<td style="{value_style}">{gr.n_perm:,}</td></tr>'
        )
        html.append(
            f'<tr><td style="{label_style}">Correction</td>'
            f'<td style="{value_style}">{gr.correction}</td></tr>'
        )
        html.append(
            f'<tr><td style="{label_style}">Random State</td>'
            f'<td style="{value_style}">{gr.random_state}</td></tr>'
        )
        html.append(
            f'<tr><td style="{label_style}">Groups</td>'
            f'<td style="{value_style}">{n_groups}</td></tr>'
        )
        html.append(
            f'<tr><td style="{label_style}">Documents Used</td>'
            f'<td style="{value_style}">{gr.n_kept:,}</td></tr>'
        )
        html.append("</table>")

        groups = getattr(gr, "groups", None)
        if groups is not None:
            unique, counts = np.unique(groups, return_counts=True)
            labels_map = gr.group_labels or {}
            html.append(f'<div style="{section_style}">Group Sizes</div>')
            html.append('<table cellspacing="6" style="width: 100%;">')
            for g, cnt in zip(unique, counts):
                display = labels_map.get(g, g)
                html.append(
                    f'<tr><td style="{label_style}">{g}</td>'
                    f'<td style="{value_style}">{display}</td>'
                    f'<td style="{value_style}">{cnt:,}</td></tr>'
                )
            html.append("</table>")

        return html

    @staticmethod
    def _build_pairwise_html(gr, section_style: str, label_style: str, p) -> list:
        """Build HTML for the pairwise comparison results table.

        ``gr`` is a GroupResult. Iterates ``gr.pairs`` directly — no intermediate
        dict layer.
        """
        labels_map = gr.group_labels or {}
        html = []
        html.append(f'<div style="{section_style}">Pairwise Comparison Results</div>')
        html.append(
            '<table cellspacing="4" style="width: 100%; font-size: 12px;">'
            f'<tr style="{label_style}">'
            "<td>Group A</td><td>Group B</td><td>n_A</td><td>n_B</td>"
            "<td>Contrast norm</td><td>p</td><td>p (corr)</td><td>Cohen's d</td></tr>"
        )
        for pr in gr.pairs:
            p_raw_str = f"{pr.p_raw:.2e}" if pr.p_raw < 0.001 else f"{pr.p_raw:.4f}"
            p_corr_str = (
                f"{pr.p_corrected:.2e}" if pr.p_corrected < 0.001
                else f"{pr.p_corrected:.4f}"
            )
            g1_label = labels_map.get(pr.g1, pr.g1)
            g2_label = labels_map.get(pr.g2, pr.g2)
            html.append(
                f'<tr>'
                f'<td>{g1_label} ({pr.g1})</td>'
                f'<td>{g2_label} ({pr.g2})</td>'
                f'<td>{pr.n_g1:,}</td>'
                f'<td>{pr.n_g2:,}</td>'
                f'<td>{pr.contrast_norm:.4f}</td>'
                f'<td>{p_raw_str}</td>'
                f'<td>{p_corr_str}</td>'
                f'<td>{pr.cohens_d:.3f}</td>'
                f'</tr>'
            )
        html.append("</table>")
        return html

    def _load_config_tab(self, result: Result):
        """Load the configuration tab data."""
        p = self._html_palette()
        # Common styles
        label_style = f"color: {p.text_secondary}; font-size: {p.font_size_sm}; text-transform: uppercase;"
        value_style = f"font-size: {p.font_size_base}; padding-left: 8px;"
        section_style = (
            f"color: {p.accent}; font-size: 12px; font-weight: 600; "
            f"border-bottom: 1px solid {p.border}; padding-bottom: 4px; margin: 12px 0 8px 0;"
        )

        is_crossgroup = result._result and isinstance(result._result, GroupResult)
        is_fulldoc = result.config_snapshot.get("concept_mode", "lexicon") == "fulldoc"

        # --- Result Information ---
        run_html = []
        run_html.append('<table cellspacing="6" style="width: 100%;">')

        run_html.append(
            f'<tr><td style="{label_style}">Result ID</td>'
            f'<td style="{value_style}">{result.result_id}</td></tr>'
        )
        run_html.append(
            f'<tr><td style="{label_style}">Timestamp</td>'
            f'<td style="{value_style}">{result.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</td></tr>'
        )
        run_html.append(
            f'<tr><td style="{label_style}">Status</td>'
            f'<td style="{value_style}">{result.status}</td></tr>'
        )
        if isinstance(result._result, PLSResult):
            atype_label = "PLS"
        elif isinstance(result._result, PCAOLSResult):
            atype_label = "PCA+OLS"
        elif isinstance(result._result, GroupResult):
            atype_label = "Group Comparison"
        else:
            atype_label = type(result._result).__name__
        run_html.append(
            f'<tr><td style="{label_style}">Analysis Type</td>'
            f'<td style="{value_style}">{atype_label}</td></tr>'
        )
        run_html.append(
            f'<tr><td style="{label_style}">Mode</td>'
            f'<td style="{value_style}">{"Full Document" if is_fulldoc else "Lexicon"}</td></tr>'
        )
        s = result.config_snapshot
        if is_crossgroup and s.get("group_column", ""):
            run_html.append(
                f'<tr><td style="{label_style}">Group Column</td>'
                f'<td style="{value_style}">{s.get("group_column", "")}</td></tr>'
            )
        elif not is_crossgroup and s.get("outcome_column", ""):
            run_html.append(
                f'<tr><td style="{label_style}">Outcome Column</td>'
                f'<td style="{value_style}">{s.get("outcome_column", "")}</td></tr>'
            )

        run_html.append("</table>")

        # Dataset statistics sub-section
        spacy_snap = result.config_snapshot
        ds_snap = result.config_snapshot
        run_html.append(f'<div style="{section_style}">Dataset</div>')
        run_html.append('<table cellspacing="6" style="width: 100%;">')
        if ds_snap.get("csv_path", ""):
            run_html.append(
                f'<tr><td colspan="2" style="{label_style}">File</td></tr>'
                f'<tr><td colspan="2" style="{value_style}; word-break: break-all;">{ds_snap.get("csv_path", "")}</td></tr>'
            )
        if ds_snap.get("text_column", ""):
            run_html.append(
                f'<tr><td style="{label_style}">Text Column</td>'
                f'<td style="{value_style}">{ds_snap.get("text_column", "")}</td></tr>'
            )
        if ds_snap.get("id_column"):
            run_html.append(
                f'<tr><td style="{label_style}">ID Column</td>'
                f'<td style="{value_style}">{ds_snap.get("id_column")}</td></tr>'
            )
        run_html.append(
            f'<tr><td style="{label_style}">Documents</td>'
            f'<td style="{value_style}">{spacy_snap.get("n_docs_processed", 0):,}</td></tr>'
        )
        run_html.append(
            f'<tr><td style="{label_style}">Valid Samples</td>'
            f'<td style="{value_style}">{ds_snap.get("n_valid", 0):,}</td></tr>'
        )
        if ds_snap.get("id_column"):
            run_html.append(
                f'<tr><td style="{label_style}">Personal Concept Vectors</td>'
                f'<td style="{value_style}">{spacy_snap.get("n_docs_processed", 0):,}</td></tr>'
            )
        if spacy_snap.get("mean_words_before_stopwords", 0) > 0:
            run_html.append(
                f'<tr><td style="{label_style}">Mean Words / Doc (pre-stopword)</td>'
                f'<td style="{value_style}">{spacy_snap.get("mean_words_before_stopwords", 0):.1f}</td></tr>'
            )
        if spacy_snap.get("n_docs_processed", 0):
            avg_tokens_post = spacy_snap.get("total_tokens", 0) / spacy_snap.get("n_docs_processed", 1)
            run_html.append(
                f'<tr><td style="{label_style}">Mean Tokens / Doc (post-stopword)</td>'
                f'<td style="{value_style}">{avg_tokens_post:.1f}</td></tr>'
            )
        run_html.append("</table>")

        if result._result:
            res = result._result

            if is_crossgroup:
                run_html.extend(
                    self._build_group_detail_html(
                        res, section_style, label_style, value_style, p
                    )
                )

            else:
                # Continuous: Model Fit
                run_html.append(f'<div style="{section_style}">Model Fit</div>')
                run_html.append('<table cellspacing="6" style="width: 100%;">')
                run_html.append(
                    f'<tr><td style="{label_style}">R\u00b2</td>'
                    f'<td style="{value_style}">{res.stats.r2:.4f}</td></tr>'
                )
                if isinstance(res, PCAOLSResult):
                    run_html.append(
                        f'<tr><td style="{label_style}">Adjusted R\u00b2</td>'
                        f'<td style="{value_style}">{res.stats.r2_adj:.4f}</td></tr>'
                    )
                pval_str = f"{res.stats.pvalue:.2e}" if res.stats.pvalue < 0.001 else f"{res.stats.pvalue:.4f}"
                run_html.append(
                    f'<tr><td style="{label_style}">p-value</td>'
                    f'<td style="{value_style}">{pval_str}</td></tr>'
                )
                run_html.append(
                    f'<tr><td style="{label_style}">Corr(y, \u0177)</td>'
                    f'<td style="{value_style}">{res.stats.y_corr_pred:.4f}</td></tr>'
                )
                run_html.append("</table>")

                # Effect Size
                run_html.append(f'<div style="{section_style}">Effect Size</div>')
                run_html.append('<table cellspacing="6" style="width: 100%;">')
                run_html.append(
                    f'<tr><td style="{label_style}">\u2016\u03b2\u2016 (SD(y) / cosine)</td>'
                    f'<td style="{value_style}">{res.stats.beta_norm:.4f}</td></tr>'
                )
                run_html.append(
                    f'<tr><td style="{label_style}">\u0394y per +0.10 cosine</td>'
                    f'<td style="{value_style}">{res.stats.delta:.4f}</td></tr>'
                )
                run_html.append(
                    f'<tr><td style="{label_style}">IQR(cos) effect</td>'
                    f'<td style="{value_style}">{res.stats.iqr_effect:.4f}</td></tr>'
                )
                run_html.append("</table>")

                # Data Retention
                run_html.append(f'<div style="{section_style}">Data Retention</div>')
                run_html.append('<table cellspacing="6" style="width: 100%;">')
                run_html.append(
                    f'<tr><td style="{label_style}">Input documents</td>'
                    f'<td style="{value_style}">{res.stats.n_raw:,}</td></tr>'
                )
                run_html.append(
                    f'<tr><td style="{label_style}">Kept</td>'
                    f'<td style="{value_style}">{res.stats.n_kept:,}</td></tr>'
                )
                if res.stats.n_raw:
                    drop_pct = res.stats.n_dropped / res.stats.n_raw * 100
                    run_html.append(
                        f'<tr><td style="{label_style}">Dropped</td>'
                        f'<td style="{value_style}">{res.stats.n_dropped:,} ({drop_pct:.1f}%)</td></tr>'
                    )
                run_html.append("</table>")

                # PCA+OLS specific: components
                if isinstance(res, PCAOLSResult):
                    run_html.append(f'<div style="{section_style}">PCA</div>')
                    run_html.append('<table cellspacing="6" style="width: 100%;">')
                    run_html.append(
                        f'<tr><td style="{label_style}">Components (K)</td>'
                        f'<td style="{value_style}">{res.n_components}</td></tr>'
                    )
                    run_html.append("</table>")
                elif isinstance(res, PLSResult):
                    run_html.append(f'<div style="{section_style}">PLS</div>')
                    run_html.append('<table cellspacing="6" style="width: 100%;">')
                    run_html.append(
                        f'<tr><td style="{label_style}">Components</td>'
                        f'<td style="{value_style}">{res.n_components}</td></tr>'
                    )
                    if res.fit_info and res.fit_info.p_method:
                        run_html.append(
                            f'<tr><td style="{label_style}">p-value Method</td>'
                            f'<td style="{value_style}">{res.fit_info.p_method}</td></tr>'
                        )
                    run_html.append("</table>")

        # Pairwise results — shown in Result Information for fulldoc,
        # in Concept Configuration for lexicon mode
        if is_crossgroup and result._result and hasattr(result._result, 'pairs'):
            pairwise_html = self._build_pairwise_html(
                result._result, section_style, label_style, p
            )
            if is_fulldoc:
                run_html.extend(pairwise_html)

        self.result_info_text.setHtml("".join(run_html))

        # --- Concept Configuration (lexicon mode only) ---
        if is_fulldoc:
            self.concept_group.hide()
        else:
            self.concept_group.show()
            concept_html = []
            concept_html.append('<table cellspacing="6" style="width: 100%;">')
            concept_html.append(
                f'<tr><td style="{label_style}">Mode</td>'
                f'<td style="{value_style}">{s.get("concept_mode", "lexicon")}</td></tr>'
            )
            if is_crossgroup and s.get("group_column", ""):
                concept_html.append(
                    f'<tr><td style="{label_style}">Group Column</td>'
                    f'<td style="{value_style}">{s.get("group_column", "")}</td></tr>'
                )
            elif not is_crossgroup and s.get("outcome_column", ""):
                concept_html.append(
                    f'<tr><td style="{label_style}">Outcome Column</td>'
                    f'<td style="{value_style}">{s.get("outcome_column", "")}</td></tr>'
                )
            concept_html.append("</table>")

            if s.get("lexicon_tokens", []):
                tokens = sorted(s.get("lexicon_tokens", []))
                concept_html.append(f'<div style="{section_style}">Lexicon ({len(tokens)} tokens)</div>')
                token_display = ", ".join(tokens[:50])
                if len(tokens) > 50:
                    token_display += f" <span style='color: {p.text_muted};'>... and {len(tokens) - 50} more</span>"
                concept_html.append(
                    f'<div style="font-size: 12px; line-height: 1.6; padding: 4px 0;">{token_display}</div>'
                )

            # Lexicon coverage summary
            if result.config_snapshot.get("lexicon_coverage_summary"):
                s = result.config_snapshot["lexicon_coverage_summary"]
                concept_html.append(f'<div style="{section_style}">Lexicon Coverage</div>')
                concept_html.append('<table cellspacing="6" style="width: 100%;">')
                concept_html.append(
                    f'<tr><td style="{label_style}">Documents with hits</td>'
                    f'<td style="{value_style}">{s.get("docs_any", 0):,} '
                    f'({s.get("cov_all", 0) * 100:.1f}%)</td></tr>'
                )

                if is_crossgroup:
                    # Categorical coverage: Cramér's V, group coverage
                    cramers_v = s.get("cramers_v", s.get("corr_any", 0))
                    concept_html.append(
                        f'<tr><td style="{label_style}">Cram\u00e9r\'s V</td>'
                        f'<td style="{value_style}">{cramers_v:.4f}</td></tr>'
                    )
                    group_cov = s.get("group_cov", {})
                    if group_cov:
                        for g, cov in sorted(group_cov.items()):
                            concept_html.append(
                                f'<tr><td style="{label_style}">Coverage ({g})</td>'
                                f'<td style="{value_style}">{cov * 100:.1f}%</td></tr>'
                            )
                else:
                    concept_html.append(
                        f'<tr><td style="{label_style}">Q1 / Q4 Coverage</td>'
                        f'<td style="{value_style}">{s.get("q1", 0) * 100:.1f}% / '
                        f'{s.get("q4", 0) * 100:.1f}%</td></tr>'
                    )
                    concept_html.append(
                        f'<tr><td style="{label_style}">Correlation</td>'
                        f'<td style="{value_style}">{s.get("corr_any", 0):.4f}</td></tr>'
                    )

                concept_html.append(
                    f'<tr><td style="{label_style}">Hits per doc</td>'
                    f'<td style="{value_style}">mean: {s.get("hits_mean", 0):.2f}, '
                    f'median: {s.get("hits_median", 0):.1f}</td></tr>'
                )
                concept_html.append("</table>")

            # Per-token coverage breakdown
            if result.config_snapshot.get("lexicon_coverage_per_token"):
                concept_html.append(f'<div style="{section_style}">Per-Token Breakdown</div>')
                concept_html.append(
                    '<table cellspacing="4" style="width: 100%; font-size: 12px;">'
                    f'<tr style="{label_style}">'
                )
                if is_crossgroup:
                    concept_html.append("<td>Word</td><td>Docs</td><td>Coverage</td><td>Cram\u00e9r's V</td></tr>")
                else:
                    concept_html.append("<td>Word</td><td>Docs</td><td>Coverage</td><td>Correlation</td></tr>")

                for t in result.config_snapshot["lexicon_coverage_per_token"]:
                    corr = t.get("corr", 0)
                    corr_color = p.success if corr > 0 else p.error if corr < 0 else p.text_secondary
                    concept_html.append(
                        f'<tr><td style="font-weight: 500;">{t.get("token", "")}</td>'
                        f'<td>{t.get("freq", 0):,}</td>'
                        f'<td>{t.get("cov_all", 0) * 100:.1f}%</td>'
                        f'<td style="color: {corr_color};">{corr:+.4f}</td></tr>'
                    )
                concept_html.append("</table>")

            # Pairwise results table (crossgroup + lexicon mode)
            if is_crossgroup and result._result and hasattr(result._result, 'pairs'):
                concept_html.extend(pairwise_html)

            self.concept_config_text.setHtml("".join(concept_html))

        # --- Model Configuration ---
        hp = result.config_snapshot
        emb = result.config_snapshot
        spacy = result.config_snapshot

        model_html = []

        # Embeddings section
        model_html.append(f'<div style="{section_style}">Embeddings</div>')
        model_html.append('<table cellspacing="6" style="width: 100%;">')
        model_html.append(
            f'<tr><td style="{label_style}">File</td>'
            f'<td style="{value_style}; word-break: break-all;">{emb.get("selected_embedding", "") or "N/A"}</td></tr>'
        )
        model_html.append(
            f'<tr><td style="{label_style}">Vocabulary</td>'
            f'<td style="{value_style}">{emb.get("vocab_size", 0):,} words</td></tr>'
        )
        model_html.append(
            f'<tr><td style="{label_style}">Dimensions</td>'
            f'<td style="{value_style}">{emb.get("embedding_dim", 0)}</td></tr>'
        )
        model_html.append(
            f'<tr><td style="{label_style}">L2 Normalize</td>'
            f'<td style="{value_style}">{"Yes" if emb.get("l2_normalized", False) else "No"}</td></tr>'
        )
        _abtt = emb.get("abtt", 0)
        model_html.append(
            f'<tr><td style="{label_style}">ABTT</td>'
            f'<td style="{value_style}">{"Yes" if _abtt > 0 else "No"} (m={_abtt})</td></tr>'
        )
        if emb.get("coverage_pct", 0) > 0:
            cov_str = f"{emb.get('coverage_pct', 0):.1f}%"
            if emb.get("n_oov", 0) > 0:
                cov_str += f" ({emb.get('n_oov', 0):,} OOV)"
            model_html.append(
                f'<tr><td style="{label_style}">Coverage</td>'
                f'<td style="{value_style}">{cov_str}</td></tr>'
            )
        model_html.append("</table>")

        # spaCy section
        model_html.append(f'<div style="{section_style}">spaCy</div>')
        model_html.append('<table cellspacing="6" style="width: 100%;">')
        _input_mode = spacy.get("input_mode", "language")
        _spacy_model = spacy.get("spacy_model", "")
        _language = spacy.get("language", "en")
        if _input_mode == "custom" and _spacy_model:
            display_model = _spacy_model
        else:
            try:
                from ssdiff.lang_config import lang_to_model
                display_model = lang_to_model(_language)
            except (ImportError, KeyError):
                display_model = f"{_language}_core_news_lg"
        model_html.append(
            f'<tr><td style="{label_style}">Model</td>'
            f'<td style="{value_style}">{display_model}</td></tr>'
        )
        model_html.append(
            f'<tr><td style="{label_style}">Language</td>'
            f'<td style="{value_style}">{_language}</td></tr>'
        )
        model_html.append(
            f'<tr><td style="{label_style}">Documents</td>'
            f'<td style="{value_style}">{spacy.get("n_docs_processed", 0):,}</td></tr>'
        )
        model_html.append(
            f'<tr><td style="{label_style}">Input Mode</td>'
            f'<td style="{value_style}">{_input_mode}</td></tr>'
        )
        stopword_labels = {"default": "Default", "none": "Disabled", "custom": "Custom file"}
        _stopword_mode = spacy.get("stopword_mode", "default")
        model_html.append(
            f'<tr><td style="{label_style}">Stopwords</td>'
            f'<td style="{value_style}">{stopword_labels.get(_stopword_mode, _stopword_mode)}</td></tr>'
        )
        model_html.append("</table>")

        # Hyperparameters section — analysis-type-specific
        atype = s.get("analysis_type", "pls")
        model_html.append(f'<div style="{section_style}">Hyperparameters</div>')
        model_html.append('<table cellspacing="6" style="width: 100%;">')

        # Common params
        if not is_fulldoc:
            model_html.append(
                f'<tr><td style="{label_style}">Context Window</td>'
                f'<td style="{value_style}">\u00b1{hp.get("context_window_size", 5)}</td></tr>'
            )
        model_html.append(
            f'<tr><td style="{label_style}">SIF Parameter</td>'
            f'<td style="{value_style}">{hp.get("sif_a", 1e-3)}</td></tr>'
        )

        # PLS-specific
        if atype == "pls":
            _pls_n_comp = hp.get("pls_n_components", 0)
            n_comp_display = "auto" if _pls_n_comp == 0 else _pls_n_comp
            model_html.append(
                f'<tr><td style="{label_style}">PLS Components</td>'
                f'<td style="{value_style}">{n_comp_display}</td></tr>'
            )
            _pls_p_method = hp.get("pls_p_method", "auto")
            model_html.append(
                f'<tr><td style="{label_style}">p-value Method</td>'
                f'<td style="{value_style}">{_pls_p_method}</td></tr>'
            )
            if _pls_p_method in ("perm", "auto"):
                model_html.append(
                    f'<tr><td style="{label_style}">n Permutations</td>'
                    f'<td style="{value_style}">{hp.get("pls_n_perm", 1000):,}</td></tr>'
                )
            if _pls_p_method in ("split", "split_cal", "auto"):
                model_html.append(
                    f'<tr><td style="{label_style}">n Splits</td>'
                    f'<td style="{value_style}">{hp.get("pls_n_splits", Project.__dataclass_fields__["pls_n_splits"].default)}</td></tr>'
                )
                model_html.append(
                    f'<tr><td style="{label_style}">Split Ratio</td>'
                    f'<td style="{value_style}">{hp.get("pls_split_ratio", 0.5)}</td></tr>'
                )
            if hp.get("pls_pca_preprocess") is not None:
                model_html.append(
                    f'<tr><td style="{label_style}">PCA Preprocess</td>'
                    f'<td style="{value_style}">{hp.get("pls_pca_preprocess")}</td></tr>'
                )
            model_html.append(
                f'<tr><td style="{label_style}">Random State</td>'
                f'<td style="{value_style}">{hp.get("pls_random_state", 42)}</td></tr>'
            )

        # PCA+OLS-specific
        elif atype == "pca_ols":
            _pcaols_n_comp = hp.get("pcaols_n_components")
            n_comp_display = "sweep" if _pcaols_n_comp is None else _pcaols_n_comp
            model_html.append(
                f'<tr><td style="{label_style}">PCA Components</td>'
                f'<td style="{value_style}">{n_comp_display}</td></tr>'
            )
            if _pcaols_n_comp is None:
                model_html.append(
                    f'<tr><td style="{label_style}">K Range</td>'
                    f'<td style="{value_style}">{hp.get("sweep_k_min", Project.__dataclass_fields__["sweep_k_min"].default)} \u2013 {hp.get("sweep_k_max", Project.__dataclass_fields__["sweep_k_max"].default)} (step {hp.get("sweep_k_step", Project.__dataclass_fields__["sweep_k_step"].default)})</td></tr>'
                )

        # Groups-specific
        elif atype == "groups":
            model_html.append(
                f'<tr><td style="{label_style}">Permutations</td>'
                f'<td style="{value_style}">{hp.get("groups_n_perm", 5000):,}</td></tr>'
            )
            model_html.append(
                f'<tr><td style="{label_style}">Correction</td>'
                f'<td style="{value_style}">{hp.get("groups_correction", "holm")}</td></tr>'
            )
            if hp.get("groups_median_split", False):
                model_html.append(
                    f'<tr><td style="{label_style}">Median Split</td>'
                    f'<td style="{value_style}">Yes</td></tr>'
                )
            model_html.append(
                f'<tr><td style="{label_style}">Random State</td>'
                f'<td style="{value_style}">{hp.get("groups_random_state", 42)}</td></tr>'
            )

        model_html.append("</table>")

        # Clustering section
        model_html.append(f'<div style="{section_style}">Clustering</div>')
        model_html.append('<table cellspacing="6" style="width: 100%;">')
        model_html.append(
            f'<tr><td style="{label_style}">Top N</td>'
            f'<td style="{value_style}">{hp.get("clustering_topn", 100)}</td></tr>'
        )
        model_html.append(
            f'<tr><td style="{label_style}">Auto K</td>'
            f'<td style="{value_style}">{"Yes (silhouette)" if hp.get("clustering_k_auto", True) else "No"}</td></tr>'
        )
        model_html.append(
            f'<tr><td style="{label_style}">K Range</td>'
            f'<td style="{value_style}">{hp.get("clustering_k_min", 2)} - {hp.get("clustering_k_max", 10)}</td></tr>'
        )
        model_html.append(
            f'<tr><td style="{label_style}">Top Words</td>'
            f'<td style="{value_style}">{hp.get("clustering_top_words", 5)}</td></tr>'
        )
        model_html.append("</table>")

        self.model_config_text.setHtml("".join(model_html))

    # ------------------------------------------------------------------ #
    #  PCA Sweep tab helpers
    # ------------------------------------------------------------------ #

    def _load_pca_sweep_tab(self, result: Result):
        """Load the PCA sweep figure into the tab."""
        selected_k = result._result.n_components if result._result and isinstance(result._result, PCAOLSResult) else None

        if selected_k is not None:
            self.pca_sweep_info.setText(f"Selected PCA K: {selected_k}")
        else:
            self.pca_sweep_info.setText("PCA K was set manually (no sweep performed).")

        # Render from live sweep_result; fall back to legacy sweep_plot.png
        # only for saved results (unsaved results have no folder on disk yet).
        pixmap = self._render_sweep_pixmap(result)
        if pixmap is None and result.result_path is not None:
            sweep_png = result.result_path / "sweep_plot.png"
            if sweep_png.exists():
                candidate = QPixmap(str(sweep_png))
                if not candidate.isNull():
                    pixmap = candidate

        if pixmap is not None:
            self._pca_sweep_pixmap = pixmap
            self._pca_sweep_zoom_pct = app_settings().value("pca_sweep_zoom_pct", 0, type=int)
            self._pca_sweep_apply_zoom()
        else:
            self._pca_sweep_pixmap = None
            self.pca_sweep_image.setText(
                "No sweep plot available.\n"
                "This run may have used manual PCA K selection."
            )

    @staticmethod
    def _render_sweep_pixmap(result: Result):
        """Render sweep chart from the live result's sweep_result. Returns QPixmap or None."""
        from ..utils.charts import render_sweep_plot

        sweep = getattr(result._result, "sweep_result", None) if result._result else None
        if sweep is not None:
            return render_sweep_plot(sweep.df_joined, sweep.best_k)
        return None

    def _pca_sweep_apply_zoom(self):
        """Redraw the sweep image at the current zoom level."""
        if self._pca_sweep_pixmap is None:
            return

        if self._pca_sweep_zoom_pct == 0:
            # Fit to the scroll area width
            available = self._pca_sweep_scroll.viewport().width()
            if available < 50:
                available = 800  # fallback before first layout
            scaled = self._pca_sweep_pixmap.scaledToWidth(
                available, Qt.SmoothTransformation
            )
            self._pca_sweep_zoom_label.setText("Fit")
        else:
            w = int(
                self._pca_sweep_pixmap.width() * self._pca_sweep_zoom_pct / 100
            )
            w = max(w, 100)
            scaled = self._pca_sweep_pixmap.scaledToWidth(
                w, Qt.SmoothTransformation
            )
            self._pca_sweep_zoom_label.setText(f"{self._pca_sweep_zoom_pct}%")

        self.pca_sweep_image.setPixmap(scaled)

    def _pca_sweep_zoom(self, delta: int):
        """Adjust zoom by *delta* percentage points."""
        if self._pca_sweep_pixmap is None:
            return

        if self._pca_sweep_zoom_pct == 0:
            # Transition from fit-to-window: compute current effective %
            available = self._pca_sweep_scroll.viewport().width()
            if available < 50:
                available = 800
            self._pca_sweep_zoom_pct = round(
                available / self._pca_sweep_pixmap.width() * 100
            )

        self._pca_sweep_zoom_pct = max(10, self._pca_sweep_zoom_pct + delta)
        self._pca_sweep_apply_zoom()
        self._pca_sweep_save_zoom()

    def _pca_sweep_zoom_reset(self):
        """Reset zoom to fit-to-window."""
        self._pca_sweep_zoom_pct = 0
        self._pca_sweep_apply_zoom()
        self._pca_sweep_save_zoom()

    def _pca_sweep_save_zoom(self):
        """Persist the current zoom preference."""
        app_settings().setValue("pca_sweep_zoom_pct", self._pca_sweep_zoom_pct)

    # ================================================================== #
    #  ACTIONS
    # ================================================================== #

    def _go_back(self):
        """Go back to Stage 2."""
        main_window = self.window()
        if hasattr(main_window, "go_to_stage"):
            main_window.go_to_stage(2)

    def _new_run(self):
        """Request a new analysis run."""
        self.new_run_requested.emit()

    def _open_report_settings(self):
        """Open the Report Settings configuration dialog."""
        from .report_settings_dialog import ReportSettingsDialog
        dlg = ReportSettingsDialog(self)
        dlg.exec()

    # ================================================================== #
    #  PUBLIC
    # ================================================================== #

    def reset(self):
        """Clear all state for project close."""
        self.project = None
        self.current_result = None
        self._unsaved_result = None
        self._is_viewing_unsaved = False

    def load_project(self, project: Project):
        """Load a project into the UI."""
        self.project = project
        self._populate_result_selector()

    def show_unsaved_result(self, result: Result):
        """Display a fresh (unsaved) result and enable the save controls."""
        self._unsaved_result = result
        self._is_viewing_unsaved = True
        self._populate_result_selector()
        # Selector index 0 is now the unsaved result — _populate already calls show_result
        self.run_name_input.clear()

    def has_unsaved_result(self) -> bool:
        """Return True if there is an unsaved result."""
        return self._unsaved_result is not None

    def _save_result(self):
        """Save an unsaved result, or re-save (overwrite in place) a saved one."""
        if self.project is None or self.current_result is None:
            return

        if self._is_viewing_unsaved:
            self._save_new_result()
        else:
            self._overwrite_saved_result()

    def _save_new_result(self):
        """Persist the unsaved result: create its folder, write all files, register in project."""
        if self._unsaved_result is None:
            return

        result = self._unsaved_result

        # Optional name from the text input — drives both display and folder name
        name = self.run_name_input.text().strip()
        if name:
            result.name = name

        # Resolve folder name: sanitized user name, else timestamp fallback.
        # Append _2, _3… on collision so repeated names never overwrite prior runs.
        folder_base = _sanitize_folder_name(name) if name else ""
        if not folder_base:
            folder_base = result.result_id
        results_dir = self.project.project_path / "results"
        folder_name = _resolve_folder_collision(results_dir, folder_base)

        result.folder_name = folder_name
        result.result_path = results_dir / folder_name

        # Write everything to disk now
        from ..utils.file_io import ProjectIO
        result.result_path.mkdir(parents=True, exist_ok=True)
        ProjectIO.save_result_config(result)
        ProjectIO.save_result(result)

        # Save sweep plot PNG if available
        if self._pca_sweep_pixmap is not None:
            try:
                sweep_png = result.result_path / "sweep_plot.png"
                self._pca_sweep_pixmap.save(str(sweep_png), "PNG")
            except Exception:
                pass  # Non-critical

        # Register in project (Save Project persists the list to project.json)
        self.project.results.append(result)
        self.project.mark_dirty()

        # Clear unsaved state
        self._unsaved_result = None
        self._is_viewing_unsaved = False

        # Update the selector entry in-place (no full reload)
        self.result_selector.setItemText(
            self.result_selector.currentIndex(), _result_dropdown_label(result),
        )

        # Now viewing a saved result — flip the button to "Overwrite"
        self.run_name_input.hide()
        self.save_result_btn.setText("Overwrite")
        self.save_result_btn.setToolTip(
            "Re-save this result in place (updates report / files)"
        )

        self.result_saved.emit()

    def _overwrite_saved_result(self):
        """Re-save a saved result in place (regenerate files, keep result_id)."""
        result = self.current_result
        if result is None or result._result is None:
            return

        from ..utils.file_io import ProjectIO
        ProjectIO.save_result_config(result)
        ProjectIO.save_result(result)

        # Keep the sweep PNG in sync with what the user is viewing
        if self._pca_sweep_pixmap is not None:
            try:
                sweep_png = result.result_path / "sweep_plot.png"
                self._pca_sweep_pixmap.save(str(sweep_png), "PNG")
            except Exception:
                pass

        self.project.mark_dirty()
        self.result_saved.emit()

    def _delete_result_by_index(self, row: int):
        """Remove a result: trash its folder (if present) and drop it from the project."""
        result = self.result_selector.itemData(row)
        if result is None or result is self._unsaved_result:
            return

        self.result_selector.hidePopup()

        display_name = result.name or result.result_id
        is_missing = result.status == "missing" or result.result_path is None

        if is_missing:
            prompt = (
                f"Remove the missing reference to \"{display_name}\" "
                "from this project?"
            )
        else:
            prompt = (
                f"Move \"{display_name}\" to the system trash?\n\n"
                "The folder will be recoverable from trash."
            )

        reply = QMessageBox.question(
            self, "Remove Result", prompt,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        if self.project and result in self.project.results:
            self.project.results.remove(result)

        if not is_missing and result.result_path and result.result_path.exists():
            try:
                from send2trash import send2trash
                send2trash(str(result.result_path))
            except Exception as e:
                QMessageBox.critical(
                    self, "Delete Failed",
                    f"Could not move to trash:\n{e}\n\n"
                    "The folder is still on disk.",
                )
                # Re-insert so the UI stays consistent with disk
                if self.project and result not in self.project.results:
                    self.project.results.append(result)
                return

        from ..utils.file_io import ProjectIO
        if self.project:
            ProjectIO.save_project(self.project)

        self._populate_result_selector()
