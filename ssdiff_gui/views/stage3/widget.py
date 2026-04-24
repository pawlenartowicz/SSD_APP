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
    QTextEdit,
    QMessageBox,
    QSplitter,
    QFrame,
    QApplication,
)
from PySide6.QtCore import Qt, Signal, QEvent, QTimer

import numpy as np

from ssdiff import PLSResult, PCAOLSResult, GroupResult

from .result_view import ResultView
from . import stats_strip, labels, pair_selector, html_helpers
from .tabs.scores import ScoresTab
from .tabs.extreme_docs import ExtremeDocsTab
from .tabs.cluster_overview import ClusterOverviewTab
from .tabs.poles import PolesTab
from .tabs.snippets import SnippetsTab
from .tabs.misdiagnosed import MisdiagnosedTab
from .tabs.pca_sweep import PcaSweepTab
from .tabs.details import DetailsTab
from ...models.project import Project, Result
from ..widgets.loading_overlay import LoadingOverlay
from ..widgets.info_button import InfoButton
from ..widgets.removable_delegate import RemovableItemDelegate


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


def _resolve_folder_collision(
    results_dir: Path, base: str, reserved: set[str] = frozenset(),
) -> str:
    """Return `base`, or `base_2`, `base_3`… so the folder name is free.

    A name is taken if its folder exists on disk OR if it's in `reserved`.
    Reserving tracked names (even ones whose folders were trashed) prevents
    re-using a folder_name that the project still references as [missing],
    which would create duplicates in project.json.
    """
    def taken(name: str) -> bool:
        return (results_dir / name).exists() or name in reserved

    if not taken(base):
        return base
    i = 2
    while taken(f"{base}_{i}"):
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
                w.word for w in list(clusters_side(c.cluster_id).words)[:5]
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
            for w in clusters_side(c.cluster_id).words:
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
        self._current_pair_key: Optional[Tuple[str, str]] = None
        self._view: ResultView | None = None

        # Pair-selector widgets (one per pair-scoped tab — Cluster Overview,
        # Contrast Snippets, Semantic Poles, Document Scores, Extreme Documents).
        # All combos are kept in sync via _on_pair_changed.
        self._pair_combos: List[QComboBox] = []
        self._pair_frames: List[QWidget] = []

        # Tab objects (own their Qt widgets; created before _setup_ui)
        self._cluster_overview_tab = ClusterOverviewTab(
            get_project=lambda: self.project,
            get_document_text=self._get_document_text,
        )
        self._scores_tab_obj = ScoresTab(get_document_text=self._get_document_text)
        self._extreme_docs_tab_obj = ExtremeDocsTab(get_document_text=self._get_document_text)
        self._poles_tab = PolesTab()
        self._snippets_tab = SnippetsTab()
        self._misdiagnosed_tab = MisdiagnosedTab(get_document_text=self._get_document_text)
        self._pca_sweep_tab = PcaSweepTab(get_current_result=lambda: self.current_result)
        self._details_tab = DetailsTab()

        # Unsaved-result tracking
        self._unsaved_result: Optional[Result] = None
        self._is_viewing_unsaved: bool = False

        self._setup_ui()

    def resizeEvent(self, event):
        """Re-apply fit-to-window zoom when the widget is resized."""
        super().resizeEvent(event)
        self._pca_sweep_tab.on_container_resized()

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
        self.tabs.addTab(
            self._cluster_overview_tab.create(self, self._on_pair_changed, self._pair_combos, self._pair_frames),
            "Cluster Overview",
        )  # 0
        self.tabs.addTab(self._details_tab.create(self), "Details")                 # 1
        self.tabs.addTab(self._pca_sweep_tab.create(self), "PCA Sweep")            # 2
        _snippets_index = self.tabs.addTab(
            self._snippets_tab.create(self, self._on_pair_changed, self._pair_combos, self._pair_frames),
            SnippetsTab.HEADER_CONTINUOUS,
        )                                                                          # 3
        self._snippets_tab.set_tab_header_callback(
            lambda text: self.tabs.setTabText(_snippets_index, text)
        )
        self.tabs.addTab(
            self._poles_tab.create(self, self._on_pair_changed, self._pair_combos, self._pair_frames),
            "Semantic Poles",
        )                                                                          # 4
        self.tabs.addTab(self._scores_tab_obj.create(self, self._on_pair_changed, self._pair_combos, self._pair_frames), "Document Scores")             # 5
        self.tabs.addTab(self._extreme_docs_tab_obj.create(self, self._on_pair_changed, self._pair_combos, self._pair_frames), "Extreme Documents")   # 6
        self.tabs.addTab(self._misdiagnosed_tab.create(self), "Misdiagnosed")     # 7

        self._tabs_by_index = {
            0: self._cluster_overview_tab,
            1: self._details_tab,
            2: self._pca_sweep_tab,
            3: self._snippets_tab,
            4: self._poles_tab,
            5: self._scores_tab_obj,
            6: self._extreme_docs_tab_obj,
            7: self._misdiagnosed_tab,
        }

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
        self._overlay_right_panel = LoadingOverlay(self._cluster_overview_tab._ov_right_panel)
        self._overlay_tabs = LoadingOverlay(self.tabs)

        # Widget dict for label functions — all 7 widgets must exist before here.
        self._label_widgets = {
            "ov_pos_group": self._cluster_overview_tab._ov_pos_group,
            "ov_neg_group": self._cluster_overview_tab._ov_neg_group,
            "pos_pole_group": self._poles_tab._pos_group,
            "neg_pole_group": self._poles_tab._neg_group,
            "pos_pole_desc": self._poles_tab._pos_desc,
            "neg_pole_desc": self._poles_tab._neg_desc,
            "snippet_side_combo": self._snippets_tab.side_combo,
            "snippet_tab_desc": self._snippets_tab.tab_desc,
        }

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
        self._stat_card_widgets: list[tuple[QLabel, QLabel]] = []
        default_labels = [
            "R-squared", "Adj. R-squared", "F-statistic",
            "p-value", "Documents Used", "PCA K", "PCA Variance Explained",
        ]
        for label in default_labels:
            card = self._create_stat_card(label, "\u2014")
            self._stat_cards.append(card)
            self._stat_card_widgets.append((card.name_label, card.value_label))
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

        save_settings_btn = QPushButton("Save Settings")
        save_settings_btn.setObjectName("btn_ghost")
        save_settings_btn.setCursor(Qt.PointingHandCursor)
        save_settings_btn.clicked.connect(self._open_save_settings)
        actions_layout.addWidget(save_settings_btn)

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

        initial_pair = None
        if isinstance(ssd_result, GroupResult) and ssd_result.pairs:
            group_labels = getattr(ssd_result, "group_labels", None) or {}
            first = ssd_result.pairs[0]
            g1 = group_labels.get(first.g1, first.g1)
            g2 = group_labels.get(first.g2, first.g2)
            initial_pair = (g1, g2)
        self._current_pair_key = initial_pair
        self._view = ResultView.build(ssd_result, current_pair=initial_pair, meta=result)

        # Re-attach shared project embeddings if missing (stripped during save)
        if getattr(ssd_result, "embeddings", None) is None and self.project._emb is not None:
            ssd_result.embeddings = self.project._emb

        is_crossgroup = isinstance(ssd_result, GroupResult)
        is_pca_ols = isinstance(ssd_result, PCAOLSResult)

        # Analysis type badge — theme-aware; pulls semantic colors from the current palette
        p = html_helpers.html_palette()
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
        for tab_index, tab in self._tabs_by_index.items():
            self.tabs.setTabVisible(tab_index, tab.is_visible_for(self._view))

        pair_selector.populate_pair_combos(self._view, self._pair_combos, self._pair_frames)

        # Update stats strip
        stats_strip.apply(self._stat_card_widgets, self._view)

        # Update group-specific labels
        if self._view.is_group:
            labels.apply_group_labels(self._view, self._label_widgets)
        else:
            labels.apply_continuous_labels(self._label_widgets)

        # Populate every tab, pumping events between each so the spinner animates
        self._cluster_overview_tab.load(self._view)
        QApplication.processEvents()
        self._poles_tab.load(self._view)
        QApplication.processEvents()
        self._snippets_tab.load(self._view)
        QApplication.processEvents()
        self._scores_tab_obj.load(self._view)
        QApplication.processEvents()
        self._details_tab.load(self._view)
        QApplication.processEvents()
        if is_pca_ols:
            self._pca_sweep_tab.load(self._view)
        self._extreme_docs_tab_obj.load(self._view)
        QApplication.processEvents()
        if not is_crossgroup:
            self._misdiagnosed_tab.load(self._view)
            QApplication.processEvents()

        self._overlay_tabs.stop()

    def _on_pair_changed(self, index: int):
        """Handle pair selection change from any per-tab combo; sync all combos."""
        if index < 0 or self._view is None or not self._view.is_multi_pair:
            return
        sender = self.sender()
        pair_key = sender.itemData(index) if sender is not None else None
        if pair_key is None:
            return
        pair_key = tuple(pair_key)
        if pair_key not in self._view.pairs:
            return

        self._current_pair_key = pair_key
        self._view = ResultView.build(self._view.source, current_pair=pair_key, meta=self._view.meta)

        for combo in self._pair_combos:
            if combo is sender:
                continue
            combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)

        stats_strip.apply(self._stat_card_widgets, self._view)
        labels.apply_group_labels(self._view, self._label_widgets)

        self._overlay_tabs.start()
        QApplication.processEvents()

        self._cluster_overview_tab.load(self._view)
        QApplication.processEvents()
        self._poles_tab.load(self._view)
        QApplication.processEvents()
        self._snippets_tab.load(self._view)
        QApplication.processEvents()
        self._scores_tab_obj.load(self._view)
        QApplication.processEvents()
        self._extreme_docs_tab_obj.load(self._view)
        QApplication.processEvents()

        self._overlay_tabs.stop()

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

    def _open_save_settings(self):
        """Open the Save Settings dialog."""
        from ..save_settings_dialog import SaveSettingsDialog
        if self.current_result is None or self.current_result._result is None:
            return
        dlg = SaveSettingsDialog(type(self.current_result._result), self)
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
        tracked_names = {
            r.folder_name for r in self.project.results if r.folder_name
        }
        folder_name = _resolve_folder_collision(
            results_dir, folder_base, reserved=tracked_names,
        )

        result.folder_name = folder_name
        result.result_path = results_dir / folder_name

        # Write everything to disk now
        from ...utils.file_io import ProjectIO
        from ...utils.result_export import export_result
        from ...utils.save_config import SaveConfig
        from ...utils.settings import app_settings

        result.result_path.mkdir(parents=True, exist_ok=True)
        ProjectIO.save_result_config(result)
        cfg = SaveConfig.from_settings(app_settings())
        export_result(result, result.result_path, cfg)

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

        from ...utils.file_io import ProjectIO
        from ...utils.result_export import export_result
        from ...utils.save_config import SaveConfig
        from ...utils.settings import app_settings

        ProjectIO.save_result_config(result)
        cfg = SaveConfig.from_settings(app_settings())
        export_result(result, result.result_path, cfg)

        self.project.mark_dirty()
        self.result_saved.emit()

    def _delete_result_by_index(self, row: int):
        """Remove a result: drop it from the project, then trash its folder (if present).

        Order matters: we persist the project.json update *before* touching the
        filesystem. If the trash step fails later, the folder stays on disk and
        will show up as an orphan on next load — the project config is already
        consistent with the user's intent.
        """
        result = self.result_selector.itemData(row)
        if result is None or result is self._unsaved_result:
            return
        if self.project is None:
            return

        self.result_selector.hidePopup()

        display_name = result.name or result.result_id
        is_missing = result.status == "missing" or result.result_path is None
        folder_path = result.result_path if not is_missing else None

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

        from ...utils.file_io import ProjectIO

        if result in self.project.results:
            self.project.results.remove(result)
        ProjectIO.save_project(self.project)
        self.project.mark_dirty()

        if folder_path is not None and folder_path.exists():
            try:
                from send2trash import send2trash
                send2trash(str(folder_path))
            except Exception as e:
                QMessageBox.critical(
                    self, "Delete Failed",
                    f"The project no longer tracks this result, "
                    f"but the folder could not be moved to trash:\n{e}\n\n"
                    f"The folder is still on disk and will appear as an "
                    f"orphan next time the project is opened.",
                )

        self._populate_result_selector()
