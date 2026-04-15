"""Stage 2: Run tab view for SSD.

Contains model selection (analysis type + column), a pre-flight review
panel, and (for lexicon mode) a lexicon builder.
"""

from typing import Optional, Set
import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QRadioButton,
    QButtonGroup,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QMessageBox,
    QInputDialog,
    QSplitter,
    QHeaderView,
    QFrame,
)
from PySide6.QtCore import Qt, Signal

from ..models.project import Project
from ..utils.settings import app_settings
from .widgets.collapsible_box import CollapsibleBox
from .widgets.info_button import InfoButton
from .widgets.overlay_info_mixin import OverlayInfoMixin


class Stage2Widget(OverlayInfoMixin, QWidget):
    """Stage 2: Run - Pre-flight review + lexicon builder."""

    run_requested = Signal()

    _info_margin_right = 4
    _info_margin_top = 18

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Optional[Project] = None
        self.lexicon: Set[str] = set()
        self._settings = app_settings()

        self._init_overlay_info()
        self._setup_ui()

    def eventFilter(self, obj, event):
        if self._overlay_info_event_filter(obj, event):
            return True
        return super().eventFilter(obj, event)

    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 12, 24, 8)
        main_layout.setSpacing(6)

        # Header
        title = QLabel("Run Analysis")
        title.setObjectName("label_title")
        main_layout.addWidget(title)

        subtitle = QLabel(
            "Review your configuration and build a lexicon (if applicable), then run."
        )
        subtitle.setObjectName("label_muted")
        main_layout.addWidget(subtitle)

        # --- Model selection header ---
        self._create_model_selection(main_layout)

        # --- Advanced settings (collapsible) ---
        self._create_advanced_settings(main_layout)
        main_layout.addWidget(self.advanced_box)

        # Main splitter: review (left) | current lexicon (middle) | suggestions (right)
        self.main_splitter = QSplitter(Qt.Horizontal)

        # --- Left: Pre-flight review panel ---
        review_group = QGroupBox("Details")
        review_group_layout = QVBoxLayout()

        self.review_panel = QTextEdit()
        self.review_panel.setReadOnly(True)
        self.review_panel.setObjectName("preflight_review")
        review_group_layout.addWidget(self.review_panel)

        review_group.setLayout(review_group_layout)
        self._review_group = review_group
        self._add_overlay_info(review_group,
            "<b>Run Details</b><br><br>"
            "A read-only summary of all your project settings: dataset, "
            "preprocessing stats, embedding coverage, analysis type/mode, "
            "and hyperparameters.<br><br>"
            "Review this before running to make sure everything looks correct.",
        )
        self.main_splitter.addWidget(review_group)

        # --- Middle: Current Lexicon panel ---
        self.lexicon_group = self._create_current_lexicon_panel()
        self.main_splitter.addWidget(self.lexicon_group)

        # --- Right: Suggestions panel ---
        self.suggestions_group = self._create_suggestions_panel()
        self.main_splitter.addWidget(self.suggestions_group)

        self.main_splitter.setSizes([350, 350, 300])
        self.main_splitter.setMinimumHeight(200)

        main_layout.addWidget(self.main_splitter, stretch=1)

        # Bottom: Run section
        self._create_run_section_widget(main_layout)

        # Restore splitter state
        self._restore_splitter_states()

    # ------------------------------------------------------------------ #
    #  Model selection & advanced settings
    # ------------------------------------------------------------------ #

    def _create_model_selection(self, parent_layout):
        """Create model selection: 3 toggle buttons + column picker + mode."""
        model_frame = QFrame()
        model_frame.setObjectName("model_selection_frame")
        model_layout = QVBoxLayout(model_frame)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(8)

        # Row 1: Three analysis type buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.analysis_type_group = QButtonGroup()
        self.analysis_type_group.setExclusive(True)

        self.pls_btn = QPushButton("PLS")
        self.pls_btn.setCheckable(True)
        self.pls_btn.setChecked(True)
        self.pls_btn.setObjectName("btn_model_select_active")
        self.pls_btn.setMinimumHeight(36)
        self.pls_btn.setCursor(Qt.PointingHandCursor)
        self.analysis_type_group.addButton(self.pls_btn, 0)
        btn_row.addWidget(self.pls_btn)

        self.pcaols_btn = QPushButton("PCA+OLS")
        self.pcaols_btn.setCheckable(True)
        self.pcaols_btn.setObjectName("btn_model_select")
        self.pcaols_btn.setMinimumHeight(36)
        self.pcaols_btn.setCursor(Qt.PointingHandCursor)
        self.analysis_type_group.addButton(self.pcaols_btn, 1)
        btn_row.addWidget(self.pcaols_btn)

        self.groups_btn = QPushButton("Groups")
        self.groups_btn.setCheckable(True)
        self.groups_btn.setObjectName("btn_model_select")
        self.groups_btn.setMinimumHeight(36)
        self.groups_btn.setCursor(Qt.PointingHandCursor)
        self.analysis_type_group.addButton(self.groups_btn, 2)
        btn_row.addWidget(self.groups_btn)

        btn_row.addStretch(1)
        model_layout.addLayout(btn_row)

        # Row 2: Column picker + mode — stays the same for all types
        config_row = QHBoxLayout()
        config_row.setSpacing(12)

        # Column label (changes between "Outcome Column" and "Group Column")
        self.column_label = QLabel("Outcome Column:")
        config_row.addWidget(self.column_label)

        self.column_combo = QComboBox()
        self.column_combo.setMinimumWidth(180)
        self.column_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.column_combo.addItem("(none)")
        self.column_combo.currentIndexChanged.connect(self._on_column_changed)
        config_row.addWidget(self.column_combo)

        self.column_summary_label = QLabel("")
        self.column_summary_label.setObjectName("label_muted")
        config_row.addWidget(self.column_summary_label, stretch=1)

        # Mode selection
        config_row.addSpacing(16)
        config_row.addWidget(QLabel("Mode:"))

        self.mode_btn_group = QButtonGroup()
        self.lexicon_radio = QRadioButton("Lexicon")
        self.lexicon_radio.setChecked(True)
        self.mode_btn_group.addButton(self.lexicon_radio, 0)
        config_row.addWidget(self.lexicon_radio)

        self.fulldoc_radio = QRadioButton("Full Document")
        self.mode_btn_group.addButton(self.fulldoc_radio, 1)
        config_row.addWidget(self.fulldoc_radio)

        model_layout.addLayout(config_row)

        parent_layout.addWidget(model_frame)

        # Connect signals
        self.pls_btn.toggled.connect(self._on_analysis_type_changed)
        self.pcaols_btn.toggled.connect(self._on_analysis_type_changed)
        self.groups_btn.toggled.connect(self._on_analysis_type_changed)
        self.lexicon_radio.toggled.connect(self._on_mode_changed)

    def _create_advanced_settings(self, parent_layout):
        """Create collapsible advanced settings section."""
        self.advanced_box = CollapsibleBox("Advanced Settings")
        content = self.advanced_box.content_layout

        def _section_label(text: str) -> QLabel:
            """Create a styled section header label."""
            lbl = QLabel(text)
            lbl.setStyleSheet(
                "QLabel { color: rgba(255,255,255,0.45); font-size: 11px;"
                " font-weight: bold; letter-spacing: 1px;"
                " border-bottom: 1px solid rgba(255,255,255,0.12);"
                " padding-bottom: 2px; margin-top: 6px; }"
            )
            return lbl

        def _form_row(label, widget, info_html, form_layout):
            """Add a form row: label | widget + info button. Returns the InfoButton."""
            row = QHBoxLayout()
            row.addWidget(widget, 1)
            ib = InfoButton(info_html)
            row.addWidget(ib)
            form_layout.addRow(label, row)
            return ib

        # -- Text Processing section (common) --
        content.addWidget(_section_label("TEXT PROCESSING"))
        text_form = QFormLayout()
        text_form.setContentsMargins(4, 2, 0, 0)
        text_form.setHorizontalSpacing(12)
        text_form.setVerticalSpacing(6)

        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(1, 20)
        self.window_size_spin.setValue(3)
        self.window_size_label = QLabel("Context window (+/-):")
        self._window_size_info = _form_row(
            self.window_size_label, self.window_size_spin,
            "Words before and after each target word included when<br>"
            "computing sentence embeddings. Larger = more context, slower.",
            text_form,
        )

        self.sif_a_spin = QDoubleSpinBox()
        self.sif_a_spin.setRange(1e-5, 1.0)
        self.sif_a_spin.setDecimals(5)
        self.sif_a_spin.setValue(1e-3)
        self.sif_a_spin.setSingleStep(1e-4)
        _form_row(
            "SIF (a):", self.sif_a_spin,
            "Smooth Inverse Frequency weight. Controls how much to<br>"
            "down-weight common words. Smaller values penalize<br>"
            "frequent words more heavily.",
            text_form,
        )
        content.addLayout(text_form)

        # -- PLS Settings section --
        self.pls_section_label = _section_label("PLS")
        content.addWidget(self.pls_section_label)

        self.pls_frame = QFrame()
        pls_form = QFormLayout(self.pls_frame)
        pls_form.setContentsMargins(4, 2, 0, 0)
        pls_form.setHorizontalSpacing(12)
        pls_form.setVerticalSpacing(6)

        self.pls_n_comp_spin = QSpinBox()
        self.pls_n_comp_spin.setRange(0, 50)
        self.pls_n_comp_spin.setValue(1)
        self.pls_n_comp_spin.setSpecialValueText("auto")
        _form_row(
            "Components:", self.pls_n_comp_spin,
            "Number of PLS components to extract.<br>"
            "'auto' uses cross-validation to select the best number.",
            pls_form,
        )

        self.pls_p_method_combo = QComboBox()
        self.pls_p_method_combo.addItems(["auto", "perm", "split", "split_cal", "none"])
        _form_row(
            "p-method:", self.pls_p_method_combo,
            "Statistical testing method.<br>"
            "<b>perm</b> \u2014 permutation test<br>"
            "<b>split</b> \u2014 formula-calibrated split-half<br>"
            "<b>split_cal</b> \u2014 permutation-calibrated split-half<br>"
            "<b>none</b> \u2014 skip testing",
            pls_form,
        )

        self.pls_n_perm_label = QLabel("Permutations:")
        self.pls_n_perm_spin = QSpinBox()
        self.pls_n_perm_spin.setRange(100, 100000)
        self.pls_n_perm_spin.setValue(1000)
        self.pls_n_perm_spin.setSingleStep(100)
        self._pls_perm_info = _form_row(
            self.pls_n_perm_label, self.pls_n_perm_spin,
            "Number of permutations for the permutation test.<br>"
            "More = more precise p-values, slower.",
            pls_form,
        )

        self.pls_n_splits_label = QLabel("Splits:")
        self.pls_n_splits_spin = QSpinBox()
        self.pls_n_splits_spin.setRange(10, 1000)
        self.pls_n_splits_spin.setValue(50)
        self._pls_splits_info = _form_row(
            self.pls_n_splits_label, self.pls_n_splits_spin,
            "Number of random splits for the split-half test.",
            pls_form,
        )

        self.pls_split_ratio_label = QLabel("Split ratio:")
        self.pls_split_ratio_spin = QDoubleSpinBox()
        self.pls_split_ratio_spin.setRange(0.1, 0.9)
        self.pls_split_ratio_spin.setValue(0.5)
        self.pls_split_ratio_spin.setSingleStep(0.05)
        self.pls_split_ratio_spin.setDecimals(2)
        self._pls_ratio_info = _form_row(
            self.pls_split_ratio_label, self.pls_split_ratio_spin,
            "Proportion of data in the training half<br>"
            "(0.5 = equal split).",
            pls_form,
        )

        self.pls_random_state_combo = QComboBox()
        self.pls_random_state_combo.setEditable(True)
        self.pls_random_state_combo.addItem("default")
        _form_row(
            "Random state:", self.pls_random_state_combo,
            "Seed for reproducibility. 'default' uses a fixed seed;<br>"
            "enter a number for a custom seed.",
            pls_form,
        )

        self.pls_frame.setVisible(True)
        content.addWidget(self.pls_frame)

        # Connect PLS p-method to show/hide perm/split params
        self.pls_p_method_combo.currentTextChanged.connect(self._on_pls_p_method_changed)

        # -- PCA+OLS section --
        self.sweep_section_label = _section_label("PCA + OLS")
        content.addWidget(self.sweep_section_label)
        self.sweep_section_label.setVisible(False)

        self.sweep_frame = QFrame()
        sweep_form = QFormLayout(self.sweep_frame)
        sweep_form.setContentsMargins(4, 2, 0, 0)
        sweep_form.setHorizontalSpacing(12)
        sweep_form.setVerticalSpacing(6)

        # Fixed K option
        fixed_k_row = QHBoxLayout()
        self.fixed_k_check = QCheckBox("Use fixed K")
        self.fixed_k_spin = QSpinBox()
        self.fixed_k_spin.setRange(2, 1000)
        self.fixed_k_spin.setValue(50)
        self.fixed_k_spin.setEnabled(False)
        fixed_k_row.addWidget(self.fixed_k_check)
        fixed_k_row.addWidget(self.fixed_k_spin)
        fixed_k_row.addWidget(InfoButton(
            "When checked, use a fixed number of PCA components<br>"
            "instead of sweeping a range to find the best K."
        ))
        fixed_k_row.addStretch()
        sweep_form.addRow("PCA K:", fixed_k_row)
        self.fixed_k_check.toggled.connect(self._on_fixed_k_toggled)

        # Sweep range
        sweep_range_row = QHBoxLayout()
        self.k_min_spin = QSpinBox()
        self.k_min_spin.setRange(2, 500)
        self.k_min_spin.setValue(20)
        sweep_range_row.addWidget(self.k_min_spin)
        sweep_range_row.addWidget(QLabel("to"))
        self.k_max_spin = QSpinBox()
        self.k_max_spin.setRange(3, 1000)
        self.k_max_spin.setValue(120)
        sweep_range_row.addWidget(self.k_max_spin)
        sweep_range_row.addWidget(QLabel("step"))
        self.k_step_spin = QSpinBox()
        self.k_step_spin.setRange(1, 50)
        self.k_step_spin.setValue(2)
        sweep_range_row.addWidget(self.k_step_spin)
        sweep_range_row.addWidget(InfoButton(
            "Range and step size for the number of clusters (K)<br>"
            "to evaluate. The best K is selected by R² on the outcome."
        ))
        sweep_range_row.addStretch()
        self.sweep_range_label = QLabel("K sweep:")
        sweep_form.addRow(self.sweep_range_label, sweep_range_row)
        # Keep references to sweep range widgets for enabling/disabling
        self._sweep_range_widgets = [
            self.k_min_spin, self.k_max_spin, self.k_step_spin,
        ]

        self.sweep_frame.setVisible(False)
        content.addWidget(self.sweep_frame)

        # -- Groups section --
        self.groups_section_label = _section_label("GROUP COMPARISON")
        content.addWidget(self.groups_section_label)
        self.groups_section_label.setVisible(False)

        self.groups_frame = QFrame()
        groups_form = QFormLayout(self.groups_frame)
        groups_form.setContentsMargins(4, 2, 0, 0)
        groups_form.setHorizontalSpacing(12)
        groups_form.setVerticalSpacing(6)

        self.groups_n_perm_spin = QSpinBox()
        self.groups_n_perm_spin.setRange(100, 100000)
        self.groups_n_perm_spin.setValue(5000)
        self.groups_n_perm_spin.setSingleStep(500)
        _form_row(
            "Permutations:", self.groups_n_perm_spin,
            "Number of permutations for the group comparison test.<br>"
            "More = more precise p-values.",
            groups_form,
        )

        self.groups_correction_combo = QComboBox()
        self.groups_correction_combo.addItems(["holm", "bonferroni", "fdr_bh"])
        _form_row(
            "Correction:", self.groups_correction_combo,
            "Multiple comparison correction method.<br>"
            "<b>holm</b> \u2014 recommended (less conservative than<br>"
            "Bonferroni, more power than FDR).<br>"
            "<b>bonferroni</b> \u2014 strict, controls family-wise error.<br>"
            "<b>fdr_bh</b> \u2014 controls false discovery rate.",
            groups_form,
        )

        self.groups_median_split_check = QCheckBox("Median split")
        _form_row(
            "", self.groups_median_split_check,
            "Split a continuous outcome variable into two groups<br>"
            "at the median, then run group comparison.",
            groups_form,
        )

        self.groups_random_state_combo = QComboBox()
        self.groups_random_state_combo.setEditable(True)
        self.groups_random_state_combo.addItem("default")
        _form_row(
            "Random state:", self.groups_random_state_combo,
            "Seed for reproducibility. 'default' uses a fixed seed;<br>"
            "enter a number for a custom seed.",
            groups_form,
        )

        self.groups_frame.setVisible(False)
        content.addWidget(self.groups_frame)

        # -- Clustering section (common) --
        content.addWidget(_section_label("CLUSTERING"))
        cluster_form = QFormLayout()
        cluster_form.setContentsMargins(4, 2, 0, 0)
        cluster_form.setHorizontalSpacing(12)
        cluster_form.setVerticalSpacing(6)

        self.cluster_topn_spin = QSpinBox()
        self.cluster_topn_spin.setRange(20, 500)
        self.cluster_topn_spin.setValue(100)
        _form_row(
            "Top-N words:", self.cluster_topn_spin,
            "Number of nearest words to consider for each<br>"
            "dimension when building word clusters.",
            cluster_form,
        )

        k_row = QHBoxLayout()
        self.cluster_k_auto_check = QCheckBox("Auto")
        self.cluster_k_auto_check.setChecked(True)
        k_row.addWidget(self.cluster_k_auto_check)
        k_row.addSpacing(8)
        k_row.addWidget(QLabel("range"))
        self.cluster_k_min_spin = QSpinBox()
        self.cluster_k_min_spin.setRange(2, 50)
        self.cluster_k_min_spin.setValue(2)
        k_row.addWidget(self.cluster_k_min_spin)
        k_row.addWidget(QLabel("\u2013"))
        self.cluster_k_max_spin = QSpinBox()
        self.cluster_k_max_spin.setRange(3, 100)
        self.cluster_k_max_spin.setValue(10)
        k_row.addWidget(self.cluster_k_max_spin)
        k_row.addWidget(InfoButton(
            "Number of clusters. When Auto is checked, the optimal K<br>"
            "is selected automatically within the given range."
        ))
        k_row.addStretch()
        cluster_form.addRow("K:", k_row)

        content.addLayout(cluster_form)

    def _on_analysis_type_changed(self, checked: bool):
        """Handle analysis type toggle button change."""
        if not checked:
            return
        # Update button styles
        for btn in [self.pls_btn, self.pcaols_btn, self.groups_btn]:
            if btn.isChecked():
                btn.setObjectName("btn_model_select_active")
            else:
                btn.setObjectName("btn_model_select")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        is_groups = self.groups_btn.isChecked()
        is_pls = self.pls_btn.isChecked()
        is_pcaols = self.pcaols_btn.isChecked()
        self.column_label.setText("Group Column:" if is_groups else "Outcome Column:")

        # Show/hide analysis-type-specific settings (frame + section label)
        self.pls_section_label.setVisible(is_pls)
        self.pls_frame.setVisible(is_pls)
        self.sweep_section_label.setVisible(is_pcaols)
        self.sweep_frame.setVisible(is_pcaols)
        self.groups_section_label.setVisible(is_groups)
        self.groups_frame.setVisible(is_groups)

        # Repopulate column combo for the new type
        self._populate_column_combo()
        self._on_column_changed()
        self._update_run_button()

    def _on_pls_p_method_changed(self, method: str):
        """Show/hide PLS sub-params based on selected p-method."""
        shows_perm = method in ("auto", "perm", "split_cal")
        shows_split = method in ("auto", "split", "split_cal")
        self.pls_n_perm_label.setVisible(shows_perm)
        self.pls_n_perm_spin.setVisible(shows_perm)
        self._pls_perm_info.setVisible(shows_perm)
        self.pls_n_splits_label.setVisible(shows_split)
        self.pls_n_splits_spin.setVisible(shows_split)
        self._pls_splits_info.setVisible(shows_split)
        self.pls_split_ratio_label.setVisible(shows_split)
        self.pls_split_ratio_spin.setVisible(shows_split)
        self._pls_ratio_info.setVisible(shows_split)

    def _on_fixed_k_toggled(self, checked: bool):
        """Enable/disable fixed K spin and sweep range widgets."""
        self.fixed_k_spin.setEnabled(checked)
        for w in self._sweep_range_widgets:
            w.setEnabled(not checked)

    def _on_mode_changed(self, lexicon_checked: bool):
        """Handle mode toggle."""
        if not self.project:
            return
        self.project.concept_mode = (
            "lexicon" if self.lexicon_radio.isChecked() else "fulldoc"
        )
        is_lexicon = self.lexicon_radio.isChecked()
        self.lexicon_group.setVisible(is_lexicon)
        self.suggestions_group.setVisible(is_lexicon)
        # Context window only relevant in lexicon mode
        self.window_size_label.setVisible(is_lexicon)
        self.window_size_spin.setVisible(is_lexicon)
        self._window_size_info.setVisible(is_lexicon)
        self._update_suggestions_btn_state()
        self._update_run_button()

    def _get_analysis_type(self) -> str:
        if self.pls_btn.isChecked():
            return "pls"
        elif self.pcaols_btn.isChecked():
            return "pca_ols"
        else:
            return "groups"

    def _populate_column_combo(self):
        """Populate column combo based on current analysis type and dataframe."""
        self.column_combo.blockSignals(True)
        old_text = self.column_combo.currentText()
        self.column_combo.clear()
        self.column_combo.addItem("(none)")

        if self.project and self.project._df is not None:
            df = self.project._df
            if self.groups_btn.isChecked():
                # All columns for groups
                self.column_combo.addItems(df.columns.tolist())
            else:
                # Only numeric columns for PLS/PCA+OLS
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                self.column_combo.addItems(numeric_cols)

        # Try to restore previous selection
        idx = self.column_combo.findText(old_text)
        if idx >= 0:
            self.column_combo.setCurrentIndex(idx)
        else:
            self.column_combo.setCurrentIndex(0)
        self.column_combo.blockSignals(False)

    def _on_column_changed(self):
        """Handle column selection change — compute y or groups."""
        if not self.project or self.project._df is None:
            self.column_summary_label.setText("")
            self._update_suggestions_btn_state()
            self._update_run_button()
            return

        col = self.column_combo.currentText()
        if not col or col == "(none)":
            self.column_summary_label.setText("")
            self.project._y = None
            self.project._y_full = None
            self.project._groups = None
            self.project._groups_full = None
            self._update_suggestions_btn_state()
            self._update_run_button()
            return

        df = self.project._df
        if col not in df.columns:
            self.column_summary_label.setText("")
            self._update_suggestions_btn_state()
            self._update_run_button()
            return

        if self.groups_btn.isChecked():
            self._compute_groups(col)
        else:
            self._compute_outcome(col)
        self._update_suggestions_btn_state()
        self._update_run_button()

    def _compute_outcome(self, col: str):
        """Compute outcome variable (y) from selected column."""
        df = self.project._df
        outcome = pd.to_numeric(df[col], errors="coerce")
        id_row_indices = self.project._id_row_indices

        if id_row_indices is not None:
            y_list = []
            for row_indices in id_row_indices:
                vals = outcome.iloc[row_indices].dropna()
                y_list.append(vals.iloc[0] if len(vals) > 0 else np.nan)
            y_series = pd.Series(y_list)
            n_valid = (~y_series.isna()).sum()
            y = y_series.dropna().to_numpy()
            y_full = y_series.to_numpy(dtype=np.float64)
        else:
            n_valid = (~outcome.isna()).sum()
            y = outcome.dropna().to_numpy()
            y_full = outcome.to_numpy(dtype=np.float64)

        if n_valid > 0:
            self.column_summary_label.setText(
                f"n={n_valid:,}, mean={y.mean():.3f}, std={y.std():.3f}"
            )
            self.project._y = y
            self.project._y_full = y_full
            self.project._groups = None
            self.project._groups_full = None
            self.project.outcome_column = col
            self.project.n_valid = int(n_valid)
            self._apply_outcome_filter(col)
        else:
            self.column_summary_label.setText("No valid numeric values")
            self.project._y = None
            self.project._y_full = None

    def _apply_outcome_filter(self, col: str):
        """Filter cached docs to match rows/profiles with valid outcome."""
        corpus = self.project._corpus
        if corpus is None:
            from ..utils.file_io import ProjectIO
            corpus = ProjectIO.load_corpus(self.project)
        if corpus is None or not corpus.docs:
            return
        all_pre_docs, all_docs = corpus.pre_docs, corpus.docs
        id_row_indices = self.project._id_row_indices
        outcome = pd.to_numeric(self.project._df[col], errors="coerce")

        if id_row_indices is not None:
            per_id_mask = [
                any(not np.isnan(outcome.iloc[ri]) for ri in row_indices)
                for row_indices in id_row_indices
            ]
            self.project._docs = [all_docs[i] for i, ok in enumerate(per_id_mask) if ok]
            self.project._pre_docs = [all_pre_docs[i] for i, ok in enumerate(per_id_mask) if ok]
        else:
            mask = ~outcome.isna()
            self.project._docs = [all_docs[i] for i in range(len(all_docs)) if mask.iat[i]]
            self.project._pre_docs = [all_pre_docs[i] for i in range(len(all_pre_docs)) if mask.iat[i]]

    def _compute_groups(self, col: str):
        """Compute group variable from selected column."""
        df = self.project._df
        id_row_indices = self.project._id_row_indices

        if id_row_indices is not None:
            per_id_groups = []
            for row_indices in id_row_indices:
                vals = df[col].iloc[row_indices].dropna()
                vals = vals[vals.astype(str).str.strip() != ""]
                per_id_groups.append(str(vals.iloc[0]) if len(vals) > 0 else "")
            groups = pd.Series(per_id_groups)
        else:
            groups = df[col].copy()

        n_missing = groups.isna().sum() + (groups.astype(str).str.strip() == "").sum()
        groups_clean = groups.dropna()
        groups_clean = groups_clean[groups_clean.astype(str).str.strip() != ""]
        unique = groups_clean.astype(str).unique()
        n_groups = len(unique)

        counts_str = ", ".join(
            f"{g}: {(groups_clean.astype(str) == g).sum()}"
            for g in sorted(unique)
        )
        summary = f"{n_groups} groups, n={len(groups_clean):,}"
        if n_missing > 0:
            summary += f" ({n_missing} dropped)"
        if n_groups <= 6:
            summary += f"  [{counts_str}]"
        self.column_summary_label.setText(summary)

        self.project._groups = groups.to_numpy()
        self.project._y = None
        self.project._y_full = None
        self.project.group_column = col
        self._apply_group_filter(col)

    def _apply_group_filter(self, col: str):
        """Filter cached docs/profiles to match rows with valid groups."""
        corpus = self.project._corpus
        if corpus is None:
            from ..utils.file_io import ProjectIO
            corpus = ProjectIO.load_corpus(self.project)
        if corpus is None or not corpus.docs:
            return
        all_pre_docs, all_docs = corpus.pre_docs, corpus.docs
        id_row_indices = self.project._id_row_indices
        df = self.project._df

        if id_row_indices is not None:
            per_id_valid = []
            per_id_group_vals = []
            for row_indices in id_row_indices:
                vals = df[col].iloc[row_indices].dropna()
                vals = vals[vals.astype(str).str.strip() != ""]
                if len(vals) > 0:
                    per_id_valid.append(True)
                    per_id_group_vals.append(str(vals.iloc[0]))
                else:
                    per_id_valid.append(False)
                    per_id_group_vals.append("")
            self.project._docs = [all_docs[i] for i, ok in enumerate(per_id_valid) if ok]
            self.project._pre_docs = [all_pre_docs[i] for i, ok in enumerate(per_id_valid) if ok]
            self.project._groups = np.array([g for g, ok in zip(per_id_group_vals, per_id_valid) if ok])
            # Full-length groups aligned with corpus.docs (None for invalid)
            self.project._groups_full = np.array(
                [g if ok else None for g, ok in zip(per_id_group_vals, per_id_valid)],
                dtype=object,
            )
        else:
            groups = df[col]
            mask = ~(groups.isna() | (groups.astype(str).str.strip() == ""))
            self.project._docs = [all_docs[i] for i in range(len(all_docs)) if mask.iat[i]]
            self.project._pre_docs = [all_pre_docs[i] for i in range(len(all_pre_docs)) if mask.iat[i]]
            self.project._groups = groups[mask].astype(str).to_numpy()
            # Full-length groups aligned with corpus.docs (None for invalid)
            groups_full = groups.astype(str).to_numpy().copy()
            groups_full[~mask.to_numpy()] = None
            self.project._groups_full = np.array(groups_full, dtype=object)
        self.project._y = None

    def _create_current_lexicon_panel(self) -> QGroupBox:
        """Create the Current Lexicon panel (middle column)."""
        group = QGroupBox("Current Lexicon")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        self._add_overlay_info(group,
            "<b>Current Lexicon</b><br><br>"
            "Your working lexicon — the word list that defines your concept.<br><br>"
            "<b>Top bar</b> — type a keyword and press Enter/Add, or paste "
            "a list of tokens.<br>"
            "<b>Coverage stats</b> — how well the lexicon covers your "
            "documents and its association with the outcome variable.<br>"
            "<b>Token table</b> — each token with its frequency, association, "
            "p-value, and effect direction. Select rows and Remove to refine.",
        )

        # -- Top bar: token input + Add + Paste --
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)

        self.token_input = QLineEdit()
        self.token_input.setPlaceholderText("Enter a keyword...")
        self.token_input.returnPressed.connect(self._add_token)
        top_bar.addWidget(self.token_input, stretch=1)

        add_btn = QPushButton("Add")
        add_btn.setMinimumWidth(60)
        add_btn.clicked.connect(self._add_token)
        top_bar.addWidget(add_btn)

        paste_btn = QPushButton("Paste Token List...")
        paste_btn.clicked.connect(self._paste_tokens)
        top_bar.addWidget(paste_btn)

        layout.addLayout(top_bar)

        # -- Coverage stats --
        self.coverage_stats = QLabel("Add tokens to see coverage statistics")
        self.coverage_stats.setWordWrap(True)
        layout.addWidget(self.coverage_stats)

        self.coverage_warnings = QLabel("")
        self.coverage_warnings.setWordWrap(True)
        layout.addWidget(self.coverage_warnings)

        # -- Lexicon table --
        self.lexicon_table = QTableWidget()
        self.lexicon_table.setColumnCount(5)
        self.lexicon_table.setHorizontalHeaderLabels([
            "Token", "Freq", "Assoc", "p-value", "Direction",
        ])
        header = self.lexicon_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setMinimumSectionSize(40)
        for col in range(1, 5):
            header.setSectionResizeMode(col, QHeaderView.Fixed)
        self.lexicon_table.setColumnWidth(1, 50)   # Freq
        self.lexicon_table.setColumnWidth(2, 62)   # Assoc
        self.lexicon_table.setColumnWidth(3, 62)   # p-value
        self.lexicon_table.setColumnWidth(4, 68)   # Direction
        self.lexicon_table.setAlternatingRowColors(True)
        self.lexicon_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.lexicon_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.lexicon_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.lexicon_table.setSortingEnabled(True)
        layout.addWidget(self.lexicon_table)

        # -- Bottom: Remove + Clear + count --
        bottom_bar = QHBoxLayout()

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected_tokens)
        bottom_bar.addWidget(remove_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.setObjectName("btn_secondary")
        clear_btn.clicked.connect(self._clear_lexicon)
        bottom_bar.addWidget(clear_btn)

        bottom_bar.addStretch()

        self.lexicon_count_label = QLabel("0 tokens")
        self.lexicon_count_label.setObjectName("label_muted")
        bottom_bar.addWidget(self.lexicon_count_label)

        layout.addLayout(bottom_bar)

        return group

    def _create_suggestions_panel(self) -> QGroupBox:
        """Create the Suggestions panel (right column)."""
        group = QGroupBox("Lexicon Suggestions")
        layout = QVBoxLayout(group)

        self._add_overlay_info(group,
            "<b>Lexicon Suggestions</b><br><br>"
            "Words suggested based on statistical association with your "
            "outcome variable.<br><br>"
            "Double-click a row to add it to your lexicon.<br>"
            "Click <b>Get Suggestions</b> to refresh after changing "
            "your lexicon.",
        )

        suggestions_desc = QLabel(
            "Double-click a row to add it to your lexicon."
        )
        suggestions_desc.setObjectName("label_muted")
        layout.addWidget(suggestions_desc)

        self.suggestions_table = QTableWidget()
        self.suggestions_table.setColumnCount(5)
        self.suggestions_table.setHorizontalHeaderLabels([
            "Token", "Freq", "Assoc", "p-value", "Direction",
        ])
        self.suggestions_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        for col in range(1, 5):
            self.suggestions_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeToContents
            )
        self.suggestions_table.setAlternatingRowColors(True)
        self.suggestions_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.suggestions_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.suggestions_table.cellDoubleClicked.connect(self._add_suggestion)
        layout.addWidget(self.suggestions_table)

        self.refresh_suggestions_btn = QPushButton("Get Suggestions")
        self.refresh_suggestions_btn.setEnabled(False)
        self.refresh_suggestions_btn.setToolTip(
            "Select an outcome or group column first"
        )
        self.refresh_suggestions_btn.clicked.connect(self._update_suggestions)
        layout.addWidget(self.refresh_suggestions_btn)

        return group

    def _create_run_section_widget(self, parent_layout):
        """Create the run analysis section and add it to the parent layout."""
        self.bottom_splitter = QSplitter(Qt.Horizontal)

        self.checks_frame = QFrame()
        self.checks_frame.setObjectName("frame_ready_pending")
        self.checks_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.checks_frame.setLineWidth(2)
        checks_layout = QVBoxLayout()

        self.checks_title = QLabel("Pre-Run Checks")
        self.checks_title.setObjectName("label_title")
        checks_layout.addWidget(self.checks_title)

        self.sanity_checks_label = QLabel("")
        self.sanity_checks_label.setWordWrap(True)
        checks_layout.addWidget(self.sanity_checks_label)

        self.checks_frame.setLayout(checks_layout)
        self.bottom_splitter.addWidget(self.checks_frame)

        nav_widget = QWidget()
        nav_layout = QVBoxLayout(nav_widget)
        nav_layout.setContentsMargins(8, 8, 0, 8)

        self.run_btn = QPushButton("Run SSD Analysis")
        self.run_btn.setMinimumHeight(48)
        self.run_btn.setObjectName("btn_run_analysis")
        self.run_btn.setEnabled(False)
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self._on_run_clicked)
        nav_layout.addWidget(self.run_btn)

        nav_layout.addStretch()

        nav_buttons_layout = QHBoxLayout()

        back_btn = QPushButton("\u2039  Back to Setup")
        back_btn.setObjectName("btn_secondary")
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.clicked.connect(self._go_back)
        nav_buttons_layout.addWidget(back_btn)

        self.view_results_btn = QPushButton("View Results")
        self.view_results_btn.setObjectName("btn_ghost")
        self.view_results_btn.setEnabled(False)
        self.view_results_btn.setCursor(Qt.PointingHandCursor)
        self.view_results_btn.clicked.connect(self._go_to_results)
        nav_buttons_layout.addWidget(self.view_results_btn)

        nav_layout.addLayout(nav_buttons_layout)

        self.bottom_splitter.addWidget(nav_widget)
        self.bottom_splitter.setSizes([500, 300])

        parent_layout.addWidget(self.bottom_splitter)

    # ------------------------------------------------------------------ #
    #  Splitter persistence
    # ------------------------------------------------------------------ #

    def _save_splitter_states(self):
        """Save splitter sizes to QSettings."""
        self._settings.setValue("run_tab/main_splitter_3col", self.main_splitter.saveState())
        self._settings.setValue("run_tab/bottom_splitter", self.bottom_splitter.saveState())

    def _restore_splitter_states(self):
        """Restore splitter sizes from QSettings."""
        try:
            state = self._settings.value("run_tab/main_splitter_3col")
            if state is not None:
                self.main_splitter.restoreState(state)
        except Exception:
            pass
        try:
            state = self._settings.value("run_tab/bottom_splitter")
            if state is not None:
                self.bottom_splitter.restoreState(state)
        except Exception:
            pass

    def hideEvent(self, event):
        """Save splitter state when tab is hidden."""
        self._save_splitter_states()
        super().hideEvent(event)

    # ------------------------------------------------------------------ #
    #  Pre-flight review panel
    # ------------------------------------------------------------------ #

    @staticmethod
    def _html_palette():
        """Return the current theme palette for HTML content styling."""
        from ..theme import build_current_palette
        return build_current_palette()

    def _build_review_html(self) -> str:
        """Build HTML content for the pre-flight review panel."""
        if not self.project:
            return "<p>No project loaded.</p>"

        pal = self._html_palette()
        p = self.project

        is_crossgroup = p.analysis_type == "groups"
        is_lexicon = p.concept_mode == "lexicon"

        label_style = (
            f"color: {pal.text_secondary}; font-size: {pal.font_size_sm}; "
            f"text-transform: uppercase;"
        )
        value_style = f"font-size: {pal.font_size_base}; padding-left: 8px;"
        section_style = (
            f"color: {pal.accent}; font-size: 12px; font-weight: 600; "
            f"border-bottom: 1px solid {pal.border}; padding-bottom: 4px; "
            f"margin: 12px 0 8px 0;"
        )

        html = []

        # --- Analysis Type ---
        html.append(f'<div style="{section_style}">Analysis Type</div>')
        html.append('<table cellspacing="6" style="width: 100%;">')
        type_label = {
            "pls": "PLS (Continuous)",
            "pca_ols": "PCA+OLS (Continuous)",
            "groups": "Group Comparison",
        }.get(p.analysis_type, p.analysis_type)
        html.append(
            f'<tr><td style="{label_style}">Type</td>'
            f'<td style="{value_style}">{type_label}</td></tr>'
        )
        html.append(
            f'<tr><td style="{label_style}">Mode</td>'
            f'<td style="{value_style}">{"Lexicon" if is_lexicon else "Full Document"}</td></tr>'
        )

        if is_crossgroup:
            col = p.group_column or "(not set)"
            html.append(
                f'<tr><td style="{label_style}">Group Column</td>'
                f'<td style="{value_style}">{col}</td></tr>'
            )
            if p._groups is not None and len(p._groups) > 0:
                unique = np.unique(p._groups)
                html.append(
                    f'<tr><td style="{label_style}">Groups</td>'
                    f'<td style="{value_style}">{len(unique)} groups, n={len(p._groups):,}</td></tr>'
                )
        else:
            col = p.outcome_column or "(not set)"
            html.append(
                f'<tr><td style="{label_style}">Outcome Column</td>'
                f'<td style="{value_style}">{col}</td></tr>'
            )
            if p._y is not None:
                y = p._y
                html.append(
                    f'<tr><td style="{label_style}">Samples</td>'
                    f'<td style="{value_style}">n={len(y):,}, '
                    f'mean={y.mean():.3f}, std={y.std():.3f}</td></tr>'
                )
        html.append("</table>")

        # --- Dataset ---
        n_docs = len(p._docs) if p._docs else 0
        mean_words = p.mean_words_before_stopwords
        html.append(f'<div style="{section_style}">Dataset</div>')
        html.append('<table cellspacing="6" style="width: 100%;">')
        if p.csv_path:
            html.append(
                f'<tr><td style="{label_style}">File</td>'
                f'<td style="{value_style}; word-break: break-all;">{p.csv_path.name}</td></tr>'
            )
        if p.text_column:
            html.append(
                f'<tr><td style="{label_style}">Text Column</td>'
                f'<td style="{value_style}">{p.text_column}</td></tr>'
            )
        html.append(
            f'<tr><td style="{label_style}">Documents</td>'
            f'<td style="{value_style}">{n_docs:,}</td></tr>'
        )
        if mean_words > 0:
            html.append(
                f'<tr><td style="{label_style}">Mean Words / Doc</td>'
                f'<td style="{value_style}">~{mean_words:.0f} (pre-stopword)</td></tr>'
            )
        html.append("</table>")

        # --- Text Processing ---
        html.append(f'<div style="{section_style}">Text Processing (spaCy)</div>')
        html.append('<table cellspacing="6" style="width: 100%;">')
        if p.input_mode == "custom" and p.spacy_model:
            _display_model = p.spacy_model
        else:
            try:
                from ssdiff.lang_config import lang_to_model
                _display_model = lang_to_model(p.language)
            except (ImportError, KeyError):
                _display_model = f"{p.language}_core_news_lg"
        html.append(
            f'<tr><td style="{label_style}">Model</td>'
            f'<td style="{value_style}">{_display_model}</td></tr>'
        )
        html.append(
            f'<tr><td style="{label_style}">Language</td>'
            f'<td style="{value_style}">{p.language}</td></tr>'
        )
        html.append(
            f'<tr><td style="{label_style}">Input Mode</td>'
            f'<td style="{value_style}">{p.input_mode}</td></tr>'
        )
        _sw_labels = {"default": "Default", "none": "Disabled", "custom": "Custom file"}
        html.append(
            f'<tr><td style="{label_style}">Stopwords</td>'
            f'<td style="{value_style}">{_sw_labels.get(p.stopword_mode, p.stopword_mode)}</td></tr>'
        )
        html.append("</table>")

        # --- Embeddings ---
        html.append(f'<div style="{section_style}">Embeddings</div>')
        html.append('<table cellspacing="6" style="width: 100%;">')
        if p.embeddings_ready:
            html.append(
                f'<tr><td style="{label_style}">Vocabulary</td>'
                f'<td style="{value_style}">{p.vocab_size:,} words, {p.embedding_dim}d</td></tr>'
            )
            if p.selected_embedding:
                html.append(
                    f'<tr><td style="{label_style}">File</td>'
                    f'<td style="{value_style}; word-break: break-all;">{p.selected_embedding}</td></tr>'
                )
            if p.emb_coverage_pct > 0:
                cov_str = f"{p.emb_coverage_pct:.1f}%"
                if p.emb_n_oov > 0:
                    cov_str += f" ({p.emb_n_oov:,} OOV)"
                html.append(
                    f'<tr><td style="{label_style}">Coverage</td>'
                    f'<td style="{value_style}">{cov_str}</td></tr>'
                )
            html.append(
                f'<tr><td style="{label_style}">L2 Normalized</td>'
                f'<td style="{value_style}">{"Yes" if p.l2_normalized else "No"}</td></tr>'
            )
            if p.abtt_m > 0:
                html.append(
                    f'<tr><td style="{label_style}">ABTT</td>'
                    f'<td style="{value_style}">m={p.abtt_m}</td></tr>'
                )
        else:
            html.append(
                f'<tr><td style="{label_style}">Status</td>'
                f'<td style="{value_style}; color: {pal.error};">Not loaded</td></tr>'
            )
        html.append("</table>")

        # --- Hyperparameters ---
        html.append(f'<div style="{section_style}">Hyperparameters</div>')
        html.append('<table cellspacing="6" style="width: 100%;">')
        if is_lexicon:
            html.append(
                f'<tr><td style="{label_style}">Context Window</td>'
                f'<td style="{value_style}">\u00b1{p.context_window_size}</td></tr>'
            )
        html.append(
            f'<tr><td style="{label_style}">SIF Parameter</td>'
            f'<td style="{value_style}">{p.sif_a:.5f}</td></tr>'
        )
        if p.clustering_k_auto:
            cluster_k_str = f"auto / silhouette ({p.clustering_k_min}\u2013{p.clustering_k_max})"
        else:
            cluster_k_str = f"{p.clustering_k_min}\u2013{p.clustering_k_max}"
        html.append(
            f'<tr><td style="{label_style}">Clustering</td>'
            f'<td style="{value_style}">Top {p.clustering_topn}, '
            f'K={cluster_k_str}, '
            f'{p.clustering_top_words} top words'
            f'</td></tr>'
        )
        html.append("</table>")

        return "".join(html)

    # ------------------------------------------------------------------ #
    #  Token management methods
    # ------------------------------------------------------------------ #

    def _add_token(self):
        """Add a token from the input field."""
        token = self.token_input.text().strip().lower()
        if not token:
            return
        self._add_token_to_lexicon(token)
        self.token_input.clear()

    def _add_token_to_lexicon(self, token: str) -> bool:
        """Add a single token to the lexicon."""
        token = token.strip().lower()
        if not token or token in self.lexicon:
            return False
        self.lexicon.add(token)
        self._update_lexicon_display()
        self._update_coverage()
        return True

    def _paste_tokens(self):
        """Paste multiple tokens from clipboard or dialog."""
        text, ok = QInputDialog.getMultiLineText(
            self,
            "Paste Token List",
            "Enter tokens (comma, space, or newline separated):",
        )
        if not ok or not text:
            return

        import re
        tokens = re.split(r'[,\s\n]+', text)
        tokens = [t.strip().lower() for t in tokens if t.strip()]

        if not tokens:
            return

        added = 0
        for token in tokens:
            if token and token not in self.lexicon:
                self.lexicon.add(token)
                added += 1

        self._update_lexicon_display()
        self._update_coverage()

        QMessageBox.information(self, "Import Complete", f"Added {added} tokens.")

    def _remove_selected_tokens(self):
        """Remove selected tokens from lexicon."""
        selected_rows = set(idx.row() for idx in self.lexicon_table.selectedIndexes())
        for row in selected_rows:
            item = self.lexicon_table.item(row, 0)
            if item:
                self.lexicon.discard(item.text())
        self._update_lexicon_display()
        self._update_coverage()

    def _clear_lexicon(self):
        """Clear all tokens from lexicon."""
        if not self.lexicon:
            return

        reply = QMessageBox.question(
            self, "Clear Lexicon",
            f"Remove all {len(self.lexicon)} tokens?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.lexicon.clear()
            self._update_lexicon_display()
            self._update_coverage()

    def _add_suggestion(self, row: int, column: int):
        """Add a suggestion from the table to the lexicon."""
        item = self.suggestions_table.item(row, 0)
        if item:
            self._add_token_to_lexicon(item.text())

    def _update_lexicon_display(self):
        """Update the current lexicon table display."""
        self.lexicon_table.setSortingEnabled(False)
        self.lexicon_table.setRowCount(0)
        tokens = sorted(self.lexicon)
        self.lexicon_table.setRowCount(len(tokens))
        for i, token in enumerate(tokens):
            self.lexicon_table.setItem(i, 0, QTableWidgetItem(token))
            for col in range(1, 5):
                self.lexicon_table.setItem(i, col, QTableWidgetItem(""))
        self.lexicon_table.setSortingEnabled(True)
        self.lexicon_count_label.setText(f"{len(self.lexicon)} tokens")
        self._update_run_button()

    # ------------------------------------------------------------------ #
    #  Coverage & Suggestions — analysis-type aware
    # ------------------------------------------------------------------ #

    def _is_crossgroup(self) -> bool:
        return self.groups_btn.isChecked()

    def _update_suggestions_btn_state(self):
        """Enable/disable 'Get Suggestions' based on coverage data availability."""
        docs, var, _ = self._get_coverage_data()
        has_data = docs is not None and var is not None
        self.refresh_suggestions_btn.setEnabled(has_data)
        if has_data:
            self.refresh_suggestions_btn.setToolTip("")
        else:
            self.refresh_suggestions_btn.setToolTip(
                "Select an outcome or group column first"
            )

    def _get_coverage_data(self):
        """Return (docs, var, var_type) for coverage functions."""
        if not self.project:
            return None, None, None
        docs = self.project._docs
        if not docs:
            return None, None, None

        if self._is_crossgroup():
            groups = self.project._groups
            if groups is None or len(groups) == 0:
                return None, None, None
            return docs, groups, "categorical"
        else:
            y = self.project._y
            if y is None:
                return None, None, None
            return docs, y, "continuous"

    def _get_corpus_y(self):
        """Return (corpus, y_full, var_type) for Corpus lexicon methods.

        ``y_full`` is aligned with ``corpus.docs`` (same length, NaN/None
        for invalid entries).  Returns ``(None, None, None)`` if data is
        not ready.
        """
        if not self.project:
            return None, None, None
        corpus = self.project._corpus
        if corpus is None or not corpus.docs:
            return None, None, None

        if self._is_crossgroup():
            groups_full = getattr(self.project, "_groups_full", None)
            if groups_full is None:
                return None, None, None
            return corpus, groups_full, "categorical"
        else:
            y_full = getattr(self.project, "_y_full", None)
            if y_full is None:
                return None, None, None
            return corpus, y_full, "continuous"

    def _update_coverage(self):
        """Update coverage statistics in the current lexicon table."""
        if not self.project or not self.lexicon:
            self.coverage_stats.setText("Add tokens to see coverage statistics")
            self.coverage_warnings.setText("")
            self._update_run_button()
            return

        corpus, y_full, var_type = self._get_corpus_y()
        if corpus is None or y_full is None:
            self.coverage_stats.setText("Data not loaded or column not selected")
            self._update_run_button()
            return

        try:
            per_token = corpus.token_stats(
                y_full, self.lexicon, var_type=var_type,
            )
            summary = corpus.coverage_summary(
                y_full, self.lexicon, var_type=var_type,
            )
        except Exception as e:
            self.coverage_stats.setText(f"Coverage computation failed: {e}")
            self._update_run_button()
            return

        n_docs = len(self.project._docs) if self.project._docs else len(corpus.docs)
        cov_pct = summary["cov_all"] * 100

        if var_type == "categorical":
            stats_text = (
                f"Documents with any hit: {summary['docs_any']:,} / {n_docs:,} "
                f"({cov_pct:.1f}%)\n"
                f"Min group coverage: {summary.get('q1', 0) * 100:.1f}%  |  "
                f"Max group coverage: {summary.get('q4', 0) * 100:.1f}%\n"
                f"Cram\u00e9r's V (any hit vs group): {summary.get('corr_any', 0):.4f}\n"
                f"Hits per doc \u2014 mean: {summary['hits_mean']:.2f}, "
                f"median: {summary['hits_median']:.1f}"
            )
            assoc_header = "Assoc (V)"
        else:
            stats_text = (
                f"Documents with any hit: {summary['docs_any']:,} / {n_docs:,} "
                f"({cov_pct:.1f}%)\n"
                f"Q1 coverage: {summary['q1'] * 100:.1f}%  |  "
                f"Q4 coverage: {summary['q4'] * 100:.1f}%\n"
                f"Correlation (any hit vs outcome): {summary['corr_any']:.4f}\n"
                f"Hits per doc \u2014 mean: {summary['hits_mean']:.2f}, "
                f"median: {summary['hits_median']:.1f}\n"
                f"Types per doc \u2014 mean: {summary['types_mean']:.2f}, "
                f"median: {summary['types_median']:.1f}"
            )
            assoc_header = "Assoc (r)"
        self.coverage_stats.setText(stats_text)

        # Update the Assoc column header to reflect the analysis type
        self.lexicon_table.setHorizontalHeaderLabels([
            "Token", "Freq", assoc_header, "p-value", "Direction",
        ])

        # Warnings
        warnings = []
        if len(self.lexicon) < 5:
            warnings.append("Small lexicon (< 5 tokens)")
        if cov_pct < 30:
            warnings.append(f"Low coverage ({cov_pct:.1f}%)")
        zero_freq = [row for row in per_token if row.get("freq", 0) == 0]
        if zero_freq:
            oov_words = ", ".join(row["token"] for row in zero_freq[:5])
            warnings.append(
                f"{len(zero_freq)} token(s) with 0 frequency: {oov_words}"
            )

        if warnings:
            self.coverage_warnings.setText("\n".join(warnings))
            self.coverage_warnings.setObjectName("label_coverage_warn")
        else:
            self.coverage_warnings.setText("Coverage looks good")
            self.coverage_warnings.setObjectName("label_coverage_ok")
        self.coverage_warnings.style().unpolish(self.coverage_warnings)
        self.coverage_warnings.style().polish(self.coverage_warnings)

        # Fill stats into the existing lexicon table rows
        stats_by_token = {row.get("token", ""): row for row in per_token}
        self.lexicon_table.setSortingEnabled(False)
        for i in range(self.lexicon_table.rowCount()):
            token_item = self.lexicon_table.item(i, 0)
            if not token_item:
                continue
            token = token_item.text()
            row = stats_by_token.get(token, {})
            self.lexicon_table.setItem(
                i, 1, QTableWidgetItem(str(row.get("freq", 0)))
            )
            self.lexicon_table.setItem(
                i, 2, QTableWidgetItem(f"{row.get('corr', 0):.4f}")
            )
            self.lexicon_table.setItem(
                i, 3, QTableWidgetItem(f"{row.get('pvalue', 1.0):.4g}")
            )
            self.lexicon_table.setItem(
                i, 4, QTableWidgetItem(str(row.get("direction", "")))
            )
        self.lexicon_table.setSortingEnabled(True)

        self._update_run_button()

    def _update_suggestions(self):
        """Update the suggestions table using Corpus.suggest_lexicon."""
        self.suggestions_table.setRowCount(0)

        if not self.project:
            return

        corpus, y_full, var_type = self._get_corpus_y()
        if corpus is None or y_full is None:
            return

        try:
            # suggest_lexicon returns LexiconResult (iterable of dicts)
            suggested = corpus.suggest_lexicon(
                y_full, var_type=var_type,
            )

            # Filter out tokens already in the lexicon
            suggested = [
                row for row in suggested
                if row["token"] not in self.lexicon
            ][:50]

            if not suggested:
                return

        except Exception as e:
            QMessageBox.warning(
                self, "Error", f"suggest_lexicon failed: {e}"
            )
            return

        if var_type == "categorical":
            self.suggestions_table.setHorizontalHeaderLabels([
                "Token", "Freq", "Assoc (V)", "p-value", "Direction",
            ])
        else:
            self.suggestions_table.setHorizontalHeaderLabels([
                "Token", "Freq", "Assoc (r)", "p-value", "Direction",
            ])

        self.suggestions_table.setRowCount(len(suggested))
        for i, row in enumerate(suggested):
            self.suggestions_table.setItem(
                i, 0, QTableWidgetItem(row["token"])
            )
            self.suggestions_table.setItem(
                i, 1, QTableWidgetItem(str(row.get("freq", 0)))
            )
            self.suggestions_table.setItem(
                i, 2, QTableWidgetItem(f"{row.get('corr', 0):.4f}")
            )
            self.suggestions_table.setItem(
                i, 3, QTableWidgetItem(f"{row.get('pvalue', 1.0):.4g}")
            )
            self.suggestions_table.setItem(
                i, 4, QTableWidgetItem(str(row.get("direction", "")))
            )

    # ------------------------------------------------------------------ #
    #  Sanity checks & run button
    # ------------------------------------------------------------------ #

    def _update_sanity_checks(self):
        """Update the pre-run sanity checks display."""
        if not self.project:
            self.sanity_checks_label.setText("No project loaded")
            return

        checks = []

        # Data checks
        if self.project._docs:
            n_docs = len(self.project._docs)
            if n_docs >= 200:
                checks.append(f"[OK] {n_docs:,} documents (sufficient)")
            elif n_docs >= 50:
                checks.append(f"[!] {n_docs:,} documents (small but workable)")
            else:
                checks.append(f"[X] {n_docs:,} documents (too few)")
        else:
            checks.append("[X] No documents loaded")

        # Analysis-type-specific checks
        if self._is_crossgroup():
            if self.project._groups is not None and len(self.project._groups) > 0:
                unique = np.unique(self.project._groups)
                n_groups = len(unique)
                if n_groups >= 2:
                    checks.append(f"[OK] {n_groups} groups selected")
                else:
                    checks.append("[X] Need at least 2 groups")
            else:
                checks.append("[X] No group column selected")
        else:
            if self.project._y is not None:
                y = self.project._y
                y_std = np.std(y)
                if y_std > 0.1:
                    checks.append(f"[OK] Outcome variance: {y_std:.3f}")
                else:
                    checks.append(f"[!] Low outcome variance: {y_std:.3f}")
            else:
                checks.append("[X] No outcome column selected")

        # Embeddings check
        if self.project._kv is not None:
            checks.append("[OK] Embeddings loaded")
        else:
            checks.append("[X] Embeddings not loaded")

        # Mode-specific checks
        is_lexicon = self.project.concept_mode == "lexicon"
        if is_lexicon:
            if len(self.lexicon) >= 3:
                checks.append(f"[OK] Lexicon: {len(self.lexicon)} tokens")
            elif len(self.lexicon) > 0:
                checks.append(f"[!] Small lexicon: {len(self.lexicon)} tokens")
            else:
                checks.append("[X] Empty lexicon")
        else:
            checks.append("[OK] Full document mode selected")

        all_ok = all("[X]" not in c for c in checks)

        if all_ok:
            self.checks_frame.setObjectName("frame_ready_ok")
            self.checks_title.setText("Ready to Run")
        else:
            self.checks_frame.setObjectName("frame_ready_pending")
            self.checks_title.setText("Pre-Run Checks")
        self.checks_frame.style().unpolish(self.checks_frame)
        self.checks_frame.style().polish(self.checks_frame)

        self.sanity_checks_label.setText("\n".join(checks))

    def _update_run_button(self):
        """Update the run button enabled state."""
        can_run = False

        if self.project and self.project._docs:
            has_embeddings = self.project._kv is not None

            has_var = False
            if self._is_crossgroup():
                has_var = (
                    self.project._groups is not None
                    and len(self.project._groups) > 0
                    and len(np.unique(self.project._groups)) >= 2
                )
            else:
                has_var = self.project._y is not None

            if has_var and has_embeddings:
                is_lexicon = self.project.concept_mode == "lexicon"
                if is_lexicon:
                    can_run = len(self.lexicon) > 0
                else:
                    can_run = True

        self.run_btn.setEnabled(can_run)
        self._update_sanity_checks()

    def _go_back(self):
        """Go back to Stage 1."""
        main_window = self.window()
        if hasattr(main_window, 'go_to_stage'):
            main_window.go_to_stage(1)

    def _go_to_results(self):
        """Navigate to Stage 3 to view saved results."""
        main_window = self.window()
        if hasattr(main_window, 'go_to_stage'):
            main_window.go_to_stage(3)

    def _on_run_clicked(self):
        """Handle run button click."""
        if not self.project:
            return

        # Save current settings to project before running
        self._save_config_to_project()

        # Pre-run validation
        atype = self._get_analysis_type()
        col = self.column_combo.currentText()

        if atype in ("pls", "pca_ols") and (not col or col == "(none)"):
            QMessageBox.warning(self, "Missing Column",
                                "Please select an outcome column before running.")
            return

        if atype == "groups" and (not col or col == "(none)"):
            QMessageBox.warning(self, "Missing Column",
                                "Please select a group column before running.")
            return

        if self.lexicon_radio.isChecked() and not self.lexicon:
            reply = QMessageBox.question(
                self, "Empty Lexicon",
                "No lexicon tokens defined. Run in full-document mode instead?",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.fulldoc_radio.setChecked(True)
            else:
                return

        # Save concept settings to project
        p = self.project
        p.concept_mode = "lexicon" if self.lexicon_radio.isChecked() else "fulldoc"
        p.analysis_type = atype
        if atype == "groups":
            p.group_column = col if col != "(none)" else None
        else:
            p.outcome_column = col if col != "(none)" else None
        if p.concept_mode == "lexicon":
            p.lexicon_tokens = list(self.lexicon)

        p.mark_dirty()
        self.run_requested.emit()

    def _save_config_to_project(self):
        """Persist current Stage 2 settings to project config."""
        if not self.project:
            return
        p = self.project
        p.analysis_type = self._get_analysis_type()
        p.concept_mode = "lexicon" if self.lexicon_radio.isChecked() else "fulldoc"

        col = self.column_combo.currentText()
        if p.analysis_type == "groups":
            p.group_column = col if col != "(none)" else None
        else:
            p.outcome_column = col if col != "(none)" else None

        # Hyperparameters — common
        p.context_window_size = self.window_size_spin.value()
        p.sif_a = self.sif_a_spin.value()
        p.clustering_topn = self.cluster_topn_spin.value()
        p.clustering_k_auto = self.cluster_k_auto_check.isChecked()
        p.clustering_k_min = self.cluster_k_min_spin.value()
        p.clustering_k_max = self.cluster_k_max_spin.value()
        # clustering_top_words is display-only (default 10); don't overwrite from topn spin

        # PLS
        p.pls_n_components = self.pls_n_comp_spin.value()
        p.pls_p_method = self.pls_p_method_combo.currentText()
        p.pls_n_perm = self.pls_n_perm_spin.value()
        p.pls_n_splits = self.pls_n_splits_spin.value()
        p.pls_split_ratio = self.pls_split_ratio_spin.value()
        p.pls_random_state = self.pls_random_state_combo.currentText()

        # PCA+OLS
        if self.fixed_k_check.isChecked():
            p.pcaols_n_components = self.fixed_k_spin.value()
        else:
            p.pcaols_n_components = None
        p.sweep_k_min = self.k_min_spin.value()
        p.sweep_k_max = self.k_max_spin.value()
        p.sweep_k_step = self.k_step_spin.value()

        # Groups
        p.groups_n_perm = self.groups_n_perm_spin.value()
        p.groups_correction = self.groups_correction_combo.currentText()
        p.groups_median_split = self.groups_median_split_check.isChecked()
        p.groups_random_state = self.groups_random_state_combo.currentText()

        # Persist lexicon tokens
        p.lexicon_tokens = list(self.lexicon)

    # ------------------------------------------------------------------ #
    #  Public methods
    # ------------------------------------------------------------------ #

    def reset(self):
        """Clear all state for project close."""
        self.project = None

    def load_project(self, project: Project):
        """Load a project into the UI."""
        self.project = project

        # Restore analysis type
        atype = project.analysis_type
        if atype == "groups":
            self.groups_btn.setChecked(True)
        elif atype == "pca_ols":
            self.pcaols_btn.setChecked(True)
        else:
            self.pls_btn.setChecked(True)
        # Update button styles
        for btn in [self.pls_btn, self.pcaols_btn, self.groups_btn]:
            btn.setObjectName("btn_model_select_active" if btn.isChecked() else "btn_model_select")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        # Populate column combo and restore selection
        self._populate_column_combo()
        if atype == "groups" and project.group_column:
            idx = self.column_combo.findText(project.group_column)
            if idx >= 0:
                self.column_combo.setCurrentIndex(idx)
        elif project.outcome_column:
            idx = self.column_combo.findText(project.outcome_column)
            if idx >= 0:
                self.column_combo.setCurrentIndex(idx)

        # Restore mode
        is_lexicon = project.concept_mode == "lexicon"
        if is_lexicon:
            self.lexicon_radio.setChecked(True)
        else:
            self.fulldoc_radio.setChecked(True)

        # Show/hide lexicon panels based on mode
        self.lexicon_group.setVisible(is_lexicon)
        self.suggestions_group.setVisible(is_lexicon)

        # Restore advanced settings — common
        self.window_size_spin.setValue(project.context_window_size)
        self.sif_a_spin.setValue(project.sif_a)
        self.cluster_topn_spin.setValue(project.clustering_topn)
        self.cluster_k_auto_check.setChecked(project.clustering_k_auto)
        self.cluster_k_min_spin.setValue(project.clustering_k_min)
        self.cluster_k_max_spin.setValue(project.clustering_k_max)

        # PLS
        self.pls_n_comp_spin.setValue(project.pls_n_components)
        idx = self.pls_p_method_combo.findText(project.pls_p_method)
        if idx >= 0:
            self.pls_p_method_combo.setCurrentIndex(idx)
        self.pls_n_perm_spin.setValue(project.pls_n_perm)
        self.pls_n_splits_spin.setValue(project.pls_n_splits)
        self.pls_split_ratio_spin.setValue(project.pls_split_ratio)
        self.pls_random_state_combo.setCurrentText(project.pls_random_state)

        # PCA+OLS
        if project.pcaols_n_components is not None:
            self.fixed_k_check.setChecked(True)
            self.fixed_k_spin.setValue(project.pcaols_n_components)
        else:
            self.fixed_k_check.setChecked(False)
        self.k_min_spin.setValue(project.sweep_k_min)
        self.k_max_spin.setValue(project.sweep_k_max)
        self.k_step_spin.setValue(project.sweep_k_step)

        # Groups
        self.groups_n_perm_spin.setValue(project.groups_n_perm)
        idx = self.groups_correction_combo.findText(project.groups_correction)
        if idx >= 0:
            self.groups_correction_combo.setCurrentIndex(idx)
        self.groups_median_split_check.setChecked(project.groups_median_split)
        self.groups_random_state_combo.setCurrentText(project.groups_random_state)

        # Context window visibility
        self.window_size_label.setVisible(is_lexicon)
        self.window_size_spin.setVisible(is_lexicon)
        self._window_size_info.setVisible(is_lexicon)

        # Analysis-type-specific frame visibility (frame + section label)
        self.pls_section_label.setVisible(atype == "pls")
        self.pls_frame.setVisible(atype == "pls")
        self.sweep_section_label.setVisible(atype == "pca_ols")
        self.sweep_frame.setVisible(atype == "pca_ols")
        self.groups_section_label.setVisible(atype == "groups")
        self.groups_frame.setVisible(atype == "groups")
        self._on_pls_p_method_changed(project.pls_p_method)

        # Column label
        self.column_label.setText("Group Column:" if atype == "groups" else "Outcome Column:")

        # Trigger column computation
        self._on_column_changed()

        # Restore lexicon from project state (preferred) or latest run (fallback)
        self.lexicon.clear()
        if project.lexicon_tokens:
            self.lexicon = set(project.lexicon_tokens)
        elif project.results:
            snap = project.results[-1].config_snapshot
            if snap.get("concept_mode") == "lexicon" and snap.get("lexicon_tokens"):
                self.lexicon = set(snap["lexicon_tokens"])

        self._update_lexicon_display()

        # Build review panel
        self.review_panel.setHtml(self._build_review_html())

        # Update coverage if in lexicon mode and we have tokens
        if is_lexicon and self.lexicon:
            self._update_coverage()

        self.view_results_btn.setEnabled(bool(project.results))
        self._update_sanity_checks()
        self._update_run_button()
