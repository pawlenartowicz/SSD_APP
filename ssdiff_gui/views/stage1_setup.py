"""Stage 1: Setup view for SSD."""

from pathlib import Path
from typing import Optional

import numpy as np
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
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QRadioButton,
    QButtonGroup,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QFrame,
    QSizePolicy,
    QApplication,
    QStyledItemDelegate,
    QStyle,
)
from PySide6.QtCore import Qt, Signal, QSettings, QEvent, QTimer, QRect, QSize, QPoint
from PySide6.QtGui import QPen, QColor, QMouseEvent

from ..models.project import Project
from ..utils.validators import Validator
from ..utils.worker_threads import PreprocessWorker, EmbeddingWorker, KvConvertWorker, SpacyDownloadWorker, find_local_model
from ..utils.file_io import ProjectIO
from .widgets.collapsible_box import CollapsibleBox
from .widgets.progress_dialog import ProgressDialog
from .widgets.info_button import InfoButton


class _RemovableItemDelegate(QStyledItemDelegate):
    """Delegate that paints an X button on hovered combo-box items."""

    _X_SIZE = 16  # clickable region width/height

    def __init__(self, combo: QComboBox, on_remove, parent=None):
        super().__init__(parent)
        self._combo = combo
        self._on_remove = on_remove
        self._hovered_row = -1

        view = combo.view()
        view.setMouseTracking(True)
        view.viewport().setMouseTracking(True)
        view.viewport().installEventFilter(self)

    # -- painting ----------------------------------------------------------

    def paint(self, painter, option, index):
        super().paint(painter, option, index)

        # Only draw X when this row is hovered
        if index.row() != self._hovered_row:
            return
        # Don't draw on disabled placeholder items
        if not (index.flags() & Qt.ItemIsEnabled):
            return

        painter.save()
        x_rect = self._x_rect(option.rect)
        painter.setPen(QPen(QColor("#8e8ea0"), 1.5))
        margin = 4
        painter.drawLine(
            x_rect.left() + margin, x_rect.top() + margin,
            x_rect.right() - margin, x_rect.bottom() - margin,
        )
        painter.drawLine(
            x_rect.right() - margin, x_rect.top() + margin,
            x_rect.left() + margin, x_rect.bottom() - margin,
        )
        painter.restore()

    def sizeHint(self, option, index):
        hint = super().sizeHint(option, index)
        return QSize(hint.width() + self._X_SIZE + 4, max(hint.height(), self._X_SIZE + 4))

    # -- event filter for hover & click ------------------------------------

    def eventFilter(self, obj, event):
        if obj is self._combo.view().viewport():
            if event.type() == QEvent.MouseMove:
                idx = self._combo.view().indexAt(event.pos())
                row = idx.row() if idx.isValid() else -1
                if row != self._hovered_row:
                    self._hovered_row = row
                    self._combo.view().viewport().update()
            elif event.type() == QEvent.Leave:
                self._hovered_row = -1
                self._combo.view().viewport().update()
            elif event.type() == QEvent.MouseButtonRelease:
                idx = self._combo.view().indexAt(event.pos())
                if idx.isValid() and idx.row() == self._hovered_row:
                    vis_rect = self._combo.view().visualRect(idx)
                    x_rect = self._x_rect(vis_rect)
                    if x_rect.contains(event.pos()):
                        self._on_remove(idx.row())
                        return True
        return super().eventFilter(obj, event)

    # -- helpers -----------------------------------------------------------

    @classmethod
    def _x_rect(cls, item_rect: QRect) -> QRect:
        """Return the clickable X region on the right side of the row."""
        return QRect(
            item_rect.right() - cls._X_SIZE - 4,
            item_rect.center().y() - cls._X_SIZE // 2,
            cls._X_SIZE,
            cls._X_SIZE,
        )


class Stage1Widget(QWidget):
    """Stage 1: Setup - Dataset, spaCy, Embeddings, Hyperparameters."""

    stage_complete = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Optional[Project] = None
        self._df: Optional[pd.DataFrame] = None
        self._worker = None
        self.next_btn = None  # Created in _create_navigation
        self._settings = QSettings("SSD", "SSD")

        self._overlay_info_buttons: list = []
        self._setup_ui()

    # -- overlay info glyphs on QGroupBoxes ----------------------------

    def _add_overlay_info(self, widget, tooltip_html: str):
        """Pin an InfoButton to the top-right corner of *widget*."""
        btn = InfoButton(tooltip_html, parent=widget)
        widget.installEventFilter(self)
        self._overlay_info_buttons.append((widget, btn))
        QTimer.singleShot(0, lambda w=widget, b=btn: self._reposition_info(w, b))

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Resize, QEvent.LayoutRequest):
            for widget, btn in self._overlay_info_buttons:
                if obj is widget:
                    self._reposition_info(widget, btn)
                    break
        return super().eventFilter(obj, event)

    @staticmethod
    def _reposition_info(widget, btn):
        x = widget.width() - btn.width() - 6
        btn.move(x, 20)
        btn.raise_()

    def _setup_ui(self):
        """Set up the user interface."""
        # Main scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)
        layout.setContentsMargins(24, 16, 24, 16)

        # Title
        title = QLabel("Project Setup")
        title.setObjectName("label_title")
        layout.addWidget(title)

        subtitle = QLabel(
            "Configure your dataset, text processing, and embedding settings. "
            "Complete all sections to proceed."
        )
        subtitle.setObjectName("label_muted")
        layout.addWidget(subtitle)

        layout.addSpacing(4)

        # 1. Dataset section
        self._create_dataset_section(layout)

        # 2. spaCy section & 3. Embeddings section — side by side
        mid_row = QHBoxLayout()
        mid_row.addWidget(self._create_spacy_section())
        mid_row.addWidget(self._create_embeddings_section())
        layout.addLayout(mid_row)

        # 4. Analysis configuration
        self._create_analysis_section(layout)

        # 5. Hyperparameters section (collapsible)
        self._create_hyperparams_section(layout)

        # Ready indicator
        self._create_ready_indicator(layout)

        # Navigation buttons
        self._create_navigation(layout)

        layout.addStretch()

        scroll.setWidget(content)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def _create_dataset_section(self, parent_layout):
        """Create the dataset configuration section."""
        group = QGroupBox("1. Dataset")
        layout = QVBoxLayout()

        # File selection row
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Data File:"))
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select a CSV or Excel file...")
        self.file_path_edit.setReadOnly(True)
        file_row.addWidget(self.file_path_edit, stretch=1)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_csv)
        file_row.addWidget(browse_btn)

        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._load_csv)
        file_row.addWidget(load_btn)
        self.load_csv_btn = load_btn

        layout.addLayout(file_row)

        # Column selection
        cols_layout = QHBoxLayout()

        # Text column
        text_col_layout = QVBoxLayout()
        text_col_layout.addWidget(QLabel("Text Column:"))
        self.text_col_combo = QComboBox()
        text_col_layout.addWidget(self.text_col_combo)
        cols_layout.addLayout(text_col_layout)

        # ID column (optional)
        id_col_layout = QVBoxLayout()
        id_col_layout.addWidget(QLabel("ID Column (optional):"))
        self.id_col_combo = QComboBox()
        self.id_col_combo.addItem("(none)")
        id_col_layout.addWidget(self.id_col_combo)
        cols_layout.addLayout(id_col_layout)

        layout.addLayout(cols_layout)

        # Dataset stats
        self.dataset_stats_label = QLabel("")
        self.dataset_stats_label.setObjectName("label_muted")
        layout.addWidget(self.dataset_stats_label)

        # Validate button
        validate_row = QHBoxLayout()
        validate_row.addStretch()
        self.validate_dataset_btn = QPushButton("Validate Dataset")
        self.validate_dataset_btn.clicked.connect(self._validate_dataset)
        self.validate_dataset_btn.setEnabled(False)
        validate_row.addWidget(self.validate_dataset_btn)
        layout.addLayout(validate_row)

        # Status
        self.dataset_status = QLabel("")
        layout.addWidget(self.dataset_status)

        group.setLayout(layout)
        self._add_overlay_info(group,
            "<b>Dataset</b><br><br>"
            "Load a CSV or Excel file containing your text data. "
            "Then select the relevant columns:<br>"
            "<b>Text Column</b> — the column with the texts to analyse.<br>"
            "<b>ID Column</b> (optional) — groups multiple rows per participant "
            "into a single profile.<br>"
            "<b>Outcome / Group Column</b> — the dependent variable "
            "(continuous or categorical).<br><br>"
            "Click <b>Validate</b> to check for missing values and confirm "
            "the dataset is ready.",
        )
        parent_layout.addWidget(group)

    def _create_spacy_section(self):
        """Create the spaCy text processing section."""
        group = QGroupBox("2. Text Processing (spaCy)")
        layout = QVBoxLayout()

        # Language and model selection
        model_row = QHBoxLayout()

        lang_layout = QVBoxLayout()
        lang_layout.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems([
            "en", "ca", "da", "de", "el", "es", "fr", "hr", "it",
            "ja", "ko", "lt", "mk", "nb", "nl", "pl", "pt", "ro",
            "ru", "sl", "sv", "uk", "zh",
        ])
        self.language_combo.currentTextChanged.connect(self._on_language_changed)
        lang_layout.addWidget(self.language_combo)
        model_row.addLayout(lang_layout)

        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel("spaCy Model:"))
        self.model_combo = QComboBox()
        self._update_model_options("en")
        model_layout.addWidget(self.model_combo)
        model_row.addLayout(model_layout)

        model_row.addStretch()
        layout.addLayout(model_row)

        # Preprocessing options
        options_layout = QHBoxLayout()

        self.lemmatize_check = QCheckBox("Lemmatize tokens")
        self.lemmatize_check.setChecked(True)
        options_layout.addWidget(self.lemmatize_check)

        self.remove_stopwords_check = QCheckBox("Remove stopwords")
        self.remove_stopwords_check.setChecked(True)
        options_layout.addWidget(self.remove_stopwords_check)

        options_layout.addStretch()
        layout.addLayout(options_layout)

        # Preprocess button and progress
        preprocess_row = QHBoxLayout()
        preprocess_row.addStretch()

        self.preprocess_btn = QPushButton("Preprocess Texts")
        self.preprocess_btn.clicked.connect(self._preprocess_texts)
        self.preprocess_btn.setEnabled(False)
        preprocess_row.addWidget(self.preprocess_btn)

        layout.addLayout(preprocess_row)

        # Status
        self.spacy_status = QLabel("")
        layout.addWidget(self.spacy_status)

        group.setLayout(layout)
        self._add_overlay_info(group,
            "<b>Text Processing</b><br><br>"
            "Select a spaCy language model, then click <b>Preprocess</b> "
            "to tokenize, lemmatize, and sentence-split your texts.<br><br>"
            "After preprocessing, statistics are shown: total sentences, "
            "mean words per document, and how many sentences were kept "
            "after short-sentence filtering.",
        )
        return group

    def _create_embeddings_section(self):
        """Create the embeddings configuration section."""
        group = QGroupBox("3. Word Embeddings")
        layout = QVBoxLayout()

        # File selection
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Embeddings File:"))
        self.emb_path_combo = QComboBox()
        self.emb_path_combo.setEditable(False)
        self.emb_path_combo.setMinimumWidth(200)
        self.emb_path_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.emb_path_combo.setMinimumContentsLength(20)
        self.emb_path_combo.view().setTextElideMode(Qt.ElideMiddle)
        self._emb_delegate = _RemovableItemDelegate(
            self.emb_path_combo, self._remove_recent_embedding, parent=self.emb_path_combo,
        )
        self.emb_path_combo.setItemDelegate(self._emb_delegate)
        self._populate_recent_embeddings()
        self.emb_path_combo.currentIndexChanged.connect(self._on_emb_combo_changed)
        file_row.addWidget(self.emb_path_combo, stretch=1)

        browse_emb_btn = QPushButton("Browse...")
        browse_emb_btn.clicked.connect(self._browse_embeddings)
        file_row.addWidget(browse_emb_btn)
        layout.addLayout(file_row)

        # Normalization options
        norm_row = QHBoxLayout()

        self.l2_norm_check = QCheckBox("L2 normalize embeddings")
        self.l2_norm_check.setChecked(True)
        norm_row.addWidget(self.l2_norm_check)

        self.abtt_check = QCheckBox("Apply ABTT (All-But-The-Top)")
        self.abtt_check.setChecked(True)
        self.abtt_check.toggled.connect(self._on_abtt_toggled)
        norm_row.addWidget(self.abtt_check)

        abtt_m_layout = QHBoxLayout()
        abtt_m_layout.addWidget(QLabel("ABTT m:"))
        self.abtt_m_spin = QSpinBox()
        self.abtt_m_spin.setRange(0, 10)
        self.abtt_m_spin.setValue(1)
        abtt_m_layout.addWidget(self.abtt_m_spin)
        norm_row.addLayout(abtt_m_layout)

        norm_row.addStretch()
        layout.addLayout(norm_row)

        # Load button
        load_row = QHBoxLayout()
        load_row.addStretch()

        self.load_emb_btn = QPushButton("Load Embeddings")
        self.load_emb_btn.clicked.connect(self._load_embeddings)
        self.load_emb_btn.setEnabled(False)
        load_row.addWidget(self.load_emb_btn)

        layout.addLayout(load_row)

        # Status
        self.emb_status = QLabel("")
        layout.addWidget(self.emb_status)

        group.setLayout(layout)
        self._add_overlay_info(group,
            "<b>Word Embeddings</b><br><br>"
            "Choose a pre-trained word-embedding file (GloVe, word2vec, "
            "or fastText format). Click <b>Load</b> to read it into memory "
            "and compute vocabulary coverage against your preprocessed "
            "corpus.<br><br>"
            "Coverage tells you what percentage of your corpus tokens "
            "are present in the embedding model.",
        )
        return group

    def _create_analysis_section(self, parent_layout):
        """Create the analysis configuration section."""
        group = QGroupBox("4. Analysis")
        layout = QVBoxLayout()
        layout.setSpacing(6)

        # Row 1: Analysis type radios + column combo + summary (all one row)
        row1 = QHBoxLayout()

        self.analysis_type_group = QButtonGroup()

        self.continuous_radio = QRadioButton("Continuous Outcome")
        self.continuous_radio.setChecked(True)
        self.analysis_type_group.addButton(self.continuous_radio, 0)
        row1.addWidget(self.continuous_radio)

        self.crossgroup_radio = QRadioButton("Group Comparison")
        self.analysis_type_group.addButton(self.crossgroup_radio, 1)
        row1.addWidget(self.crossgroup_radio)

        row1.addStretch()
        layout.addLayout(row1)

        # Row 2: Continuous outcome column
        self.cont_row = QWidget()
        self.cont_row.setAttribute(Qt.WA_StyledBackground, False)
        self.cont_row.setStyleSheet("background: transparent;")
        cont_layout = QHBoxLayout(self.cont_row)
        cont_layout.setContentsMargins(0, 0, 0, 0)
        cont_layout.addWidget(QLabel("Outcome Column:"))
        self.outcome_col_combo = QComboBox()
        self.outcome_col_combo.setMinimumWidth(180)
        self.outcome_col_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.outcome_col_combo.addItem("(none)")
        self.outcome_col_combo.currentIndexChanged.connect(self._on_outcome_column_changed)
        cont_layout.addWidget(self.outcome_col_combo)

        self.outcome_summary_label = QLabel("")
        self.outcome_summary_label.setObjectName("label_muted")
        cont_layout.addWidget(self.outcome_summary_label, stretch=1)

        layout.addWidget(self.cont_row)

        # Row 2b: Crossgroup column + n_perm (hidden by default)
        self.cross_row = QWidget()
        self.cross_row.setAttribute(Qt.WA_StyledBackground, False)
        self.cross_row.setStyleSheet("background: transparent;")
        cross_layout = QHBoxLayout(self.cross_row)
        cross_layout.setContentsMargins(0, 0, 0, 0)
        cross_layout.addWidget(QLabel("Group Column:"))
        self.group_col_combo = QComboBox()
        self.group_col_combo.setMinimumWidth(180)
        self.group_col_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.group_col_combo.addItem("(none)")
        self.group_col_combo.currentIndexChanged.connect(self._on_group_column_changed)
        cross_layout.addWidget(self.group_col_combo)

        cross_layout.addWidget(QLabel("Permutations:"))
        self.n_perm_spin = QSpinBox()
        self.n_perm_spin.setRange(100, 50000)
        self.n_perm_spin.setValue(5000)
        self.n_perm_spin.setSingleStep(500)
        cross_layout.addWidget(self.n_perm_spin)

        self.group_summary_label = QLabel("")
        self.group_summary_label.setObjectName("label_muted")
        cross_layout.addWidget(self.group_summary_label, stretch=1)

        self.cross_row.setVisible(False)
        layout.addWidget(self.cross_row)

        # Connect analysis type toggle
        self.continuous_radio.toggled.connect(self._on_analysis_type_changed)

        # Row 3: Mode selection
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))

        self.mode_btn_group = QButtonGroup()

        self.lexicon_radio = QRadioButton("Lexicon")
        self.lexicon_radio.setChecked(True)
        self.mode_btn_group.addButton(self.lexicon_radio, 0)
        mode_row.addWidget(self.lexicon_radio)

        self.fulldoc_radio = QRadioButton("Full Document")
        self.mode_btn_group.addButton(self.fulldoc_radio, 1)
        mode_row.addWidget(self.fulldoc_radio)

        mode_row.addStretch()
        layout.addLayout(mode_row)

        # Connect mode toggle to adaptive advanced settings
        self.lexicon_radio.toggled.connect(self._on_mode_changed)

        group.setLayout(layout)
        self._add_overlay_info(group,
            "<b>Analysis</b><br><br>"
            "<b>Type</b> — <i>Continuous Outcome</i> predicts a numeric "
            "variable; <i>Group Comparison</i> contrasts two or more "
            "categorical groups.<br>"
            "<b>Mode</b> — <i>Full-Document</i> uses the entire text; "
            "<i>Lexicon</i> restricts to a hand-picked word list; "
            "<i>Hybrid</i> combines both approaches.",
        )
        parent_layout.addWidget(group)

    def _on_analysis_type_changed(self, continuous_checked: bool):
        """Handle analysis type radio toggle."""
        self.cont_row.setVisible(continuous_checked)
        self.cross_row.setVisible(not continuous_checked)
        if continuous_checked:
            self._on_outcome_column_changed()
        else:
            self._on_group_column_changed()
        self._update_advanced_settings_visibility()
        self._update_ready_indicator()

    def _on_mode_changed(self, lexicon_checked: bool):
        """Handle mode radio toggle."""
        self._update_advanced_settings_visibility()
        self._update_ready_indicator()

    def _update_advanced_settings_visibility(self):
        """Show/hide advanced settings groups based on analysis type and mode."""
        if not hasattr(self, 'pca_group_widget'):
            return
        is_continuous = self.continuous_radio.isChecked()
        is_lexicon = self.lexicon_radio.isChecked()

        # PCA Dimensionality: visible only for continuous
        self.pca_group_widget.setVisible(is_continuous)
        # Context window size: visible only for lexicon mode
        self.window_size_label.setVisible(is_lexicon)
        self.window_size_spin.setVisible(is_lexicon)
        # Unit beta: visible only for continuous
        self.unit_beta_check.setVisible(is_continuous)

    def _on_outcome_column_changed(self):
        """Handle outcome column selection change."""
        if not self.project or self.project._cached_df is None:
            return
        if not self.continuous_radio.isChecked():
            return

        col = self.outcome_col_combo.currentText()
        if not col or col == "(none)":
            self.outcome_summary_label.setText("")
            self.project._cached_y = None
            self._update_ready_indicator()
            return

        df = self.project._cached_df
        if col not in df.columns:
            self.outcome_summary_label.setText("")
            self.project._cached_y = None
            self._update_ready_indicator()
            return

        outcome = pd.to_numeric(df[col], errors="coerce")
        id_row_indices = self.project._cached_id_row_indices

        if id_row_indices is not None:
            # Grouped mode: one outcome per unique ID (first non-NaN value)
            y_list = []
            for row_indices in id_row_indices:
                vals = outcome.iloc[row_indices].dropna()
                y_list.append(vals.iloc[0] if len(vals) > 0 else np.nan)
            y_series = pd.Series(y_list)
            n_valid = (~y_series.isna()).sum()
            y = y_series.dropna().to_numpy()
        else:
            n_valid = (~outcome.isna()).sum()
            y = outcome.dropna().to_numpy()

        if n_valid > 0:
            self.outcome_summary_label.setText(
                f"n={n_valid:,}, mean={y.mean():.3f}, std={y.std():.3f}"
            )
            self.project._cached_y = y
            self.project.dataset_config.outcome_column = col
            self.project.dataset_config.n_valid = int(n_valid)
            self._apply_outcome_filter()
        else:
            self.outcome_summary_label.setText("No valid numeric values")
            self.project._cached_y = None

        self._update_ready_indicator()

    def _apply_outcome_filter(self):
        """Filter cached docs to match rows/profiles with valid outcome."""
        if not self.project or self.project._cached_df is None:
            return

        col = self.outcome_col_combo.currentText()
        if not col or col not in self.project._cached_df.columns:
            return

        from ..utils.file_io import ProjectIO
        cached = ProjectIO.load_preprocessed_docs(self.project)
        if not cached:
            return

        all_pre_docs, all_docs, id_row_indices = cached
        outcome = pd.to_numeric(self.project._cached_df[col], errors="coerce")

        if id_row_indices is not None:
            # Grouped: per-ID mask — an ID is valid if it has at least one
            # non-NaN outcome value among its rows.
            per_id_mask = [
                any(not np.isnan(outcome.iloc[ri]) for ri in row_indices)
                for row_indices in id_row_indices
            ]
            self.project._cached_docs = [all_docs[i] for i, ok in enumerate(per_id_mask) if ok]
            self.project._cached_pre_docs = [all_pre_docs[i] for i, ok in enumerate(per_id_mask) if ok]
        else:
            mask = ~outcome.isna()
            self.project._cached_docs = [all_docs[i] for i in range(len(all_docs)) if mask.iat[i]]
            self.project._cached_pre_docs = [all_pre_docs[i] for i in range(len(all_pre_docs)) if mask.iat[i]]

    def _on_group_column_changed(self):
        """Handle group column selection change."""
        if not self.project or self.project._cached_df is None:
            return
        if not self.crossgroup_radio.isChecked():
            return

        col = self.group_col_combo.currentText()
        if not col or col == "(none)":
            self.group_summary_label.setText("")
            self.project._cached_groups = None
            self._update_ready_indicator()
            return

        df = self.project._cached_df
        if col not in df.columns:
            self.group_summary_label.setText("")
            self.project._cached_groups = None
            self._update_ready_indicator()
            return

        id_row_indices = self.project._cached_id_row_indices

        if id_row_indices is not None:
            # Grouped: build per-ID group array (first non-empty value)
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
            summary += f" ({n_missing} dropped: NaN/empty)"
        summary += f"  [{counts_str}]"
        self.group_summary_label.setText(summary)

        self.project._cached_groups = groups.to_numpy()
        self.project.dataset_config.group_column = col

        self._apply_group_filter()
        self._update_ready_indicator()

    def _apply_group_filter(self):
        """Filter cached docs/profiles to match rows with valid groups."""
        if not self.project or self.project._cached_df is None:
            return

        col = self.group_col_combo.currentText()
        if not col or col not in self.project._cached_df.columns:
            return

        from ..utils.file_io import ProjectIO
        cached = ProjectIO.load_preprocessed_docs(self.project)
        if not cached:
            return

        all_pre_docs, all_docs, id_row_indices = cached

        if id_row_indices is not None:
            # Grouped: per-ID mask based on first non-empty group value
            df = self.project._cached_df
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

            self.project._cached_docs = [all_docs[i] for i, ok in enumerate(per_id_valid) if ok]
            self.project._cached_pre_docs = [all_pre_docs[i] for i, ok in enumerate(per_id_valid) if ok]
            self.project._cached_groups = np.array([g for g, ok in zip(per_id_group_vals, per_id_valid) if ok])
        else:
            groups = self.project._cached_df[col]
            mask = ~(groups.isna() | (groups.astype(str).str.strip() == ""))
            self.project._cached_docs = [all_docs[i] for i in range(len(all_docs)) if mask.iat[i]]
            self.project._cached_pre_docs = [all_pre_docs[i] for i in range(len(all_pre_docs)) if mask.iat[i]]
            self.project._cached_groups = groups[mask].astype(str).to_numpy()

        self.project._cached_y = None

    def _is_crossgroup(self) -> bool:
        return self.crossgroup_radio.isChecked()

    def _create_hyperparams_section(self, parent_layout):
        """Create the hyperparameters section (collapsible)."""
        self.hyperparams_box = CollapsibleBox("5. Advanced Settings (click to expand)")
        content_layout = self.hyperparams_box.content_layout

        # PCA settings
        pca_group = QGroupBox("PCA Dimensionality")
        pca_layout = QVBoxLayout()

        pca_mode_row = QHBoxLayout()
        self.pca_mode_group = QButtonGroup()

        self.pca_auto_radio = QRadioButton("Auto (PCA sweep)")
        self.pca_auto_radio.setChecked(True)
        self.pca_mode_group.addButton(self.pca_auto_radio, 0)
        pca_mode_row.addWidget(self.pca_auto_radio)

        self.pca_manual_radio = QRadioButton("Manual:")
        self.pca_mode_group.addButton(self.pca_manual_radio, 1)
        pca_mode_row.addWidget(self.pca_manual_radio)

        self.pca_k_spin = QSpinBox()
        self.pca_k_spin.setRange(2, 500)
        self.pca_k_spin.setValue(20)
        self.pca_k_spin.setEnabled(False)
        self.pca_manual_radio.toggled.connect(self.pca_k_spin.setEnabled)
        pca_mode_row.addWidget(self.pca_k_spin)

        pca_mode_row.addStretch()
        pca_layout.addLayout(pca_mode_row)

        sweep_row = QHBoxLayout()
        sweep_row.addWidget(QLabel("Sweep K range:"))
        self.sweep_k_min_spin = QSpinBox()
        self.sweep_k_min_spin.setRange(1, 200)
        self.sweep_k_min_spin.setValue(1)
        sweep_row.addWidget(self.sweep_k_min_spin)
        sweep_row.addWidget(QLabel("to"))
        self.sweep_k_max_spin = QSpinBox()
        self.sweep_k_max_spin.setRange(2, 500)
        self.sweep_k_max_spin.setValue(80)
        sweep_row.addWidget(self.sweep_k_max_spin)
        sweep_row.addWidget(QLabel("step:"))
        self.sweep_k_step_spin = QSpinBox()
        self.sweep_k_step_spin.setRange(1, 50)
        self.sweep_k_step_spin.setValue(1)
        sweep_row.addWidget(self.sweep_k_step_spin)
        sweep_row.addStretch()
        pca_layout.addLayout(sweep_row)

        sweep_row2 = QHBoxLayout()
        sweep_row2.addWidget(QLabel("AUCK radius:"))
        self.auck_radius_spin = QSpinBox()
        self.auck_radius_spin.setRange(1, 20)
        self.auck_radius_spin.setValue(3)
        sweep_row2.addWidget(self.auck_radius_spin)

        sweep_row2.addWidget(QLabel("Beta smooth window:"))
        self.beta_smooth_win_spin = QSpinBox()
        self.beta_smooth_win_spin.setRange(1, 31)
        self.beta_smooth_win_spin.setValue(7)
        sweep_row2.addWidget(self.beta_smooth_win_spin)

        sweep_row2.addWidget(QLabel("Smooth kind:"))
        self.beta_smooth_kind_combo = QComboBox()
        self.beta_smooth_kind_combo.addItems(["median", "mean"])
        sweep_row2.addWidget(self.beta_smooth_kind_combo)

        self.weight_by_size_check = QCheckBox("Weight by cluster size")
        self.weight_by_size_check.setChecked(True)
        sweep_row2.addWidget(self.weight_by_size_check)

        sweep_row2.addStretch()
        pca_layout.addLayout(sweep_row2)

        pca_group.setLayout(pca_layout)
        self.pca_group_widget = pca_group
        content_layout.addWidget(pca_group)

        # Context settings
        context_group = QGroupBox("Context Window")
        context_layout = QHBoxLayout()

        self.window_size_label = QLabel("Window size (+/-):")
        context_layout.addWidget(self.window_size_label)
        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(1, 20)
        self.window_size_spin.setValue(3)
        context_layout.addWidget(self.window_size_spin)

        context_layout.addWidget(QLabel("SIF parameter (a):"))
        self.sif_a_spin = QDoubleSpinBox()
        self.sif_a_spin.setRange(1e-5, 1.0)
        self.sif_a_spin.setDecimals(5)
        self.sif_a_spin.setValue(1e-3)
        self.sif_a_spin.setSingleStep(1e-4)
        context_layout.addWidget(self.sif_a_spin)

        context_layout.addStretch()
        context_group.setLayout(context_layout)
        content_layout.addWidget(context_group)

        # Model settings
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout()

        model_row1 = QHBoxLayout()
        self.unit_beta_check = QCheckBox("Use unit-length beta for interpretation")
        self.unit_beta_check.setChecked(True)
        model_row1.addWidget(self.unit_beta_check)

        self.l2_docs_check = QCheckBox("L2 normalize document vectors")
        self.l2_docs_check.setChecked(True)
        model_row1.addWidget(self.l2_docs_check)
        model_row1.addStretch()
        model_layout.addLayout(model_row1)

        model_group.setLayout(model_layout)
        content_layout.addWidget(model_group)

        # Clustering settings
        cluster_group = QGroupBox("Clustering")
        cluster_layout = QVBoxLayout()

        cluster_row1 = QHBoxLayout()
        cluster_row1.addWidget(QLabel("Top N neighbors:"))
        self.cluster_topn_spin = QSpinBox()
        self.cluster_topn_spin.setRange(20, 500)
        self.cluster_topn_spin.setValue(100)
        cluster_row1.addWidget(self.cluster_topn_spin)

        self.cluster_k_auto_check = QCheckBox("Auto-select K (silhouette)")
        self.cluster_k_auto_check.setChecked(True)
        cluster_row1.addWidget(self.cluster_k_auto_check)
        cluster_row1.addStretch()
        cluster_layout.addLayout(cluster_row1)

        cluster_row2 = QHBoxLayout()
        cluster_row2.addWidget(QLabel("K range:"))
        self.cluster_k_min_spin = QSpinBox()
        self.cluster_k_min_spin.setRange(2, 50)
        self.cluster_k_min_spin.setValue(2)
        cluster_row2.addWidget(self.cluster_k_min_spin)
        cluster_row2.addWidget(QLabel("to"))
        self.cluster_k_max_spin = QSpinBox()
        self.cluster_k_max_spin.setRange(3, 100)
        self.cluster_k_max_spin.setValue(10)
        cluster_row2.addWidget(self.cluster_k_max_spin)

        cluster_row2.addStretch()
        cluster_layout.addLayout(cluster_row2)

        cluster_group.setLayout(cluster_layout)
        content_layout.addWidget(cluster_group)

        self._add_overlay_info(self.hyperparams_box,
            "<b>Advanced Settings</b><br><br>"
            "Fine-tune the SSD hyperparameters:<br>"
            "<b>PCA</b> — auto-sweep or fix the number of components.<br>"
            "<b>Context Window</b> — sentence window size for building "
            "document vectors.<br>"
            "<b>Model</b> — permutation count, minimum word frequency, "
            "and SIF weighting.<br>"
            "<b>Clustering</b> — number of semantic clusters and top-N "
            "words per cluster.",
        )
        parent_layout.addWidget(self.hyperparams_box)

    def _create_ready_indicator(self, parent_layout):
        """Create the project ready indicator."""
        self.ready_frame = QFrame()
        self.ready_frame.setObjectName("frame_ready")
        self.ready_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.ready_frame.setLineWidth(2)

        layout = QVBoxLayout()

        self.ready_title = QLabel("Project Status")
        self.ready_title.setObjectName("label_title")
        layout.addWidget(self.ready_title)

        self.ready_details = QLabel("")
        layout.addWidget(self.ready_details)

        self.ready_frame.setLayout(layout)
        self._update_ready_indicator()

        parent_layout.addWidget(self.ready_frame)

    def _create_navigation(self, parent_layout):
        """Create navigation buttons."""
        parent_layout.addSpacing(8)

        nav_layout = QHBoxLayout()
        nav_layout.addStretch()

        self.next_btn = QPushButton("Continue to Run  \u203A")
        self.next_btn.setObjectName("btn_success")
        self.next_btn.setEnabled(False)
        self.next_btn.setMinimumHeight(44)
        self.next_btn.setMinimumWidth(260)
        self.next_btn.setCursor(Qt.PointingHandCursor)
        self.next_btn.clicked.connect(self._on_next_clicked)
        nav_layout.addWidget(self.next_btn)

        parent_layout.addLayout(nav_layout)

    # --- Helper methods ---

    def _populate_recent_embeddings(self):
        """Populate the embeddings combo from QSettings."""
        self.emb_path_combo.blockSignals(True)
        self.emb_path_combo.clear()
        recent = self._settings.value("embeddings/recent_paths", [])
        if not isinstance(recent, list):
            recent = []
        if recent:
            for p in recent:
                self.emb_path_combo.addItem(p)
        else:
            self.emb_path_combo.addItem("(no recent models)")
            model = self.emb_path_combo.model()
            item = model.item(0)
            item.setEnabled(False)
        self.emb_path_combo.blockSignals(False)

    def _add_recent_embedding_path(self, path: str):
        """Add a path to the recent embeddings list and refresh the combo."""
        if not path:
            return
        recent = self._settings.value("embeddings/recent_paths", [])
        if not isinstance(recent, list):
            recent = []
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        recent = recent[:10]
        self._settings.setValue("embeddings/recent_paths", recent)

        self.emb_path_combo.blockSignals(True)
        self.emb_path_combo.clear()
        for p in recent:
            self.emb_path_combo.addItem(p)
        idx = self.emb_path_combo.findText(path)
        if idx >= 0:
            self.emb_path_combo.setCurrentIndex(idx)
        self.emb_path_combo.blockSignals(False)
        self._on_emb_combo_changed()

    def _remove_recent_embedding(self, row: int):
        """Remove an embedding path from the recent list by combo row index."""
        path = self.emb_path_combo.itemText(row)
        if not path or path == "(no recent models)":
            return

        recent = self._settings.value("embeddings/recent_paths", [])
        if not isinstance(recent, list):
            recent = []
        if path in recent:
            recent.remove(path)
        self._settings.setValue("embeddings/recent_paths", recent)

        self.emb_path_combo.blockSignals(True)
        self.emb_path_combo.removeItem(row)
        if self.emb_path_combo.count() == 0:
            self.emb_path_combo.addItem("(no recent models)")
            model = self.emb_path_combo.model()
            model.item(0).setEnabled(False)
        self.emb_path_combo.blockSignals(False)
        self._on_emb_combo_changed()

    def _on_emb_combo_changed(self):
        """Enable/disable Load button based on combo selection."""
        text = self.emb_path_combo.currentText()
        valid = bool(text) and text != "(no recent models)"
        self.load_emb_btn.setEnabled(valid)

    def _update_model_options(self, language: str):
        """Update spaCy model options based on language."""
        self.model_combo.clear()

        models = {
            "en": ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
            "ca": ["ca_core_news_sm", "ca_core_news_md", "ca_core_news_lg"],
            "da": ["da_core_news_sm", "da_core_news_md", "da_core_news_lg"],
            "de": ["de_core_news_sm", "de_core_news_md", "de_core_news_lg"],
            "el": ["el_core_news_sm", "el_core_news_md", "el_core_news_lg"],
            "es": ["es_core_news_sm", "es_core_news_md", "es_core_news_lg"],
            "fr": ["fr_core_news_sm", "fr_core_news_md", "fr_core_news_lg"],
            "hr": ["hr_core_news_sm", "hr_core_news_md", "hr_core_news_lg"],
            "it": ["it_core_news_sm", "it_core_news_md", "it_core_news_lg"],
            "ja": ["ja_core_news_sm", "ja_core_news_md", "ja_core_news_lg"],
            "ko": ["ko_core_news_sm", "ko_core_news_md", "ko_core_news_lg"],
            "lt": ["lt_core_news_sm", "lt_core_news_md", "lt_core_news_lg"],
            "mk": ["mk_core_news_sm", "mk_core_news_md", "mk_core_news_lg"],
            "nb": ["nb_core_news_sm", "nb_core_news_md", "nb_core_news_lg"],
            "nl": ["nl_core_news_sm", "nl_core_news_md", "nl_core_news_lg"],
            "pl": ["pl_core_news_sm", "pl_core_news_md", "pl_core_news_lg"],
            "pt": ["pt_core_news_sm", "pt_core_news_md", "pt_core_news_lg"],
            "ro": ["ro_core_news_sm", "ro_core_news_md", "ro_core_news_lg"],
            "ru": ["ru_core_news_sm", "ru_core_news_md", "ru_core_news_lg"],
            "sl": ["sl_core_news_sm", "sl_core_news_md", "sl_core_news_lg"],
            "sv": ["sv_core_news_sm", "sv_core_news_md", "sv_core_news_lg"],
            "uk": ["uk_core_news_sm", "uk_core_news_md", "uk_core_news_lg"],
            "zh": ["zh_core_web_sm", "zh_core_web_md", "zh_core_web_lg"],
        }

        self.model_combo.addItems(models.get(language, ["en_core_web_sm"]))

    def _on_language_changed(self, language: str):
        """Handle language selection change."""
        self._update_model_options(language)

    def _on_abtt_toggled(self, checked: bool):
        """Handle ABTT checkbox toggle."""
        self.abtt_m_spin.setEnabled(checked)

    def _browse_csv(self):
        """Open file dialog to select data file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "All Supported Files (*.csv *.tsv *.xlsx *.xls);;"
            "CSV Files (*.csv *.tsv);;"
            "Excel Files (*.xlsx *.xls);;"
            "All Files (*)",
        )
        if filepath:
            self.file_path_edit.setText(filepath)
            self.load_csv_btn.setEnabled(True)

    def _load_csv(self):
        """Load the selected data file (CSV or Excel)."""
        filepath = self.file_path_edit.text()
        if not filepath:
            return

        try:
            ext = Path(filepath).suffix.lower()

            if ext in (".xlsx", ".xls"):
                self._df = pd.read_excel(filepath)
            elif ext == ".tsv":
                self._df = pd.read_csv(filepath, sep="\t")
            else:
                self._df = pd.read_csv(filepath)

            # Populate column combos
            columns = self._df.columns.tolist()

            self.text_col_combo.clear()
            self.text_col_combo.addItems(columns)

            self.id_col_combo.clear()
            self.id_col_combo.addItem("(none)")
            self.id_col_combo.addItems(columns)

            # Populate analysis column combos
            numeric_cols = self._df.select_dtypes(include="number").columns.tolist()
            self.outcome_col_combo.blockSignals(True)
            self.outcome_col_combo.clear()
            self.outcome_col_combo.addItem("(none)")
            self.outcome_col_combo.addItems(numeric_cols)
            self.outcome_col_combo.setCurrentIndex(0)
            self.outcome_col_combo.blockSignals(False)

            self.group_col_combo.blockSignals(True)
            self.group_col_combo.clear()
            self.group_col_combo.addItem("(none)")
            self.group_col_combo.addItems(columns)
            self.group_col_combo.setCurrentIndex(0)
            self.group_col_combo.blockSignals(False)

            # Show stats
            n_rows = len(self._df)
            n_cols = len(columns)
            self.dataset_stats_label.setText(
                f"Loaded {n_rows:,} rows, {n_cols} columns"
            )

            self.validate_dataset_btn.setEnabled(True)
            self.preprocess_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {e}")

    def _validate_dataset(self):
        """Validate the loaded dataset."""
        if self._df is None:
            return

        text_col = self.text_col_combo.currentText()
        id_col = self.id_col_combo.currentText()
        if id_col == "(none)":
            id_col = None

        errors, warnings, id_stats = Validator.validate_dataset_text(
            self._df, text_col, id_col
        )

        if errors:
            self.dataset_status.setText("Validation errors found")
            self.dataset_status.setObjectName("label_status_error")
            self.dataset_status.style().unpolish(self.dataset_status)
            self.dataset_status.style().polish(self.dataset_status)
            QMessageBox.critical(
                self, "Validation Errors", "\n".join(errors)
            )
            return

        if warnings:
            self.dataset_status.setText("Validation passed with warnings")
            self.dataset_status.setObjectName("label_status_warn")
            self.dataset_status.style().unpolish(self.dataset_status)
            self.dataset_status.style().polish(self.dataset_status)
            QMessageBox.warning(
                self, "Validation Warnings", "\n".join(warnings)
            )
        else:
            # Build success message with ID stats
            n_rows = len(self._df)
            msg = f"Dataset validated successfully  |  {n_rows:,} text rows"
            if id_stats:
                n_unique = id_stats["n_unique_ids"]
                if id_stats["has_duplicates"]:
                    avg = id_stats["avg_texts_per_id"]
                    msg += f", {n_unique:,} unique IDs ({avg:.1f} avg texts/ID)"
                else:
                    msg += f", {n_unique:,} unique IDs"
            self.dataset_status.setText(msg)
            self.dataset_status.setObjectName("label_status_ok")
            self.dataset_status.style().unpolish(self.dataset_status)
            self.dataset_status.style().polish(self.dataset_status)

        # Store validated columns
        if self.project:
            self.project.dataset_config.csv_path = Path(self.file_path_edit.text())
            self.project.dataset_config.text_column = text_col
            self.project.dataset_config.id_column = id_col
            self.project.dataset_config.n_rows = len(self._df)
            self.project.dataset_config.n_valid = len(self._df)
            self.project.dataset_config.cached = True

            # Cache dataframe
            self.project._cached_df = self._df

    def _browse_embeddings(self):
        """Open file dialog to select embeddings file."""
        current = self.emb_path_combo.currentText()
        start_dir = ""
        if current and current != "(no recent models)":
            parent = Path(current).parent
            if parent.exists():
                start_dir = str(parent)

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Embeddings File",
            start_dir,
            "Embedding Files (*.kv *.bin *.txt *.gz *.vec);;All Files (*)",
        )
        if filepath:
            self._add_recent_embedding_path(filepath)

    def _preprocess_texts(self):
        """Run spaCy preprocessing on the texts.

        When an ID column is selected, texts are grouped by ID so that
        ssdiff produces one PreprocessedProfile (and later one PCV) per
        unique ID.  Rows with NaN IDs become individual single-text
        profiles.
        """
        if self._df is None or self.project is None:
            return

        text_col = self.text_col_combo.currentText()
        id_col = self.id_col_combo.currentText()
        if id_col == "(none)":
            id_col = None

        if id_col and id_col in self._df.columns:
            # Group texts by ID (preserve order)
            texts_raw = []          # List[List[str]]
            id_row_indices = []     # List[List[int]]
            grouped = self._df.groupby(
                self._df[id_col].fillna(
                    pd.Series(
                        [f"__nan_{i}__" for i in range(len(self._df))],
                        index=self._df.index,
                    )
                ),
                sort=False,
            )
            for _id_val, grp in grouped:
                row_texts = grp[text_col].fillna("").astype(str).tolist()
                texts_raw.append(row_texts)
                id_row_indices.append(grp.index.tolist())
            self.project._cached_id_row_indices = id_row_indices
        else:
            texts_raw = self._df[text_col].fillna("").astype(str).tolist()
            self.project._cached_id_row_indices = None

        model_name = self.model_combo.currentText()

        # Check if the spaCy model is available (pip-installed or locally downloaded)
        import spacy.util
        local_path = find_local_model(model_name)
        if not spacy.util.is_package(model_name) and not local_path:
            reply = QMessageBox.question(
                self,
                "spaCy Model Not Found",
                f"The model '{model_name}' is not installed.\n\n"
                "Would you like to download it now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return
            # Store texts_raw for use after download completes
            self._pending_preprocess_texts = texts_raw
            self._download_spacy_model(model_name)
            return

        self._run_preprocess_worker(texts_raw, model_name, model_path=local_path)

    def _download_spacy_model(self, model_name: str):
        """Download a spaCy model in a background thread."""
        self._spacy_dl_worker = SpacyDownloadWorker(model_name)
        self._progress_dialog = ProgressDialog(f"Downloading {model_name}", self)

        self._spacy_dl_worker.progress.connect(self._progress_dialog.update_progress)
        self._spacy_dl_worker.finished.connect(self._on_spacy_download_finished)
        self._spacy_dl_worker.error.connect(self._on_spacy_download_error)

        self._progress_dialog.show()
        QApplication.processEvents()
        self._spacy_dl_worker.start()
        self._progress_dialog.exec()

    def _on_spacy_download_finished(self, model_path_str: str):
        """After model download, proceed with preprocessing using the same dialog."""
        texts_raw = self._pending_preprocess_texts
        self._pending_preprocess_texts = None
        if texts_raw is None:
            self._progress_dialog.accept()
            return
        model_name = self.model_combo.currentText()
        model_path = Path(model_path_str) if model_path_str else None
        # Reuse the download dialog for preprocessing (no accept/new dialog)
        self._progress_dialog.setWindowTitle("Preprocessing Texts")
        self._run_preprocess_worker(texts_raw, model_name, model_path=model_path, reuse_dialog=True)

    def _on_spacy_download_error(self, error_message: str):
        """Handle spaCy model download failure."""
        self._pending_preprocess_texts = None
        self._progress_dialog.set_error(error_message)

    def _run_preprocess_worker(
        self, texts_raw, model_name: str,
        model_path: Optional[Path] = None,
        reuse_dialog: bool = False,
    ):
        """Start the preprocessing worker thread."""
        self._worker = PreprocessWorker(
            texts_raw=texts_raw,
            language=self.language_combo.currentText(),
            model=model_name,
            model_path=model_path,
        )

        if not reuse_dialog:
            self._progress_dialog = ProgressDialog("Preprocessing Texts", self)

        self._worker.progress.connect(self._progress_dialog.update_progress)
        self._worker.finished.connect(self._on_preprocess_finished)
        self._worker.error.connect(self._on_preprocess_error)

        self._progress_dialog.cancel_button.clicked.connect(self._worker.cancel)

        if not reuse_dialog:
            self._progress_dialog.show()
            QApplication.processEvents()
        self._worker.start()
        if not reuse_dialog:
            self._progress_dialog.exec()

    def _on_preprocess_finished(self, pre_docs, docs, stats):
        """Handle preprocessing completion."""
        if self._progress_dialog.is_cancelled():
            return
        self._progress_dialog.accept()

        self.project._cached_pre_docs = pre_docs
        self.project._cached_docs = docs

        # Update project config
        self.project.spacy_config.language = self.language_combo.currentText()
        self.project.spacy_config.model = self.model_combo.currentText()
        self.project.spacy_config.lemmatize = self.lemmatize_check.isChecked()
        self.project.spacy_config.remove_stopwords = self.remove_stopwords_check.isChecked()
        self.project.spacy_config.processed = True
        self.project.spacy_config.n_docs_processed = stats["n_docs"]
        self.project.spacy_config.total_tokens = stats["total_tokens"]
        self.project.spacy_config.mean_words_before_stopwords = stats["mean_words_before_stopwords"]

        id_row_indices = self.project._cached_id_row_indices
        ProjectIO.save_preprocessed_docs(self.project, pre_docs, docs, id_row_indices)
        ProjectIO.save_project(self.project)

        if stats.get("is_grouped"):
            status_text = (
                f"Preprocessed {stats['n_docs']:,} profiles "
                f"(from {stats['n_total_rows']:,} rows), "
                f"{stats['total_tokens']:,} total tokens, "
                f"{stats['avg_tokens_per_doc']:.1f} avg/profile"
            )
        else:
            status_text = (
                f"Preprocessed {stats['n_docs']:,} documents, "
                f"{stats['total_tokens']:,} total tokens, "
                f"{stats['avg_tokens_per_doc']:.1f} avg/doc"
            )
        self.spacy_status.setText(status_text)
        self.spacy_status.setObjectName("label_status_ok")
        self.spacy_status.style().unpolish(self.spacy_status)
        self.spacy_status.style().polish(self.spacy_status)

        self._update_ready_indicator()

    def _on_preprocess_error(self, error_message):
        """Handle preprocessing error."""
        if self._progress_dialog.is_cancelled():
            return
        self._progress_dialog.set_error(error_message)
        self.spacy_status.setText("Preprocessing failed")
        self.spacy_status.setObjectName("label_status_error")
        self.spacy_status.style().unpolish(self.spacy_status)
        self.spacy_status.style().polish(self.spacy_status)

    @staticmethod
    def _kv_target_path(emb_path: str) -> Optional[Path]:
        """Return the .kv path for a text-format embedding, or None."""
        lower = emb_path.lower()
        is_text = (
            lower.endswith(".txt")
            or lower.endswith(".vec")
            or lower.endswith(".txt.gz")
            or lower.endswith(".vec.gz")
        )
        if not is_text:
            return None
        p = Path(emb_path)
        if p.suffix.lower() == ".gz":
            return p.with_suffix("").with_suffix(".kv")
        return p.with_suffix(".kv")

    def _load_embeddings(self, reuse_dialog: bool = False):
        """Load word embeddings.

        If *reuse_dialog* is True, the caller has already created and shown
        ``self._progress_dialog`` so we skip creating a new one.
        """
        if self.project is None:
            return

        emb_path = self.emb_path_combo.currentText()
        if not emb_path:
            return

        # Ask about .kv conversion FIRST, before any slow work
        self._pending_kv_path = None
        kv_target = self._kv_target_path(emb_path)
        if kv_target is not None and not kv_target.exists():
            reply = QMessageBox.question(
                self,
                "Convert to .kv Format",
                "Would you like to save these embeddings in .kv format?\n\n"
                "This will take some extra time now, but will significantly "
                "speed up loading in future sessions.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self._pending_kv_path = str(kv_target)

        errors, warnings = Validator.validate_embeddings_path(emb_path)
        if errors:
            QMessageBox.critical(self, "Error", "\n".join(errors))
            return

        self._worker = EmbeddingWorker(
            embedding_path=Path(emb_path),
            l2_normalize=self.l2_norm_check.isChecked(),
            abtt_enabled=self.abtt_check.isChecked(),
            abtt_m=self.abtt_m_spin.value(),
            docs=self.project._cached_docs,
        )

        if not reuse_dialog:
            title = "Loading & Converting Embeddings" if self._pending_kv_path else "Loading Embeddings"
            self._progress_dialog = ProgressDialog(title, self)

        self._worker.progress.connect(self._progress_dialog.update_progress)
        self._worker.finished.connect(self._on_embeddings_loaded)
        self._worker.error.connect(self._on_embeddings_error)

        self._progress_dialog.cancel_button.clicked.connect(self._worker.cancel)

        if not reuse_dialog:
            self._progress_dialog.show()
            QApplication.processEvents()
        self._worker.start()
        self._progress_dialog.exec()

    def _on_embeddings_loaded(self, kv, stats):
        """Handle embeddings loaded."""
        if self._progress_dialog.is_cancelled():
            return

        self.project._cached_kv = kv

        self.project.embedding_config.model_path = Path(self.emb_path_combo.currentText())
        self._add_recent_embedding_path(self.emb_path_combo.currentText())
        self.project.embedding_config.l2_normalize = self.l2_norm_check.isChecked()
        self.project.embedding_config.abtt_enabled = self.abtt_check.isChecked()
        self.project.embedding_config.abtt_m = self.abtt_m_spin.value()
        self.project.embedding_config.loaded = True
        self.project.embedding_config.vocab_size = stats["vocab_size"]
        self.project.embedding_config.embedding_dim = stats["embedding_dim"]

        if "coverage_pct" in stats:
            self.project.embedding_config.coverage_pct = stats["coverage_pct"]
            self.project.embedding_config.n_oov = stats.get("oov", 0)

        status_text = (
            f"Loaded {stats['vocab_size']:,} words, "
            f"{stats['embedding_dim']}d vectors"
        )
        if "coverage_pct" in stats:
            status_text += f", {stats['coverage_pct']:.1f}% coverage"

        self.emb_status.setText(status_text)
        self.emb_status.setObjectName("label_status_ok")
        self.emb_status.style().unpolish(self.emb_status)
        self.emb_status.style().polish(self.emb_status)

        self._update_ready_indicator()

        # Chain .kv conversion if the user opted in before loading
        if self._pending_kv_path:
            self._kv_convert_worker = KvConvertWorker(kv, self._pending_kv_path)
            self._kv_convert_worker.progress.connect(self._progress_dialog.update_progress)
            self._kv_convert_worker.finished.connect(self._on_kv_convert_finished)
            self._kv_convert_worker.error.connect(self._on_kv_convert_error)
            self._kv_convert_worker.start()
        else:
            self._progress_dialog.accept()

    def _on_kv_convert_finished(self, kv_path_str: str):
        """Handle successful .kv conversion."""
        self._progress_dialog.accept()
        self._pending_kv_path = None

        # Update combo box and recent paths to the new .kv file
        self._add_recent_embedding_path(kv_path_str)
        self.project.embedding_config.model_path = Path(kv_path_str)

    def _on_kv_convert_error(self, error_message: str):
        """Handle .kv conversion error."""
        self._pending_kv_path = None
        self._progress_dialog.set_error(error_message)

    def _on_embeddings_error(self, error_message):
        """Handle embeddings loading error."""
        if self._progress_dialog.is_cancelled():
            return
        self._progress_dialog.set_error(error_message)
        self.emb_status.setText("Failed to load embeddings")
        self.emb_status.setObjectName("label_status_error")
        self.emb_status.style().unpolish(self.emb_status)
        self.emb_status.style().polish(self.emb_status)

    def _update_ready_indicator(self):
        """Update the ready indicator based on current state."""
        if not self.project:
            self.ready_frame.setObjectName("frame_ready_pending")
            self.ready_frame.style().unpolish(self.ready_frame)
            self.ready_frame.style().polish(self.ready_frame)
            self.ready_title.setText("No Project Loaded")
            self.ready_details.setText("Create or open a project to begin")
            if self.next_btn:
                self.next_btn.setEnabled(False)
            return

        checks = []

        if self.project.dataset_config.cached:
            checks.append(("Dataset", True, f"{self.project.dataset_config.n_valid:,} valid samples"))
        else:
            checks.append(("Dataset", False, "Not loaded"))

        if self.project.spacy_config.processed:
            checks.append(("Text Processing", True, f"{self.project.spacy_config.n_docs_processed:,} docs processed"))
        else:
            checks.append(("Text Processing", False, "Not preprocessed"))

        if self.project._cached_kv is not None:
            checks.append(("Embeddings", True, f"{self.project.embedding_config.vocab_size:,} words"))
        elif self.project.embedding_config.loaded:
            checks.append(("Embeddings", False, "Not in memory — click Load Embeddings"))
        else:
            checks.append(("Embeddings", False, "Not loaded"))

        # Analysis config check
        if hasattr(self, 'continuous_radio'):
            if self.continuous_radio.isChecked():
                col = self.outcome_col_combo.currentText()
                if self.project._cached_y is not None and col and col != "(none)":
                    checks.append(("Analysis", True, f"Continuous - {col}"))
                else:
                    checks.append(("Analysis", False, "No outcome column selected"))
            else:
                col = self.group_col_combo.currentText()
                if (self.project._cached_groups is not None
                        and len(self.project._cached_groups) > 0
                        and len(np.unique(self.project._cached_groups)) >= 2
                        and col and col != "(none)"):
                    checks.append(("Analysis", True, f"Crossgroup - {col}"))
                else:
                    checks.append(("Analysis", False, "No group column selected"))

        all_ready = all(ok for _, ok, _ in checks)

        if all_ready:
            self.ready_frame.setObjectName("frame_ready_ok")
            self.ready_title.setText("Setup Complete")
            if self.next_btn:
                self.next_btn.setEnabled(True)
            self.project.update_ready_state()
        else:
            self.ready_frame.setObjectName("frame_ready_pending")
            self.ready_title.setText("Setup Incomplete")
            if self.next_btn:
                self.next_btn.setEnabled(False)

        self.ready_frame.style().unpolish(self.ready_frame)
        self.ready_frame.style().polish(self.ready_frame)

        details = []
        for name, ok, detail in checks:
            icon = "[OK]" if ok else "[  ]"
            details.append(f"{icon} {name}: {detail}")

        self.ready_details.setText("\n".join(details))

    def _on_next_clicked(self):
        """Handle next button click."""
        self.save_to_project(self.project)
        self.stage_complete.emit()

    # --- Public methods ---

    def load_project(self, project: Project) -> bool:
        """Load a project into the UI.

        Returns ``False`` if the user cancelled a required loading step
        (e.g. embeddings), signalling that the caller should abort.
        """
        self.project = project

        # Show a progress dialog immediately so the user sees feedback
        # while synchronous loading (CSV, preprocessed docs) runs.
        needs_embeddings = (
            project.embedding_config.loaded
            and project.embedding_config.model_path
            and project.embedding_config.model_path.exists()
        )
        self._progress_dialog = ProgressDialog("Loading Project", self)
        self._progress_dialog.update_progress(0, "Loading dataset...")
        self._progress_dialog.show()
        QApplication.processEvents()

        if project.dataset_config.csv_path:
            self.file_path_edit.setText(str(project.dataset_config.csv_path))

            try:
                csv_path = project.dataset_config.csv_path
                if csv_path.exists():
                    ext = csv_path.suffix.lower()
                    if ext in (".xlsx", ".xls"):
                        self._df = pd.read_excel(csv_path)
                    elif ext == ".tsv":
                        self._df = pd.read_csv(csv_path, sep="\t")
                    else:
                        self._df = pd.read_csv(csv_path)
                    project._cached_df = self._df

                    columns = self._df.columns.tolist()
                    self.text_col_combo.clear()
                    self.text_col_combo.addItems(columns)
                    self.id_col_combo.clear()
                    self.id_col_combo.addItem("(none)")
                    self.id_col_combo.addItems(columns)

                    if project.dataset_config.text_column:
                        idx = self.text_col_combo.findText(project.dataset_config.text_column)
                        if idx >= 0:
                            self.text_col_combo.setCurrentIndex(idx)

                    if project.dataset_config.id_column:
                        idx = self.id_col_combo.findText(project.dataset_config.id_column)
                        if idx >= 0:
                            self.id_col_combo.setCurrentIndex(idx)

                    self.validate_dataset_btn.setEnabled(True)
                    self.preprocess_btn.setEnabled(True)

            except Exception as e:
                print(f"Failed to reload data file: {e}")

        self._progress_dialog.update_progress(20, "Loading configuration...")
        QApplication.processEvents()

        # Load spaCy config
        lang_idx = self.language_combo.findText(project.spacy_config.language)
        if lang_idx >= 0:
            self.language_combo.setCurrentIndex(lang_idx)

        self._update_model_options(project.spacy_config.language)
        model_idx = self.model_combo.findText(project.spacy_config.model)
        if model_idx >= 0:
            self.model_combo.setCurrentIndex(model_idx)

        self.lemmatize_check.setChecked(project.spacy_config.lemmatize)
        self.remove_stopwords_check.setChecked(project.spacy_config.remove_stopwords)

        if project.spacy_config.processed:
            self.spacy_status.setText(
                f"Preprocessed {project.spacy_config.n_docs_processed:,} documents"
            )
            self.spacy_status.setObjectName("label_status_ok")
            self.spacy_status.style().unpolish(self.spacy_status)
            self.spacy_status.style().polish(self.spacy_status)

            self._progress_dialog.update_progress(35, "Loading preprocessed texts...")
            QApplication.processEvents()

            cached = ProjectIO.load_preprocessed_docs(project)
            if cached:
                pre_docs, docs, id_row_indices = cached
                project._cached_pre_docs = pre_docs
                project._cached_docs = docs
                project._cached_id_row_indices = id_row_indices

                # Back-fill mean_words_before_stopwords for older projects
                if not project.spacy_config.mean_words_before_stopwords and pre_docs:
                    try:
                        words_per_doc = [
                            sum(len(s.split()) for s in pdoc.sents_surface)
                            for pdoc in pre_docs
                        ]
                        project.spacy_config.mean_words_before_stopwords = (
                            sum(words_per_doc) / len(words_per_doc)
                        )
                    except Exception:
                        pass

        self._progress_dialog.update_progress(70, "Loading configuration...")
        QApplication.processEvents()

        # Load embedding config
        if project.embedding_config.model_path:
            self._add_recent_embedding_path(str(project.embedding_config.model_path))

        self.l2_norm_check.setChecked(project.embedding_config.l2_normalize)
        self.abtt_check.setChecked(project.embedding_config.abtt_enabled)
        self.abtt_m_spin.setValue(project.embedding_config.abtt_m)

        if project.embedding_config.loaded:
            self.emb_status.setText(
                f"Loaded {project.embedding_config.vocab_size:,} words, "
                f"{project.embedding_config.embedding_dim}d"
            )
            self.emb_status.setObjectName("label_status_ok")
            self.emb_status.style().unpolish(self.emb_status)
            self.emb_status.style().polish(self.emb_status)

        # Load hyperparameters
        if project.hyperparameters.n_pca_mode == "manual":
            self.pca_manual_radio.setChecked(True)
            self.pca_k_spin.setValue(project.hyperparameters.n_pca_manual or 50)
        else:
            self.pca_auto_radio.setChecked(True)

        self.sweep_k_min_spin.setValue(project.hyperparameters.sweep_k_min)
        self.sweep_k_max_spin.setValue(project.hyperparameters.sweep_k_max)
        self.sweep_k_step_spin.setValue(project.hyperparameters.sweep_k_step)
        self.window_size_spin.setValue(project.hyperparameters.context_window_size)
        self.sif_a_spin.setValue(project.hyperparameters.sif_a)
        self.unit_beta_check.setChecked(project.hyperparameters.use_unit_beta)
        self.l2_docs_check.setChecked(project.hyperparameters.l2_normalize_docs)
        self.cluster_topn_spin.setValue(project.hyperparameters.clustering_topn)
        self.cluster_k_auto_check.setChecked(project.hyperparameters.clustering_k_auto)
        self.cluster_k_min_spin.setValue(project.hyperparameters.clustering_k_min)
        self.cluster_k_max_spin.setValue(project.hyperparameters.clustering_k_max)
        self.auck_radius_spin.setValue(project.hyperparameters.auck_radius)
        self.beta_smooth_win_spin.setValue(project.hyperparameters.beta_smooth_win)
        self.beta_smooth_kind_combo.setCurrentText(project.hyperparameters.beta_smooth_kind)
        self.weight_by_size_check.setChecked(project.hyperparameters.weight_by_size)

        # Load analysis configuration
        if self._df is not None:
            columns = self._df.columns.tolist()
            numeric_cols = self._df.select_dtypes(include="number").columns.tolist()

            self.outcome_col_combo.blockSignals(True)
            self.outcome_col_combo.clear()
            self.outcome_col_combo.addItem("(none)")
            self.outcome_col_combo.addItems(numeric_cols)
            self.outcome_col_combo.setCurrentIndex(0)
            self.outcome_col_combo.blockSignals(False)

            self.group_col_combo.blockSignals(True)
            self.group_col_combo.clear()
            self.group_col_combo.addItem("(none)")
            self.group_col_combo.addItems(columns)
            self.group_col_combo.setCurrentIndex(0)
            self.group_col_combo.blockSignals(False)

        # Restore analysis type
        if project.dataset_config.analysis_type == "crossgroup":
            self.crossgroup_radio.setChecked(True)
            if project.dataset_config.group_column:
                idx = self.group_col_combo.findText(project.dataset_config.group_column)
                if idx >= 0:
                    self.group_col_combo.setCurrentIndex(idx)
            self.n_perm_spin.setValue(project.dataset_config.n_perm)
        else:
            self.continuous_radio.setChecked(True)
            col = project.dataset_config.outcome_column
            if col:
                idx = self.outcome_col_combo.findText(col)
                if idx >= 0:
                    self.outcome_col_combo.setCurrentIndex(idx)

        # Restore mode
        if project.dataset_config.concept_mode == "fulldoc":
            self.fulldoc_radio.setChecked(True)
        else:
            self.lexicon_radio.setChecked(True)

        # Trigger column change to populate cached y/groups
        if self.continuous_radio.isChecked():
            self._on_outcome_column_changed()
        else:
            self._on_group_column_changed()

        self._update_advanced_settings_visibility()
        self._update_ready_indicator()

        # Auto-reload embeddings if they were previously loaded AND the
        # user has opted in via Settings (default: off for faster opens).
        from .settings_dialog import get_autoload_embeddings

        if needs_embeddings and project._cached_kv is None and get_autoload_embeddings():
            self._progress_dialog.update_progress(50, "Loading embeddings...")
            QApplication.processEvents()
            self._load_embeddings(reuse_dialog=True)
            if (
                hasattr(self, "_progress_dialog")
                and self._progress_dialog is not None
                and self._progress_dialog.is_cancelled()
            ):
                self.project = None
                return False
        else:
            # No embeddings to load — set progress to 100%.
            # The caller (_load_project_into_ui) will close the dialog
            # after all stage widgets have finished loading.
            self._progress_dialog.update_progress(100, "Project loaded.")
            QApplication.processEvents()

        return True

    def save_to_project(self, project: Project):
        """Save current UI state to project."""
        if not project:
            return

        project.dataset_config.text_column = self.text_col_combo.currentText()

        id_col = self.id_col_combo.currentText()
        project.dataset_config.id_column = id_col if id_col != "(none)" else None

        project.spacy_config.language = self.language_combo.currentText()
        project.spacy_config.model = self.model_combo.currentText()
        project.spacy_config.lemmatize = self.lemmatize_check.isChecked()
        project.spacy_config.remove_stopwords = self.remove_stopwords_check.isChecked()

        if self.emb_path_combo.currentText() and self.emb_path_combo.currentText() != "(no recent models)":
            project.embedding_config.model_path = Path(self.emb_path_combo.currentText())
        project.embedding_config.l2_normalize = self.l2_norm_check.isChecked()
        project.embedding_config.abtt_enabled = self.abtt_check.isChecked()
        project.embedding_config.abtt_m = self.abtt_m_spin.value()

        project.hyperparameters.n_pca_mode = "manual" if self.pca_manual_radio.isChecked() else "auto"
        project.hyperparameters.n_pca_manual = self.pca_k_spin.value() if self.pca_manual_radio.isChecked() else None
        project.hyperparameters.sweep_k_min = self.sweep_k_min_spin.value()
        project.hyperparameters.sweep_k_max = self.sweep_k_max_spin.value()
        project.hyperparameters.sweep_k_step = self.sweep_k_step_spin.value()
        project.hyperparameters.context_window_size = self.window_size_spin.value()
        project.hyperparameters.sif_a = self.sif_a_spin.value()
        project.hyperparameters.use_unit_beta = self.unit_beta_check.isChecked()
        project.hyperparameters.l2_normalize_docs = self.l2_docs_check.isChecked()
        project.hyperparameters.clustering_topn = self.cluster_topn_spin.value()
        project.hyperparameters.clustering_k_auto = self.cluster_k_auto_check.isChecked()
        project.hyperparameters.clustering_k_min = self.cluster_k_min_spin.value()
        project.hyperparameters.clustering_k_max = self.cluster_k_max_spin.value()
        project.hyperparameters.clustering_top_words = self.cluster_topn_spin.value()
        project.hyperparameters.auck_radius = self.auck_radius_spin.value()
        project.hyperparameters.beta_smooth_win = self.beta_smooth_win_spin.value()
        project.hyperparameters.beta_smooth_kind = self.beta_smooth_kind_combo.currentText()
        project.hyperparameters.weight_by_size = self.weight_by_size_check.isChecked()

        # Analysis configuration
        project.dataset_config.analysis_type = "crossgroup" if self.crossgroup_radio.isChecked() else "continuous"
        project.dataset_config.concept_mode = "fulldoc" if self.fulldoc_radio.isChecked() else "lexicon"
        project.dataset_config.n_perm = self.n_perm_spin.value()
        if self.continuous_radio.isChecked():
            col = self.outcome_col_combo.currentText()
            project.dataset_config.outcome_column = None if col == "(none)" else (col or None)
        else:
            col = self.group_col_combo.currentText()
            project.dataset_config.group_column = None if col == "(none)" else (col or None)
