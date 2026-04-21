"""Stage 1: Setup view for SSD."""

from pathlib import Path
from typing import Optional

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
    QFileDialog,
    QMessageBox,
    QScrollArea,
    QFrame,
    QApplication,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QRadioButton,
    QButtonGroup,
)
from PySide6.QtCore import Qt, Signal

from ..models.project import Project
from ..utils.worker_threads import PreprocessWorker, EmbeddingPrepareWorker, EmbeddingLoadWorker, SpacyDownloadWorker, find_local_model
from ..utils.file_io import ProjectIO
from ..utils.settings import app_settings
from ..utils.paths import embeddings_dir, embeddings_autoload_enabled
from .widgets.progress_dialog import ProgressDialog
from .widgets.overlay_info_mixin import OverlayInfoMixin


class Stage1Widget(OverlayInfoMixin, QWidget):
    """Stage 1: Setup - Dataset, spaCy, Embeddings, Hyperparameters."""

    stage_complete = Signal()

    _info_margin_right = 6
    _info_margin_top = 20

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Optional[Project] = None
        self._df: Optional[pd.DataFrame] = None
        self._worker = None
        self.next_btn = None  # Created in _create_navigation
        self._settings = app_settings()
        self._custom_stopwords: list = []  # loaded from file in custom stopword mode

        self._init_overlay_info()
        self._setup_ui()

    def eventFilter(self, obj, event):
        if self._overlay_info_event_filter(obj, event):
            return True
        return super().eventFilter(obj, event)

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

        layout.addLayout(file_row)

        # Encoding row
        enc_row = QHBoxLayout()
        enc_row.addWidget(QLabel("Encoding:"))
        self.encoding_combo = QComboBox()
        for label, value in [
            ("UTF-8", "utf-8-sig"),
            ("Latin-1 / ISO-8859-1", "latin-1"),
            ("CP1252 (Windows)", "cp1252"),
            ("UTF-16", "utf-16"),
        ]:
            self.encoding_combo.addItem(label, userData=value)
        self.encoding_combo.setFixedWidth(180)
        enc_row.addWidget(self.encoding_combo)
        enc_row.addStretch()

        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._load_csv)
        enc_row.addWidget(load_btn)
        self.load_csv_btn = load_btn

        layout.addLayout(enc_row)

        # Column selection
        cols_layout = QHBoxLayout()

        # Text column
        text_col_layout = QVBoxLayout()
        text_col_layout.addWidget(QLabel("Text Column:"))
        self.text_col_combo = QComboBox()
        self.text_col_combo.currentTextChanged.connect(self._on_text_column_changed)
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
            "into a single profile.<br><br>"
            "<b>Encoding</b> — the character encoding of your CSV/TSV file. "
            "Use <i>UTF-8</i> for most files. If you see garbled characters "
            "or a load error, try <i>Latin-1</i> (common for Western-European "
            "survey exports) or <i>CP1252</i> (common for files exported from "
            "Excel on Windows). Encoding is ignored for Excel files.<br><br>"
            "Click <b>Validate</b> to check for missing values and confirm "
            "the dataset is ready.",
        )
        parent_layout.addWidget(group)

    def _create_spacy_section(self):
        """Create the spaCy text processing section."""
        group = QGroupBox("2. Text Processing (spaCy)")
        outer = QVBoxLayout()

        # --- Model mode toggle ---
        mode_row = QHBoxLayout()
        self.radio_language = QRadioButton("Language")
        self.radio_language.setChecked(True)
        self.radio_custom = QRadioButton("Custom model")
        self._model_mode_group = QButtonGroup(self)
        self._model_mode_group.addButton(self.radio_language)
        self._model_mode_group.addButton(self.radio_custom)
        mode_row.addWidget(self.radio_language)
        mode_row.addWidget(self.radio_custom)
        mode_row.addStretch()
        outer.addLayout(mode_row)

        # --- Language mode widgets (shown by default) ---
        self.language_row = QWidget()
        self.language_row.setObjectName("language_row")
        self.language_row.setAttribute(Qt.WA_StyledBackground, False)
        self.language_row.setStyleSheet("QWidget#language_row { background: transparent; }")
        lang_layout = QHBoxLayout(self.language_row)
        lang_layout.setContentsMargins(20, 0, 0, 0)

        self.language_combo = QComboBox()
        self.language_combo.setMinimumWidth(140)
        try:
            from ssdiff.lang_config import LANGUAGES, _ALIASES
            code_to_name = {v: k.replace("_", " ").title() for k, v in _ALIASES.items()}
            items = [(code_to_name.get(c, c), c) for c in sorted(LANGUAGES.keys())]
        except ImportError:
            items = [
                ("Catalan", "ca"), ("Danish", "da"), ("German", "de"),
                ("Greek", "el"), ("English", "en"), ("Spanish", "es"),
                ("French", "fr"), ("Croatian", "hr"), ("Italian", "it"),
                ("Lithuanian", "lt"), ("Macedonian", "mk"), ("Norwegian", "nb"),
                ("Dutch", "nl"), ("Polish", "pl"), ("Portuguese", "pt"),
                ("Romanian", "ro"), ("Russian", "ru"), ("Slovenian", "sl"),
                ("Swedish", "sv"), ("Ukrainian", "uk"),
            ]
        for display, code in items:
            self.language_combo.addItem(display, userData=code)
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        lang_layout.addWidget(self.language_combo)

        self.auto_model_label = QLabel("")
        self.auto_model_label.setStyleSheet("color: gray; font-style: italic;")
        lang_layout.addWidget(self.auto_model_label)
        lang_layout.addStretch()

        outer.addWidget(self.language_row)

        en_idx = self.language_combo.findData("en")
        if en_idx >= 0:
            self.language_combo.setCurrentIndex(en_idx)
        self._update_auto_model_label(self.language_combo.currentData() or "en")

        # --- Custom model widgets (hidden by default) ---
        self.custom_row = QWidget()
        self.custom_row.setObjectName("custom_row")
        self.custom_row.setAttribute(Qt.WA_StyledBackground, False)
        self.custom_row.setStyleSheet("QWidget#custom_row { background: transparent; }")
        custom_layout = QHBoxLayout(self.custom_row)
        custom_layout.setContentsMargins(20, 0, 0, 0)

        self.custom_model_input = QLineEdit()
        self.custom_model_input.setPlaceholderText("Model name (e.g. en_core_web_lg) or path")
        custom_layout.addWidget(self.custom_model_input)
        custom_layout.addStretch()

        self.custom_row.setVisible(False)
        outer.addWidget(self.custom_row)

        self.radio_language.toggled.connect(self._on_model_mode_changed)

        # --- Stopwords ---
        stop_row = QHBoxLayout()
        stop_row.addWidget(QLabel("Stopwords:"))

        self.stopword_combo = QComboBox()
        self.stopword_combo.addItems(["Default", "Do not remove", "Custom file"])
        self.stopword_combo.currentIndexChanged.connect(self._on_stopword_mode_changed)
        stop_row.addWidget(self.stopword_combo)

        self.stopword_file_btn = QPushButton("Browse...")
        self.stopword_file_btn.setVisible(False)
        self.stopword_file_btn.clicked.connect(self._browse_stopword_file)
        stop_row.addWidget(self.stopword_file_btn)

        self.stopword_file_label = QLabel("")
        self.stopword_file_label.setStyleSheet("color: gray;")
        self.stopword_file_label.setVisible(False)
        stop_row.addWidget(self.stopword_file_label)

        stop_row.addStretch()
        outer.addLayout(stop_row)

        # --- Preprocess button ---
        preprocess_row = QHBoxLayout()
        preprocess_row.addStretch()

        self.preprocess_btn = QPushButton("Preprocess Texts")
        self.preprocess_btn.clicked.connect(self._preprocess_texts)
        self.preprocess_btn.setEnabled(False)
        preprocess_row.addWidget(self.preprocess_btn)

        outer.addLayout(preprocess_row)

        # Status
        self.spacy_status = QLabel("")
        outer.addWidget(self.spacy_status)

        group.setLayout(outer)
        self._add_overlay_info(group,
            "<b>Text Processing</b><br><br>"
            "Choose a <b>language</b> (model auto-resolved) or specify a "
            "<b>custom spaCy model</b> name/path.<br><br>"
            "Stopwords can be left at default (resolved per language), "
            "disabled, or loaded from a custom file.<br><br>"
            "Click <b>Preprocess</b> to tokenize, lemmatize, and "
            "sentence-split your texts.",
        )
        return group

    def _update_auto_model_label(self, lang: str):
        """Show the auto-resolved model name for the selected language."""
        try:
            from ssdiff.lang_config import lang_to_model
            model = lang_to_model(lang)
        except (ImportError, KeyError):
            model = f"{lang}_core_news_lg"
        self.auto_model_label.setText(f"→ {model}")

    def _on_model_mode_changed(self, language_checked: bool):
        """Toggle between Language and Custom model modes."""
        self.language_row.setVisible(language_checked)
        self.custom_row.setVisible(not language_checked)

    def _on_stopword_mode_changed(self, index: int):
        """Show/hide the custom stopword file controls."""
        is_custom = (index == 2)  # "Custom file"
        self.stopword_file_btn.setVisible(is_custom)
        self.stopword_file_label.setVisible(is_custom)

    def _browse_stopword_file(self):
        """Open file dialog to select a custom stopwords file (one word per line)."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Stopwords File", "",
            "Text files (*.txt);;All files (*)",
        )
        if not filepath:
            return
        try:
            with open(filepath, encoding="utf-8") as f:
                words = [line.strip() for line in f if line.strip()]
            self._custom_stopwords = words
            self.stopword_file_label.setText(f"{len(words)} words loaded")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to read stopwords file:\n{e}")
            self._custom_stopwords = []
            self.stopword_file_label.setText("")

    def _create_embeddings_section(self):
        """Create the embedding management section with prepare + select workflow."""
        group = QGroupBox("3. Word Embeddings")
        layout = QVBoxLayout()

        # --- Top: Prepare Embeddings ---
        prepare_group = QGroupBox("Prepare Embeddings")
        prepare_layout = QVBoxLayout()

        # Source file selection
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Source File:"))
        self.emb_source_path = QLineEdit()
        self.emb_source_path.setPlaceholderText("Select embeddings file (.ssdembed, .bin, .txt, .kv, .vec, .gz)")
        self.emb_source_path.setReadOnly(True)
        file_row.addWidget(self.emb_source_path, stretch=1)

        browse_emb_btn = QPushButton("Browse...")
        browse_emb_btn.clicked.connect(self._browse_embeddings_source)
        file_row.addWidget(browse_emb_btn)
        prepare_layout.addLayout(file_row)

        # Normalization row
        norm_row = QHBoxLayout()
        self.l2_check = QCheckBox("L2 Normalize")
        self.l2_check.setChecked(True)
        norm_row.addWidget(self.l2_check)
        norm_row.addWidget(QLabel("ABTT:"))
        self.abtt_spin = QSpinBox()
        self.abtt_spin.setRange(0, 10)
        self.abtt_spin.setValue(1)
        norm_row.addWidget(self.abtt_spin)
        norm_row.addStretch()
        self.prepare_btn = QPushButton("Load && Prepare")
        self.prepare_btn.setEnabled(False)
        self.prepare_btn.clicked.connect(self._prepare_embedding)
        norm_row.addWidget(self.prepare_btn)
        prepare_layout.addLayout(norm_row)

        self.prepare_status = QLabel("")
        prepare_layout.addWidget(self.prepare_status)
        prepare_group.setLayout(prepare_layout)
        layout.addWidget(prepare_group)

        # --- Bottom: Prepared Embeddings Table ---
        select_group = QGroupBox("Prepared Embeddings")
        select_layout = QVBoxLayout()

        self.emb_table = QTableWidget()
        self.emb_table.setColumnCount(6)
        self.emb_table.setHorizontalHeaderLabels([
            "Name", "Vocab", "Dim", "L2", "ABTT", "Size (MB)",
        ])
        self.emb_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 6):
            self.emb_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.emb_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.emb_table.setSelectionMode(QTableWidget.SingleSelection)
        self.emb_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.emb_table.setAlternatingRowColors(True)
        self.emb_table.setMinimumHeight(120)
        select_layout.addWidget(self.emb_table)

        btn_row = QHBoxLayout()
        self.select_emb_btn = QPushButton("Select")
        self.select_emb_btn.setEnabled(False)
        self.select_emb_btn.clicked.connect(self._select_embedding)
        btn_row.addWidget(self.select_emb_btn)
        self.delete_emb_btn = QPushButton("Delete")
        self.delete_emb_btn.setEnabled(False)
        self.delete_emb_btn.clicked.connect(self._delete_embedding)
        btn_row.addWidget(self.delete_emb_btn)
        btn_row.addStretch()
        self.emb_status = QLabel("")
        self.emb_status.setObjectName("label_muted")
        btn_row.addWidget(self.emb_status)
        select_layout.addLayout(btn_row)

        select_group.setLayout(select_layout)
        layout.addWidget(select_group)

        group.setLayout(layout)
        self._add_overlay_info(group,
            "<b>Word Embeddings</b><br><br>"
            "<b>Prepare:</b> Select a source embedding file, configure L2 "
            "normalization and ABTT, then click <i>Load &amp; Prepare</i>. "
            "The embedding is normalized and saved as <code>.ssdembed</code> "
            "to this project.<br><br>"
            "<b>Select:</b> Choose a prepared embedding for analysis. "
            "Exactly one embedding is loaded into RAM at a time.",
        )
        return group

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

    def _on_language_changed(self, index: int):
        """Update the auto-resolved model label when language changes.

        Also saves the chosen language to the project so the readiness check
        can detect a mismatch with the language used to preprocess the corpus.
        The corpus is preserved — switching back to the preprocessed language
        flips the indicator green again without re-running preprocessing.
        """
        lang = self.language_combo.currentData()
        if not lang:
            return
        self._update_auto_model_label(lang)

        if self.project is not None and self.project.language != lang:
            self.project.language = lang
            self.project.mark_dirty()
            self._update_ready_indicator()

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
            encoding = self.encoding_combo.currentData()

            if ext in (".xlsx", ".xls"):
                self._df = pd.read_excel(filepath)
            elif ext == ".tsv":
                self._df = pd.read_csv(filepath, sep="\t", encoding=encoding)
            else:
                self._df = pd.read_csv(filepath, encoding=encoding)

            # Populate column combos
            columns = self._df.columns.tolist()

            self.text_col_combo.clear()
            self.text_col_combo.addItems(columns)

            self.id_col_combo.clear()
            self.id_col_combo.addItem("(none)")
            self.id_col_combo.addItems(columns)

            # Show stats
            n_rows = len(self._df)
            n_cols = len(columns)
            self.dataset_stats_label.setText(
                f"Loaded {n_rows:,} rows, {n_cols} columns"
            )

            self.validate_dataset_btn.setEnabled(True)
            self.preprocess_btn.setEnabled(True)

            # New CSV = nothing validated yet
            if self.project:
                self.project._df = self._df
                self.project.n_valid = 0
                self.project.preprocessed_text_column = None
                self.project._pre_docs = None
                self.project._docs = None
                self.project._corpus = None
                self.project._id_row_indices = None
                self.project.mark_dirty()

                self.spacy_status.setText("Not preprocessed")
                self.spacy_status.setObjectName("label_muted")
                self.spacy_status.style().unpolish(self.spacy_status)
                self.spacy_status.style().polish(self.spacy_status)
                self._update_ready_indicator()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {e}")

    def _validate_dataset(self):
        """Validate the loaded dataset."""
        if self._df is None or self.project is None:
            return

        text_col = self.text_col_combo.currentText()
        id_col = self.id_col_combo.currentText()
        if id_col == "(none)":
            id_col = None

        # Store columns on project so validate_text() can read them
        if self.project:
            self.project.csv_path = Path(self.file_path_edit.text())
            self.project.csv_encoding = self.encoding_combo.currentData()
            self.project.text_column = text_col
            self.project.id_column = id_col
            self.project._df = self._df

        errors, warnings, id_stats = self.project.validate_text()

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
            self.project.n_rows = len(self._df)
            self.project.n_valid = len(self._df)
            self.project.mark_dirty()

        self._update_ready_indicator()

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
            self.project._id_row_indices = id_row_indices
        else:
            texts_raw = self._df[text_col].fillna("").astype(str).tolist()
            self.project._id_row_indices = None

        # Resolve model name from input mode
        if self.radio_language.isChecked():
            lang = self.language_combo.currentData()
            try:
                from ssdiff.lang_config import lang_to_model
                model_name = lang_to_model(lang)
            except (ImportError, KeyError):
                model_name = f"{lang}_core_news_lg"
        else:
            model_name = self.custom_model_input.text().strip()
            if not model_name:
                QMessageBox.warning(self, "No Model", "Enter a spaCy model name or path.")
                return

        # Resolve stopwords from stopword mode
        stop_idx = self.stopword_combo.currentIndex()
        if stop_idx == 0:  # Default
            stopwords_override = None  # worker calls load_stopwords(language)
        elif stop_idx == 1:  # Do not remove
            stopwords_override = []
        else:  # Custom file
            if not self._custom_stopwords:
                QMessageBox.warning(self, "No Stopwords", "Load a custom stopwords file first.")
                return
            stopwords_override = list(self._custom_stopwords)

        # Check if the spaCy model is available (pip-installed or locally downloaded)
        import spacy.util
        local_path = find_local_model(model_name)
        # If custom model looks like a path, try loading from that path directly
        is_path = not self.radio_language.isChecked() and ("/" in model_name or "\\" in model_name)
        if is_path:
            local_path = Path(model_name) if Path(model_name).exists() else None
            if local_path is None:
                QMessageBox.warning(self, "Model Not Found", f"Path does not exist: {model_name}")
                return
        elif not spacy.util.is_package(model_name) and not local_path:
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
            self._pending_stopwords_override = stopwords_override
            self._download_spacy_model(model_name)
            return

        self._run_preprocess_worker(texts_raw, model_name, model_path=local_path,
                                    stopwords_override=stopwords_override)

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
        stopwords_override = getattr(self, "_pending_stopwords_override", None)
        self._pending_preprocess_texts = None
        self._pending_stopwords_override = None
        if texts_raw is None:
            self._progress_dialog.accept()
            return
        # Resolve model name from current UI state
        if self.radio_language.isChecked():
            lang = self.language_combo.currentData()
            try:
                from ssdiff.lang_config import lang_to_model
                model_name = lang_to_model(lang)
            except (ImportError, KeyError):
                model_name = f"{lang}_core_news_lg"
        else:
            model_name = self.custom_model_input.text().strip()
        model_path = Path(model_path_str) if model_path_str else None
        # Reuse the download dialog for preprocessing (no accept/new dialog)
        self._progress_dialog.setWindowTitle("Preprocessing Texts")
        self._run_preprocess_worker(texts_raw, model_name, model_path=model_path,
                                    stopwords_override=stopwords_override, reuse_dialog=True)

    def _on_spacy_download_error(self, error_message: str):
        """Handle spaCy model download failure."""
        self._pending_preprocess_texts = None
        self._pending_stopwords_override = None
        self._progress_dialog.set_error(error_message)

    def _run_preprocess_worker(
        self, texts_raw, model_name: str,
        model_path: Optional[Path] = None,
        stopwords_override=None,
        reuse_dialog: bool = False,
    ):
        """Start the preprocessing worker thread."""
        self._worker = PreprocessWorker(
            texts_raw=texts_raw,
            language=self.language_combo.currentData(),
            model=model_name,
            model_path=model_path,
            stopwords_override=stopwords_override,
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

        self.project._pre_docs = pre_docs
        self.project._docs = docs

        # Update project config
        self.project.language = self.language_combo.currentData()
        self.project.input_mode = "language" if self.radio_language.isChecked() else "custom"
        self.project.spacy_model = self.custom_model_input.text().strip() if not self.radio_language.isChecked() else ""
        stop_idx = self.stopword_combo.currentIndex()
        self.project.stopword_mode = ["default", "none", "custom"][stop_idx]
        self.project.custom_stopwords = list(self._custom_stopwords) if stop_idx == 2 else []
        self.project.preprocessed_text_column = self.text_col_combo.currentText()
        self.project.preprocessed_language = self.project.language
        self.project.n_docs_processed = stats["n_docs"]
        self.project.total_tokens = stats["total_tokens"]
        self.project.mean_words_before_stopwords = stats["mean_words_before_stopwords"]

        # Build and save a library Corpus object (Phase 6: corpus.pkl)
        from ssdiff.corpus import Corpus
        lang = self.project.language
        corpus = Corpus(docs, pretokenized=True, lang=lang)
        self.project._corpus = corpus
        ProjectIO.save_corpus(self.project, corpus, pre_docs=pre_docs)

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

        self.project.mark_dirty()
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

    def _on_text_column_changed(self, new_col: str):
        """Invalidate dataset validation and preprocessing when text column changes."""
        if self.project is None or not new_col:
            return

        p = self.project
        changed = False

        # Invalidate dataset validation if validated for a different column
        if p.text_ready and p.text_column and new_col != p.text_column:
            p.n_valid = 0
            self.dataset_status.setText("Text column changed — re-validate dataset")
            self.dataset_status.setObjectName("label_muted")
            self.dataset_status.style().unpolish(self.dataset_status)
            self.dataset_status.style().polish(self.dataset_status)
            changed = True

        # Invalidate preprocessing if done for a different column
        if p.preprocessing_ready and p.preprocessed_text_column and new_col != p.preprocessed_text_column:
            p.preprocessed_text_column = None
            p._pre_docs = None
            p._docs = None
            p._corpus = None
            p._id_row_indices = None
            self.spacy_status.setText("Preprocessed texts discarded (column changed)")
            self.spacy_status.setObjectName("label_muted")
            self.spacy_status.style().unpolish(self.spacy_status)
            self.spacy_status.style().polish(self.spacy_status)
            changed = True

        if changed:
            self.project.mark_dirty()
        self._update_ready_indicator()

    # --- Embedding prepare / select methods ---

    def _browse_embeddings_source(self):
        """Browse for a source embedding file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Embeddings File",
            str(self._settings.value("last_emb_dir", "")),
            "Embeddings (*.ssdembed *.bin *.txt *.kv *.vec *.gz);;All Files (*)",
        )
        if path:
            self.emb_source_path.setText(path)
            self._settings.setValue("last_emb_dir", str(Path(path).parent))
            self.prepare_btn.setEnabled(True)
            # If source is .ssdembed, read flags to constrain L2/ABTT
            if path.endswith(".ssdembed"):
                self._apply_ssdembed_constraints(Path(path))
            else:
                self.l2_check.setEnabled(True)
                self.abtt_spin.setMinimum(0)

    def _apply_ssdembed_constraints(self, path: Path):
        """Apply irreversibility constraints for .ssdembed source files.

        Reads metadata from the pickle file WITHOUT loading the heavy
        .vectors.npy sidecar.
        """
        try:
            import pickle
            with open(path, "rb") as f:
                obj = pickle.load(f)
            is_l2 = getattr(obj, "l2_normalized", False)
            cur_abtt = getattr(obj, "abtt", 0)
            del obj
            if is_l2:
                self.l2_check.setChecked(True)
                self.l2_check.setEnabled(False)
            else:
                self.l2_check.setEnabled(True)
            if cur_abtt > 0:
                self.abtt_spin.setMinimum(cur_abtt)
                self.abtt_spin.setValue(cur_abtt)
            else:
                self.abtt_spin.setMinimum(0)
        except Exception:
            self.l2_check.setEnabled(True)
            self.abtt_spin.setMinimum(0)

    def _prepare_embedding(self):
        """Start the embedding preparation worker."""
        source = self.emb_source_path.text()
        if not source or not Path(source).exists():
            QMessageBox.warning(self, "Error", "Please select a valid source file.")
            return
        if not self.project:
            return
        output_dir = embeddings_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        l2 = self.l2_check.isChecked()
        abtt = self.abtt_spin.value()
        self.prepare_btn.setEnabled(False)
        self.prepare_status.setText("Preparing...")
        self._prepare_worker = EmbeddingPrepareWorker(
            source_path=Path(source),
            output_dir=output_dir,
            l2_normalize=l2,
            abtt=abtt,
            parent=self,
        )

        self._progress_dialog = ProgressDialog("Preparing Embeddings", self)
        self._prepare_worker.progress.connect(self._progress_dialog.update_progress)
        self._prepare_worker.progress.connect(
            lambda pct, msg: self.prepare_status.setText(msg)
        )
        self._prepare_worker.finished.connect(self._on_embedding_prepared)
        self._prepare_worker.error.connect(self._on_embedding_prepare_error)
        self._progress_dialog.cancel_button.clicked.connect(self._prepare_worker.cancel)

        self._progress_dialog.show()
        QApplication.processEvents()
        self._prepare_worker.start()
        self._progress_dialog.exec()

    def _on_embedding_prepared(self, saved_path: str, meta: dict):
        """Handle embedding preparation completion."""
        if hasattr(self, "_progress_dialog") and self._progress_dialog:
            self._progress_dialog.accept()
        self.prepare_btn.setEnabled(True)
        new_hash = ProjectIO.compute_embedding_hash(Path(saved_path))
        existing = ProjectIO.find_duplicate_embedding(self.project, new_hash)
        saved_name = Path(saved_path).name
        if existing and existing != saved_name:
            Path(saved_path).unlink(missing_ok=True)
            sidecar = Path(saved_path + ".vectors.npy")
            if sidecar.exists():
                sidecar.unlink()
            QMessageBox.information(
                self, "Already Prepared",
                f"This embedding is identical to: {existing}\n"
                f"The duplicate was not saved.",
            )
            self.prepare_status.setText(f"Already prepared: {existing}")
        else:
            self.prepare_status.setText(
                f"Saved: {saved_name} "
                f"({meta['vocab_size']:,} words, {meta['embedding_dim']}d)"
            )
        self._refresh_embeddings_table()
        # Auto-select if none selected yet
        if not self.project.selected_embedding:
            target = existing or saved_name
            for row in range(self.emb_table.rowCount()):
                if self.emb_table.item(row, 0) and self.emb_table.item(row, 0).text() == target:
                    self.emb_table.selectRow(row)
                    self._select_embedding()
                    break

    def _on_embedding_prepare_error(self, error_msg: str):
        if hasattr(self, "_progress_dialog") and self._progress_dialog:
            self._progress_dialog.accept()
        self.prepare_btn.setEnabled(True)
        self.prepare_status.setText("Preparation failed")
        QMessageBox.critical(self, "Error", error_msg)

    def _refresh_embeddings_table(self):
        """Refresh the prepared embeddings table from project's embeddings/ dir."""
        if not self.project:
            return
        embeddings = ProjectIO.list_prepared_embeddings(self.project)
        self.emb_table.setRowCount(len(embeddings))
        selected = self.project.selected_embedding
        for i, meta in enumerate(embeddings):
            self.emb_table.setItem(i, 0, QTableWidgetItem(meta["filename"]))
            self.emb_table.setItem(i, 1, QTableWidgetItem(f"{meta['vocab_size']:,}"))
            self.emb_table.setItem(i, 2, QTableWidgetItem(str(meta["embedding_dim"])))
            self.emb_table.setItem(i, 3, QTableWidgetItem("Yes" if meta["l2_normalized"] else "No"))
            self.emb_table.setItem(i, 4, QTableWidgetItem(str(meta["abtt"])))
            self.emb_table.setItem(i, 5, QTableWidgetItem(f"{meta['file_size_mb']:.1f}"))
            if meta["filename"] == selected:
                self.emb_table.selectRow(i)
        # Enable/disable buttons
        has_rows = self.emb_table.rowCount() > 0
        self.select_emb_btn.setEnabled(has_rows)
        self.delete_emb_btn.setEnabled(has_rows)

    def _select_embedding(self):
        """Load the selected embedding into RAM."""
        if not self.project:
            return
        rows = self.emb_table.selectionModel().selectedRows()
        if rows:
            row = rows[0].row()
            filename = self.emb_table.item(row, 0).text()
        elif self.project.selected_embedding:
            # Fallback: no table selection but config has a file — find it
            target = self.project.selected_embedding
            filename = None
            for i in range(self.emb_table.rowCount()):
                item = self.emb_table.item(i, 0)
                if item and item.text() == target:
                    self.emb_table.selectRow(i)
                    filename = target
                    break
            if not filename:
                return
        else:
            return
        emb_path = embeddings_dir() / filename
        if not emb_path.exists():
            QMessageBox.warning(self, "Error", f"File not found: {filename}")
            return
        # Unload current embedding and detach from results
        self.project._emb = None
        self.project._kv = None
        for r in self.project.results:
            if r._result is not None and hasattr(r._result, "embeddings"):
                r._result.embeddings = None
        self.emb_status.setText(f"Loading {filename}...")
        self.select_emb_btn.setEnabled(False)
        self._load_worker = EmbeddingLoadWorker(
            ssdembed_path=emb_path,
            docs=self.project._docs,
            parent=self,
        )

        # Show progress dialog for manual Select (autoload has its own)
        is_autoload = getattr(self, "_autoloading_embedding", False)
        if not is_autoload:
            self._progress_dialog = ProgressDialog("Loading Embeddings", self)
            self._load_worker.progress.connect(self._progress_dialog.update_progress)
            self._progress_dialog.cancel_button.clicked.connect(self._load_worker.cancel)

        self._load_worker.progress.connect(
            lambda pct, msg: self.emb_status.setText(msg)
        )
        self._load_worker.finished.connect(
            lambda emb, stats: self._on_embedding_loaded(emb, stats, filename)
        )
        self._load_worker.error.connect(
            lambda err: self._on_embedding_load_error(err)
        )

        if not is_autoload:
            self._progress_dialog.show()
            QApplication.processEvents()
        self._load_worker.start()
        if not is_autoload:
            self._progress_dialog.exec()

    def _on_embedding_load_error(self, err: str):
        """Handle embedding load failure."""
        if hasattr(self, "_progress_dialog") and self._progress_dialog:
            self._progress_dialog.accept()
        self.emb_status.setText("Load failed")
        self.select_emb_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", err)
        if getattr(self, "_autoloading_embedding", False):
            self._autoloading_embedding = False

    def _on_embedding_loaded(self, emb, stats: dict, filename: str):
        """Handle embedding load completion."""
        if hasattr(self, "_progress_dialog") and self._progress_dialog:
            self._progress_dialog.accept()
        self.project._emb = emb
        self.project._kv = emb
        p = self.project
        p.selected_embedding = filename
        p.vocab_size = stats.get("vocab_size", 0)
        p.embedding_dim = stats.get("embedding_dim", 0)
        p.l2_normalized = stats.get("l2_normalized", False)
        p.abtt = stats.get("abtt", 0)
        p.emb_coverage_pct = stats.get("coverage_pct", 0.0)
        p.emb_n_oov = stats.get("n_oov", 0)
        cov = f", coverage: {p.emb_coverage_pct:.1f}%" if p.emb_coverage_pct > 0 else ""
        self.emb_status.setText(
            f"Selected: {filename} ({p.vocab_size:,} words, {p.embedding_dim}d{cov})"
        )
        self.select_emb_btn.setEnabled(True)
        self.project.mark_dirty()
        self._update_ready_indicator()

        if getattr(self, "_autoloading_embedding", False):
            self._autoloading_embedding = False

    def _delete_embedding(self):
        """Delete a prepared embedding from the project."""
        rows = self.emb_table.selectionModel().selectedRows()
        if not rows or not self.project:
            return
        row = rows[0].row()
        filename = self.emb_table.item(row, 0).text()
        reply = QMessageBox.question(
            self, "Delete Embedding",
            f"Delete {filename} from this project?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        emb_path = embeddings_dir() / filename
        sidecar = Path(str(emb_path) + ".vectors.npy")
        import subprocess
        subprocess.run(["trash-put", str(emb_path)], check=False)
        if sidecar.exists():
            subprocess.run(["trash-put", str(sidecar)], check=False)
        if self.project.selected_embedding == filename:
            self.project.selected_embedding = None
            self.project._emb = None
            self.project._kv = None
            self.emb_status.setText("")
            self._update_ready_indicator()
        self._refresh_embeddings_table()

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

        # 1. Dataset loaded (CSV in memory)
        if self.project._df is not None and len(self.project._df) > 0:
            n_rows = len(self.project._df)
            n_cols = len(self.project._df.columns)
            checks.append(("Dataset", True, f"{n_rows:,} rows, {n_cols} columns"))
        else:
            checks.append(("Dataset", False, "Not loaded"))

        # 2. Text column validated
        if self.project.text_ready and self.project.n_valid > 0:
            checks.append(("Text Column", True, f"{self.project.n_valid:,} valid samples"))
        else:
            checks.append(("Text Column", False, "Not validated"))

        # 3. Text preprocessing
        preprocessed = self.project.preprocessing_ready
        if preprocessed:
            n_docs = self.project.n_docs_processed or len(self.project._docs or [])
            pp_col = self.project.preprocessed_text_column
            if pp_col:
                detail = f"{n_docs:,} docs (column: '{pp_col}')"
                current_col = self.text_col_combo.currentText()
                if current_col and current_col != pp_col:
                    detail += " [!] re-preprocess needed"
            else:
                detail = f"{n_docs:,} docs processed"
            checks.append(("Text Processing", True, detail))
        else:
            p = self.project
            if (p._corpus is not None and p.preprocessed_language
                    and p.language != p.preprocessed_language):
                detail = (f"Language changed ('{p.preprocessed_language}' "
                          f"-> '{p.language}') — re-preprocess needed")
            else:
                detail = "Not preprocessed"
            checks.append(("Text Processing", False, detail))

        # 4. Embeddings
        if self.project._kv is not None:
            checks.append(("Embeddings", True, f"{self.project.vocab_size:,} words"))
        elif self.project.selected_embedding:
            checks.append(("Embeddings", False, "Not in memory — click Select"))
        else:
            checks.append(("Embeddings", False, "Not loaded"))

        all_ready = all(ok for _, ok, _ in checks)

        if all_ready:
            self.ready_frame.setObjectName("frame_ready_ok")
            self.ready_title.setText("Setup Complete")
            if self.next_btn:
                self.next_btn.setEnabled(True)
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

    def reset(self):
        """Clear all state for project close."""
        self.project = None
        self._df = None
        self._update_ready_indicator()

    def _validate_project_on_load(self):
        """Check for inconsistencies after loading a project and fix them."""
        project = self.project
        if not project:
            return
        warnings = []

        # 1. CSV file missing
        csv_path = project.csv_path
        if csv_path and not Path(csv_path).exists():
            warnings.append(f"CSV file not found: {csv_path}")

        # 2. Text column missing from loaded DataFrame
        text_col = project.text_column
        if text_col and self._df is not None and text_col not in self._df.columns:
            warnings.append(f"Text column '{text_col}' not found in CSV columns")

        # 2b. Text column exists but is all-empty
        if (text_col and self._df is not None
                and text_col in self._df.columns
                and project.text_ready
                and self._df[text_col].dropna().astype(str).str.strip().eq('').all()):
            warnings.append(f"Text column '{text_col}' contains no non-empty values")

        # 3. Preprocessed docs missing from disk
        if project.preprocessing_ready:
            if not ProjectIO.corpus_exists(project):
                project.preprocessed_text_column = None
                warnings.append("Preprocessed documents not found on disk")

        # 4. Embedding file missing
        if project.selected_embedding:
            emb_path = embeddings_dir() / project.selected_embedding
            if not emb_path.exists():
                project.selected_embedding = None
                warnings.append(f"Embedding file not found: {project.selected_embedding}")

        # 5. Column mismatch — preprocessed with different text column
        pp_col = project.preprocessed_text_column
        ds_col = project.text_column
        if (pp_col and ds_col and pp_col != ds_col
                and project.preprocessing_ready):
            project.preprocessed_text_column = None
            project._pre_docs = None
            project._docs = None
            project._corpus = None
            project._id_row_indices = None
            warnings.append(
                f"Text column changed ('{pp_col}' -> '{ds_col}'): "
                f"preprocessed texts discarded"
            )

        if warnings:
            QMessageBox.warning(
                self, "Project Validation", "\n".join(warnings)
            )

    def load_project(self, project: Project) -> bool:
        """Load a project into the UI.

        Returns ``False`` if the user cancelled a required loading step
        (e.g. embeddings), signalling that the caller should abort.
        """
        self.project = project

        # Show a progress dialog immediately so the user sees feedback
        # while synchronous loading (CSV, preprocessed docs) runs.
        needs_embeddings = bool(
            project.selected_embedding
            and (embeddings_dir() / project.selected_embedding).exists()
        )
        self._progress_dialog = ProgressDialog("Loading Project", self)
        self._progress_dialog.update_progress(0, "Loading dataset...")
        self._progress_dialog.show()
        QApplication.processEvents()

        if project.csv_path:
            self.file_path_edit.setText(str(project.csv_path))

            # Restore encoding selection
            saved_enc = project.csv_encoding
            for i in range(self.encoding_combo.count()):
                if self.encoding_combo.itemData(i) == saved_enc:
                    self.encoding_combo.setCurrentIndex(i)
                    break

            try:
                csv_path = project.csv_path
                if csv_path.exists():
                    ext = csv_path.suffix.lower()
                    encoding = self.encoding_combo.currentData()
                    if ext in (".xlsx", ".xls"):
                        self._df = pd.read_excel(csv_path)
                    elif ext == ".tsv":
                        self._df = pd.read_csv(csv_path, sep="\t", encoding=encoding)
                    else:
                        self._df = pd.read_csv(csv_path, encoding=encoding)
                    project._df = self._df

                    columns = self._df.columns.tolist()
                    self.text_col_combo.blockSignals(True)
                    self.text_col_combo.clear()
                    self.text_col_combo.addItems(columns)
                    if project.text_column:
                        idx = self.text_col_combo.findText(project.text_column)
                        if idx >= 0:
                            self.text_col_combo.setCurrentIndex(idx)
                    self.text_col_combo.blockSignals(False)

                    self.id_col_combo.clear()
                    self.id_col_combo.addItem("(none)")
                    self.id_col_combo.addItems(columns)
                    if project.id_column:
                        idx = self.id_col_combo.findText(project.id_column)
                        if idx >= 0:
                            self.id_col_combo.setCurrentIndex(idx)

                    self.validate_dataset_btn.setEnabled(True)
                    self.preprocess_btn.setEnabled(True)

            except Exception as e:
                print(f"Failed to reload data file: {e}")

        self._progress_dialog.update_progress(20, "Loading configuration...")
        QApplication.processEvents()

        # Load spaCy config
        lang_idx = self.language_combo.findData(project.language)
        if lang_idx >= 0:
            self.language_combo.setCurrentIndex(lang_idx)

        # Restore input mode
        if project.input_mode == "custom":
            self.radio_custom.setChecked(True)
            self.custom_model_input.setText(project.spacy_model)
        else:
            self.radio_language.setChecked(True)

        # Restore stopword mode
        stop_map = {"default": 0, "none": 1, "custom": 2}
        self.stopword_combo.setCurrentIndex(stop_map.get(project.stopword_mode, 0))
        if project.stopword_mode == "custom":
            self._custom_stopwords = list(project.custom_stopwords)
            self.stopword_file_label.setText(f"{len(self._custom_stopwords)} words loaded")

        if (ProjectIO.corpus_exists(project)
                and project.preprocessed_text_column == project.text_column):
            self.spacy_status.setText(
                f"Preprocessed {project.n_docs_processed:,} documents"
            )
            self.spacy_status.setObjectName("label_status_ok")
            self.spacy_status.style().unpolish(self.spacy_status)
            self.spacy_status.style().polish(self.spacy_status)

            self._progress_dialog.update_progress(35, "Loading preprocessed texts...")
            QApplication.processEvents()

            corpus = ProjectIO.load_corpus(project)
            if corpus is not None:
                project._corpus = corpus
                project._pre_docs = corpus.pre_docs
                project._docs = corpus.docs
                project._id_row_indices = None
                if not project.preprocessed_language:
                    project.preprocessed_language = project.language

        self._progress_dialog.update_progress(70, "Loading configuration...")
        QApplication.processEvents()

        # Load embedding config — refresh table and restore status
        self._refresh_embeddings_table()
        self.l2_check.setChecked(project.l2_normalized)
        self.abtt_spin.setValue(project.abtt)

        if project.selected_embedding:
            self.emb_status.setText(
                f"Selected: {project.selected_embedding} "
                f"({project.vocab_size:,} words, {project.embedding_dim}d)"
            )

        self._validate_project_on_load()
        self._update_ready_indicator()

        # Auto-load the previously selected embedding if available
        # (can be disabled via Settings → "Autoload embedding on project open")
        if needs_embeddings and project._kv is None and embeddings_autoload_enabled():
            self._progress_dialog.update_progress(50, "Loading embeddings...")
            QApplication.processEvents()
            self._autoloading_embedding = True
            self._select_embedding()
        else:
            self._autoloading_embedding = False

        self._progress_dialog.update_progress(100, "Project loaded.")
        QApplication.processEvents()

        if not self._autoloading_embedding:
            self._progress_dialog.accept()

        return True

    def save_to_project(self, project: Project):
        """Save current UI state to project."""
        if not project:
            return

        project.text_column = self.text_col_combo.currentText()

        id_col = self.id_col_combo.currentText()
        project.id_column = id_col if id_col != "(none)" else None

        project.language = self.language_combo.currentData()
        project.input_mode = "language" if self.radio_language.isChecked() else "custom"
        project.spacy_model = self.custom_model_input.text().strip() if not self.radio_language.isChecked() else ""
        stop_idx = self.stopword_combo.currentIndex()
        project.stopword_mode = ["default", "none", "custom"][stop_idx]
        project.custom_stopwords = list(self._custom_stopwords) if stop_idx == 2 else []

        # Embedding config is managed by the prepare/select workflow;
        # only persist the ABTT/L2 defaults from the "Prepare" panel.
        project.l2_normalized = self.l2_check.isChecked()
        project.abtt = self.abtt_spin.value()
