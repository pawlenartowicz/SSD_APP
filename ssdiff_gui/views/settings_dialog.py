"""Application settings dialog for SSD."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QGroupBox,
    QFileDialog,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
)
from ..utils.settings import app_settings
from ..utils.paths import (
    SHARED_EMB_SUBDIR,
    embeddings_dir,
    projects_dir,
    _qsetting_bool,
)


# QSettings keys
_KEY_PROJECTS_DIR = "projects_directory"
_KEY_AUTOLOAD = "embeddings/autoload_on_open"
_KEY_EMB_MODE = "embeddings/location_mode"
_KEY_EMB_CUSTOM = "embeddings/custom_path"


class SettingsDialog(QDialog):
    """Dialog for general application settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(620)
        self.setMinimumHeight(640)

        self._settings = app_settings()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 20, 24, 20)

        # --- Projects directory ---
        dir_group = QGroupBox("Projects Directory")
        dir_layout = QVBoxLayout()

        dir_desc = QLabel(
            "Default location used when creating or opening projects."
        )
        dir_desc.setObjectName("label_muted")
        dir_desc.setWordWrap(True)
        dir_layout.addWidget(dir_desc)

        row = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setReadOnly(True)
        self._dir_edit.setPlaceholderText("Not set")
        self._dir_edit.setText(self._settings.value(_KEY_PROJECTS_DIR, ""))
        row.addWidget(self._dir_edit, stretch=1)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_projects_directory)
        row.addWidget(browse_btn)

        dir_layout.addLayout(row)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        # --- Autoload embedding ---
        autoload_group = QGroupBox("Autoload Embedding")
        autoload_layout = QVBoxLayout()

        self._autoload_check = QCheckBox(
            "Autoload selected embedding when opening a project"
        )
        self._autoload_check.setChecked(
            _qsetting_bool(self._settings.value(_KEY_AUTOLOAD, True), default=True)
        )
        autoload_desc = QLabel(
            "Turn off to inspect saved results without paying the RAM "
            "cost of loading embeddings."
        )
        autoload_desc.setObjectName("label_muted")
        autoload_desc.setWordWrap(True)

        autoload_layout.addWidget(self._autoload_check)
        autoload_layout.addWidget(autoload_desc)
        autoload_group.setLayout(autoload_layout)
        layout.addWidget(autoload_group)

        # --- Embeddings location ---
        loc_group = QGroupBox("Embeddings Location")
        loc_layout = QVBoxLayout()

        loc_desc = QLabel(
            "Shared across projects — no per-project embeddings folder."
        )
        loc_desc.setObjectName("label_muted")
        loc_desc.setWordWrap(True)
        loc_layout.addWidget(loc_desc)

        self._loc_group = QButtonGroup(self)
        self._loc_shared_radio = QRadioButton(
            f"Default  —  \u201C{SHARED_EMB_SUBDIR}\u201D folder inside the "
            f"projects directory"
        )
        self._loc_custom_radio = QRadioButton("Custom path")
        self._loc_group.addButton(self._loc_shared_radio, 0)
        self._loc_group.addButton(self._loc_custom_radio, 1)

        mode = self._settings.value(_KEY_EMB_MODE, "shared")
        if mode == "custom":
            self._loc_custom_radio.setChecked(True)
        else:
            self._loc_shared_radio.setChecked(True)

        loc_layout.addWidget(self._loc_shared_radio)

        self._shared_hint = QLabel("")
        self._shared_hint.setObjectName("label_muted")
        self._shared_hint.setWordWrap(True)
        self._shared_hint.setContentsMargins(24, 0, 0, 0)
        loc_layout.addWidget(self._shared_hint)

        loc_layout.addWidget(self._loc_custom_radio)

        custom_row = QHBoxLayout()
        custom_row.setContentsMargins(24, 0, 0, 0)
        self._custom_edit = QLineEdit()
        self._custom_edit.setReadOnly(True)
        self._custom_edit.setPlaceholderText("Select a folder...")
        self._custom_edit.setText(self._settings.value(_KEY_EMB_CUSTOM, ""))
        custom_row.addWidget(self._custom_edit, stretch=1)

        self._custom_browse_btn = QPushButton("Browse...")
        self._custom_browse_btn.clicked.connect(self._browse_custom_embeddings)
        custom_row.addWidget(self._custom_browse_btn)
        loc_layout.addLayout(custom_row)

        loc_group.setLayout(loc_layout)
        layout.addWidget(loc_group)

        # React to radio / projects-dir changes to refresh hints and enable state
        self._loc_shared_radio.toggled.connect(self._update_embeddings_ui)
        self._dir_edit.textChanged.connect(self._update_embeddings_ui)
        self._update_embeddings_ui()

        layout.addStretch()

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("btn_secondary")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

    def _browse_projects_directory(self):
        current = self._dir_edit.text()
        directory = QFileDialog.getExistingDirectory(
            self, "Select Default Projects Directory", current
        )
        if directory:
            self._dir_edit.setText(directory)

    def _browse_custom_embeddings(self):
        current = self._custom_edit.text() or str(projects_dir())
        directory = QFileDialog.getExistingDirectory(
            self, "Select Embeddings Folder", current
        )
        if directory:
            self._custom_edit.setText(directory)
            self._loc_custom_radio.setChecked(True)

    def _update_embeddings_ui(self):
        """Refresh the resolved shared path hint and enable/disable custom row."""
        shared_base = self._dir_edit.text().strip()
        if shared_base:
            from pathlib import Path
            resolved = Path(shared_base) / SHARED_EMB_SUBDIR
        else:
            resolved = projects_dir() / SHARED_EMB_SUBDIR
        self._shared_hint.setText(str(resolved))

        is_custom = self._loc_custom_radio.isChecked()
        self._custom_edit.setEnabled(is_custom)
        self._custom_browse_btn.setEnabled(is_custom)

    def _save(self):
        self._settings.setValue(_KEY_PROJECTS_DIR, self._dir_edit.text())
        self._settings.setValue(_KEY_AUTOLOAD, self._autoload_check.isChecked())
        self._settings.setValue(
            _KEY_EMB_MODE,
            "custom" if self._loc_custom_radio.isChecked() else "shared",
        )
        self._settings.setValue(_KEY_EMB_CUSTOM, self._custom_edit.text())
        self.accept()
