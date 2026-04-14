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
)
from ..utils.settings import app_settings


# QSettings keys
_KEY_PROJECTS_DIR = "projects_directory"


class SettingsDialog(QDialog):
    """Dialog for general application settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(520)

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
        self._dir_edit.setText(
            self._settings.value(_KEY_PROJECTS_DIR, "")
        )
        row.addWidget(self._dir_edit, stretch=1)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_directory)
        row.addWidget(browse_btn)

        dir_layout.addLayout(row)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

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

    def _browse_directory(self):
        current = self._dir_edit.text()
        directory = QFileDialog.getExistingDirectory(
            self, "Select Default Projects Directory", current
        )
        if directory:
            self._dir_edit.setText(directory)

    def _save(self):
        self._settings.setValue(_KEY_PROJECTS_DIR, self._dir_edit.text())
        self.accept()
