"""Report Settings dialog — controls counts for the saved results.txt report."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QPushButton,
)
from ..utils.settings import app_settings
from ..utils.report_settings import (
    KEY_TOP_WORDS,
    KEY_CLUSTERS,
    KEY_EXTREME_DOCS,
    KEY_MISDIAGNOSED,
    get_report_setting,
)


class ReportSettingsDialog(QDialog):
    """Dialog for configuring report output counts (4 spin boxes)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Report Settings")
        self.setMinimumWidth(380)
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 18, 20, 18)

        info = QLabel(
            "These counts control the results.txt report saved with each result.\n"
            "Set to 0 to skip a section."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        group = QGroupBox("Report Sections")
        form = QFormLayout()
        form.setContentsMargins(12, 10, 12, 10)
        form.setSpacing(10)

        self._spin_top_words = QSpinBox()
        self._spin_top_words.setRange(0, 200)
        self._spin_top_words.setSuffix(" per pole")
        form.addRow("Top words:", self._spin_top_words)

        self._spin_clusters = QSpinBox()
        self._spin_clusters.setRange(0, 500)
        self._spin_clusters.setSuffix(" neighbors")
        form.addRow("Clusters (topn):", self._spin_clusters)

        self._spin_extreme_docs = QSpinBox()
        self._spin_extreme_docs.setRange(0, 100)
        self._spin_extreme_docs.setSuffix(" per side")
        form.addRow("Extreme docs:", self._spin_extreme_docs)

        self._spin_misdiagnosed = QSpinBox()
        self._spin_misdiagnosed.setRange(0, 100)
        self._spin_misdiagnosed.setSuffix(" per side")
        form.addRow("Misdiagnosed:", self._spin_misdiagnosed)

        group.setLayout(form)
        layout.addWidget(group)

        layout.addStretch()

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("btn_secondary")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply)
        btn_row.addWidget(apply_btn)

        layout.addLayout(btn_row)

    def _load_settings(self):
        self._spin_top_words.setValue(get_report_setting(KEY_TOP_WORDS))
        self._spin_clusters.setValue(get_report_setting(KEY_CLUSTERS))
        self._spin_extreme_docs.setValue(get_report_setting(KEY_EXTREME_DOCS))
        self._spin_misdiagnosed.setValue(get_report_setting(KEY_MISDIAGNOSED))

    def _apply(self):
        s = app_settings()
        s.setValue(KEY_TOP_WORDS, self._spin_top_words.value())
        s.setValue(KEY_CLUSTERS, self._spin_clusters.value())
        s.setValue(KEY_EXTREME_DOCS, self._spin_extreme_docs.value())
        s.setValue(KEY_MISDIAGNOSED, self._spin_misdiagnosed.value())
        self.accept()
