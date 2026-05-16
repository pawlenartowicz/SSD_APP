"""Save Settings dialog — picks extensions + which result artifacts to save."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLayout, QLayoutItem, QPushButton, QScrollArea, QSpinBox,
    QVBoxLayout, QWidget,
)

from ssdiff import GroupResult, PCAOLSResult, PLSResult
from ssdiff.results.display import NARRATIVE_EXTS, TABULAR_EXTS
from ssdiff.results.lexicon_result import LexiconResult
from ssdiff.results.multi_pls_result import MultiPLSResult

from ..utils.artifact_registry import get_columns
from ..utils.save_config import (
    DEFAULT_ITEM_KEYS,
    DEFAULT_REPORT_SECTION_KEYS,
    ItemConfig,
    ReportSectionConfig,
    SaveConfig,
)
from ..utils.settings import app_settings


_ROW_SPEC: tuple[tuple[str, str, tuple[type, ...] | None], ...] = (
    ("Sweep plot (PNG)",                 "sweep_plot",       (PCAOLSResult,)),
    ("Sweep table",                      "sweep",            (PCAOLSResult,)),
    ("Words",                            "words",            None),
    ("Clusters (pos + neg)",             "clusters",         None),
    ("Cluster member words (pos + neg)", "cluster_words",    None),
    ("Snippets",                         "snippets",         None),
    ("Extreme docs (pos + neg)",         "docs_extreme",     (PLSResult, PCAOLSResult)),
    ("Misdiagnosed docs",                "docs_misdiagnosed", (PLSResult, PCAOLSResult)),
    ("Pairs list",                       "pairs",            (GroupResult, MultiPLSResult)),
)

# Report sections — (label, section_key, applicable result types).
# Spinbox defaults track the SSDLite library defaults; bumping the library
# defaults later still works (the dialog just lags one bump until the user
# re-saves).
_REPORT_SECTION_SPEC: tuple[
    tuple[str, str, tuple[type, ...]], ...
] = (
    ("Top words section",   "top_words",
        (PLSResult, PCAOLSResult, GroupResult, MultiPLSResult)),
    ("Clusters section",    "clusters",
        (PLSResult, PCAOLSResult, GroupResult)),
    ("Extreme docs section", "extreme_docs",
        (PLSResult, PCAOLSResult)),
    ("Misdiagnosed section", "misdiagnosed",
        (PLSResult, PCAOLSResult)),
    ("Suggestions (top)",   "top",
        (LexiconResult,)),
)

# Initial spinbox values when ReportSectionConfig has None for a knob.
# Mirrors SSDLite defaults so the on-disk QSettings stays predictable.
_REPORT_SECTION_DEFAULTS: dict[str, dict[str, int]] = {
    "top_words":    {"n": 5},
    "clusters":     {"n": 10, "n_words": 5, "n_snippets": 1},
    "extreme_docs": {"n": 5},
    "misdiagnosed": {"n": 5},
    "top":          {"n": 20},
}

# Keys for rows that are plain checkboxes only — no column/row controls.
_FLAT_KEYS = frozenset({"sweep_plot"})

_TABLES_EXTS = tuple(e for e in TABULAR_EXTS if e != "txt")

# Indent applied to the inline cols/rows panel so it visually attaches to
# its parent checkbox.
_DETAIL_INDENT = 24

# Spinbox default for the per-artifact row cap. Library treats k as a row
# cap with no separate "all rows" sentinel in this dialog.
_DEFAULT_K = 100


class _FlowLayout(QLayout):
    """Horizontal layout that wraps items to the next line when space runs out."""

    def __init__(self, parent: QWidget | None = None, spacing: int = 8) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._h_spacing = spacing
        self._v_spacing = 4
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientations:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        m = self.contentsMargins()
        x = rect.x() + m.left()
        y = rect.y() + m.top()
        right = rect.right() - m.right()
        line_height = 0

        for item in self._items:
            wid = item.sizeHint()
            next_x = x + wid.width() + self._h_spacing
            if next_x - self._h_spacing > right and line_height > 0:
                x = rect.x() + m.left()
                y = y + line_height + self._v_spacing
                next_x = x + wid.width() + self._h_spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), wid))
            x = next_x
            line_height = max(line_height, wid.height())

        return y + line_height + m.bottom() - rect.y()


class _ClickableLabel(QLabel):
    """QLabel that emits ``clicked`` on left-button press."""

    clicked = Signal()

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class _ArtifactRow(QWidget):
    """One artifact row.

    For tabular items: checkbox indicator + clickable label. The label toggles
    a hidden panel below with the column picker and rows-cap spinbox; enable
    state and expand state are independent. Plain rows (in ``_FLAT_KEYS``) are
    a single labelled checkbox with no panel.
    """

    def __init__(self, label: str, item_key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._item_key = item_key
        self._col_info = get_columns(item_key)
        self._has_columns = bool(self._col_info.all_columns)
        self._is_tabular = item_key not in _FLAT_KEYS
        self._has_content = self._is_tabular

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QHBoxLayout()
        header.setContentsMargins(0, 2, 0, 2)
        header.setSpacing(6)

        if self._has_content:
            self.enabled_checkbox = QCheckBox()
            header.addWidget(self.enabled_checkbox)
            self._title = _ClickableLabel(label)
            self._title.clicked.connect(self._toggle_content)
            header.addWidget(self._title)
        else:
            self.enabled_checkbox = QCheckBox(label)
            header.addWidget(self.enabled_checkbox)

        header.addStretch()
        root.addLayout(header)

        if self._has_content:
            self._content = QWidget()
            self._content.setVisible(False)
            content_layout = QVBoxLayout(self._content)
            content_layout.setContentsMargins(_DETAIL_INDENT, 4, 0, 8)
            content_layout.setSpacing(6)

            self._col_checkboxes: dict[str, QCheckBox] = {}
            if self._has_columns:
                content_layout.addWidget(QLabel("Columns:"))
                cols_host = QWidget()
                cols_flow = _FlowLayout(cols_host, spacing=12)
                for col in self._col_info.all_columns:
                    cb = QCheckBox(col)
                    cols_flow.addWidget(cb)
                    self._col_checkboxes[col] = cb
                content_layout.addWidget(cols_host)

            rows_row = QHBoxLayout()
            rows_row.setSpacing(8)
            rows_row.addWidget(QLabel("Rows:"))
            self._k_spinbox = QSpinBox()
            self._k_spinbox.setRange(1, 10000)
            self._k_spinbox.setSingleStep(5)
            self._k_spinbox.setValue(_DEFAULT_K)
            rows_row.addWidget(self._k_spinbox)
            rows_row.addStretch()
            content_layout.addLayout(rows_row)

            root.addWidget(self._content)

    # ── public API ────────────────────────────────────────────────────────────

    def set_value(self, item: ItemConfig) -> None:
        self.enabled_checkbox.setChecked(item.enabled)

        if self._has_columns:
            if item.cols is None:
                active_cols = set(self._col_info.default_columns)
            else:
                active_cols = set(item.cols) & set(self._col_info.all_columns)
            for col, cb in self._col_checkboxes.items():
                cb.setChecked(col in active_cols)

        if self._is_tabular:
            self._k_spinbox.setValue(item.k if item.k is not None else _DEFAULT_K)

    def value(self) -> ItemConfig:
        enabled = self.enabled_checkbox.isChecked()

        cols: tuple[str, ...] | None = None
        if self._has_columns:
            selected = tuple(
                col for col in self._col_info.all_columns
                if self._col_checkboxes[col].isChecked()
            )
            # Persist None when user's selection matches library default so the
            # library's default remains authoritative across version bumps.
            if selected == tuple(self._col_info.default_columns):
                cols = None
            else:
                cols = selected if selected else None

        k: int | None = self._k_spinbox.value() if self._is_tabular else None

        return ItemConfig(enabled=enabled, cols=cols, k=k)

    def set_visible_with_content(self, visible: bool) -> None:
        self.setVisible(visible)

    # ── private helpers ───────────────────────────────────────────────────────

    def _toggle_content(self) -> None:
        self._content.setVisible(not self._content.isVisible())


class _ReportSectionRow(QWidget):
    """One report-section row: enable checkbox + label + per-knob spinboxes.

    Only the ``clusters`` section exposes ``n_words`` and ``n_snippets``
    spinboxes; every other section has just ``n``. ``n_snippets`` allows
    0 (drops the excerpt column).
    """

    def __init__(self, label: str, section_key: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._section_key = section_key
        self._defaults = _REPORT_SECTION_DEFAULTS[section_key]

        row = QHBoxLayout(self)
        row.setContentsMargins(_DETAIL_INDENT, 2, 0, 2)
        row.setSpacing(8)

        self.enabled_checkbox = QCheckBox(label)
        row.addWidget(self.enabled_checkbox)
        row.addSpacing(8)

        self._n_spinbox = self._make_spinbox(self._defaults["n"], min_value=1)
        row.addWidget(QLabel("n:"))
        row.addWidget(self._n_spinbox)

        if section_key == "clusters":
            self._n_words_spinbox = self._make_spinbox(
                self._defaults["n_words"], min_value=1,
            )
            row.addSpacing(6)
            row.addWidget(QLabel("words:"))
            row.addWidget(self._n_words_spinbox)

            self._n_snippets_spinbox = self._make_spinbox(
                self._defaults["n_snippets"], min_value=0,
            )
            row.addSpacing(6)
            row.addWidget(QLabel("snippets:"))
            row.addWidget(self._n_snippets_spinbox)
        else:
            self._n_words_spinbox = None
            self._n_snippets_spinbox = None

        row.addStretch()

    @staticmethod
    def _make_spinbox(initial: int, *, min_value: int) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(min_value, 10000)
        sb.setSingleStep(1)
        sb.setValue(initial)
        return sb

    def set_value(self, section: ReportSectionConfig) -> None:
        self.enabled_checkbox.setChecked(section.enabled)
        self._n_spinbox.setValue(
            section.n if section.n is not None else self._defaults["n"]
        )
        if self._n_words_spinbox is not None:
            self._n_words_spinbox.setValue(
                section.n_words
                if section.n_words is not None
                else self._defaults["n_words"]
            )
        if self._n_snippets_spinbox is not None:
            self._n_snippets_spinbox.setValue(
                section.n_snippets
                if section.n_snippets is not None
                else self._defaults["n_snippets"]
            )

    def value(self) -> ReportSectionConfig:
        return ReportSectionConfig(
            enabled=self.enabled_checkbox.isChecked(),
            n=self._n_spinbox.value(),
            n_words=(self._n_words_spinbox.value()
                     if self._n_words_spinbox is not None else None),
            n_snippets=(self._n_snippets_spinbox.value()
                        if self._n_snippets_spinbox is not None else None),
        )


class SaveSettingsDialog(QDialog):
    def __init__(self, result_type: type, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Settings")
        self.setMinimumSize(720, 560)
        self.resize(760, 820)
        self._result_type = result_type
        self._initial_cfg = SaveConfig.from_settings(app_settings())
        self._build_ui()
        self._apply_cfg(self._initial_cfg)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(14)

        root.addWidget(QLabel(
            "Pick formats and which artifacts to save. Files land in the "
            "result folder; tabular outputs go under <code>tables/</code>."
        ))

        formats = QGroupBox("Formats")
        form = QFormLayout(formats)
        self._report_format = QComboBox()
        self._report_format.addItems(list(NARRATIVE_EXTS))
        form.addRow("Report format:", self._report_format)
        self._tables_format = QComboBox()
        self._tables_format.addItems(list(_TABLES_EXTS))
        form.addRow("Tables format:", self._tables_format)
        root.addWidget(formats)

        items_box = QGroupBox("Artifacts")
        items_layout = QVBoxLayout(items_box)
        items_layout.setContentsMargins(8, 8, 8, 8)
        items_layout.setSpacing(2)

        self._report_checkbox = QCheckBox("Report")
        items_layout.addWidget(self._report_checkbox)

        self._report_section_rows: dict[str, _ReportSectionRow] = {}
        for label, key, show_for in _REPORT_SECTION_SPEC:
            row = _ReportSectionRow(label, key, parent=items_box)
            row.setVisible(issubclass(self._result_type, show_for))
            self._report_section_rows[key] = row
            items_layout.addWidget(row)

        # Section rows are only meaningful when the master Report checkbox is
        # ticked. Greying them out makes that wiring legible.
        def _sync_sections_enabled(checked: bool) -> None:
            for r in self._report_section_rows.values():
                r.setEnabled(checked)

        self._report_checkbox.toggled.connect(_sync_sections_enabled)
        _sync_sections_enabled(self._report_checkbox.isChecked())

        self._item_rows: dict[str, _ArtifactRow] = {}
        for label, key, show_for in _ROW_SPEC:
            row = _ArtifactRow(label, key, parent=items_box)
            visible = show_for is None or issubclass(self._result_type, show_for)
            row.set_visible_with_content(visible)
            self._item_rows[key] = row
            items_layout.addWidget(row)

        items_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(items_box)
        root.addWidget(scroll, stretch=1)

        button_row = QHBoxLayout()
        reset_btn = QPushButton("Reset to default")
        reset_btn.setObjectName("btn_secondary")
        reset_btn.clicked.connect(self._reset_defaults)
        button_row.addWidget(reset_btn)
        button_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_row.addWidget(cancel_btn)
        ok_btn = QPushButton("Save")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_accept)
        button_row.addWidget(ok_btn)

        for btn in (reset_btn, cancel_btn, ok_btn):
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        root.addLayout(button_row)

    def _apply_cfg(self, cfg: SaveConfig) -> None:
        self._report_checkbox.setChecked(bool(cfg.report_enabled))
        idx_report = self._report_format.findText(cfg.report_format)
        self._report_format.setCurrentIndex(idx_report if idx_report >= 0 else 0)
        idx_tables = self._tables_format.findText(cfg.tables_format)
        self._tables_format.setCurrentIndex(idx_tables if idx_tables >= 0 else 0)

        for key, row in self._item_rows.items():
            row.set_value(cfg.items.get(key, ItemConfig()))

        for key, sec_row in self._report_section_rows.items():
            sec_row.set_value(cfg.report_sections.get(key, ReportSectionConfig()))

    def _current_cfg(self) -> SaveConfig:
        items: dict[str, ItemConfig] = {}
        for key, row in self._item_rows.items():
            if row.isVisible():
                items[key] = row.value()
            else:
                items[key] = self._initial_cfg.items.get(key, ItemConfig())
        for key in DEFAULT_ITEM_KEYS:
            if key not in items:
                items[key] = self._initial_cfg.items.get(key, ItemConfig())

        sections: dict[str, ReportSectionConfig] = {}
        for key, sec_row in self._report_section_rows.items():
            if sec_row.isVisible():
                sections[key] = sec_row.value()
            else:
                sections[key] = self._initial_cfg.report_sections.get(
                    key, ReportSectionConfig()
                )
        for key in DEFAULT_REPORT_SECTION_KEYS:
            if key not in sections:
                sections[key] = self._initial_cfg.report_sections.get(
                    key, ReportSectionConfig()
                )

        return replace(
            self._initial_cfg,
            report_enabled=self._report_checkbox.isChecked(),
            report_format=self._report_format.currentText(),
            tables_format=self._tables_format.currentText(),
            items=MappingProxyType(items),
            report_sections=MappingProxyType(sections),
        )

    def _reset_defaults(self) -> None:
        self._apply_cfg(SaveConfig.default())

    def _on_accept(self) -> None:
        self._current_cfg().to_settings(app_settings())
        self.accept()
