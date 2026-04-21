"""Appearance settings dialog for SSD."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QFrame,
    QGroupBox,
    QSizePolicy,
)
from PySide6.QtCore import Qt

from ..theme import (
    THEME_PRESETS,
    FONT_SIZE_OPTIONS,
    ThemePalette,
    generate_stylesheet,
    build_qpalette,
    scale_font_sizes,
    get_saved_theme_name,
    get_saved_font_size,
    save_appearance,
)


class _ThemeCard(QFrame):
    """A clickable card representing a single theme preset."""

    def __init__(
        self, name: str, palette: ThemePalette, parent: AppearanceDialog
    ):
        super().__init__(parent)
        self.theme_name = name
        self.palette = palette
        self._dialog = parent
        self._selected = False

        self.setFixedHeight(110)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Color swatches row
        swatch_row = QHBoxLayout()
        swatch_row.setSpacing(3)
        for color in (palette.bg_base, palette.accent, palette.bg_card):
            swatch = QLabel()
            swatch.setFixedSize(28, 28)
            swatch.setStyleSheet(
                f"background-color: {color};"
                f"border-radius: 4px;"
                f"border: 1px solid {palette.border};"
            )
            swatch_row.addWidget(swatch)
        swatch_row.addStretch()
        layout.addLayout(swatch_row)

        layout.addStretch()

        # Name label
        self._name_label = QLabel(name)
        self._name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._name_label)

        self._update_style()

    def set_selected(self, selected: bool):
        self._selected = selected
        self._update_style()

    def _update_style(self):
        p = self.palette
        if self._selected:
            self.setStyleSheet(
                f"_ThemeCard {{"
                f"  background-color: {p.bg_card};"
                f"  border: 2px solid {p.accent};"
                f"  border-radius: 8px;"
                f"}}"
            )
            self._name_label.setStyleSheet(
                f"color: {p.accent}; font-weight: bold; background: transparent;"
            )
        else:
            self.setStyleSheet(
                f"_ThemeCard {{"
                f"  background-color: {p.bg_surface};"
                f"  border: 1px solid {p.border};"
                f"  border-radius: 8px;"
                f"}}"
            )
            self._name_label.setStyleSheet(
                f"color: {p.text_secondary}; background: transparent;"
            )

    def mousePressEvent(self, event):
        self._dialog._select_theme(self.theme_name)
        super().mousePressEvent(event)


class AppearanceDialog(QDialog):
    """Dialog for choosing theme preset and font size."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Appearance")
        self.setMinimumWidth(700)
        self.setMinimumHeight(580)

        self._selected_theme = get_saved_theme_name()
        self._selected_font_size = get_saved_font_size()

        self._setup_ui()
        self._update_preview()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 20, 24, 20)

        # --- Theme presets ---
        theme_group = QGroupBox("Theme")
        theme_layout = QVBoxLayout()
        theme_layout.setSpacing(8)

        self._cards: dict[str, _ThemeCard] = {}
        presets = list(THEME_PRESETS.items())
        mid = (len(presets) + 1) // 2  # ceiling split → row 1 is never shorter

        for row_items in (presets[:mid], presets[mid:]):
            row = QHBoxLayout()
            row.setSpacing(10)
            for name, palette in row_items:
                card = _ThemeCard(name, palette, self)
                card.set_selected(name == self._selected_theme)
                self._cards[name] = card
                row.addWidget(card)
            theme_layout.addLayout(row)

        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)

        # --- Font size ---
        font_group = QGroupBox("Text Size")
        font_layout = QHBoxLayout()

        font_layout.addWidget(QLabel("Base font size:"))
        self._font_combo = QComboBox()
        for label, px in FONT_SIZE_OPTIONS.items():
            self._font_combo.addItem(label, px)
            if px == self._selected_font_size:
                self._font_combo.setCurrentIndex(self._font_combo.count() - 1)
        self._font_combo.currentIndexChanged.connect(self._on_font_changed)
        font_layout.addWidget(self._font_combo)
        font_layout.addStretch()

        font_group.setLayout(font_layout)
        layout.addWidget(font_group)

        # --- Live preview ---
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()

        self._preview_frame = QFrame()
        self._preview_frame.setFixedHeight(120)
        pf_layout = QVBoxLayout(self._preview_frame)
        pf_layout.setContentsMargins(16, 12, 16, 12)
        pf_layout.setSpacing(6)

        self._preview_title = QLabel("SSD")
        self._preview_subtitle = QLabel("This is how your interface will look.")
        self._preview_button = QPushButton("Sample Button")
        self._preview_button.setFixedWidth(140)
        self._preview_muted = QLabel("Secondary text and muted labels appear like this.")

        pf_layout.addWidget(self._preview_title)
        pf_layout.addWidget(self._preview_subtitle)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._preview_button)
        btn_row.addWidget(self._preview_muted)
        btn_row.addStretch()
        pf_layout.addLayout(btn_row)

        preview_layout.addWidget(self._preview_frame)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        layout.addStretch()

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("btn_secondary")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply)
        btn_layout.addWidget(apply_btn)

        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------ #
    #  Interaction
    # ------------------------------------------------------------------ #

    def _select_theme(self, name: str):
        self._selected_theme = name
        for card_name, card in self._cards.items():
            card.set_selected(card_name == name)
        self._update_preview()

    def _on_font_changed(self, index: int):
        self._selected_font_size = self._font_combo.itemData(index)
        self._update_preview()

    def _update_preview(self):
        palette = THEME_PRESETS.get(self._selected_theme, THEME_PRESETS["Midnight"])
        if self._selected_font_size != 13:
            palette = scale_font_sizes(palette, self._selected_font_size)

        self._preview_frame.setStyleSheet(
            f"QFrame {{"
            f"  background-color: {palette.bg_base};"
            f"  border: 1px solid {palette.border};"
            f"  border-radius: 8px;"
            f"}}"
        )
        self._preview_title.setStyleSheet(
            f"color: {palette.text_primary};"
            f"font-size: {palette.font_size_xl};"
            f"font-weight: bold;"
            f"background: transparent;"
        )
        self._preview_subtitle.setStyleSheet(
            f"color: {palette.text_secondary};"
            f"font-size: {palette.font_size_base};"
            f"background: transparent;"
        )
        self._preview_button.setStyleSheet(
            f"QPushButton {{"
            f"  background-color: {palette.accent};"
            f"  color: {palette.text_on_accent};"
            f"  border: none;"
            f"  border-radius: 6px;"
            f"  padding: 6px 16px;"
            f"  font-weight: 600;"
            f"  font-size: {palette.font_size_base};"
            f"}}"
        )
        self._preview_muted.setStyleSheet(
            f"color: {palette.text_muted};"
            f"font-size: {palette.font_size_sm};"
            f"background: transparent;"
        )

    def _apply(self):
        save_appearance(self._selected_theme, self._selected_font_size)

        # Build palette and re-apply to running app
        palette = THEME_PRESETS.get(self._selected_theme, THEME_PRESETS["Midnight"])
        if self._selected_font_size != 13:
            palette = scale_font_sizes(palette, self._selected_font_size)

        app = QApplication.instance()
        if app:
            app.setPalette(build_qpalette(palette))
            app.setStyleSheet(generate_stylesheet(palette))

            # Update the app icon to match the new theme
            from ..logo import create_app_icon
            icon = create_app_icon(palette, self._selected_theme)
            if icon:
                app.setWindowIcon(icon)

            # Re-register Linux desktop icon with new theme colours
            from ..utils.linux_install import register as _linux_register
            _linux_register(palette, self._selected_theme, force=True)

        # Refresh the welcome-page logo if the main window is the parent
        main_win = self.parent()
        if main_win and hasattr(main_win, "_refresh_welcome_logo"):
            main_win._refresh_welcome_logo()

        self.accept()
