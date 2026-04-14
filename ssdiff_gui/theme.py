"""Centralized theme system for SSD.

All colors, fonts, and spacing are defined here in a single palette.
The generate_stylesheet() function produces the full QSS from a palette,
making it straightforward to add theme switching / personalization later.

To create a custom theme:
    palette = ThemePalette(accent="#e06c75", ...)
    qss = generate_stylesheet(palette)
    app.setStyleSheet(qss)
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class ThemePalette:
    """Complete color and typography palette for the application."""

    # -- Background layers (darkest to lightest) --
    bg_base: str = "#14151f"
    bg_surface: str = "#1c1d2e"
    bg_card: str = "#222336"
    bg_input: str = "#272840"
    bg_hover: str = "#2f3048"
    bg_elevated: str = "#2a2b42"

    # -- Borders --
    border: str = "#353650"
    border_subtle: str = "#2c2d44"
    border_focus: str = "#6c9ce6"

    # -- Text --
    text_primary: str = "#e4e4f0"
    text_secondary: str = "#9c9cb4"
    text_muted: str = "#6e6e88"
    text_on_accent: str = "#ffffff"

    # -- Accent --
    accent: str = "#6c9ce6"
    accent_hover: str = "#5a8ad4"
    accent_pressed: str = "#4878be"
    accent_muted: str = "rgba(108,156,230,0.12)"

    # -- Semantic colors --
    success: str = "#5cb85c"
    success_hover: str = "#4cae4c"
    success_bg: str = "rgba(92,184,92,0.10)"
    success_border: str = "#3d8b3d"

    warning: str = "#f0ad4e"
    warning_hover: str = "#ec971f"
    warning_bg: str = "rgba(240,173,78,0.10)"
    warning_border: str = "#c49242"

    error: str = "#e25555"
    error_hover: str = "#d43f3f"
    error_bg: str = "rgba(226,85,85,0.10)"
    error_border: str = "#b83a3a"

    # -- Interactive / selection --
    selection: str = "#3d5a8a"

    # -- Disabled --
    disabled_bg: str = "#2a2b3c"
    disabled_text: str = "#555566"

    # -- Typography --
    font_family: str = '"Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif'
    font_size_base: str = "13px"
    font_size_sm: str = "11px"
    font_size_md: str = "14px"
    font_size_lg: str = "16px"
    font_size_xl: str = "20px"
    font_size_xxl: str = "24px"
    font_size_hero: str = "42px"

    # -- Spacing --
    radius_sm: str = "4px"
    radius_md: str = "6px"
    radius_lg: str = "8px"
    radius_xl: str = "12px"


# ================================================================
#  Built-in Theme Presets
# ================================================================

# The default dark-blue theme
DARK_PALETTE = ThemePalette()

# Violet  — inspired by a PyCharm "Violet" color scheme
VIOLET_PALETTE = ThemePalette(
    bg_base="#1a0520",
    bg_surface="#200124",
    bg_card="#2b0a30",
    bg_input="#32103a",
    bg_hover="#3c1a46",
    bg_elevated="#351440",
    border="#4a2858",
    border_subtle="#3a1c48",
    border_focus="#8f4ce8",
    text_primary="#e3e3ff",
    text_secondary="#baa3d4",
    text_muted="#7e6894",
    text_on_accent="#ffffff",
    accent="#8f4ce8",
    accent_hover="#7c3cd0",
    accent_pressed="#6a2db8",
    accent_muted="rgba(143,76,232,0.14)",
    success="#5cb85c",
    success_hover="#4cae4c",
    success_bg="rgba(92,184,92,0.10)",
    success_border="#3d8b3d",
    warning="#f0ad4e",
    warning_hover="#ec971f",
    warning_bg="rgba(240,173,78,0.10)",
    warning_border="#c49242",
    error="#e25555",
    error_hover="#d43f3f",
    error_bg="rgba(226,85,85,0.10)",
    error_border="#b83a3a",
    selection="#4e0b4d",
    disabled_bg="#2e1836",
    disabled_text="#5c4668",
)

# Emerald  — dark theme with teal-green accent
EMERALD_PALETTE = ThemePalette(
    bg_base="#0f1714",
    bg_surface="#141e1a",
    bg_card="#1a2822",
    bg_input="#1f2e28",
    bg_hover="#263830",
    bg_elevated="#22322c",
    border="#2e4a3e",
    border_subtle="#243a32",
    border_focus="#4ec9a0",
    text_primary="#e0f0ea",
    text_secondary="#98b8aa",
    text_muted="#608878",
    text_on_accent="#ffffff",
    accent="#4ec9a0",
    accent_hover="#3eb88e",
    accent_pressed="#30a67c",
    accent_muted="rgba(78,201,160,0.12)",
    success="#5cb85c",
    success_hover="#4cae4c",
    success_bg="rgba(92,184,92,0.10)",
    success_border="#3d8b3d",
    warning="#f0ad4e",
    warning_hover="#ec971f",
    warning_bg="rgba(240,173,78,0.10)",
    warning_border="#c49242",
    error="#e25555",
    error_hover="#d43f3f",
    error_bg="rgba(226,85,85,0.10)",
    error_border="#b83a3a",
    selection="#1e5040",
    disabled_bg="#1c2c26",
    disabled_text="#486058",
)

# Rose  — dark theme with pink accent
ROSE_PALETTE = ThemePalette(
    bg_base="#181115",
    bg_surface="#1e171c",
    bg_card="#281e26",
    bg_input="#30242e",
    bg_hover="#3a2c38",
    bg_elevated="#322834",
    border="#4e3a4a",
    border_subtle="#3e2e3c",
    border_focus="#e06c8a",
    text_primary="#f0e4ec",
    text_secondary="#bca4b4",
    text_muted="#887080",
    text_on_accent="#ffffff",
    accent="#e06c8a",
    accent_hover="#d05878",
    accent_pressed="#c04668",
    accent_muted="rgba(224,108,138,0.12)",
    success="#5cb85c",
    success_hover="#4cae4c",
    success_bg="rgba(92,184,92,0.10)",
    success_border="#3d8b3d",
    warning="#f0ad4e",
    warning_hover="#ec971f",
    warning_bg="rgba(240,173,78,0.10)",
    warning_border="#c49242",
    error="#e25555",
    error_hover="#d43f3f",
    error_bg="rgba(226,85,85,0.10)",
    error_border="#b83a3a",
    selection="#5a2840",
    disabled_bg="#2c2028",
    disabled_text="#665060",
)

# Amber  — dark theme with warm orange accent
AMBER_PALETTE = ThemePalette(
    bg_base="#16130e",
    bg_surface="#1e1a14",
    bg_card="#28221a",
    bg_input="#302a20",
    bg_hover="#3a3228",
    bg_elevated="#332c24",
    border="#4e4434",
    border_subtle="#3e3628",
    border_focus="#e0a050",
    text_primary="#f0ece4",
    text_secondary="#bcb4a0",
    text_muted="#887e6c",
    text_on_accent="#1a1408",
    accent="#e0a050",
    accent_hover="#d09040",
    accent_pressed="#c08030",
    accent_muted="rgba(224,160,80,0.12)",
    success="#5cb85c",
    success_hover="#4cae4c",
    success_bg="rgba(92,184,92,0.10)",
    success_border="#3d8b3d",
    warning="#f0ad4e",
    warning_hover="#ec971f",
    warning_bg="rgba(240,173,78,0.10)",
    warning_border="#c49242",
    error="#e25555",
    error_hover="#d43f3f",
    error_bg="rgba(226,85,85,0.10)",
    error_border="#b83a3a",
    selection="#5a4020",
    disabled_bg="#2c2820",
    disabled_text="#665c4c",
)

# Slate  — neutral gray theme
SLATE_PALETTE = ThemePalette(
    bg_base="#161819",
    bg_surface="#1c1f21",
    bg_card="#232729",
    bg_input="#292d30",
    bg_hover="#303538",
    bg_elevated="#2c3033",
    border="#424a4e",
    border_subtle="#363c40",
    border_focus="#8899aa",
    text_primary="#e4e8ec",
    text_secondary="#98a4b0",
    text_muted="#687480",
    text_on_accent="#ffffff",
    accent="#8899aa",
    accent_hover="#788898",
    accent_pressed="#687888",
    accent_muted="rgba(136,153,170,0.12)",
    success="#5cb85c",
    success_hover="#4cae4c",
    success_bg="rgba(92,184,92,0.10)",
    success_border="#3d8b3d",
    warning="#f0ad4e",
    warning_hover="#ec971f",
    warning_bg="rgba(240,173,78,0.10)",
    warning_border="#c49242",
    error="#e25555",
    error_hover="#d43f3f",
    error_bg="rgba(226,85,85,0.10)",
    error_border="#b83a3a",
    selection="#3a4854",
    disabled_bg="#282c2e",
    disabled_text="#505860",
)

# Crisp  — light mode, near-white with deep navy accent
CRISP_PALETTE = ThemePalette(
    bg_base="#F0F0F0",
    bg_surface="#E8E8E8",
    bg_card="#FFFFFF",
    bg_input="#FFFFFF",
    bg_hover="#E0E0E0",
    bg_elevated="#ECECEC",

    border="#C0C0C0",
    border_subtle="#D4D4D4",
    border_focus="#163060",

    text_primary="#0A0A0A",
    text_secondary="#3A3A3A",
    text_muted="#686868",
    text_on_accent="#FFFFFF",

    accent="#163060",
    accent_hover="#0E2248",
    accent_pressed="#081830",
    accent_muted="rgba(22,48,96,0.10)",

    success="#1A5C2A",
    success_hover="#144A22",
    success_bg="rgba(26,92,42,0.10)",
    success_border="#144A22",

    warning="#7A5800",
    warning_hover="#624600",
    warning_bg="rgba(122,88,0,0.10)",
    warning_border="#624600",

    error="#8C2020",
    error_hover="#701818",
    error_bg="rgba(140,32,32,0.10)",
    error_border="#701818",

    selection="#B8CCE8",

    disabled_bg="#E0E0E0",
    disabled_text="#A0A0A0",
)

# Warm  — light mode, off-white parchment with dark warm navy accent
WARM_PALETTE = ThemePalette(
    bg_base="#EEEBE4",
    bg_surface="#E6E2DA",
    bg_card="#F4F1EC",
    bg_input="#FDFAF6",
    bg_hover="#DED9D0",
    bg_elevated="#EAE6DF",

    border="#C4BAB0",
    border_subtle="#D4CEC6",
    border_focus="#1A2E52",

    text_primary="#141010",
    text_secondary="#3A3230",
    text_muted="#6A6260",
    text_on_accent="#FFFFFF",

    accent="#1A2E52",
    accent_hover="#142444",
    accent_pressed="#0E1A34",
    accent_muted="rgba(26,46,82,0.10)",

    success="#1E5230",
    success_hover="#184228",
    success_bg="rgba(30,82,48,0.10)",
    success_border="#184228",

    warning="#7A5000",
    warning_hover="#603E00",
    warning_bg="rgba(122,80,0,0.10)",
    warning_border="#603E00",

    error="#882020",
    error_hover="#6C1818",
    error_bg="rgba(136,32,32,0.10)",
    error_border="#6C1818",

    selection="#B8C8DC",

    disabled_bg="#E2DED6",
    disabled_text="#ACA89E",
)

# Ordered mapping of display name -> palette
THEME_PRESETS: dict[str, ThemePalette] = {
    "Midnight": DARK_PALETTE,
    "Violet": VIOLET_PALETTE,
    "Emerald": EMERALD_PALETTE,
    "Rose": ROSE_PALETTE,
    "Amber": AMBER_PALETTE,
    "Slate": SLATE_PALETTE,
    "Crisp": CRISP_PALETTE,
    "Warm": WARM_PALETTE,
}

# ================================================================
#  Font-size scaling
# ================================================================

# base_px -> (sm, md, lg, xl, xxl, hero) mapping
_FONT_SCALE = {
    12: ("10px", "13px", "15px", "18px", "22px", "38px"),
    13: ("11px", "14px", "16px", "20px", "24px", "42px"),
    15: ("12px", "16px", "18px", "22px", "27px", "46px"),
    17: ("14px", "18px", "20px", "25px", "30px", "50px"),
}

FONT_SIZE_OPTIONS: dict[str, int] = {
    "Small (12px)": 12,
    "Default (13px)": 13,
    "Large (15px)": 15,
    "Extra Large (17px)": 17,
}


def scale_font_sizes(palette: ThemePalette, base_px: int) -> ThemePalette:
    """Return a copy of *palette* with font sizes adjusted to *base_px*."""
    sm, md, lg, xl, xxl, hero = _FONT_SCALE.get(
        base_px, _FONT_SCALE[13]
    )
    return replace(
        palette,
        font_size_base=f"{base_px}px",
        font_size_sm=sm,
        font_size_md=md,
        font_size_lg=lg,
        font_size_xl=xl,
        font_size_xxl=xxl,
        font_size_hero=hero,
    )


# ================================================================
#  QSettings persistence helpers
# ================================================================

def get_saved_theme_name() -> str:
    """Return the persisted theme name (or 'Midnight')."""
    from .utils.settings import app_settings
    return app_settings().value("appearance/theme", "Midnight")


def get_saved_font_size() -> int:
    """Return the persisted base font size in px (or 13)."""
    from .utils.settings import app_settings
    return int(app_settings().value("appearance/font_size", 13))


def save_appearance(theme_name: str, font_size: int) -> None:
    """Persist theme + font choices."""
    from .utils.settings import app_settings
    s = app_settings()
    s.setValue("appearance/theme", theme_name)
    s.setValue("appearance/font_size", font_size)


def build_current_palette() -> ThemePalette:
    """Build the palette from saved settings (theme + font size)."""
    name = get_saved_theme_name()
    palette = THEME_PRESETS.get(name, DARK_PALETTE)
    font_size = get_saved_font_size()
    if font_size != 13:
        palette = scale_font_sizes(palette, font_size)
    return palette


def generate_stylesheet(p: ThemePalette | None = None) -> str:
    """Generate the full application QSS from a ThemePalette."""
    if p is None:
        p = DARK_PALETTE

    return f"""
/* ============================================================
   SSD - Generated Theme
   ============================================================ */

/* --- Base --- */
QWidget {{
    font-family: {p.font_family};
    font-size: {p.font_size_base};
    color: {p.text_primary};
    background-color: {p.bg_base};
}}

QMainWindow {{
    background-color: {p.bg_base};
}}

/* --- Menu Bar --- */
QMenuBar {{
    background-color: {p.bg_surface};
    color: {p.text_primary};
    border-bottom: 1px solid {p.border};
    padding: 2px 0;
}}

QMenuBar::item {{
    padding: 6px 14px;
    background: transparent;
    border-radius: {p.radius_sm};
    margin: 2px 1px;
}}

QMenuBar::item:selected {{
    background-color: {p.bg_hover};
}}

QMenu {{
    background-color: {p.bg_card};
    color: {p.text_primary};
    border: 1px solid {p.border};
    border-radius: {p.radius_md};
    padding: 4px 0;
}}

QMenu::item {{
    padding: 7px 32px 7px 20px;
    border-radius: {p.radius_sm};
    margin: 1px 4px;
}}

QMenu::item:selected {{
    background-color: {p.accent_muted};
    color: {p.accent};
}}

QMenu::separator {{
    height: 1px;
    background-color: {p.border};
    margin: 4px 12px;
}}

/* --- Status Bar --- */
QStatusBar {{
    background-color: {p.bg_surface};
    color: {p.text_secondary};
    border-top: 1px solid {p.border};
    padding: 3px 12px;
    font-size: {p.font_size_sm};
}}

/* --- Group Boxes --- */
QGroupBox {{
    font-weight: 600;
    font-size: {p.font_size_base};
    border: 1px solid {p.border_subtle};
    border-radius: {p.radius_sm};
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    background-color: {p.bg_card};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: {p.text_secondary};
    font-size: {p.font_size_base};
}}

/* --- Buttons --- */
QPushButton {{
    background-color: {p.accent};
    color: {p.text_on_accent};
    border: none;
    border-radius: {p.radius_md};
    padding: 8px 20px;
    min-width: 80px;
    font-weight: 600;
    font-size: {p.font_size_base};
}}

QPushButton:hover {{
    background-color: {p.accent_hover};
}}

QPushButton:pressed {{
    background-color: {p.accent_pressed};
}}

QPushButton:disabled {{
    background-color: {p.disabled_bg};
    color: {p.disabled_text};
}}

/* Button roles */
QPushButton#btn_success {{
    background-color: {p.success};
}}
QPushButton#btn_success:hover {{
    background-color: {p.success_hover};
}}
QPushButton#btn_success:disabled {{
    background-color: {p.disabled_bg};
    color: {p.disabled_text};
}}

QPushButton#btn_danger {{
    background-color: {p.error};
}}
QPushButton#btn_danger:hover {{
    background-color: {p.error_hover};
}}
QPushButton#btn_danger:disabled {{
    background-color: {p.disabled_bg};
    color: {p.disabled_text};
}}

QPushButton#btn_secondary {{
    background-color: {p.bg_hover};
    color: {p.text_primary};
    border: 1px solid {p.border};
}}
QPushButton#btn_secondary:hover {{
    background-color: {p.bg_elevated};
    border-color: {p.text_muted};
}}

QPushButton#btn_ghost {{
    background-color: transparent;
    color: {p.text_secondary};
    border: 1px solid {p.border};
}}
QPushButton#btn_ghost:hover {{
    background-color: {p.bg_hover};
    color: {p.text_primary};
    border-color: {p.text_muted};
}}

/* Welcome page buttons */
QPushButton#btn_welcome_primary {{
    font-size: {p.font_size_lg};
    font-weight: bold;
    padding: 16px 36px;
    border-radius: {p.radius_lg};
    background-color: {p.accent};
}}
QPushButton#btn_welcome_primary:hover {{
    background-color: {p.accent_hover};
}}

QPushButton#btn_welcome_secondary {{
    font-size: {p.font_size_lg};
    font-weight: bold;
    padding: 16px 36px;
    border-radius: {p.radius_lg};
    background-color: {p.bg_hover};
    border: 1px solid {p.border};
}}
QPushButton#btn_welcome_secondary:hover {{
    background-color: {p.bg_elevated};
    border-color: {p.text_muted};
}}

/* Big action buttons */
QPushButton#btn_run_analysis {{
    font-size: {p.font_size_md};
    font-weight: bold;
    padding: 10px 32px;
    border-radius: {p.radius_md};
    background-color: {p.error};
}}
QPushButton#btn_run_analysis:hover {{
    background-color: {p.error_hover};
}}
QPushButton#btn_run_analysis:disabled {{
    background-color: {p.disabled_bg};
    color: {p.disabled_text};
}}

QPushButton#btn_export {{
    font-size: {p.font_size_md};
    font-weight: bold;
    padding: 10px 32px;
    border-radius: {p.radius_md};
    background-color: {p.accent};
}}
QPushButton#btn_export:hover {{
    background-color: {p.accent_hover};
}}

/* Model selection toggle buttons */
QPushButton#btn_model_select {{
    font-size: {p.font_size_base};
    font-weight: 600;
    padding: 6px 20px;
    border-radius: {p.radius_md};
    background-color: {p.bg_hover};
    color: {p.text_secondary};
    border: 1px solid {p.border};
}}
QPushButton#btn_model_select:hover {{
    background-color: {p.bg_elevated};
    color: {p.text_primary};
    border-color: {p.text_muted};
}}
QPushButton#btn_model_select_active {{
    font-size: {p.font_size_base};
    font-weight: 600;
    padding: 6px 20px;
    border-radius: {p.radius_md};
    background-color: {p.accent};
    color: {p.text_on_accent};
    border: 1px solid {p.accent};
}}
QPushButton#btn_model_select_active:hover {{
    background-color: {p.accent_hover};
    border-color: {p.accent_hover};
}}

/* --- Input Fields --- */
QLineEdit {{
    border: 1px solid {p.border};
    border-radius: {p.radius_sm};
    padding: 7px 12px;
    background-color: {p.bg_input};
    color: {p.text_primary};
    selection-background-color: {p.selection};
}}

QLineEdit:focus {{
    border-color: {p.accent};
}}

QLineEdit:disabled, QLineEdit:read-only {{
    background-color: {p.bg_surface};
    color: {p.text_secondary};
    border-color: {p.border_subtle};
}}

QLineEdit::placeholder {{
    color: {p.text_muted};
}}

/* --- Combo Boxes --- */
QComboBox {{
    border: 1px solid {p.border};
    border-radius: {p.radius_sm};
    padding: 7px 12px;
    background-color: {p.bg_input};
    color: {p.text_primary};
    min-width: 100px;
}}

QComboBox:focus, QComboBox:on {{
    border-color: {p.accent};
}}

QComboBox::drop-down {{
    border: none;
    width: 26px;
    background-color: transparent;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {p.text_secondary};
    margin-right: 8px;
}}

QComboBox QAbstractItemView {{
    background-color: {p.bg_card};
    color: {p.text_primary};
    border: 1px solid {p.border};
    selection-background-color: {p.accent_muted};
    selection-color: {p.accent};
    outline: none;
    border-radius: {p.radius_sm};
}}

QComboBox QAbstractItemView::item {{
    padding: 7px 12px;
    min-height: 26px;
}}

/* --- Spin Boxes --- */
QSpinBox, QDoubleSpinBox {{
    border: 1px solid {p.border};
    border-radius: {p.radius_sm};
    padding: 5px 10px;
    background-color: {p.bg_input};
    color: {p.text_primary};
    selection-background-color: {p.selection};
    selection-color: {p.text_primary};
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {p.accent};
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: {p.bg_hover};
    border: none;
    width: 18px;
}}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {p.bg_elevated};
}}

/* --- Check Boxes --- */
QCheckBox {{
    spacing: 8px;
    color: {p.text_primary};
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 1px solid {p.border};
    border-radius: {p.radius_sm};
    background-color: {p.bg_input};
}}

QCheckBox::indicator:checked {{
    background-color: {p.accent};
    border-color: {p.accent};
}}

QCheckBox::indicator:hover {{
    border-color: {p.accent};
}}

/* --- Radio Buttons --- */
QRadioButton {{
    spacing: 8px;
    color: {p.text_primary};
}}

QRadioButton::indicator {{
    width: 18px;
    height: 18px;
    border: 1px solid {p.border};
    border-radius: 9px;
    background-color: {p.bg_input};
}}

QRadioButton::indicator:checked {{
    background-color: {p.accent};
    border-color: {p.accent};
}}

QRadioButton::indicator:hover {{
    border-color: {p.accent};
}}

/* --- Tables --- */
QTableWidget, QTableView {{
    border: 1px solid {p.border_subtle};
    border-radius: {p.radius_sm};
    gridline-color: {p.border_subtle};
    background-color: {p.bg_surface};
    alternate-background-color: {p.bg_card};
    color: {p.text_primary};
    selection-background-color: {p.selection};
    selection-color: {p.accent};
}}

QTableWidget::item, QTableView::item {{
    padding: 4px 8px;
}}

QHeaderView::section {{
    background-color: {p.bg_card};
    color: {p.text_secondary};
    padding: 5px 8px;
    border: none;
    border-bottom: 1px solid {p.border};
    border-right: 1px solid {p.border_subtle};
    font-weight: 600;
    font-size: {p.font_size_sm};
    text-transform: uppercase;
}}

/* --- List Widgets --- */
QListWidget {{
    border: 1px solid {p.border_subtle};
    border-radius: {p.radius_sm};
    background-color: {p.bg_surface};
    color: {p.text_primary};
}}

QListWidget::item {{
    padding: 5px 10px;
    border-radius: {p.radius_sm};
    margin: 1px 2px;
}}

QListWidget::item:selected {{
    background-color: {p.accent_muted};
    color: {p.accent};
}}

QListWidget::item:hover {{
    background-color: {p.bg_hover};
}}

/* --- Text Edits --- */
QTextEdit, QPlainTextEdit {{
    border: 1px solid {p.border_subtle};
    border-radius: {p.radius_sm};
    background-color: {p.bg_surface};
    color: {p.text_primary};
    selection-background-color: {p.selection};
    padding: 4px;
}}

/* --- Labels --- */
QLabel {{
    color: {p.text_primary};
    background: transparent;
}}

QLabel#label_title {{
    font-size: {p.font_size_xxl};
    font-weight: bold;
    color: {p.text_primary};
}}

QLabel#label_section_title {{
    font-size: {p.font_size_lg};
    font-weight: 600;
    color: {p.text_primary};
}}

QLabel#label_welcome_title {{
    font-size: {p.font_size_hero};
    font-weight: bold;
    color: {p.text_primary};
    letter-spacing: -1px;
}}

QLabel#label_welcome_subtitle {{
    font-size: 18px;
    color: {p.text_secondary};
    font-weight: 300;
}}

QLabel#label_welcome_desc {{
    font-size: {p.font_size_md};
    color: {p.text_muted};
    line-height: 1.5;
}}

QLabel#label_muted {{
    color: {p.text_secondary};
    font-size: {p.font_size_base};
}}

QLabel#label_status_ok {{
    color: {p.success};
    font-weight: 600;
}}

QLabel#label_status_warn {{
    color: {p.warning};
    font-weight: 600;
}}

QLabel#label_status_error {{
    color: {p.error};
    font-weight: 600;
}}

QLabel#label_link {{
    color: {p.accent};
}}

QLabel#label_link:hover {{
    color: {p.accent_hover};
    text-decoration: underline;
}}

/* --- Progress Bars --- */
QProgressBar {{
    border: 1px solid {p.border};
    border-radius: {p.radius_sm};
    text-align: center;
    background-color: {p.bg_surface};
    color: {p.text_primary};
    font-size: {p.font_size_sm};
    min-height: 18px;
}}

QProgressBar::chunk {{
    background-color: {p.accent};
    border-radius: 3px;
}}

/* --- Tab Widgets --- */
QTabWidget::pane {{
    border: 1px solid {p.border_subtle};
    border-radius: {p.radius_sm};
    background-color: {p.bg_card};
    top: -1px;
}}

QTabBar::tab {{
    background-color: {p.bg_surface};
    color: {p.text_secondary};
    border: 1px solid {p.border_subtle};
    border-bottom: none;
    border-top-left-radius: {p.radius_sm};
    border-top-right-radius: {p.radius_sm};
    padding: 7px 14px;
    margin-right: 1px;
    font-weight: 500;
}}

QTabBar::tab:selected {{
    background-color: {p.bg_card};
    color: {p.text_primary};
    border-bottom-color: {p.bg_card};
    font-weight: 600;
}}

QTabBar::tab:hover:!selected {{
    background-color: {p.bg_hover};
    color: {p.text_primary};
}}

/* --- Frames --- */
QFrame#frame_ready {{
    background-color: {p.bg_card};
    border: 1px solid {p.border};
    border-radius: {p.radius_sm};
    padding: 4px;
}}

QFrame#frame_ready_ok {{
    background-color: {p.success_bg};
    border: 1px solid {p.success_border};
    border-radius: {p.radius_sm};
    padding: 4px;
}}

QFrame#frame_ready_pending {{
    background-color: {p.warning_bg};
    border: 1px solid {p.warning_border};
    border-radius: {p.radius_sm};
    padding: 4px;
}}

QFrame#frame_summary {{
    background-color: {p.bg_card};
    border: 1px solid {p.border_subtle};
    border-radius: {p.radius_sm};
    padding: 4px;
}}

/* --- Stage Navigation Bar --- */
QFrame#stage_nav_bar {{
    background-color: {p.bg_surface};
    border-bottom: 1px solid {p.border};
    padding: 0;
}}

QPushButton#stage_step {{
    background: transparent;
    color: {p.text_muted};
    border: none;
    border-radius: 0;
    padding: 12px 20px;
    font-size: {p.font_size_base};
    font-weight: 500;
    min-width: 120px;
    border-bottom: 3px solid transparent;
}}

QPushButton#stage_step:hover {{
    color: {p.text_primary};
    background-color: {p.bg_hover};
}}

QPushButton#stage_step_active {{
    background: transparent;
    color: {p.accent};
    border: none;
    border-radius: 0;
    padding: 12px 20px;
    font-size: {p.font_size_base};
    font-weight: 700;
    min-width: 120px;
    border-bottom: 3px solid {p.accent};
}}

QPushButton#stage_step_active:hover {{
    background-color: {p.bg_hover};
}}

QPushButton#stage_step:disabled {{
    color: {p.disabled_text};
    background: transparent;
}}

QLabel#stage_step_number {{
    background-color: {p.bg_hover};
    color: {p.text_muted};
    border-radius: 11px;
    font-size: {p.font_size_sm};
    font-weight: bold;
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    max-height: 22px;
    qproperty-alignment: AlignCenter;
}}

QLabel#stage_step_number_active {{
    background-color: {p.accent};
    color: {p.text_on_accent};
    border-radius: 11px;
    font-size: {p.font_size_sm};
    font-weight: bold;
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    max-height: 22px;
    qproperty-alignment: AlignCenter;
}}

QLabel#stage_step_separator {{
    color: {p.border};
    font-size: {p.font_size_lg};
    background: transparent;
}}

/* --- Scroll Area --- */
QScrollArea {{
    border: none;
    background-color: {p.bg_base};
}}

/* --- Scroll Bars --- */
QScrollBar:vertical {{
    border: none;
    background: transparent;
    width: 6px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background-color: {p.border};
    border-radius: 0px;
    min-height: 24px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {p.accent};
}}

QScrollBar::handle:vertical:pressed {{
    background-color: {p.accent_hover};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}

QScrollBar:horizontal {{
    border: none;
    background: transparent;
    height: 6px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background-color: {p.border};
    border-radius: 0px;
    min-width: 24px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {p.accent};
}}

QScrollBar::handle:horizontal:pressed {{
    background-color: {p.accent_hover};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
}}

/* --- Splitter --- */
QSplitter::handle {{
    background-color: transparent;
}}

QSplitter::handle:hover, QSplitter::handle:pressed {{
    background-color: {p.border};
    border-radius: 1px;
}}

QSplitter::handle:vertical {{
    height: 5px;
}}

QSplitter::handle:horizontal {{
    width: 5px;
}}

/* --- Tooltips --- */
QToolTip {{
    background-color: {p.bg_card};
    color: {p.text_primary};
    border: 1px solid {p.border};
    padding: 8px 12px;
    border-radius: {p.radius_md};
    font-size: {p.font_size_sm};
}}

/* --- Dialogs --- */
QMessageBox {{
    background-color: {p.bg_card};
}}

QMessageBox QLabel {{
    color: {p.text_primary};
}}

QDialog {{
    background-color: {p.bg_card};
}}

QInputDialog {{
    background-color: {p.bg_card};
}}

/* --- Collapsible Box toggle --- */
QToolButton {{
    color: {p.text_primary};
    border: none;
    padding: 8px 10px;
    background: transparent;
    font-weight: 600;
    font-size: {p.font_size_base};
}}

QToolButton:hover {{
    background-color: {p.bg_hover};
    border-radius: {p.radius_sm};
}}

/* --- Coverage / status panels --- */
QLabel#label_coverage_ok {{
    color: {p.success};
    background-color: {p.success_bg};
    border: 1px solid {p.success_border};
    border-radius: {p.radius_sm};
    padding: 10px;
}}

QLabel#label_coverage_warn {{
    color: {p.warning};
    background-color: {p.warning_bg};
    border: 1px solid {p.warning_border};
    border-radius: {p.radius_sm};
    padding: 10px;
}}

/* --- Stat cards in Stage 3 --- */
QLabel#label_stat_value {{
    font-size: {p.font_size_xl};
    font-weight: bold;
    color: {p.text_primary};
}}

QLabel#label_stat_name {{
    font-size: {p.font_size_sm};
    color: {p.text_secondary};
    font-weight: 500;
}}

/* --- Update banner --- */
QFrame#update_banner {{
    background-color: {p.accent_muted};
    border-top: 1px solid {p.border_focus};
    padding: 4px 8px;
}}

QLabel#update_banner_icon {{
    font-size: 14px;
    color: {p.accent};
    background-color: transparent;
}}

QLabel#update_banner_text {{
    color: {p.text_primary};
    background-color: transparent;
}}

QPushButton#update_banner_download {{
    background-color: {p.accent};
    color: {p.text_on_accent};
    border: none;
    border-radius: {p.radius_sm};
    padding: 4px 14px;
    min-width: 0;
    font-weight: 600;
}}

QPushButton#update_banner_download:hover {{
    background-color: {p.accent_hover};
}}

QPushButton#update_banner_dismiss {{
    background-color: transparent;
    color: {p.text_muted};
    border: 1px solid {p.border};
    border-radius: {p.radius_sm};
    padding: 2px;
    min-width: 0;
    font-weight: normal;
}}

QPushButton#update_banner_dismiss:hover {{
    color: {p.text_primary};
    border-color: {p.text_muted};
}}

/* Export Options dialog — required-column badges */
QLabel#export_required_tag {{
    background-color: {p.bg_elevated};
    color: {p.text_muted};
    border: 1px solid {p.border};
    border-radius: 4px;
    padding: 1px 6px;
    font-size: {p.font_size_sm};
}}
"""
