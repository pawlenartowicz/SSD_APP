"""Entry point for SSD application."""

import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from ssdiff_gui import __version__
from ssdiff_gui.views.main_window import MainWindow
from ssdiff_gui.theme import generate_stylesheet, build_current_palette, get_saved_theme_name
from ssdiff_gui.logo import create_app_icon
from ssdiff_gui.utils.linux_install import register as _linux_register


def main():
    """Main entry point."""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("SSD")
    app.setOrganizationName("SSD")
    app.setApplicationVersion(__version__)

    # Set application style
    app.setStyle("Fusion")

    # Disable tooltip animation (instant show)
    app.setEffectEnabled(Qt.UI_AnimateTooltip, False)

    # Generate and apply theme stylesheet from saved preferences
    theme_name = get_saved_theme_name()
    palette = build_current_palette()
    app.setStyleSheet(generate_stylesheet(palette))

    # Set application icon (theme-aware)
    icon = create_app_icon(palette, theme_name)
    if icon:
        app.setWindowIcon(icon)

    # Register desktop entry + icon on Linux (silent no-op elsewhere)
    _linux_register(palette, theme_name)

    # Create and show main window
    window = MainWindow()
    window.show()

    # Prompt for projects directory on first launch
    window.check_first_run_settings()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
