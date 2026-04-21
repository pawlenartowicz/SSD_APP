"""Entry point for SSD application."""

import os
import sys
import multiprocessing
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtNetwork import QLocalServer, QLocalSocket

from ssdiff_gui import __version__
from ssdiff_gui.views.main_window import MainWindow
from ssdiff_gui.theme import generate_stylesheet, build_current_palette, build_qpalette, get_saved_theme_name
from ssdiff_gui.logo import create_app_icon
from ssdiff_gui.utils.linux_install import register as _linux_register


def main():
    """Main entry point."""
    multiprocessing.freeze_support()

    # Suppress spurious Wayland text-input warnings (Qt bug, not actionable)
    rules = os.environ.get("QT_LOGGING_RULES", "")
    suppress = "qt.qpa.wayland.textinput=false"
    os.environ["QT_LOGGING_RULES"] = (
        suppress + ";" + rules if rules else suppress
    )

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
    app.setPalette(build_qpalette(palette))
    app.setStyleSheet(generate_stylesheet(palette))

    # Set application icon (theme-aware)
    icon = create_app_icon(palette, theme_name)
    if icon:
        app.setWindowIcon(icon)

    # Register desktop entry + icon on Linux (silent no-op elsewhere)
    _linux_register(palette, theme_name)

    # Single-instance guard — if another copy is already running, bring it to
    # the front and exit this one instead of opening a second window.
    _SOCKET_NAME = "SSD_single_instance"
    socket = QLocalSocket()
    socket.connectToServer(_SOCKET_NAME)
    if socket.waitForConnected(300):
        # Another instance is running — ask it to raise its window and quit.
        socket.write(b"raise")
        socket.flush()
        socket.disconnectFromServer()
        sys.exit(0)

    server = QLocalServer()
    QLocalServer.removeServer(_SOCKET_NAME)  # clean up any stale socket
    server.listen(_SOCKET_NAME)

    # Create and show main window
    window = MainWindow()
    window.show()

    def _on_new_connection():
        conn = server.nextPendingConnection()  # noqa: F821 (closure)
        conn.waitForReadyRead(200)
        conn.readAll()  # consume the message
        conn.disconnectFromServer()
        window.setWindowState(window.windowState() & ~Qt.WindowMinimized)  # noqa: F821 (closure)
        window.raise_()  # noqa: F821 (closure)
        window.activateWindow()  # noqa: F821 (closure)

    server.newConnection.connect(_on_new_connection)

    # Prompt for projects directory on first launch
    window.check_first_run_settings()

    ret = app.exec()
    # Explicitly delete Qt objects while Python is still fully alive.
    # Without this, sys.exit() triggers Python shutdown before Qt finishes
    # destroying widgets, causing callbacks into partially-torn-down Python
    # objects and a SIGSEGV crash on macOS.
    del window
    del server
    del app
    sys.exit(ret)


if __name__ == "__main__":
    main()
