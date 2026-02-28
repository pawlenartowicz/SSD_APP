"""Linux desktop integration: self-register on first run.

Writes ~/.local/share/applications/ssd-app.desktop and a PNG icon so the
app appears with its custom icon in file managers and application launchers.

Silently no-ops on non-Linux platforms and when running from source (not a
frozen PyInstaller binary).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_DESKTOP_FILENAME = "ssd-app.desktop"
_ICON_FILENAME = "ssd-app.png"
_ICON_SIZE = 256


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def _exe_path() -> Path:
    return Path(sys.executable).resolve()


def _desktop_path() -> Path:
    return Path.home() / ".local" / "share" / "applications" / _DESKTOP_FILENAME


def _icon_path() -> Path:
    return Path.home() / ".local" / "share" / "icons" / "hicolor" / "256x256" / "apps" / _ICON_FILENAME


def _desktop_content(exe: Path) -> str:
    return (
        "[Desktop Entry]\n"
        "Version=1.0\n"
        "Type=Application\n"
        "Name=SSD\n"
        "Comment=Supervised Semantic Differential\n"
        f"Exec={exe}\n"
        "Icon=ssd-app\n"
        "Terminal=false\n"
        "Categories=Science;Education;\n"
        "StartupWMClass=SSD\n"
    )


def _needs_update(desktop: Path, exe: Path, icon: Path) -> bool:
    """True if the .desktop file is missing, the icon is missing, or the Exec
    path no longer matches the current executable (e.g. the binary was moved)."""
    if not desktop.exists() or not icon.exists():
        return True
    return f"Exec={exe}" not in desktop.read_text()




def _write_icon_png(palette, theme_name: str, out_path: Path) -> None:
    from PySide6.QtCore import QBuffer, QIODevice, Qt
    from PySide6.QtGui import QPainter, QPixmap

    from ssdiff_gui.logo import paint_logo

    pixmap = QPixmap(_ICON_SIZE, _ICON_SIZE)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    paint_logo(painter, _ICON_SIZE, palette, theme_name, draw_bg=False, content_scale=1.2)
    painter.end()

    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    pixmap.save(buf, "PNG")
    out_path.write_bytes(bytes(buf.data()))
    buf.close()


def register(palette, theme_name: str = "Midnight", force: bool = False) -> None:
    """Register the app with the desktop environment if needed.

    Pass force=True to re-register even if already registered (e.g. after a
    theme change).  Safe to call unconditionally — skips silently when not on
    Linux or not running as a frozen binary.  Never raises.
    """
    if sys.platform != "linux" or not _is_frozen():
        return

    exe = _exe_path()
    desktop = _desktop_path()
    icon = _icon_path()

    if not force and not _needs_update(desktop, exe, icon):
        return

    try:
        icon.parent.mkdir(parents=True, exist_ok=True)
        _write_icon_png(palette, theme_name, icon)

        desktop.parent.mkdir(parents=True, exist_ok=True)
        desktop.write_text(_desktop_content(exe))

        # Register icon with the hicolor theme so GNOME resolves it by name
        subprocess.run(
            ["gtk-update-icon-cache", "-f", "-t",
             str(Path.home() / ".local" / "share" / "icons" / "hicolor")],
            capture_output=True,
            timeout=5,
        )

        # Tell the desktop environment to pick up the new .desktop entry
        subprocess.run(
            ["update-desktop-database", str(desktop.parent)],
            capture_output=True,
            timeout=5,
        )

        # Set the custom icon directly on the binary file so file managers
        # (Nautilus/Files etc.) show it when browsing to the executable.
        # This uses GIO metadata (stored in ~/.local/share/gvfs-metadata).
        subprocess.run(
            ["gio", "set", "-t", "string", str(exe),
             "metadata::custom-icon", f"file://{icon}"],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass
