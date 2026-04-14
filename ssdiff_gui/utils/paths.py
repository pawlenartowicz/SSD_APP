"""Platform-specific path utilities for SSD."""

import os
import sys
from pathlib import Path


def get_app_data_dir() -> Path:
    """Return the platform-appropriate application data directory for SSD.

    - Windows: ``%LOCALAPPDATA%/SSD``
    - macOS:   ``~/Library/Application Support/SSD``
    - Linux:   ``~/.local/share/SSD``
    """
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            return Path(local_appdata) / "SSD"
        return Path.home() / "AppData" / "Local" / "SSD"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "SSD"
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        if xdg:
            return Path(xdg) / "SSD"
        return Path.home() / ".local" / "share" / "SSD"
