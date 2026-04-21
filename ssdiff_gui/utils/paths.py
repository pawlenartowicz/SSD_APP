"""Platform-specific path utilities for SSD."""

import os
import sys
from pathlib import Path

# Shared-embeddings subfolder name (lives under the default projects directory)
SHARED_EMB_SUBDIR = "SSD-Embeddings"


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


def _qsetting_bool(value, default: bool) -> bool:
    """QSettings stores booleans as strings on Linux — normalize here."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def projects_dir() -> Path:
    """Configured default projects directory, with a home fallback."""
    from .settings import app_settings
    val = app_settings().value("projects_directory", "")
    if val:
        return Path(val)
    return Path.home() / "SSD-Projects"


def embeddings_dir() -> Path:
    """Return the shared directory where ``.ssdembed`` files live.

    Controlled by the ``embeddings/location_mode`` setting:
      - ``"shared"`` (default): ``<projects_dir>/SSD-Embeddings``
      - ``"custom"``: user-picked path stored in ``embeddings/custom_path``
    """
    from .settings import app_settings
    s = app_settings()
    mode = s.value("embeddings/location_mode", "shared")
    if mode == "custom":
        custom = s.value("embeddings/custom_path", "")
        if custom:
            return Path(custom)
    return projects_dir() / SHARED_EMB_SUBDIR


def embeddings_autoload_enabled() -> bool:
    """Return True if embeddings should auto-load when a project is opened."""
    from .settings import app_settings
    return _qsetting_bool(
        app_settings().value("embeddings/autoload_on_open", True),
        default=True,
    )
