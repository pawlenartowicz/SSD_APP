"""Application-level QSettings accessor."""

from PySide6.QtCore import QSettings

_ORG = "SSD"
_APP = "SSD"


def app_settings() -> QSettings:
    """Return a QSettings instance for the SSD application."""
    return QSettings(_ORG, _APP)
