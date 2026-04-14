"""SSD - Desktop application for Supervised Semantic Differential analysis."""

from importlib.metadata import version as _get_version

try:
    __version__ = _get_version("ssd-app")
except Exception:
    __version__ = "2.0.0"

try:
    __ssdiff_version__ = _get_version("ssdiff")
except Exception:
    __ssdiff_version__ = None

__author__ = "SSD Team"
