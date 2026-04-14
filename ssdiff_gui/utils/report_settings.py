"""Shared QSettings keys and helpers for report options."""

from __future__ import annotations

# ------------------------------------------------------------------ #
#  QSettings keys — map 1:1 to ssdiff result.report() parameters
# ------------------------------------------------------------------ #

KEY_TOP_WORDS = "report/top_words"          # int; words per pole. 0 = skip section
KEY_CLUSTERS = "report/clusters"            # int; topn for clustering. 0 = skip
KEY_EXTREME_DOCS = "report/extreme_docs"    # int; docs per side. 0 = skip
KEY_MISDIAGNOSED = "report/misdiagnosed"    # int; docs per side. 0 = skip

_DEFAULTS: dict[str, int] = {
    KEY_TOP_WORDS: 10,
    KEY_CLUSTERS: 50,
    KEY_EXTREME_DOCS: 5,
    KEY_MISDIAGNOSED: 5,
}


def get_report_setting(key: str) -> int:
    """Read a single report setting from QSettings, returning the typed default."""
    from .settings import app_settings
    default = _DEFAULTS.get(key, 0)
    return int(app_settings().value(key, default))
