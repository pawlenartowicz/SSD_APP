"""Static registry: item_key → columns available from the library view.

Loaded at import time by introspecting ssdiff view classes. Qt-free.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactColumns:
    all_columns: tuple[str, ...]      # full _columns from the view class; () if unavailable
    default_columns: tuple[str, ...]  # DEFAULT_COLS subset, or all_columns if no entry


def _load_registry() -> dict[str, ArtifactColumns]:
    """Build the registry by introspecting ssdiff view classes.

    Each import is wrapped in try/except so a library shape change degrades
    gracefully to empty columns rather than crashing the app.
    """
    registry: dict[str, ArtifactColumns] = {}

    # --- load DEFAULT_COLS once ---
    try:
        from ssdiff.results.core import DEFAULT_COLS as _DC
        _default_cols: dict[str, tuple[str, ...]] = dict(_DC)
    except Exception:
        _default_cols = {}

    def _spec(cls_name: str, cls) -> ArtifactColumns:
        """Build ArtifactColumns from a view class."""
        try:
            all_cols: tuple[str, ...] = tuple(cls._columns)
        except Exception:
            all_cols = ()
        default = _default_cols.get(cls_name, all_cols)
        return ArtifactColumns(all_columns=all_cols, default_columns=tuple(default))

    # --- words ---
    try:
        from ssdiff.results.continuous_result import WordsView
        registry["words"] = _spec("WordsView", WordsView)
    except Exception:
        registry["words"] = ArtifactColumns((), ())

    # --- clusters ---
    try:
        from ssdiff.results.continuous_result import ClustersViewSided
        registry["clusters"] = _spec("ClustersViewSided", ClustersViewSided)
    except Exception:
        registry["clusters"] = ArtifactColumns((), ())

    # --- cluster_words ---
    try:
        from ssdiff.results.continuous_result import ClusterWordsViewSided
        registry["cluster_words"] = _spec("ClusterWordsViewSided", ClusterWordsViewSided)
    except Exception:
        registry["cluster_words"] = ArtifactColumns((), ())

    # --- snippets ---
    try:
        from ssdiff.results.continuous_result import SnippetsView
        registry["snippets"] = _spec("SnippetsView", SnippetsView)
    except Exception:
        registry["snippets"] = ArtifactColumns((), ())

    # --- docs_extreme ---
    try:
        from ssdiff.results.continuous_result import DocsView
        registry["docs_extreme"] = _spec("DocsView", DocsView)
    except Exception:
        registry["docs_extreme"] = ArtifactColumns((), ())

    # --- docs_misdiagnosed (same class as docs_extreme) ---
    try:
        from ssdiff.results.continuous_result import DocsView
        registry["docs_misdiagnosed"] = _spec("DocsView", DocsView)
    except Exception:
        registry["docs_misdiagnosed"] = ArtifactColumns((), ())

    # --- pairs ---
    try:
        from ssdiff.results.group_result import PairsListView
        registry["pairs"] = _spec("PairsListView", PairsListView)
    except Exception:
        registry["pairs"] = ArtifactColumns((), ())

    # --- sweep ---
    try:
        from ssdiff.results.continuous_result import SweepView
        registry["sweep"] = _spec("SweepView", SweepView)
    except Exception:
        registry["sweep"] = ArtifactColumns((), ())

    # --- sweep_plot: PNG, no columns ---
    registry["sweep_plot"] = ArtifactColumns((), ())

    return registry


# Keys match save_config.DEFAULT_ITEM_KEYS. sweep_plot and any unknown key map to
# ArtifactColumns((), ()) — the UI uses this as "no column controls".
ARTIFACT_COLUMNS: dict[str, ArtifactColumns] = _load_registry()


def get_columns(item_key: str) -> ArtifactColumns:
    """Return the column spec for an item key, or empty if unknown."""
    return ARTIFACT_COLUMNS.get(item_key, ArtifactColumns((), ()))
