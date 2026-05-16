"""Persistable configuration for the Save Settings dialog.

Adapter-style: SaveConfig does not import PySide6. Consumers pass any
object with QSettings-compatible ``value(key, default)``,
``setValue(key, value)``, and ``remove(key)`` methods.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol


DEFAULT_REPORT_FORMAT = "md"
DEFAULT_TABLES_FORMAT = "csv"

DEFAULT_ITEM_KEYS: tuple[str, ...] = (
    "sweep_plot",
    "sweep",
    "words",
    "clusters",
    "cluster_words",
    "snippets",
    "docs_extreme",
    "docs_misdiagnosed",
    "pairs",
)

# Report sections that .report(**kwargs) understands across result classes.
# Per-class applicability lives next to the consumer (result_export); this
# tuple is the union, in display order.
DEFAULT_REPORT_SECTION_KEYS: tuple[str, ...] = (
    "top_words",
    "clusters",
    "extreme_docs",
    "misdiagnosed",
    "top",
)

# Keys on ReportSectionConfig that only apply to the "clusters" section.
_CLUSTERS_ONLY_FIELDS: tuple[str, ...] = ("n_words", "n_snippets")

_LEGACY_REPORT_KEYS = (
    "report/top_words",
    "report/clusters",
    "report/extreme_docs",
    "report/misdiagnosed",
    "report/snippet_preview",
)

# Sentinel stored in QSettings when cols=None (library default).
_COLS_DEFAULT_SENTINEL = "__default__"


class _SettingsLike(Protocol):
    def value(self, key: str, default=None): ...
    def setValue(self, key: str, value) -> None: ...
    def remove(self, key: str) -> None: ...


@dataclass(frozen=True)
class ItemConfig:
    """Per-artifact export settings matching the `.save()` signature."""
    enabled: bool = False
    cols: tuple[str, ...] | None = None   # None = library default
    k: int | None = None                  # None = all rows


@dataclass(frozen=True)
class ReportSectionConfig:
    """Per-section knobs for ``result.report(...)``.

    ``enabled`` flips the section on/off. The numeric fields each map to a
    key in the section's dict argument; ``None`` means "use the library
    default", so version bumps to those defaults flow through unchanged.
    ``n_words`` and ``n_snippets`` only apply to the ``clusters`` section.
    """
    enabled: bool = True
    n: int | None = None
    n_words: int | None = None        # clusters only
    n_snippets: int | None = None     # clusters only


def _default_items() -> Mapping[str, ItemConfig]:
    return MappingProxyType({k: ItemConfig() for k in DEFAULT_ITEM_KEYS})


def _default_report_sections() -> Mapping[str, ReportSectionConfig]:
    return MappingProxyType(
        {k: ReportSectionConfig() for k in DEFAULT_REPORT_SECTION_KEYS}
    )


def _coerce_bool(raw, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _coerce_int_or_none(raw, default: int | None) -> int | None:
    """Coerce QSettings value to int or None. -1 sentinel means None."""
    if raw is None:
        return default
    try:
        v = int(raw)
    except (ValueError, TypeError):
        return default
    return None if v == -1 else v


def _coerce_cols(raw) -> tuple[str, ...] | None:
    """Coerce QSettings value for cols. Returns None for default sentinel."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return tuple(str(c) for c in raw)
    if isinstance(raw, str):
        s = raw.strip()
        if s in {_COLS_DEFAULT_SENTINEL, ""}:
            return None
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return tuple(str(c) for c in parsed)
        except (json.JSONDecodeError, ValueError):
            pass
        return None
    return None


def _load_item_config(qs: _SettingsLike, key: str) -> ItemConfig:
    """Read ItemConfig for a single item key from QSettings."""
    enabled = _coerce_bool(qs.value(f"save/tables/{key}_enabled", False), False)
    raw_cols = qs.value(f"save/tables/{key}_cols", _COLS_DEFAULT_SENTINEL)
    raw_k = qs.value(f"save/tables/{key}_k", -1)
    cols = _coerce_cols(raw_cols)
    k = _coerce_int_or_none(raw_k, None)
    return ItemConfig(enabled=enabled, cols=cols, k=k)


def _save_item_config(qs: _SettingsLike, key: str, item: ItemConfig) -> None:
    """Write ItemConfig for a single item key to QSettings."""
    qs.setValue(f"save/tables/{key}_enabled", bool(item.enabled))
    if item.cols is None:
        qs.setValue(f"save/tables/{key}_cols", _COLS_DEFAULT_SENTINEL)
    else:
        qs.setValue(f"save/tables/{key}_cols", json.dumps(list(item.cols)))
    qs.setValue(f"save/tables/{key}_k", int(item.k) if item.k is not None else -1)


def _load_report_section(qs: _SettingsLike, key: str) -> ReportSectionConfig:
    """Read ReportSectionConfig for a single section key from QSettings."""
    base = f"save/report_sections/{key}"
    enabled = _coerce_bool(qs.value(f"{base}_enabled", True), True)
    n = _coerce_int_or_none(qs.value(f"{base}_n", -1), None)
    if key == "clusters":
        n_words = _coerce_int_or_none(qs.value(f"{base}_n_words", -1), None)
        n_snippets = _coerce_int_or_none(qs.value(f"{base}_n_snippets", -1), None)
    else:
        n_words = None
        n_snippets = None
    return ReportSectionConfig(
        enabled=enabled, n=n, n_words=n_words, n_snippets=n_snippets,
    )


def _save_report_section(
    qs: _SettingsLike, key: str, section: ReportSectionConfig,
) -> None:
    """Write ReportSectionConfig for a single section key to QSettings."""
    base = f"save/report_sections/{key}"
    qs.setValue(f"{base}_enabled", bool(section.enabled))
    qs.setValue(f"{base}_n", int(section.n) if section.n is not None else -1)
    if key == "clusters":
        qs.setValue(
            f"{base}_n_words",
            int(section.n_words) if section.n_words is not None else -1,
        )
        qs.setValue(
            f"{base}_n_snippets",
            int(section.n_snippets) if section.n_snippets is not None else -1,
        )


@dataclass(frozen=True)
class SaveConfig:
    report_enabled: bool = True
    report_format: str = DEFAULT_REPORT_FORMAT
    tables_format: str = DEFAULT_TABLES_FORMAT
    items: Mapping[str, ItemConfig] = field(default_factory=_default_items)
    report_sections: Mapping[str, ReportSectionConfig] = field(
        default_factory=_default_report_sections,
    )

    @classmethod
    def default(cls) -> "SaveConfig":
        return cls()

    @classmethod
    def from_settings(cls, qs: _SettingsLike) -> "SaveConfig":
        for legacy_key in _LEGACY_REPORT_KEYS:
            qs.remove(legacy_key)

        report_enabled = _coerce_bool(qs.value("save/report_enabled", True), True)
        report_format = str(qs.value("save/report_format", DEFAULT_REPORT_FORMAT))
        tables_format = str(qs.value("save/tables_format", DEFAULT_TABLES_FORMAT))

        items = {
            key: _load_item_config(qs, key)
            for key in DEFAULT_ITEM_KEYS
        }

        report_sections = {
            key: _load_report_section(qs, key)
            for key in DEFAULT_REPORT_SECTION_KEYS
        }

        return cls(
            report_enabled=report_enabled,
            report_format=report_format,
            tables_format=tables_format,
            items=MappingProxyType(items),
            report_sections=MappingProxyType(report_sections),
        )

    def to_settings(self, qs: _SettingsLike) -> None:
        qs.setValue("save/report_enabled", bool(self.report_enabled))
        qs.setValue("save/report_format", str(self.report_format))
        qs.setValue("save/tables_format", str(self.tables_format))
        for key in DEFAULT_ITEM_KEYS:
            item = self.items.get(key, ItemConfig())
            _save_item_config(qs, key, item)
        for key in DEFAULT_REPORT_SECTION_KEYS:
            section = self.report_sections.get(key, ReportSectionConfig())
            _save_report_section(qs, key, section)

    def report_kwargs(self, allowed: tuple[str, ...]) -> dict:
        """Build the ``kwargs`` for ``result.report(**kwargs)``.

        ``allowed`` lists section keys the concrete result type accepts.
        Sections outside ``allowed`` are silently dropped (passing them to
        ``.report`` would raise ``TypeError``). For each allowed section:

        - ``enabled=False`` → ``False`` (skip the section)
        - all numeric overrides ``None`` → ``True`` (use library defaults)
        - any non-``None`` override → an explicit dict (e.g. ``{"n": 20}``)
        """
        kwargs: dict = {}
        for key in allowed:
            section = self.report_sections.get(key, ReportSectionConfig())
            if not section.enabled:
                kwargs[key] = False
                continue
            overrides: dict = {}
            if section.n is not None:
                overrides["n"] = section.n
            if key == "clusters":
                if section.n_words is not None:
                    overrides["n_words"] = section.n_words
                if section.n_snippets is not None:
                    overrides["n_snippets"] = section.n_snippets
            kwargs[key] = overrides if overrides else True
        return kwargs
