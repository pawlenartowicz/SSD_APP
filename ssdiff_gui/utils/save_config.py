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


def _default_items() -> Mapping[str, ItemConfig]:
    return MappingProxyType({k: ItemConfig() for k in DEFAULT_ITEM_KEYS})


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


@dataclass(frozen=True)
class SaveConfig:
    report_enabled: bool = True
    report_format: str = DEFAULT_REPORT_FORMAT
    tables_format: str = DEFAULT_TABLES_FORMAT
    items: Mapping[str, ItemConfig] = field(default_factory=_default_items)

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

        return cls(
            report_enabled=report_enabled,
            report_format=report_format,
            tables_format=tables_format,
            items=MappingProxyType(items),
        )

    def to_settings(self, qs: _SettingsLike) -> None:
        qs.setValue("save/report_enabled", bool(self.report_enabled))
        qs.setValue("save/report_format", str(self.report_format))
        qs.setValue("save/tables_format", str(self.tables_format))
        for key in DEFAULT_ITEM_KEYS:
            item = self.items.get(key, ItemConfig())
            _save_item_config(qs, key, item)
