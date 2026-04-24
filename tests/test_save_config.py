"""Unit tests for SaveConfig and ItemConfig."""

from dataclasses import replace

import pytest


class FakeSettings:
    """Minimal in-memory stand-in for QSettings."""

    def __init__(self, initial: dict | None = None):
        self._store: dict = dict(initial or {})
        self.removed: list[str] = []

    def value(self, key: str, default=None):
        return self._store.get(key, default)

    def setValue(self, key: str, value) -> None:
        self._store[key] = value

    def remove(self, key: str) -> None:
        self.removed.append(key)
        self._store.pop(key, None)


class StringFakeSettings(FakeSettings):
    """FakeSettings that stores everything as strings, simulating QSettings text backend."""

    def setValue(self, key: str, value) -> None:
        self._store[key] = str(value)


# ---------------------------------------------------------------------------
# Existing tests — adapted to ItemConfig shape
# ---------------------------------------------------------------------------

def test_defaults_report_ticked_others_unticked():
    from ssdiff_gui.utils.save_config import SaveConfig, ItemConfig

    cfg = SaveConfig.default()

    assert cfg.report_enabled is True
    assert cfg.report_format == "md"
    assert cfg.tables_format == "csv"
    # All items should be disabled ItemConfig instances
    for k, v in cfg.items.items():
        assert isinstance(v, ItemConfig)
        assert v.enabled is False


def test_round_trip_through_fake_settings():
    from ssdiff_gui.utils.save_config import SaveConfig, ItemConfig

    words_item = ItemConfig(enabled=True)
    snippets_item = ItemConfig(enabled=True)
    cfg = replace(
        SaveConfig.default(),
        report_format="txt",
        tables_format="xlsx",
        items={**SaveConfig.default().items, "words": words_item, "snippets": snippets_item},
    )

    qs = FakeSettings()
    cfg.to_settings(qs)

    round_tripped = SaveConfig.from_settings(qs)
    assert round_tripped == cfg


def test_from_settings_falls_back_to_defaults():
    from ssdiff_gui.utils.save_config import SaveConfig

    cfg = SaveConfig.from_settings(FakeSettings())

    assert cfg == SaveConfig.default()


def test_legacy_report_keys_are_removed_on_load():
    from ssdiff_gui.utils.save_config import SaveConfig

    qs = FakeSettings(initial={
        "report/top_words": 10,
        "report/clusters": 50,
        "report/extreme_docs": 5,
        "report/misdiagnosed": 5,
        "report/snippet_preview": 600,
    })

    _ = SaveConfig.from_settings(qs)

    assert set(qs.removed) == {
        "report/top_words",
        "report/clusters",
        "report/extreme_docs",
        "report/misdiagnosed",
        "report/snippet_preview",
    }


# ---------------------------------------------------------------------------
# New ItemConfig tests
# ---------------------------------------------------------------------------

def test_item_config_defaults_are_none():
    from ssdiff_gui.utils.save_config import ItemConfig

    item = ItemConfig()
    assert item.enabled is False
    assert item.cols is None
    assert item.k is None


def test_round_trip_preserves_cols_and_k():
    from ssdiff_gui.utils.save_config import SaveConfig, ItemConfig

    words_item = ItemConfig(enabled=True, cols=("word", "rank"), k=50)
    cfg = replace(
        SaveConfig.default(),
        items={**SaveConfig.default().items, "words": words_item},
    )

    qs = FakeSettings()
    cfg.to_settings(qs)
    round_tripped = SaveConfig.from_settings(qs)

    assert round_tripped.items["words"] == words_item
    assert round_tripped.items["words"].cols == ("word", "rank")
    assert round_tripped.items["words"].k == 50


def test_round_trip_default_cols_and_k_are_preserved_as_none():
    from ssdiff_gui.utils.save_config import SaveConfig, ItemConfig

    words_item = ItemConfig(enabled=True)  # cols=None, k=None
    cfg = replace(
        SaveConfig.default(),
        items={**SaveConfig.default().items, "words": words_item},
    )

    qs = FakeSettings()
    cfg.to_settings(qs)
    round_tripped = SaveConfig.from_settings(qs)

    loaded = round_tripped.items["words"]
    assert loaded.enabled is True
    assert loaded.cols is None
    assert loaded.k is None


def test_legacy_boolean_only_upgrade_reads_cleanly():
    """QSettings with only _enabled (no _cols/_k) should load as ItemConfig(enabled=True, cols=None, k=None)."""
    from ssdiff_gui.utils.save_config import SaveConfig, ItemConfig

    # Seed only the old-style key — no _cols or _k keys
    qs = FakeSettings(initial={
        "save/tables/words_enabled": True,
    })

    cfg = SaveConfig.from_settings(qs)
    loaded = cfg.items["words"]

    assert loaded == ItemConfig(enabled=True, cols=None, k=None)


def test_string_round_trip_via_qsettings_style_values():
    """StringFakeSettings stores everything as strings (like QSettings text backend).

    Round-trip must still produce the correct typed values.
    """
    from ssdiff_gui.utils.save_config import SaveConfig, ItemConfig

    words_item = ItemConfig(enabled=True, cols=("word", "rank"), k=50)
    cfg = replace(
        SaveConfig.default(),
        report_format="txt",
        items={**SaveConfig.default().items, "words": words_item},
    )

    qs = StringFakeSettings()
    cfg.to_settings(qs)

    # Verify that the store indeed contains strings
    assert isinstance(qs._store.get("save/tables/words_k"), str)
    assert isinstance(qs._store.get("save/tables/words_enabled"), str)

    round_tripped = SaveConfig.from_settings(qs)

    loaded = round_tripped.items["words"]
    assert loaded.enabled is True
    assert loaded.cols == ("word", "rank")
    assert loaded.k == 50
    assert round_tripped.report_format == "txt"
