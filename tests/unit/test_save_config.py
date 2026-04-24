"""SaveConfig + ItemConfig — QSettings migration and string-backend round-trip.

Kept verbatim:
  - test_string_round_trip_via_qsettings_style_values (QSettings string trap)
  - test_legacy_report_keys_are_removed_on_load (migration guard)
  - test_legacy_boolean_only_upgrade_reads_cleanly (migration guard)

Dropped: test_item_config_defaults_are_none (trivial),
         test_round_trip_default_cols_and_k_are_preserved_as_none (redundant).
"""

from dataclasses import replace


from ssdiff_gui.utils.save_config import SaveConfig, ItemConfig


def test_defaults_report_ticked_others_unticked():
    cfg = SaveConfig.default()
    assert cfg.report_enabled is True
    assert cfg.report_format == "md"
    assert cfg.tables_format == "csv"
    for v in cfg.items.values():
        assert isinstance(v, ItemConfig)
        assert v.enabled is False


def test_round_trip_preserves_cols_and_k(fake_settings):
    words_item = ItemConfig(enabled=True, cols=("word", "rank"), k=50)
    cfg = replace(
        SaveConfig.default(),
        items={**SaveConfig.default().items, "words": words_item},
    )
    cfg.to_settings(fake_settings)
    round_tripped = SaveConfig.from_settings(fake_settings)
    assert round_tripped.items["words"] == words_item


def test_from_settings_falls_back_to_defaults(fake_settings):
    assert SaveConfig.from_settings(fake_settings) == SaveConfig.default()


def test_legacy_report_keys_are_removed_on_load(fake_settings):
    """VERBATIM migration guard."""
    qs = fake_settings
    for key in ("report/top_words", "report/clusters", "report/extreme_docs",
                "report/misdiagnosed", "report/snippet_preview"):
        qs.setValue(key, 10)

    SaveConfig.from_settings(qs)

    assert set(qs.removed) == {
        "report/top_words", "report/clusters", "report/extreme_docs",
        "report/misdiagnosed", "report/snippet_preview",
    }


def test_legacy_boolean_only_upgrade_reads_cleanly(fake_settings):
    """VERBATIM: pre-upgrade QSettings with only _enabled key loads as full ItemConfig."""
    fake_settings.setValue("save/tables/words_enabled", True)
    cfg = SaveConfig.from_settings(fake_settings)
    assert cfg.items["words"] == ItemConfig(enabled=True, cols=None, k=None)


def test_string_round_trip_via_qsettings_style_values(string_fake_settings):
    """VERBATIM: QSettings Linux backend returns strings; types must round-trip correctly."""
    words_item = ItemConfig(enabled=True, cols=("word", "rank"), k=50)
    cfg = replace(
        SaveConfig.default(),
        report_format="txt",
        items={**SaveConfig.default().items, "words": words_item},
    )
    cfg.to_settings(string_fake_settings)
    assert isinstance(string_fake_settings._store.get("save/tables/words_k"), str)

    round_tripped = SaveConfig.from_settings(string_fake_settings)
    loaded = round_tripped.items["words"]
    assert loaded.enabled is True
    assert loaded.cols == ("word", "rank")
    assert loaded.k == 50
    assert round_tripped.report_format == "txt"
