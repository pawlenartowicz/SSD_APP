"""SaveConfig + ItemConfig — QSettings migration and string-backend round-trip.

Kept verbatim:
  - test_string_round_trip_via_qsettings_style_values (QSettings string trap)
  - test_legacy_report_keys_are_removed_on_load (migration guard)
  - test_legacy_boolean_only_upgrade_reads_cleanly (migration guard)

Dropped: test_item_config_defaults_are_none (trivial),
         test_round_trip_default_cols_and_k_are_preserved_as_none (redundant).
"""

from dataclasses import replace

from types import MappingProxyType

from ssdiff_gui.utils.save_config import (
    DEFAULT_REPORT_SECTION_KEYS,
    ItemConfig,
    ReportSectionConfig,
    SaveConfig,
)


def test_defaults_report_ticked_others_unticked():
    cfg = SaveConfig.default()
    assert cfg.report_enabled is True
    assert cfg.report_format == "md"
    assert cfg.tables_format == "csv"
    for v in cfg.items.values():
        assert isinstance(v, ItemConfig)
        assert v.enabled is False


def test_defaults_report_sections_all_enabled_with_library_defaults():
    cfg = SaveConfig.default()
    assert set(cfg.report_sections) == set(DEFAULT_REPORT_SECTION_KEYS)
    for v in cfg.report_sections.values():
        assert isinstance(v, ReportSectionConfig)
        assert v.enabled is True
        assert v.n is None
        assert v.n_words is None
        assert v.n_snippets is None


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


def test_round_trip_preserves_report_section_overrides(fake_settings):
    clusters = ReportSectionConfig(
        enabled=True, n=20, n_words=8, n_snippets=2,
    )
    misdiagnosed = ReportSectionConfig(enabled=False, n=None)
    cfg = replace(
        SaveConfig.default(),
        report_sections={
            **SaveConfig.default().report_sections,
            "clusters": clusters,
            "misdiagnosed": misdiagnosed,
        },
    )
    cfg.to_settings(fake_settings)
    round_tripped = SaveConfig.from_settings(fake_settings)
    assert round_tripped.report_sections["clusters"] == clusters
    assert round_tripped.report_sections["misdiagnosed"] == misdiagnosed
    # Untouched sections stay at the library-default sentinel.
    assert round_tripped.report_sections["top_words"] == ReportSectionConfig()


def test_string_round_trip_preserves_report_sections(string_fake_settings):
    clusters = ReportSectionConfig(
        enabled=True, n=15, n_words=7, n_snippets=0,
    )
    cfg = replace(
        SaveConfig.default(),
        report_sections={
            **SaveConfig.default().report_sections,
            "clusters": clusters,
        },
    )
    cfg.to_settings(string_fake_settings)
    assert isinstance(
        string_fake_settings._store.get("save/report_sections/clusters_n"),
        str,
    )
    round_tripped = SaveConfig.from_settings(string_fake_settings)
    assert round_tripped.report_sections["clusters"] == clusters


def test_report_kwargs_filters_to_allowed_sections_and_skips_disabled():
    cfg = replace(
        SaveConfig.default(),
        report_sections=MappingProxyType({
            "top_words":    ReportSectionConfig(enabled=True, n=10),
            "clusters":     ReportSectionConfig(
                enabled=True, n=20, n_words=8, n_snippets=2,
            ),
            "extreme_docs": ReportSectionConfig(enabled=False),
            "misdiagnosed": ReportSectionConfig(),  # all defaults
            "top":          ReportSectionConfig(enabled=True, n=50),
        }),
    )

    kwargs = cfg.report_kwargs(
        ("top_words", "clusters", "extreme_docs", "misdiagnosed"),
    )

    # No "top" — it wasn't in the allowed list (Lexicon-only).
    assert set(kwargs) == {
        "top_words", "clusters", "extreme_docs", "misdiagnosed",
    }
    assert kwargs["top_words"] == {"n": 10}
    assert kwargs["clusters"] == {"n": 20, "n_words": 8, "n_snippets": 2}
    assert kwargs["extreme_docs"] is False
    # All-default section => True (let the library pick).
    assert kwargs["misdiagnosed"] is True


def test_report_kwargs_n_snippets_zero_passes_through():
    """0 means "drop excerpt column"; it must not collapse to "use default"."""
    cfg = replace(
        SaveConfig.default(),
        report_sections=MappingProxyType({
            **SaveConfig.default().report_sections,
            "clusters": ReportSectionConfig(
                enabled=True, n=None, n_words=None, n_snippets=0,
            ),
        }),
    )
    assert cfg.report_kwargs(("clusters",))["clusters"] == {"n_snippets": 0}
