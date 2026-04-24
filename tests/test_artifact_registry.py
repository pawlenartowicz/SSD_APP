"""Smoke tests for the artifact column registry."""

import pytest

ssdiff = pytest.importorskip("ssdiff", reason="ssdiff library not available")


def test_all_item_keys_have_an_entry():
    from ssdiff_gui.utils.save_config import DEFAULT_ITEM_KEYS
    from ssdiff_gui.utils.artifact_registry import ARTIFACT_COLUMNS

    for key in DEFAULT_ITEM_KEYS:
        assert key in ARTIFACT_COLUMNS, f"Missing registry entry for item_key={key!r}"


def test_sweep_plot_has_empty_columns():
    from ssdiff_gui.utils.artifact_registry import get_columns

    spec = get_columns("sweep_plot")
    assert spec.all_columns == ()
    assert spec.default_columns == ()


def test_words_has_nonempty_columns():
    from ssdiff_gui.utils.artifact_registry import get_columns

    spec = get_columns("words")
    assert "word" in spec.all_columns
    assert "rank" in spec.all_columns
    # default_columns must be a (non-strict) subset of all_columns
    assert set(spec.default_columns).issubset(set(spec.all_columns))


def test_default_columns_is_subset_of_all_columns():
    from ssdiff_gui.utils.artifact_registry import ARTIFACT_COLUMNS

    for key, spec in ARTIFACT_COLUMNS.items():
        if spec.all_columns:
            assert set(spec.default_columns).issubset(set(spec.all_columns)), (
                f"default_columns not a subset of all_columns for {key!r}: "
                f"extra = {set(spec.default_columns) - set(spec.all_columns)}"
            )


def test_unknown_key_returns_empty():
    from ssdiff_gui.utils.artifact_registry import get_columns

    spec = get_columns("nonsense")
    assert spec.all_columns == ()
    assert spec.default_columns == ()
