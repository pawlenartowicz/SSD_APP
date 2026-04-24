"""Artifact column registry consistency."""

import pytest

ssdiff = pytest.importorskip("ssdiff", reason="ssdiff library not available")


def test_all_item_keys_have_an_entry():
    """Drift guard: every SaveConfig item key must have a registry entry."""
    from ssdiff_gui.utils.save_config import DEFAULT_ITEM_KEYS
    from ssdiff_gui.utils.artifact_registry import ARTIFACT_COLUMNS
    for key in DEFAULT_ITEM_KEYS:
        assert key in ARTIFACT_COLUMNS, f"Missing registry entry for item_key={key!r}"


def test_default_columns_is_subset_of_all_columns():
    """Registry self-consistency: every default column must also be listed in all_columns."""
    from ssdiff_gui.utils.artifact_registry import ARTIFACT_COLUMNS
    for key, spec in ARTIFACT_COLUMNS.items():
        if spec.all_columns:
            extra = set(spec.default_columns) - set(spec.all_columns)
            assert not extra, (
                f"default_columns has entries missing from all_columns for {key!r}: {extra}"
            )
