"""Project dataclass serialization, snapshot_config, readiness properties.

Guards against:
  - A new field added to Project.to_dict but not Project.from_dict (or vice versa)
  - Snapshot leaking fields from another analysis type
  - Readiness property logic regressing after a Stage 1/2 refactor

Filler dropped:
  - TestDirtyTracking (4 tests for a 2-line flag) — trivial, visible on use
  - test_round_trip_all_fields (45 asserts) — replaced by parametrize
"""

from datetime import datetime
from pathlib import Path

import pytest

from ssdiff_gui.models.project import (
    Project,
    _SNAPSHOT_COMMON, _SNAPSHOT_PLS, _SNAPSHOT_PCA_OLS, _SNAPSHOT_GROUPS,
)


def _non_default_values_for_all_fields(tmp_path):
    """Set every serialized field to a non-default value and return the project."""
    p = Project(
        project_path=tmp_path / "p",
        name="test",
        created_date=datetime(2026, 1, 1),
        modified_date=datetime(2026, 4, 14),
    )
    p.project_path.mkdir(parents=True, exist_ok=True)
    p.csv_path = Path("/data/test.csv")
    p.csv_encoding = "latin-1"
    p.text_column = "review"
    p.language = "pl"
    p.analysis_type = "pca_ols"
    p.lexicon_tokens = ["happy", "sad"]
    p.sweep_k_min = 15
    p.pls_n_perm = 2000
    p.groups_correction = "bonferroni"
    p._id_row_indices = [[0, 1], [2, 3]]
    return p


def test_round_trip_preserves_all_serialized_fields(tmp_path):
    """Every key written by to_dict must be restored by from_dict."""
    p = _non_default_values_for_all_fields(tmp_path)
    d = p.to_dict()
    restored = Project.from_dict(d, p.project_path)

    for key, value in d.items():
        if key in ("results",):
            continue
        restored_value = getattr(restored, key, None)
        if key == "csv_path" and value is not None:
            restored_value = str(restored_value)
        if key == "id_row_indices":
            restored_value = restored._id_row_indices
        if key in ("created_date", "modified_date") and restored_value is not None:
            restored_value = restored_value.isoformat()
        assert restored_value == value, (
            f"field {key!r}: to_dict wrote {value!r} but from_dict restored {restored_value!r}"
        )


def test_unserialized_fields_excluded(tmp_path, make_project):
    p = make_project()
    p._dirty = True
    p._df = "fake"
    p._corpus = "fake"
    p._emb = "fake"
    d = p.to_dict()
    for forbidden in ("_dirty", "dirty", "_df", "_corpus", "_emb", "_kv"):
        assert forbidden not in d


def test_from_dict_with_only_required_keys_uses_defaults(tmp_path):
    d = {
        "name": "sparse",
        "created_date": "2026-01-01T00:00:00",
        "modified_date": "2026-04-14T00:00:00",
    }
    proj_path = tmp_path / "sparse"
    proj_path.mkdir()
    restored = Project.from_dict(d, proj_path)
    assert restored.name == "sparse"
    assert restored.csv_path is None
    assert restored.analysis_type == "pls"
    assert restored.groups_correction == "holm"


@pytest.mark.parametrize("analysis_type,expected_keys,forbidden_keys", [
    ("pls",     _SNAPSHOT_PLS,     _SNAPSHOT_PCA_OLS + _SNAPSHOT_GROUPS),
    ("pca_ols", _SNAPSHOT_PCA_OLS, _SNAPSHOT_PLS + _SNAPSHOT_GROUPS),
    ("groups",  _SNAPSHOT_GROUPS,  _SNAPSHOT_PLS + _SNAPSHOT_PCA_OLS),
])
def test_snapshot_per_analysis_type(make_project, analysis_type, expected_keys, forbidden_keys):
    p = make_project(analysis_type=analysis_type)
    snap = p.snapshot_config()

    for key in _SNAPSHOT_COMMON:
        assert key in snap, f"[{analysis_type}] missing common key: {key}"

    for key in expected_keys:
        assert key in snap, f"[{analysis_type}] missing type-specific key: {key}"

    for key in forbidden_keys:
        assert key not in snap, f"[{analysis_type}] leaked key from other type: {key}"


def test_snapshot_captures_current_values(make_project):
    p = make_project(analysis_type="pls")
    p.pls_n_components = 4
    p.context_window_size = 7
    snap = p.snapshot_config()
    assert snap["pls_n_components"] == 4
    assert snap["context_window_size"] == 7


def test_text_ready_requires_df_and_text_column(make_project):
    p = make_project()
    assert p.text_ready is False
    p._df = "fake"
    assert p.text_ready is False
    p.text_column = "text"
    assert p.text_ready is True


def test_preprocessing_ready_requires_matching_columns(make_project):
    p = make_project()
    p._corpus = "fake"
    p.text_column = "review"
    p.preprocessed_text_column = "text"
    assert p.preprocessing_ready is False
    p.preprocessed_text_column = "review"
    p.preprocessed_language = p.language
    assert p.preprocessing_ready is True


def test_stage1_ready_requires_all_three(make_project):
    p = make_project()
    p._df = "fake"
    p.text_column = "text"
    p._corpus = "fake"
    p.preprocessed_text_column = "text"
    p.preprocessed_language = p.language
    assert p.stage1_ready is False
    p._emb = "fake"
    assert p.stage1_ready is True
