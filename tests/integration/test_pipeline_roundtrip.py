"""Save → load round-trip for each analysis type.

Dropped from the old suite:
  - TestSaveArtifacts (6 tests) — substring-checked the report for "Plisiecki";
    missing reports are immediately visible on first export and cheap to fix.
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pytest

pytestmark = pytest.mark.spacy


class _FakeSettings:
    def __init__(self):
        self._store = {}

    def value(self, key, default=None):
        return self._store.get(key, default)

    def setValue(self, key, value):
        self._store[key] = value

    def remove(self, key):
        self._store.pop(key, None)


def _make_project(tmp_path):
    from ssdiff_gui.models.project import Project
    project_path = tmp_path / "pipeline_test"
    project_path.mkdir(parents=True, exist_ok=True)
    return Project(
        project_path=project_path,
        name="pipeline_test",
        created_date=datetime(2026, 1, 1),
        modified_date=datetime(2026, 4, 14),
    )


def _make_result_wrapper(project, result_id, config_snapshot):
    from ssdiff_gui.models.project import Result
    result_path = project.project_path / "results" / result_id
    result_path.mkdir(parents=True, exist_ok=True)
    return Result(
        result_id=result_id,
        timestamp=datetime(2026, 4, 14, 12, 0, 0),
        result_path=result_path,
        config_snapshot=config_snapshot,
        status="complete",
    )


def test_pls_roundtrip(tmp_path, tiny_embeddings, synthetic_corpus):
    from ssdiff import SSD, PLSResult
    from ssdiff_gui.utils.file_io import ProjectIO

    rng = np.random.RandomState(42)
    y = rng.randn(len(synthetic_corpus.docs))
    lexicon = ["happy", "sad", "angry", "love", "hate"]

    ssd = SSD(tiny_embeddings, synthetic_corpus, y, lexicon, window=3)
    ssd_result = ssd.fit_pls(n_components=1, p_method=None)

    project = _make_project(tmp_path)
    ProjectIO.create_project_structure(project.project_path)
    result = _make_result_wrapper(project, "20260414_pls", {
        "analysis_type": "pls", "pls_n_components": 1,
    })
    result._result = ssd_result

    ProjectIO.save_result_config(result)
    with patch("ssdiff_gui.utils.settings.app_settings", return_value=_FakeSettings()):
        ProjectIO.save_result(result)

    loaded = ProjectIO.load_result(project.project_path, "20260414_pls")
    assert loaded.status == "complete"
    assert isinstance(loaded._result, PLSResult)
    assert loaded.config_snapshot["analysis_type"] == "pls"


def test_pcaols_roundtrip(tmp_path, tiny_embeddings, synthetic_corpus):
    from ssdiff import SSD, PCAOLSResult
    from ssdiff_gui.utils.file_io import ProjectIO

    rng = np.random.RandomState(42)
    y = rng.randn(len(synthetic_corpus.docs))
    lexicon = ["happy", "sad", "angry", "love", "hate"]

    ssd = SSD(tiny_embeddings, synthetic_corpus, y, lexicon, window=3)
    ssd_result = ssd.fit_ols(fixed_k=10)

    project = _make_project(tmp_path)
    ProjectIO.create_project_structure(project.project_path)
    result = _make_result_wrapper(project, "20260414_pcaols", {
        "analysis_type": "pca_ols", "pcaols_n_components": 10,
    })
    result._result = ssd_result

    ProjectIO.save_result_config(result)
    with patch("ssdiff_gui.utils.settings.app_settings", return_value=_FakeSettings()):
        ProjectIO.save_result(result)

    loaded = ProjectIO.load_result(project.project_path, "20260414_pcaols")
    assert loaded.status == "complete"
    assert isinstance(loaded._result, PCAOLSResult)


def test_groups_roundtrip(tmp_path, tiny_embeddings, synthetic_corpus):
    from ssdiff import SSD, GroupResult
    from ssdiff_gui.utils.file_io import ProjectIO

    n = len(synthetic_corpus.docs)
    groups = ["A"] * (n // 2) + ["B"] * (n - n // 2)
    lexicon = ["happy", "sad", "angry", "love", "hate"]

    ssd = SSD(tiny_embeddings, synthetic_corpus, groups, lexicon, window=3)
    ssd_result = ssd.fit_groups(n_perm=100)

    project = _make_project(tmp_path)
    ProjectIO.create_project_structure(project.project_path)
    result = _make_result_wrapper(project, "20260414_groups", {
        "analysis_type": "groups", "groups_n_perm": 100,
    })
    result._result = ssd_result

    ProjectIO.save_result_config(result)
    with patch("ssdiff_gui.utils.settings.app_settings", return_value=_FakeSettings()):
        ProjectIO.save_result(result)

    loaded = ProjectIO.load_result(project.project_path, "20260414_groups")
    assert loaded.status == "complete"
    assert isinstance(loaded._result, GroupResult)


def test_replication_script_generated_and_executes(tmp_path, tiny_embeddings, synthetic_corpus):
    """Integration: generated script must be written + compile."""
    from ssdiff import SSD
    from ssdiff_gui.utils.file_io import ProjectIO

    rng = np.random.RandomState(42)
    y = rng.randn(len(synthetic_corpus.docs))
    lexicon = ["happy", "sad", "angry", "love", "hate"]

    ssd = SSD(tiny_embeddings, synthetic_corpus, y, lexicon, window=3)
    ssd_result = ssd.fit_pls(n_components=1, p_method=None)

    project = _make_project(tmp_path)
    ProjectIO.create_project_structure(project.project_path)
    result = _make_result_wrapper(project, "20260414_script", {
        "analysis_type": "pls", "pls_n_components": 1,
        "pls_p_method": None, "pls_random_state": "default",
        "csv_path": "/data/test.csv", "csv_encoding": "utf-8-sig",
        "text_column": "text", "language": "en",
        "selected_embedding": "glove.ssdembed",
        "concept_mode": "lexicon",
        "lexicon_tokens": ["happy", "sad", "angry", "love", "hate"],
        "context_window_size": 3, "sif_a": 1e-3,
    })
    result._result = ssd_result

    with patch("ssdiff_gui.utils.settings.app_settings", return_value=_FakeSettings()):
        ProjectIO.save_result(result)

    script_path = result.result_path / "replication_script.py"
    assert script_path.exists()
    content = script_path.read_text()
    assert "fit_pls(" in content
    assert "n_components=1" in content
    compile(content, "<replication_script>", "exec")
