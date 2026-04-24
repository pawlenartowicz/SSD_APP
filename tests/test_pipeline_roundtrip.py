"""Integration tests: spaCy → Corpus → SSD fit → ProjectIO save/load.

Skipped if spaCy / en_core_web_sm cannot be loaded (auto-install attempted by conftest).
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pytest


class FakeSettings:
    def __init__(self, initial=None):
        self._store = dict(initial or {})

    def value(self, key, default=None):
        return self._store.get(key, default)

    def setValue(self, key, value):
        self._store[key] = value

    def remove(self, key):
        self._store.pop(key, None)

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    _HAS_SPACY = True
except (ImportError, OSError):
    _HAS_SPACY = False
    _nlp = None

pytestmark = [
    pytest.mark.spacy,
    pytest.mark.skipif(not _HAS_SPACY, reason="spaCy / en_core_web_sm not available"),
]


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spacy_nlp():
    return _nlp


@pytest.fixture(scope="session")
def synthetic_texts():
    """100 short English sentences using a controlled vocabulary."""
    import random
    rng = random.Random(42)
    vocab = [
        "happy", "sad", "angry", "love", "hate",
        "joy", "fear", "trust", "surprise", "disgust",
        "good", "bad", "great", "terrible", "wonderful",
        "hope", "despair", "calm", "fury", "peace",
    ]
    templates = [
        "The {} feeling was very {}",
        "I felt {} and {} today",
        "This is a {} {} experience",
        "They expressed {} {} emotions",
    ]
    texts = []
    for i in range(100):
        tmpl = rng.choice(templates)
        words = [rng.choice(vocab) for _ in range(tmpl.count("{}"))]
        texts.append(tmpl.format(*words))
    return texts


@pytest.fixture(scope="session")
def tokenized_docs(spacy_nlp, synthetic_texts):
    """spaCy-tokenized versions (lemmatized, stopwords removed)."""
    docs = []
    for text in synthetic_texts:
        doc = spacy_nlp(text)
        tokens = [t.lemma_.lower() for t in doc
                  if not t.is_stop and not t.is_punct and len(t.text) > 1]
        docs.append(tokens)
    return docs


@pytest.fixture(scope="session")
def tiny_embeddings():
    """50-word × 10-dim ssdiff.Embeddings, seeded, L2-normalized."""
    from ssdiff import Embeddings
    rng = np.random.RandomState(42)
    words = [
        "happy", "sad", "angry", "love", "hate",
        "joy", "fear", "trust", "surprise", "disgust",
        "good", "bad", "great", "terrible", "wonderful",
        "hope", "despair", "calm", "fury", "peace",
        "bright", "dark", "warm", "cold", "gentle",
        "kind", "cruel", "brave", "weak", "strong",
        "feeling", "emotion", "experience", "express", "today",
        "the", "a", "is", "was", "very",
        "i", "they", "this", "felt", "and",
        "beautiful", "ugly", "nice", "mean", "sweet",
    ]
    vecs = rng.randn(len(words), 10).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norms > 0, norms, 1.0)
    emb = Embeddings(words, vecs)
    return emb


@pytest.fixture(scope="session")
def synthetic_corpus(tokenized_docs):
    """ssdiff.Corpus from tokenized docs (pretokenized mode)."""
    from ssdiff import Corpus
    return Corpus(tokenized_docs, pretokenized=True, lang="en")


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

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


class TestSaveArtifacts:
    """Verify that save_result() produces report.md and a replication script."""

    def test_pls_results_txt(self, tmp_path, tiny_embeddings, synthetic_corpus):
        from ssdiff import SSD
        from ssdiff_gui.utils.file_io import ProjectIO

        rng = np.random.RandomState(42)
        y = rng.randn(len(synthetic_corpus.docs))
        lexicon = ["happy", "sad", "angry", "love", "hate"]

        ssd = SSD(tiny_embeddings, synthetic_corpus, y, lexicon, window=3)
        ssd_result = ssd.fit_pls(n_components=1, p_method=None)

        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        result = _make_result_wrapper(project, "20260414_pls_report", {
            "analysis_type": "pls", "pls_n_components": 1,
        })
        result._result = ssd_result

        with patch("ssdiff_gui.utils.settings.app_settings", return_value=FakeSettings()):
            ProjectIO.save_result(result)

        report_path = result.result_path / "report.md"
        assert report_path.exists(), "report.md not generated"
        content = report_path.read_text(encoding="utf-8")
        assert "Please cite as:" in content
        assert "Plisiecki" in content
        assert "PLSResult" in content

    def test_groups_results_txt(self, tmp_path, tiny_embeddings, synthetic_corpus):
        from ssdiff import SSD
        from ssdiff_gui.utils.file_io import ProjectIO

        n = len(synthetic_corpus.docs)
        groups = ["A"] * (n // 2) + ["B"] * (n - n // 2)
        lexicon = ["happy", "sad", "angry", "love", "hate"]

        ssd = SSD(tiny_embeddings, synthetic_corpus, groups, lexicon, window=3)
        ssd_result = ssd.fit_groups(n_perm=100)

        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        result = _make_result_wrapper(project, "20260414_grp_report", {
            "analysis_type": "groups", "groups_n_perm": 100,
        })
        result._result = ssd_result

        with patch("ssdiff_gui.utils.settings.app_settings", return_value=FakeSettings()):
            ProjectIO.save_result(result)

        report_path = result.result_path / "report.md"
        assert report_path.exists(), "report.md not generated for group result"
        content = report_path.read_text(encoding="utf-8")
        assert "Please cite as:" in content
        assert "Plisiecki" in content

    def test_pcaols_results_txt(self, tmp_path, tiny_embeddings, synthetic_corpus):
        from ssdiff import SSD
        from ssdiff_gui.utils.file_io import ProjectIO

        rng = np.random.RandomState(42)
        y = rng.randn(len(synthetic_corpus.docs))
        lexicon = ["happy", "sad", "angry", "love", "hate"]

        ssd = SSD(tiny_embeddings, synthetic_corpus, y, lexicon, window=3)
        ssd_result = ssd.fit_ols(fixed_k=5)

        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        result = _make_result_wrapper(project, "20260414_pca_report", {
            "analysis_type": "pca_ols", "pcaols_n_components": 5,
        })
        result._result = ssd_result

        with patch("ssdiff_gui.utils.settings.app_settings", return_value=FakeSettings()):
            ProjectIO.save_result(result)

        report_path = result.result_path / "report.md"
        assert report_path.exists(), "report.md not generated for PCA+OLS result"
        content = report_path.read_text(encoding="utf-8")
        assert "Please cite as:" in content
        assert "Plisiecki" in content
        assert "PCA+OLS" in content

    def test_pcaols_sweep_png(self, tmp_path, tiny_embeddings, synthetic_corpus):
        from ssdiff import SSD
        from ssdiff_gui.utils.result_export import export_result
        from ssdiff_gui.utils.save_config import ItemConfig, SaveConfig
        from dataclasses import replace
        from unittest.mock import MagicMock, patch

        rng = np.random.RandomState(42)
        y = rng.randn(len(synthetic_corpus.docs))
        lexicon = ["happy", "sad", "angry", "love", "hate"]

        ssd = SSD(tiny_embeddings, synthetic_corpus, y, lexicon, window=3)
        ssd_result = ssd.fit_ols(fixed_k=5)

        project = _make_project(tmp_path)
        result = _make_result_wrapper(project, "20260414_sweep", {
            "analysis_type": "pca_ols",
        })
        result._result = ssd_result

        # Attach mock sweep so the sweep_plot branch fires.
        # Patch pickle.dump in result_export to avoid serialising the unpicklable mock,
        # and patch render_sweep_plot so we don't need real sweep_result data here.
        ssd_result.sweep_result = MagicMock()

        cfg = replace(
            SaveConfig.default(),
            items={**SaveConfig.default().items, "sweep_plot": ItemConfig(enabled=True)},
        )
        fake_pixmap = MagicMock()
        with patch("ssdiff_gui.utils.result_export.pickle.dump"), \
             patch("ssdiff_gui.utils.charts.render_sweep_plot", return_value=fake_pixmap) as mock_render:
            export_result(result, result.result_path, cfg)

        expected_path = str(result.result_path / "tables" / "sweep_plot.png")
        mock_render.assert_called_once_with(
            ssd_result.sweep_result.df_joined, ssd_result.sweep_result.best_k,
        )
        fake_pixmap.save.assert_called_once_with(expected_path, "PNG")

    def test_no_sweep_png_for_pls(self, tmp_path, tiny_embeddings, synthetic_corpus):
        from ssdiff import SSD
        from ssdiff_gui.utils.file_io import ProjectIO

        rng = np.random.RandomState(42)
        y = rng.randn(len(synthetic_corpus.docs))
        lexicon = ["happy", "sad", "angry", "love", "hate"]

        ssd = SSD(tiny_embeddings, synthetic_corpus, y, lexicon, window=3)
        ssd_result = ssd.fit_pls(n_components=1, p_method=None)

        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        result = _make_result_wrapper(project, "20260414_pls_nosweep", {
            "analysis_type": "pls", "pls_n_components": 1,
        })
        result._result = ssd_result

        with patch("ssdiff_gui.utils.settings.app_settings", return_value=FakeSettings()):
            ProjectIO.save_result(result)

        sweep_path = result.result_path / "sweep_plot.png"
        assert not sweep_path.exists(), "sweep_plot.png should not exist for PLS results"

    def test_no_sweep_png_for_explicit_k(self, tmp_path, tiny_embeddings, synthetic_corpus):
        from ssdiff import SSD
        from ssdiff_gui.utils.file_io import ProjectIO

        rng = np.random.RandomState(42)
        y = rng.randn(len(synthetic_corpus.docs))
        lexicon = ["happy", "sad", "angry", "love", "hate"]

        ssd = SSD(tiny_embeddings, synthetic_corpus, y, lexicon, window=3)
        ssd_result = ssd.fit_ols(fixed_k=5)  # explicit K — no sweep
        assert ssd_result.sweep_result is None

        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        result = _make_result_wrapper(project, "20260414_pca_nosweep", {
            "analysis_type": "pca_ols", "pcaols_n_components": 5,
        })
        result._result = ssd_result

        with patch("ssdiff_gui.utils.settings.app_settings", return_value=FakeSettings()):
            ProjectIO.save_result(result)

        sweep_path = result.result_path / "sweep_plot.png"
        assert not sweep_path.exists(), "sweep_plot.png should not exist when K was set explicitly"


class TestPipelineRoundtrip:
    def test_pls_roundtrip(self, tmp_path, tiny_embeddings, synthetic_corpus):
        from ssdiff import SSD
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
        with patch("ssdiff_gui.utils.settings.app_settings", return_value=FakeSettings()):
            ProjectIO.save_result(result)

        from ssdiff import PLSResult

        loaded = ProjectIO.load_result(project.project_path, "20260414_pls")
        assert loaded.status == "complete"
        assert loaded._result is not None
        assert isinstance(loaded._result, PLSResult)
        assert loaded.config_snapshot["analysis_type"] == "pls"
        assert loaded.config_snapshot["pls_n_components"] == 1

    def test_pcaols_roundtrip(self, tmp_path, tiny_embeddings, synthetic_corpus):
        from ssdiff import SSD
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
        with patch("ssdiff_gui.utils.settings.app_settings", return_value=FakeSettings()):
            ProjectIO.save_result(result)

        from ssdiff import PCAOLSResult

        loaded = ProjectIO.load_result(project.project_path, "20260414_pcaols")
        assert loaded.status == "complete"
        assert loaded._result is not None
        assert isinstance(loaded._result, PCAOLSResult)
        assert loaded.config_snapshot["analysis_type"] == "pca_ols"

    def test_groups_roundtrip(self, tmp_path, tiny_embeddings, synthetic_corpus):
        from ssdiff import SSD
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
        with patch("ssdiff_gui.utils.settings.app_settings", return_value=FakeSettings()):
            ProjectIO.save_result(result)

        from ssdiff import GroupResult

        loaded = ProjectIO.load_result(project.project_path, "20260414_groups")
        assert loaded.status == "complete"
        assert loaded._result is not None
        assert isinstance(loaded._result, GroupResult)
        assert loaded.config_snapshot["analysis_type"] == "groups"

    def test_replication_script_generated(self, tmp_path, tiny_embeddings, synthetic_corpus):
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
            "analysis_type": "pls",
            "pls_n_components": 1,
            "pls_p_method": None,
            "pls_random_state": "default",
            "csv_path": "/data/test.csv",
            "csv_encoding": "utf-8-sig",
            "text_column": "text",
            "language": "en",
            "selected_embedding": "glove.ssdembed",
            "concept_mode": "lexicon",
            "lexicon_tokens": ["happy", "sad", "angry", "love", "hate"],
            "context_window_size": 3,
            "sif_a": 1e-3,
        })
        result._result = ssd_result

        with patch("ssdiff_gui.utils.settings.app_settings", return_value=FakeSettings()):
            ProjectIO.save_result(result)

        script_path = result.result_path / "replication_script.py"
        assert script_path.exists()
        content = script_path.read_text()
        assert "fit_pls(" in content
        assert "n_components=1" in content
        assert 'read_csv("/data/test.csv"' in content
        compile(content, "<replication_script>", "exec")
