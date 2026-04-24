"""SSDRunner helpers — pure / lightly-mocked logic.

Does NOT cover:
  - QThread.run() or progress signal plumbing (Qt-specific, caught by manual use)
  - _run_pls / _run_pca_ols / _run_groups orchestration (covered in integration
    via test_pipeline_roundtrip)
"""

from unittest.mock import MagicMock

import pytest

from ssdiff_gui.controllers.ssd_runner import SSDRunner
from ssdiff_gui.models.project import DEFAULT_RANDOM_SEED


@pytest.fixture
def bare_runner():
    """SSDRunner instance with no QThread init — only use for pure helpers."""
    return SSDRunner.__new__(SSDRunner)


class TestResolveRandomState:
    def test_default_keyword_returns_seed(self, bare_runner):
        assert bare_runner._resolve_random_state("default") == DEFAULT_RANDOM_SEED

    def test_numeric_string_parsed(self, bare_runner):
        assert bare_runner._resolve_random_state("42") == 42

    def test_garbage_falls_back_to_seed(self, bare_runner):
        assert bare_runner._resolve_random_state("garbage") == DEFAULT_RANDOM_SEED


class TestBuildSSD:
    """_build_ssd branches on concept_mode; test both without constructing real SSD."""

    def test_raises_without_cached_data(self, bare_runner, make_project):
        bare_runner.project = make_project()
        with pytest.raises(ValueError, match="Cached data not available"):
            bare_runner._build_ssd()

    def test_lexicon_mode_without_tokens_raises(self, bare_runner, make_project,
                                                 monkeypatch):
        p = make_project(concept_mode="lexicon", lexicon_tokens=[])
        p._emb = MagicMock()
        p._docs = [["happy"], ["sad"]]
        p._y = [1.0, 2.0]
        p._corpus = None
        p._pre_docs = None
        bare_runner.project = p

        fake_corpus = MagicMock()
        monkeypatch.setattr("ssdiff.Corpus", lambda *a, **k: fake_corpus)

        with pytest.raises(ValueError, match="Lexicon mode selected but no tokens"):
            bare_runner._build_ssd()

    def test_fulldoc_mode_builds_pseudo_lexicon_from_tokens(
        self, bare_runner, make_project, monkeypatch,
    ):
        p = make_project(concept_mode="fulldoc")
        p._emb = MagicMock()
        p._docs = [["happy", "sad"], ["happy", "angry"]]
        p._y = [1.0, 2.0]
        p._corpus = None
        p._pre_docs = None
        bare_runner.project = p

        fake_corpus = MagicMock()
        fake_corpus.pre_docs = None
        monkeypatch.setattr("ssdiff.Corpus", lambda *a, **k: fake_corpus)
        captured = {}

        def fake_ssd_ctor(emb, corpus, y, lexicon, **kwargs):
            captured["lexicon"] = lexicon
            captured["use_full_doc"] = kwargs.get("use_full_doc")
            return MagicMock()

        monkeypatch.setattr("ssdiff.SSD", fake_ssd_ctor)

        bare_runner._build_ssd()

        assert captured["use_full_doc"] is True
        assert captured["lexicon"] == {"happy", "sad", "angry"}


class TestComputeCoverage:
    def test_returns_none_when_not_lexicon_mode(self, bare_runner, make_project):
        bare_runner.project = make_project(concept_mode="fulldoc")
        cov_summary, cov_per_token = bare_runner._compute_coverage()
        assert (cov_summary, cov_per_token) == (None, None)

    def test_returns_none_when_no_corpus(self, bare_runner, make_project):
        p = make_project(concept_mode="lexicon", lexicon_tokens=["happy"])
        p._corpus = None
        bare_runner.project = p
        assert bare_runner._compute_coverage() == (None, None)

    def test_calls_coverage_summary_when_ready(self, bare_runner, make_project):
        p = make_project(concept_mode="lexicon", lexicon_tokens=["happy"])
        p._corpus = MagicMock()
        p._corpus.coverage_summary.return_value = {"summary": "ok"}
        p._corpus.token_stats.return_value = {"per_token": "ok"}
        p._y_full = [1.0, 2.0, 3.0]
        bare_runner.project = p

        cov_summary, cov_per_token = bare_runner._compute_coverage()
        assert cov_summary == {"summary": "ok"}
        assert cov_per_token == {"per_token": "ok"}
        p._corpus.coverage_summary.assert_called_once()
