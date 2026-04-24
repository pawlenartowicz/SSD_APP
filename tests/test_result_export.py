"""Unit tests for result_export.export_result."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

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


# ── Shared fixtures (duplicated from test_pipeline_roundtrip.py on purpose) ──

@pytest.fixture(scope="module")
def synthetic_texts():
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
    for _ in range(100):
        tmpl = rng.choice(templates)
        words = [rng.choice(vocab) for _ in range(tmpl.count("{}"))]
        texts.append(tmpl.format(*words))
    return texts


@pytest.fixture(scope="module")
def tokenized_docs(synthetic_texts):
    docs = []
    for text in synthetic_texts:
        doc = _nlp(text)
        tokens = [t.lemma_.lower() for t in doc
                  if not t.is_stop and not t.is_punct and len(t.text) > 1]
        docs.append(tokens)
    return docs


@pytest.fixture(scope="module")
def tiny_embeddings():
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
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norms > 0, norms, 1.0)
    return Embeddings(words, vecs)


@pytest.fixture(scope="module")
def synthetic_corpus(tokenized_docs):
    from ssdiff import Corpus
    return Corpus(tokenized_docs, pretokenized=True, lang="en")


# ── Helpers ──

def _make_pls(tiny_embeddings, synthetic_corpus):
    from ssdiff import SSD
    rng = np.random.RandomState(42)
    y = rng.randn(len(synthetic_corpus.docs))
    lexicon = ["happy", "sad", "angry", "love", "hate"]
    return SSD(tiny_embeddings, synthetic_corpus, y, lexicon, window=3).fit_pls(
        n_components=1, p_method=None,
    )


def _make_pcaols(tiny_embeddings, synthetic_corpus):
    from ssdiff import SSD
    rng = np.random.RandomState(42)
    y = rng.randn(len(synthetic_corpus.docs))
    lexicon = ["happy", "sad", "angry", "love", "hate"]
    return SSD(tiny_embeddings, synthetic_corpus, y, lexicon, window=3).fit_ols(fixed_k=5)


def _make_pcaols_with_sweep(tiny_embeddings, synthetic_corpus):
    from ssdiff import SSD
    rng = np.random.RandomState(42)
    y = rng.randn(len(synthetic_corpus.docs))
    lexicon = ["happy", "sad", "angry", "love", "hate"]
    # fixed_k=None triggers the PCA-K sweep; sweep_result will be populated.
    # Use a narrow k range so the test stays fast.
    return SSD(tiny_embeddings, synthetic_corpus, y, lexicon, window=3).fit_ols(
        k_min=2, k_max=6, k_step=2,
    )


def _make_groups(tiny_embeddings, synthetic_corpus):
    from ssdiff import SSD
    n = len(synthetic_corpus.docs)
    groups = ["A"] * (n // 2) + ["B"] * (n - n // 2)
    lexicon = ["happy", "sad", "angry", "love", "hate"]
    return SSD(tiny_embeddings, synthetic_corpus, groups, lexicon, window=3).fit_groups(n_perm=100)


def _make_result_wrapper(result_path: Path, ssd_result, config_snapshot=None):
    from ssdiff_gui.models.project import Result
    result_path.mkdir(parents=True, exist_ok=True)
    r = Result(
        result_id=result_path.name,
        timestamp=datetime(2026, 4, 23, 12, 0, 0),
        result_path=result_path,
        config_snapshot=config_snapshot or {"analysis_type": "pls"},
        status="complete",
    )
    r._result = ssd_result
    return r


# ── PLS ──

@pytest.mark.slow
class TestExportResultPLS:
    def test_default_config_writes_only_pkl_script_and_report(
        self, tmp_path, tiny_embeddings, synthetic_corpus,
    ):
        from ssdiff_gui.utils.result_export import export_result
        from ssdiff_gui.utils.save_config import SaveConfig

        result = _make_result_wrapper(tmp_path / "r", _make_pls(tiny_embeddings, synthetic_corpus))
        export_result(result, result.result_path, SaveConfig.default())

        expected = {"results.pkl", "replication_script.py", "report.md"}
        assert {p.name for p in result.result_path.iterdir()} == expected

    def test_ticked_items_create_tables_dir(
        self, tmp_path, tiny_embeddings, synthetic_corpus,
    ):
        from ssdiff_gui.utils.result_export import export_result
        from ssdiff_gui.utils.save_config import ItemConfig, SaveConfig

        cfg = replace(
            SaveConfig.default(),
            items={
                **SaveConfig.default().items,
                "words": ItemConfig(enabled=True),
                "clusters": ItemConfig(enabled=True),
            },
        )

        result = _make_result_wrapper(tmp_path / "r", _make_pls(tiny_embeddings, synthetic_corpus))
        export_result(result, result.result_path, cfg)

        tables = result.result_path / "tables"
        assert tables.is_dir()
        assert (tables / "words.csv").exists()
        assert (tables / "clusters_pos.csv").exists()
        assert (tables / "clusters_neg.csv").exists()

    def test_resave_rewrites_tables_and_deletes_stale_report(
        self, tmp_path, tiny_embeddings, synthetic_corpus,
    ):
        from ssdiff_gui.utils.result_export import export_result
        from ssdiff_gui.utils.save_config import ItemConfig, SaveConfig

        result = _make_result_wrapper(tmp_path / "r", _make_pls(tiny_embeddings, synthetic_corpus))
        cfg_first = replace(
            SaveConfig.default(),
            report_format="md",
            tables_format="csv",
            items={**SaveConfig.default().items, "words": ItemConfig(enabled=True)},
        )
        export_result(result, result.result_path, cfg_first)
        assert (result.result_path / "report.md").exists()
        assert (result.result_path / "tables" / "words.csv").exists()

        cfg_second = replace(
            SaveConfig.default(),
            report_format="txt",
            tables_format="csv",
            items={**SaveConfig.default().items, "clusters": ItemConfig(enabled=True)},
        )
        export_result(result, result.result_path, cfg_second)

        assert not (result.result_path / "report.md").exists()
        assert (result.result_path / "report.txt").exists()
        tables = result.result_path / "tables"
        assert not (tables / "words.csv").exists()
        assert (tables / "clusters_pos.csv").exists()

    def test_report_disabled_removes_all_report_files(
        self, tmp_path, tiny_embeddings, synthetic_corpus,
    ):
        from ssdiff_gui.utils.result_export import export_result
        from ssdiff_gui.utils.save_config import SaveConfig

        result = _make_result_wrapper(tmp_path / "r", _make_pls(tiny_embeddings, synthetic_corpus))
        export_result(result, result.result_path, SaveConfig.default())
        assert (result.result_path / "report.md").exists()

        cfg = replace(SaveConfig.default(), report_enabled=False)
        export_result(result, result.result_path, cfg)
        assert not list(result.result_path.glob("report.*"))


# ── PCA+OLS ──

@pytest.mark.slow
class TestExportResultPCAOLS:
    def test_sweep_items_write_expected_files(
        self, tmp_path, tiny_embeddings, synthetic_corpus,
    ):
        from ssdiff_gui.utils.result_export import export_result
        from ssdiff_gui.utils.save_config import ItemConfig, SaveConfig

        cfg = replace(
            SaveConfig.default(),
            items={
                **SaveConfig.default().items,
                "sweep": ItemConfig(enabled=True),
                "sweep_plot": ItemConfig(enabled=True),
            },
        )

        result = _make_result_wrapper(
            tmp_path / "r", _make_pcaols(tiny_embeddings, synthetic_corpus),
            config_snapshot={"analysis_type": "pca_ols"},
        )
        export_result(result, result.result_path, cfg)

        tables = result.result_path / "tables"
        assert (tables / "sweep.csv").exists()
        # sweep_plot.png is NOT expected: _make_pcaols uses fixed_k=5, so
        # sweep_result is None and plot_sweep() raises RuntimeError (silently
        # swallowed by result_export). This matches the real API behaviour.

    def test_sweep_plot_writes_png(
        self, tmp_path, tiny_embeddings, synthetic_corpus,
    ):
        from ssdiff_gui.utils.result_export import export_result
        from ssdiff_gui.utils.save_config import ItemConfig, SaveConfig

        cfg = replace(
            SaveConfig.default(),
            items={**SaveConfig.default().items, "sweep_plot": ItemConfig(enabled=True)},
        )

        result = _make_result_wrapper(
            tmp_path / "r",
            _make_pcaols_with_sweep(tiny_embeddings, synthetic_corpus),
            config_snapshot={"analysis_type": "pca_ols"},
        )
        export_result(result, result.result_path, cfg)

        assert (result.result_path / "tables" / "sweep_plot.png").exists()


# ── Groups ──

@pytest.mark.slow
class TestExportResultGroup:
    def test_pairs_writes_fanned_out_or_single_file(
        self, tmp_path, tiny_embeddings, synthetic_corpus,
    ):
        from ssdiff_gui.utils.result_export import export_result
        from ssdiff_gui.utils.save_config import ItemConfig, SaveConfig

        cfg = replace(
            SaveConfig.default(),
            items={**SaveConfig.default().items, "pairs": ItemConfig(enabled=True)},
        )

        result = _make_result_wrapper(
            tmp_path / "r", _make_groups(tiny_embeddings, synthetic_corpus),
            config_snapshot={"analysis_type": "groups"},
        )
        export_result(result, result.result_path, cfg)

        tables = result.result_path / "tables"
        assert tables.exists()
        assert any(p.name.startswith("pairs") or p.is_dir() for p in tables.iterdir())


# ── cols / k flow ──

@pytest.mark.slow
class TestColsAndKFlow:
    def test_words_cols_subset_lands_in_csv(
        self, tmp_path, tiny_embeddings, synthetic_corpus,
    ):
        """cols=('word', 'rank') must produce a CSV with exactly that header."""
        import csv as _csv

        from ssdiff_gui.utils.result_export import export_result
        from ssdiff_gui.utils.save_config import ItemConfig, SaveConfig

        cfg = replace(
            SaveConfig.default(),
            items={
                **SaveConfig.default().items,
                "words": ItemConfig(enabled=True, cols=("word", "rank")),
            },
        )
        result = _make_result_wrapper(
            tmp_path / "r", _make_pls(tiny_embeddings, synthetic_corpus),
        )
        export_result(result, result.result_path, cfg)

        csv_path = result.result_path / "tables" / "words.csv"
        assert csv_path.exists()
        with open(csv_path, newline="") as f:
            header = next(_csv.reader(f))
        assert header == ["word", "rank"]

    def test_words_k_limits_rows(
        self, tmp_path, tiny_embeddings, synthetic_corpus,
    ):
        """k=3 must produce exactly 3 data rows (plus 1 header row)."""
        import csv as _csv

        from ssdiff_gui.utils.result_export import export_result
        from ssdiff_gui.utils.save_config import ItemConfig, SaveConfig

        cfg = replace(
            SaveConfig.default(),
            items={
                **SaveConfig.default().items,
                "words": ItemConfig(enabled=True, k=3),
            },
        )
        result = _make_result_wrapper(
            tmp_path / "r", _make_pls(tiny_embeddings, synthetic_corpus),
        )
        export_result(result, result.result_path, cfg)

        csv_path = result.result_path / "tables" / "words.csv"
        assert csv_path.exists()
        with open(csv_path, newline="") as f:
            rows = [r for r in _csv.reader(f) if r]
        # 1 header + 3 data rows
        assert len(rows) == 4

    def test_pairs_cols_subset_lands_in_csv(
        self, tmp_path, tiny_embeddings, synthetic_corpus,
    ):
        """cols=('contrast', 'T') on pairs must produce a CSV with exactly that header."""
        import csv as _csv

        from ssdiff_gui.utils.result_export import export_result
        from ssdiff_gui.utils.save_config import ItemConfig, SaveConfig

        cfg = replace(
            SaveConfig.default(),
            items={
                **SaveConfig.default().items,
                "pairs": ItemConfig(enabled=True, cols=("contrast", "T")),
            },
        )
        result = _make_result_wrapper(
            tmp_path / "r",
            _make_groups(tiny_embeddings, synthetic_corpus),
            config_snapshot={"analysis_type": "groups"},
        )
        export_result(result, result.result_path, cfg)

        csv_path = result.result_path / "tables" / "pairs.csv"
        assert csv_path.exists()
        with open(csv_path, newline="") as f:
            header = next(_csv.reader(f))
        assert header == ["contrast", "T"]


class TestSaveKwargs:
    """Unit tests for _save_kwargs — no fixtures needed (no I/O)."""

    def test_default_cols_and_k_omit_kwargs(self):
        """When cols=None and k=None, _save_kwargs must return an empty dict."""
        from ssdiff_gui.utils.result_export import _save_kwargs
        from ssdiff_gui.utils.save_config import ItemConfig, SaveConfig

        cfg_default = replace(
            SaveConfig.default(),
            items={
                **SaveConfig.default().items,
                "words": ItemConfig(enabled=True),  # cols=None, k=None
            },
        )
        result = _save_kwargs(cfg_default, "words")
        assert result == {}, f"Expected empty dict, got {result!r}"

        cfg_with_cols = replace(
            SaveConfig.default(),
            items={
                **SaveConfig.default().items,
                "words": ItemConfig(enabled=True, cols=("word", "rank"), k=5),
            },
        )
        result_with = _save_kwargs(cfg_with_cols, "words")
        assert result_with == {"cols": ["word", "rank"], "k": 5}
        # cols must be list (not tuple) so library error messages are unambiguous
        assert isinstance(result_with["cols"], list)
