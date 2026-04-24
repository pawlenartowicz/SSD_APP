"""Tests for path validators (Validator class) and Project.validate_* methods.

Path validators test against the Validator class in utils/validators.py.
Text/outcome/lexicon validators test against Project instance methods.
"""

import pandas as pd

from ssdiff_gui.utils.validators import Validator
from ssdiff_gui.models.project import Project


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_project(tmp_path, **kwargs):
    """Create a minimal Project for validation tests."""
    defaults = dict(
        project_path=tmp_path,
        name="test",
        created_date=None,
        modified_date=None,
    )
    defaults.update(kwargs)
    from datetime import datetime
    if defaults["created_date"] is None:
        defaults["created_date"] = datetime(2026, 1, 1)
    if defaults["modified_date"] is None:
        defaults["modified_date"] = datetime(2026, 1, 1)
    return Project(**defaults)


def _make_df(n=100, text_col="text", outcome_col="score", id_col=None,
             empty_pct=0.0, outcome_std=1.0):
    """Build a synthetic DataFrame for validator tests."""
    import random
    random.seed(42)

    texts = [f"This is a sample text number {i} with enough words" for i in range(n)]
    n_empty = int(n * empty_pct)
    for i in range(n_empty):
        texts[i] = "" if i % 2 == 0 else None

    data = {text_col: texts}

    if outcome_col:
        base = [random.gauss(3.0, outcome_std) for _ in range(n)]
        data[outcome_col] = base

    if id_col:
        data[id_col] = [f"id_{i}" for i in range(n)]

    return pd.DataFrame(data)


# ===================================================================
# validate_embeddings_path (Validator — static, 4 tests)
# ===================================================================

class TestValidateEmbeddingsPath:
    def test_empty_path(self):
        errors, warnings = Validator.validate_embeddings_path("")
        assert len(errors) == 1
        assert "No embedding file specified" in errors[0]

    def test_missing_file(self):
        errors, warnings = Validator.validate_embeddings_path("/nonexistent/file.bin")
        assert len(errors) == 1
        assert "File not found" in errors[0]

    def test_unusual_extension(self, tmp_path):
        f = tmp_path / "embeddings.xyz"
        f.write_text("dummy")
        errors, warnings = Validator.validate_embeddings_path(str(f))
        assert len(errors) == 0
        assert len(warnings) == 1
        assert "Unusual file extension: .xyz" in warnings[0]

    def test_valid_file(self, tmp_path):
        f = tmp_path / "embeddings.bin"
        f.write_bytes(b"\x00" * 1024)
        errors, warnings = Validator.validate_embeddings_path(str(f))
        assert len(errors) == 0
        assert len(warnings) == 0


# ===================================================================
# validate_csv_path (Validator — static, 5 tests)
# ===================================================================

class TestValidateCsvPath:
    def test_empty_path(self):
        errors, warnings = Validator.validate_csv_path("")
        assert len(errors) == 1
        assert "No CSV file specified" in errors[0]

    def test_missing_file(self):
        errors, warnings = Validator.validate_csv_path("/nonexistent/data.csv")
        assert len(errors) == 1
        assert "File not found" in errors[0]

    def test_unusual_extension(self, tmp_path):
        f = tmp_path / "data.xlsx"
        f.write_text("dummy")
        errors, warnings = Validator.validate_csv_path(str(f))
        assert len(errors) == 0
        assert len(warnings) == 1
        assert "Unusual file extension: .xlsx" in warnings[0]

    def test_valid_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        errors, warnings = Validator.validate_csv_path(str(f))
        assert len(errors) == 0
        assert len(warnings) == 0

    def test_valid_tsv(self, tmp_path):
        f = tmp_path / "data.tsv"
        f.write_text("a\tb\n1\t2\n")
        errors, warnings = Validator.validate_csv_path(str(f))
        assert len(errors) == 0
        assert len(warnings) == 0


# ===================================================================
# Project.validate_text() (11 tests)
# ===================================================================

class TestValidateText:
    def test_no_dataset_loaded(self, tmp_path):
        p = _make_project(tmp_path)
        # _df is None by default
        errors, warnings, id_stats = p.validate_text()
        assert any("No dataset loaded" in e for e in errors)

    def test_missing_text_column(self, tmp_path):
        p = _make_project(tmp_path)
        p._df = _make_df(text_col="text")
        p.text_column = "nonexistent"
        errors, warnings, _ = p.validate_text()
        assert len(errors) == 1
        assert "Text column 'nonexistent' not found" in errors[0]

    def test_high_empty_pct(self, tmp_path):
        """60% empty → error (threshold >50%)."""
        p = _make_project(tmp_path)
        p._df = _make_df(n=100, empty_pct=0.6)
        p.text_column = "text"
        errors, warnings, _ = p.validate_text()
        empty_errors = [e for e in errors if "empty or missing" in e]
        assert len(empty_errors) == 1
        assert "60.0%" in empty_errors[0]

    def test_moderate_empty_pct(self, tmp_path):
        """20% empty → warning (>10%, <=50%)."""
        p = _make_project(tmp_path)
        p._df = _make_df(n=100, empty_pct=0.2)
        p.text_column = "text"
        errors, warnings, _ = p.validate_text()
        assert len(errors) == 0
        empty_warnings = [w for w in warnings if "empty or missing" in w]
        assert len(empty_warnings) == 1
        assert "20.0%" in empty_warnings[0]

    def test_low_empty_pct_no_warning(self, tmp_path):
        """5% empty → no warning."""
        p = _make_project(tmp_path)
        p._df = _make_df(n=100, empty_pct=0.05)
        p.text_column = "text"
        errors, warnings, _ = p.validate_text()
        assert len(errors) == 0
        assert not any("empty or missing" in w for w in warnings)

    def test_too_few_rows(self, tmp_path):
        """20 rows → error (<30)."""
        p = _make_project(tmp_path)
        p._df = _make_df(n=20)
        p.text_column = "text"
        errors, warnings, _ = p.validate_text()
        assert any("at least 30" in e for e in errors)

    def test_small_sample_warning(self, tmp_path):
        """50 rows → warning (30-99)."""
        p = _make_project(tmp_path)
        p._df = _make_df(n=50)
        p.text_column = "text"
        errors, warnings, _ = p.validate_text()
        assert len(errors) == 0
        assert any("Small sample" in w for w in warnings)

    def test_id_stats_returned(self, tmp_path):
        p = _make_project(tmp_path)
        p._df = _make_df(n=100, id_col="participant")
        p.text_column = "text"
        p.id_column = "participant"
        _, _, id_stats = p.validate_text()
        assert id_stats is not None
        assert id_stats["n_unique_ids"] == 100
        assert bool(id_stats["has_duplicates"]) is False
        assert id_stats["avg_texts_per_id"] == 1.0

    def test_id_stats_with_duplicates(self, tmp_path):
        p = _make_project(tmp_path)
        df = _make_df(n=100, id_col="participant")
        df.loc[:9, "participant"] = "id_0"
        p._df = df
        p.text_column = "text"
        p.id_column = "participant"
        _, _, id_stats = p.validate_text()
        assert bool(id_stats["has_duplicates"]) is True
        assert id_stats["n_unique_ids"] == 91

    def test_no_id_stats_when_no_id_col(self, tmp_path):
        p = _make_project(tmp_path)
        p._df = _make_df(n=100)
        p.text_column = "text"
        _, _, id_stats = p.validate_text()
        assert id_stats is None


# ===================================================================
# Project.validate_outcome() (12 tests)
# ===================================================================

class TestValidateOutcome:
    def test_no_dataset(self, tmp_path):
        p = _make_project(tmp_path)
        errors, warnings = p.validate_outcome()
        assert any("No dataset loaded" in e for e in errors)

    def test_missing_outcome_column(self, tmp_path):
        p = _make_project(tmp_path)
        p._df = _make_df(n=100)
        p.outcome_column = "nonexistent"
        p.analysis_type = "pls"
        errors, warnings = p.validate_outcome()
        assert any("not found" in e for e in errors)

    def test_non_numeric_outcome_high(self, tmp_path):
        """61% non-numeric → error with percentage."""
        p = _make_project(tmp_path)
        df = _make_df(n=100)
        vals = list(df["score"].values)
        for i in range(61):
            vals[i] = "not_a_number"
        df["score"] = pd.array(vals, dtype="object")
        p._df = df
        p.outcome_column = "score"
        p.analysis_type = "pls"
        errors, warnings = p.validate_outcome()
        nn_errors = [e for e in errors if "non-numeric" in e]
        assert len(nn_errors) == 1
        assert "61.0%" in nn_errors[0]

    def test_non_numeric_outcome_low(self, tmp_path):
        """6% non-numeric → warning with count."""
        p = _make_project(tmp_path)
        df = _make_df(n=100)
        vals = list(df["score"].values)
        for i in range(6):
            vals[i] = "not_a_number"
        df["score"] = pd.array(vals, dtype="object")
        p._df = df
        p.outcome_column = "score"
        p.analysis_type = "pls"
        errors, warnings = p.validate_outcome()
        assert not any("non-numeric" in e for e in errors)
        nn_warnings = [w for w in warnings if "non-numeric" in w]
        assert len(nn_warnings) == 1
        assert "6 outcome values" in nn_warnings[0]

    def test_near_zero_variance(self, tmp_path):
        """std=0.001 → error."""
        p = _make_project(tmp_path)
        p._df = _make_df(n=100, outcome_std=0.001)
        p.outcome_column = "score"
        p.analysis_type = "pls"
        errors, warnings = p.validate_outcome()
        assert any("near-zero variance" in e for e in errors)

    def test_low_variance_warning(self, tmp_path):
        """std=0.05 → warning."""
        p = _make_project(tmp_path)
        p._df = _make_df(n=100, outcome_std=0.05)
        p.outcome_column = "score"
        p.analysis_type = "pls"
        errors, warnings = p.validate_outcome()
        assert len(errors) == 0
        assert any("low variance" in w for w in warnings)

    def test_too_few_valid(self, tmp_path):
        """20 valid → error."""
        p = _make_project(tmp_path)
        df = _make_df(n=100)
        df.loc[:79, "score"] = None
        p._df = df
        p.outcome_column = "score"
        p.analysis_type = "pls"
        errors, warnings = p.validate_outcome()
        assert any("Only 20 valid samples" in e for e in errors)

    def test_small_valid_sample(self, tmp_path):
        """50 valid → warning."""
        p = _make_project(tmp_path)
        df = _make_df(n=100)
        df.loc[:49, "score"] = None
        p._df = df
        p.outcome_column = "score"
        p.analysis_type = "pls"
        errors, warnings = p.validate_outcome()
        assert not any("valid samples" in e for e in errors)
        assert any("Small sample" in w for w in warnings)

    def test_groups_too_few(self, tmp_path):
        """1 group → error."""
        p = _make_project(tmp_path)
        df = pd.DataFrame({"text": ["t"] * 100, "grp": ["A"] * 100})
        p._df = df
        p.analysis_type = "groups"
        p.group_column = "grp"
        errors, warnings = p.validate_outcome()
        assert any("at least 2 groups" in e for e in errors)

    def test_groups_small_members(self, tmp_path):
        """Group with <10 members → error."""
        p = _make_project(tmp_path)
        df = pd.DataFrame({
            "text": ["t"] * 100,
            "grp": ["A"] * 95 + ["B"] * 5,
        })
        p._df = df
        p.analysis_type = "groups"
        p.group_column = "grp"
        errors, warnings = p.validate_outcome()
        assert any("<10 members" in e for e in errors)

    def test_clean_continuous(self, tmp_path):
        p = _make_project(tmp_path)
        p._df = _make_df(n=200)
        p.outcome_column = "score"
        p.analysis_type = "pls"
        errors, warnings = p.validate_outcome()
        assert len(errors) == 0
        assert len(warnings) == 0

    def test_constant_outcome(self, tmp_path):
        """All-identical outcome values → near-zero variance error."""
        p = _make_project(tmp_path)
        df = _make_df(n=100)
        df["score"] = 3.0
        p._df = df
        p.outcome_column = "score"
        p.analysis_type = "pls"
        errors, warnings = p.validate_outcome()
        assert any("near-zero variance" in e for e in errors)


# ===================================================================
# Project.validate_lexicon() (9 tests)
# ===================================================================

class TestValidateLexicon:
    def _make_mock_kv(self, words):
        """Create a mock _kv with key_to_index."""
        from unittest.mock import MagicMock
        kv = MagicMock()
        kv.key_to_index = {w: i for i, w in enumerate(words)}
        return kv

    def _make_docs(self, n=100, vocab_tokens=None):
        """Create synthetic tokenized docs."""
        if vocab_tokens is None:
            vocab_tokens = ["happy", "sad", "angry", "love", "hate",
                            "joy", "fear", "trust", "surprise", "disgust"]
        import random
        rng = random.Random(42)
        docs = []
        for _ in range(n):
            doc_len = rng.randint(5, 20)
            doc = [rng.choice(vocab_tokens + ["the", "a", "is", "of", "and"])
                   for _ in range(doc_len)]
            docs.append(doc)
        return docs

    def test_empty_lexicon(self, tmp_path):
        p = _make_project(tmp_path)
        p.lexicon_tokens = []
        errors, warnings = p.validate_lexicon()
        assert any("Lexicon is empty" in e for e in errors)

    def test_no_embeddings(self, tmp_path):
        p = _make_project(tmp_path)
        p.lexicon_tokens = ["happy"]
        p._kv = None
        errors, warnings = p.validate_lexicon()
        assert any("No embeddings loaded" in e for e in errors)

    def test_all_oov(self, tmp_path):
        p = _make_project(tmp_path)
        p.lexicon_tokens = ["zzz", "yyy"]
        p._kv = self._make_mock_kv(["happy", "sad"])
        p._docs = self._make_docs()
        errors, warnings = p.validate_lexicon()
        assert any("None of the lexicon tokens" in e for e in errors)

    def test_partial_oov_high(self, tmp_path):
        """75% OOV → warning with percentage."""
        p = _make_project(tmp_path)
        p.lexicon_tokens = ["happy", "zzz1", "zzz2", "zzz3"]
        p._kv = self._make_mock_kv(["happy", "sad"])
        p._docs = self._make_docs()
        errors, warnings = p.validate_lexicon()
        assert len(errors) == 0
        assert any("75.0%" in w and "not in vocabulary" in w for w in warnings)

    def test_partial_oov_low(self, tmp_path):
        """25% OOV → warning with count."""
        p = _make_project(tmp_path)
        p.lexicon_tokens = ["happy", "sad", "angry", "zzz"]
        p._kv = self._make_mock_kv(["happy", "sad", "angry", "love", "hate"])
        p._docs = self._make_docs()
        errors, warnings = p.validate_lexicon()
        assert len(errors) == 0
        assert any("1 tokens not in vocabulary" in w for w in warnings)

    def test_low_coverage(self, tmp_path):
        """0% doc coverage → error (<10%)."""
        p = _make_project(tmp_path)
        p.lexicon_tokens = ["rarissimo"]
        p._kv = self._make_mock_kv(["rarissimo"])
        p._docs = [["the", "quick", "brown"] for _ in range(100)]
        errors, warnings = p.validate_lexicon()
        assert any("Very low coverage" in e for e in errors)

    def test_moderate_coverage(self, tmp_path):
        """20% coverage → warning (10-30%)."""
        p = _make_project(tmp_path)
        p.lexicon_tokens = ["happy"]
        p._kv = self._make_mock_kv(["happy"])
        docs = [["other", "words"] for _ in range(100)]
        for i in range(20):
            docs[i] = ["happy", "text"]
        p._docs = docs
        errors, warnings = p.validate_lexicon()
        assert len(errors) == 0
        assert any("Low coverage" in w and "20.0%" in w for w in warnings)

    def test_very_small_lexicon(self, tmp_path):
        """2 valid tokens → warning."""
        p = _make_project(tmp_path)
        p.lexicon_tokens = ["happy", "sad"]
        p._kv = self._make_mock_kv(["happy", "sad"])
        p._docs = self._make_docs()
        errors, warnings = p.validate_lexicon()
        assert any("Very small lexicon (2 tokens)" in w for w in warnings)

    def test_good_lexicon(self, tmp_path):
        """5+ tokens, good coverage → no errors, no warnings."""
        p = _make_project(tmp_path)
        p.lexicon_tokens = ["happy", "sad", "angry", "love", "hate"]
        p._kv = self._make_mock_kv(
            ["happy", "sad", "angry", "love", "hate",
             "joy", "fear", "trust", "surprise", "disgust"]
        )
        p._docs = self._make_docs()
        errors, warnings = p.validate_lexicon()
        assert len(errors) == 0
        assert len(warnings) == 0
