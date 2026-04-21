"""Tests for Project and Result dataclass serialization, dirty tracking,
snapshot_config, and computed readiness.
"""

from datetime import datetime
from pathlib import Path


from ssdiff_gui.models.project import (
    Project, Result,
    _SNAPSHOT_COMMON, _SNAPSHOT_PLS, _SNAPSHOT_PCA_OLS, _SNAPSHOT_GROUPS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_project(tmp_path, **kwargs):
    defaults = dict(
        project_path=tmp_path / "proj",
        name="test",
        created_date=datetime(2026, 1, 1),
        modified_date=datetime(2026, 4, 14),
    )
    defaults.update(kwargs)
    defaults["project_path"].mkdir(parents=True, exist_ok=True)
    return Project(**defaults)


def _make_result(tmp_path, **kwargs):
    result_path = tmp_path / "results" / "20260414_120000"
    result_path.mkdir(parents=True, exist_ok=True)
    defaults = dict(
        result_id="20260414_120000",
        timestamp=datetime(2026, 4, 14, 12, 0, 0),
        result_path=result_path,
        config_snapshot={"analysis_type": "pls", "pls_n_components": 1},
    )
    defaults.update(kwargs)
    return Result(**defaults)


# ===================================================================
# Project serialization (~6 tests)
# ===================================================================

class TestProjectSerialization:
    def test_round_trip_minimal(self, tmp_path):
        """Empty Project survives to_dict/from_dict."""
        p = _make_project(tmp_path)
        d = p.to_dict()
        restored = Project.from_dict(d, p.project_path)
        assert restored.name == "test"
        assert restored.created_date == datetime(2026, 1, 1)
        assert restored.modified_date == datetime(2026, 4, 14)

    def test_round_trip_all_fields(self, tmp_path):
        """Set every serialized field, verify round-trip."""
        p = _make_project(tmp_path)
        p.csv_path = Path("/data/test.csv")
        p.csv_encoding = "latin-1"
        p.text_column = "review"
        p.id_column = "user_id"
        p.n_rows = 500
        p.n_valid = 480
        p.language = "pl"
        p.spacy_model = "pl_core_news_lg"
        p.input_mode = "custom"
        p.stopword_mode = "custom"
        p.custom_stopwords = ["i", "a"]
        p.preprocessed_text_column = "review"
        p.n_docs_processed = 480
        p.total_tokens = 5000
        p.mean_words_before_stopwords = 12.5
        p.selected_embedding = "glove.ssdembed"
        p.vocab_size = 100000
        p.embedding_dim = 300
        p.l2_normalized = True
        p.abtt = 1
        p.emb_coverage_pct = 95.0
        p.emb_n_oov = 50
        p.analysis_type = "pca_ols"
        p.concept_mode = "fulldoc"
        p.outcome_column = "score"
        p.group_column = "group"
        p.lexicon_tokens = ["happy", "sad"]
        p.min_hits_per_doc = 2
        p.drop_no_hits = False
        p.fulldoc_stoplist = ["the"]
        p.context_window_size = 5
        p.sif_a = 2e-3
        p.pls_n_components = 2
        p.pls_pca_preprocess = 50
        p.pls_p_method = "perm"
        p.pls_n_perm = 2000
        p.pls_n_splits = 100
        p.pls_split_ratio = 0.6
        p.pls_random_state = "42"
        p.pcaols_n_components = 40
        p.sweep_k_min = 10
        p.sweep_k_max = 200
        p.sweep_k_step = 5
        p.groups_n_perm = 10000
        p.groups_correction = "bonferroni"
        p.groups_median_split = True
        p.groups_random_state = "99"

        d = p.to_dict()
        restored = Project.from_dict(d, p.project_path)

        # Verify every field that was explicitly set
        assert restored.csv_path == Path("/data/test.csv")
        assert restored.csv_encoding == "latin-1"
        assert restored.text_column == "review"
        assert restored.id_column == "user_id"
        assert restored.n_rows == 500
        assert restored.n_valid == 480
        assert restored.language == "pl"
        assert restored.spacy_model == "pl_core_news_lg"
        assert restored.input_mode == "custom"
        assert restored.stopword_mode == "custom"
        assert restored.custom_stopwords == ["i", "a"]
        assert restored.preprocessed_text_column == "review"
        assert restored.n_docs_processed == 480
        assert restored.total_tokens == 5000
        assert restored.mean_words_before_stopwords == 12.5
        assert restored.selected_embedding == "glove.ssdembed"
        assert restored.vocab_size == 100000
        assert restored.embedding_dim == 300
        assert restored.l2_normalized is True
        assert restored.abtt == 1
        assert restored.emb_coverage_pct == 95.0
        assert restored.emb_n_oov == 50
        assert restored.analysis_type == "pca_ols"
        assert restored.concept_mode == "fulldoc"
        assert restored.outcome_column == "score"
        assert restored.group_column == "group"
        assert restored.lexicon_tokens == ["happy", "sad"]
        assert restored.min_hits_per_doc == 2
        assert restored.drop_no_hits is False
        assert restored.fulldoc_stoplist == ["the"]
        assert restored.context_window_size == 5
        assert restored.sif_a == 2e-3
        assert restored.pls_n_components == 2
        assert restored.pls_pca_preprocess == 50
        assert restored.pls_p_method == "perm"
        assert restored.pls_n_perm == 2000
        assert restored.pls_n_splits == 100
        assert restored.pls_split_ratio == 0.6
        assert restored.pls_random_state == "42"
        assert restored.pcaols_n_components == 40
        assert restored.sweep_k_min == 10
        assert restored.sweep_k_max == 200
        assert restored.sweep_k_step == 5
        assert restored.groups_n_perm == 10000
        assert restored.groups_correction == "bonferroni"
        assert restored.groups_median_split is True
        assert restored.groups_random_state == "99"

    def test_csv_path_coerced_to_path(self, tmp_path):
        """str → Path coercion in from_dict."""
        p = _make_project(tmp_path)
        p.csv_path = Path("/data/test.csv")
        d = p.to_dict()
        assert isinstance(d["csv_path"], str)
        restored = Project.from_dict(d, p.project_path)
        assert isinstance(restored.csv_path, Path)

    def test_id_row_indices_persisted(self, tmp_path):
        p = _make_project(tmp_path)
        p._id_row_indices = [[0, 1], [2, 3], [4]]
        d = p.to_dict()
        assert d["id_row_indices"] == [[0, 1], [2, 3], [4]]
        restored = Project.from_dict(d, p.project_path)
        assert restored._id_row_indices == [[0, 1], [2, 3], [4]]

    def test_unserialized_fields_excluded(self, tmp_path):
        p = _make_project(tmp_path)
        p._dirty = True
        p._df = "fake_df"
        p._corpus = "fake_corpus"
        d = p.to_dict()
        assert "_dirty" not in d
        assert "dirty" not in d
        assert "_df" not in d
        assert "_corpus" not in d
        assert "_emb" not in d

    def test_results_serialized_as_ids(self, tmp_path):
        p = _make_project(tmp_path)
        r = _make_result(tmp_path)
        p.results.append(r)
        d = p.to_dict()
        assert d["results"] == ["20260414_120000"]

    def test_from_dict_missing_keys_uses_defaults(self, tmp_path):
        """from_dict with only required keys → defaults for everything else."""
        d = {
            "name": "sparse",
            "created_date": "2026-01-01T00:00:00",
            "modified_date": "2026-04-14T00:00:00",
        }
        proj_path = tmp_path / "sparse"
        proj_path.mkdir()
        restored = Project.from_dict(d, proj_path)
        assert restored.name == "sparse"
        # All optional fields should have their defaults
        assert restored.csv_path is None
        assert restored.csv_encoding == "utf-8-sig"
        assert restored.language == "en"
        assert restored.analysis_type == "pls"
        assert restored.pls_n_components == 1
        assert restored.groups_correction == "holm"
        assert restored.sweep_k_min == Project.__dataclass_fields__["sweep_k_min"].default


# ===================================================================
# Result serialization (~4 tests)
# ===================================================================

class TestResultSerialization:
    def test_round_trip(self, tmp_path):
        r = _make_result(tmp_path, name="my run", status="complete")
        d = r.to_dict()
        restored = Result.from_dict(d, r.result_path)
        assert restored.result_id == "20260414_120000"
        assert restored.name == "my run"
        assert restored.status == "complete"
        assert restored.config_snapshot["analysis_type"] == "pls"
        assert restored.config_snapshot["pls_n_components"] == 1
        assert restored.timestamp == datetime(2026, 4, 14, 12, 0, 0)
        assert restored.error_message is None

    def test_round_trip_with_error(self, tmp_path):
        """error_message survives round-trip."""
        r = _make_result(tmp_path, status="error",
                         error_message="OOM during fit")
        d = r.to_dict()
        restored = Result.from_dict(d, r.result_path)
        assert restored.status == "error"
        assert restored.error_message == "OOM during fit"

    def test_status_recovery_running(self, tmp_path):
        """status 'running' + no results.pkl → 'interrupted'."""
        r = _make_result(tmp_path, status="running")
        d = r.to_dict()
        # No results.pkl exists in result_path
        restored = Result.from_dict(d, r.result_path)
        assert restored.status == "interrupted"

    def test_status_running_with_pkl_stays_running(self, tmp_path):
        """status 'running' + results.pkl exists → stays 'running'."""
        r = _make_result(tmp_path, status="running")
        (r.result_path / "results.pkl").write_bytes(b"fake")
        d = r.to_dict()
        restored = Result.from_dict(d, r.result_path)
        assert restored.status == "running"

    def test_config_snapshot_preserved(self, tmp_path):
        snap = {
            "analysis_type": "groups",
            "groups_n_perm": 5000,
            "groups_correction": "holm",
            "lexicon_tokens": ["a", "b"],
        }
        r = _make_result(tmp_path, config_snapshot=snap)
        d = r.to_dict()
        restored = Result.from_dict(d, r.result_path)
        assert restored.config_snapshot == snap


# ===================================================================
# Dirty tracking (~4 tests)
# ===================================================================

class TestDirtyTracking:
    def test_starts_clean(self, tmp_path):
        p = _make_project(tmp_path)
        assert p._dirty is False

    def test_mark_dirty(self, tmp_path):
        p = _make_project(tmp_path)
        p.mark_dirty()
        assert p._dirty is True

    def test_mark_clean(self, tmp_path):
        p = _make_project(tmp_path)
        p.mark_dirty()
        p.mark_clean()
        assert p._dirty is False

    def test_dirty_not_serialized(self, tmp_path):
        p = _make_project(tmp_path)
        p.mark_dirty()
        d = p.to_dict()
        assert "_dirty" not in d
        assert "dirty" not in d


# ===================================================================
# snapshot_config (~5 tests)
# ===================================================================

class TestSnapshotConfig:
    def test_common_fields_always_present(self, tmp_path):
        """Every key in _SNAPSHOT_COMMON must appear in the snapshot."""
        p = _make_project(tmp_path)
        p.csv_path = Path("/data/test.csv")
        p.language = "en"
        p.analysis_type = "pls"
        snap = p.snapshot_config()
        for key in _SNAPSHOT_COMMON:
            assert key in snap, f"missing common key: {key}"

    def test_pls_fields_included(self, tmp_path):
        """All _SNAPSHOT_PLS keys present when analysis_type='pls'."""
        p = _make_project(tmp_path)
        p.analysis_type = "pls"
        p.pls_n_components = 3
        p.pls_p_method = "perm"
        snap = p.snapshot_config()
        for key in _SNAPSHOT_PLS:
            assert key in snap, f"missing PLS key: {key}"
        assert snap["pls_n_components"] == 3
        assert snap["pls_p_method"] == "perm"

    def test_pcaols_fields_included(self, tmp_path):
        """All _SNAPSHOT_PCA_OLS keys present when analysis_type='pca_ols'."""
        p = _make_project(tmp_path)
        p.analysis_type = "pca_ols"
        p.sweep_k_min = 10
        snap = p.snapshot_config()
        for key in _SNAPSHOT_PCA_OLS:
            assert key in snap, f"missing PCA+OLS key: {key}"
        assert snap["sweep_k_min"] == 10

    def test_groups_fields_included(self, tmp_path):
        """All _SNAPSHOT_GROUPS keys present when analysis_type='groups'."""
        p = _make_project(tmp_path)
        p.analysis_type = "groups"
        p.groups_n_perm = 10000
        snap = p.snapshot_config()
        for key in _SNAPSHOT_GROUPS:
            assert key in snap, f"missing groups key: {key}"
        assert snap["groups_n_perm"] == 10000

    def test_pls_excludes_other_type_fields(self, tmp_path):
        """PLS snapshot must not contain PCA+OLS or groups fields."""
        p = _make_project(tmp_path)
        p.analysis_type = "pls"
        snap = p.snapshot_config()
        for key in _SNAPSHOT_PCA_OLS + _SNAPSHOT_GROUPS:
            assert key not in snap, f"unexpected key in PLS snap: {key}"

    def test_groups_excludes_other_type_fields(self, tmp_path):
        """Groups snapshot must not contain PLS or PCA+OLS fields."""
        p = _make_project(tmp_path)
        p.analysis_type = "groups"
        snap = p.snapshot_config()
        for key in _SNAPSHOT_PLS + _SNAPSHOT_PCA_OLS:
            assert key not in snap, f"unexpected key in groups snap: {key}"

    def test_pcaols_excludes_other_type_fields(self, tmp_path):
        """PCA+OLS snapshot must not contain PLS or groups fields."""
        p = _make_project(tmp_path)
        p.analysis_type = "pca_ols"
        snap = p.snapshot_config()
        for key in _SNAPSHOT_PLS + _SNAPSHOT_GROUPS:
            assert key not in snap, f"unexpected key in PCA+OLS snap: {key}"

    def test_snapshot_values_match_project(self, tmp_path):
        """Snapshot values must match the project state at capture time."""
        p = _make_project(tmp_path)
        p.analysis_type = "pls"
        p.csv_path = Path("/data/my.csv")
        p.language = "de"
        p.context_window_size = 7
        p.sif_a = 5e-4
        p.pls_n_components = 4
        p.pls_n_perm = 3000
        snap = p.snapshot_config()
        assert snap["csv_path"] == "/data/my.csv"
        assert snap["language"] == "de"
        assert snap["context_window_size"] == 7
        assert snap["sif_a"] == 5e-4
        assert snap["pls_n_components"] == 4
        assert snap["pls_n_perm"] == 3000


# ===================================================================
# Computed readiness (~4 tests)
# ===================================================================

class TestComputedReadiness:
    def test_text_ready(self, tmp_path):
        p = _make_project(tmp_path)
        assert p.text_ready is False
        p._df = "fake_df"
        p.text_column = "text"
        assert p.text_ready is True

    def test_text_ready_requires_both(self, tmp_path):
        """text_ready is False with df but no text_column, and vice versa."""
        p = _make_project(tmp_path)
        p._df = "fake_df"
        assert p.text_ready is False
        p._df = None
        p.text_column = "text"
        assert p.text_ready is False

    def test_preprocessing_ready(self, tmp_path):
        p = _make_project(tmp_path)
        assert p.preprocessing_ready is False
        p._corpus = "fake"
        p.text_column = "text"
        p.preprocessed_text_column = "text"
        p.preprocessed_language = p.language
        assert p.preprocessing_ready is True

    def test_preprocessing_ready_column_mismatch(self, tmp_path):
        """preprocessing_ready is False when preprocessed column != text column."""
        p = _make_project(tmp_path)
        p._corpus = "fake"
        p.text_column = "review"
        p.preprocessed_text_column = "text"
        assert p.preprocessing_ready is False

    def test_embeddings_ready(self, tmp_path):
        p = _make_project(tmp_path)
        assert p.embeddings_ready is False
        p._emb = "fake"
        assert p.embeddings_ready is True

    def test_stage1_ready(self, tmp_path):
        p = _make_project(tmp_path)
        assert p.stage1_ready is False
        # Set all three
        p._df = "fake_df"
        p.text_column = "text"
        p._corpus = "fake"
        p.preprocessed_text_column = "text"
        p.preprocessed_language = p.language
        p._emb = "fake"
        assert p.stage1_ready is True

    def test_stage1_not_ready_missing_one(self, tmp_path):
        """stage1_ready is False when any one prerequisite is missing."""
        p = _make_project(tmp_path)
        # text + preprocessing OK, but no embeddings
        p._df = "fake_df"
        p.text_column = "text"
        p._corpus = "fake"
        p.preprocessed_text_column = "text"
        assert p.stage1_ready is False
