"""Tests for Result.to_replication_script() — pure string generation."""

from datetime import datetime


from ssdiff_gui.models.project import Result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(tmp_path, config_snapshot):
    result_path = tmp_path / "results" / "20260414_120000"
    result_path.mkdir(parents=True, exist_ok=True)
    return Result(
        result_id="20260414_120000",
        timestamp=datetime(2026, 4, 14, 12, 0, 0),
        result_path=result_path,
        config_snapshot=config_snapshot,
    )


_BASE_CONFIG = {
    "csv_path": "/data/test.csv",
    "csv_encoding": "utf-8-sig",
    "text_column": "text",
    "language": "en",
    "selected_embedding": "glove.ssdembed",
    "concept_mode": "lexicon",
    "lexicon_tokens": ["happy", "sad", "angry"],
    "outcome_column": "score",
    "group_column": "group",
    "context_window_size": 3,
    "sif_a": 1e-3,
}


# ===================================================================
# Script generation (~6 tests)
# ===================================================================

class TestReplicationScript:
    def test_pls_script(self, tmp_path):
        config = {
            **_BASE_CONFIG,
            "analysis_type": "pls",
            "pls_n_components": 2,
            "pls_p_method": "perm",
            "pls_n_perm": 1000,
            "pls_n_splits": 50,
            "pls_split_ratio": 0.5,
            "pls_random_state": "default",
        }
        r = _make_result(tmp_path, config)
        script = r.to_replication_script()
        assert "fit_pls(" in script
        assert "n_components=2" in script
        assert '"perm"' in script
        assert "n_perm=1000" in script
        assert "n_splits=50" in script
        assert "split_ratio=0.5" in script
        assert "random_state=2137" in script

    def test_pcaols_script(self, tmp_path):
        config = {
            **_BASE_CONFIG,
            "analysis_type": "pca_ols",
            "pcaols_n_components": 40,
            "sweep_k_min": 20,
            "sweep_k_max": 120,
            "sweep_k_step": 2,
        }
        r = _make_result(tmp_path, config)
        script = r.to_replication_script()
        assert "fit_ols(" in script
        assert "fixed_k=40" in script
        assert "k_min=20" in script
        assert "k_max=120" in script
        assert "k_step=2" in script
        assert "fit_pls(" not in script
        assert "fit_groups(" not in script

    def test_groups_script(self, tmp_path):
        config = {
            **_BASE_CONFIG,
            "analysis_type": "groups",
            "groups_n_perm": 5000,
            "groups_correction": "holm",
            "groups_median_split": False,
            "groups_random_state": "42",
        }
        r = _make_result(tmp_path, config)
        script = r.to_replication_script()
        assert "fit_groups(" in script
        assert '"holm"' in script
        assert "n_perm=5000" in script
        assert "random_state=42" in script
        assert "median_split=False" in script

    def test_groups_median_split_true(self, tmp_path):
        config = {
            **_BASE_CONFIG,
            "analysis_type": "groups",
            "groups_n_perm": 1000,
            "groups_correction": "fdr_bh",
            "groups_median_split": True,
            "groups_random_state": "default",
        }
        r = _make_result(tmp_path, config)
        script = r.to_replication_script()
        assert "median_split=True" in script
        assert "random_state=2137" in script
        assert '"fdr_bh"' in script

    def test_lexicon_mode(self, tmp_path):
        config = {
            **_BASE_CONFIG,
            "analysis_type": "pls",
            "pls_n_components": 1,
            "pls_p_method": "auto",
            "pls_random_state": "default",
        }
        r = _make_result(tmp_path, config)
        script = r.to_replication_script()
        assert "lexicon = ['happy', 'sad', 'angry']" in script

    def test_fulldoc_mode(self, tmp_path):
        config = {
            **_BASE_CONFIG,
            "analysis_type": "pls",
            "concept_mode": "fulldoc",
            "pls_n_components": 1,
            "pls_p_method": "auto",
            "pls_random_state": "default",
        }
        r = _make_result(tmp_path, config)
        script = r.to_replication_script()
        assert "use_full_doc=True" in script
        assert "set(t for doc" in script

    def test_scripts_compile(self, tmp_path):
        """All generated scripts pass compile() (valid Python syntax)."""
        configs = [
            {**_BASE_CONFIG, "analysis_type": "pls",
             "pls_n_components": 1, "pls_p_method": "auto",
             "pls_random_state": "default"},
            {**_BASE_CONFIG, "analysis_type": "pca_ols",
             "pcaols_n_components": None, "sweep_k_min": 20,
             "sweep_k_max": 120, "sweep_k_step": 2},
            {**_BASE_CONFIG, "analysis_type": "groups",
             "groups_n_perm": 5000, "groups_correction": "holm",
             "groups_median_split": False, "groups_random_state": "default"},
        ]
        for cfg in configs:
            r = _make_result(tmp_path, cfg)
            script = r.to_replication_script()
            compile(script, "<replication_script>", "exec")
