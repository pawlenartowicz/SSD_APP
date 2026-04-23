"""Tests for ssdiff_gui/utils/file_io.py — ProjectIO and NumpyEncoder."""

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from ssdiff_gui.utils.file_io import ProjectIO, _NumpyEncoder
from ssdiff_gui.models.project import Project, Result


@pytest.fixture
def shared_emb_dir(tmp_path, monkeypatch):
    """Redirect the shared embeddings_dir() to a tmp location for the test."""
    emb_dir = tmp_path / "shared_embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("ssdiff_gui.utils.paths.embeddings_dir", lambda: emb_dir)
    return emb_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_project(tmp_path: Path, name: str = "test_project") -> Project:
    """Create a minimal Project rooted in tmp_path."""
    project_path = tmp_path / name
    project_path.mkdir(parents=True, exist_ok=True)
    return Project(
        project_path=project_path,
        name=name,
        created_date=datetime(2026, 1, 15, 10, 0, 0),
        modified_date=datetime(2026, 4, 14, 14, 30, 0),
    )


def _make_result(project: Project, result_id: str = "20260414_143000",
                 status: str = "pending", config_snapshot: dict = None) -> Result:
    """Create a minimal Result attached to a project."""
    result_path = project.project_path / "results" / result_id
    result_path.mkdir(parents=True, exist_ok=True)
    return Result(
        result_id=result_id,
        timestamp=datetime(2026, 4, 14, 14, 30, 0),
        result_path=result_path,
        config_snapshot=config_snapshot or {
            "analysis_type": "pls",
            "csv_path": "/data/test.csv",
            "text_column": "text",
            "pls_n_components": 1,
        },
        status=status,
    )


# ===================================================================
# create_project_structure
# ===================================================================

class TestCreateProjectStructure:
    def test_creates_all_dirs(self, tmp_path):
        proj_path = tmp_path / "new_project"
        ProjectIO.create_project_structure(proj_path)
        assert (proj_path / "data").is_dir()
        assert (proj_path / "results").is_dir()

    def test_idempotent(self, tmp_path):
        proj_path = tmp_path / "new_project"
        ProjectIO.create_project_structure(proj_path)
        ProjectIO.create_project_structure(proj_path)
        assert (proj_path / "data").is_dir()


# ===================================================================
# Project save/load
# ===================================================================

class TestProjectSaveLoad:
    def test_empty_project(self, tmp_path):
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        ProjectIO.save_project(project)

        loaded = ProjectIO.load_project(project.project_path)
        assert loaded.name == "test_project"
        assert loaded.created_date == datetime(2026, 1, 15, 10, 0, 0)
        assert len(loaded.results) == 0
        # Defaults should be intact
        assert loaded.csv_path is None
        assert loaded.language == "en"
        assert loaded.analysis_type == "pls"

    def test_project_with_config(self, tmp_path):
        project = _make_project(tmp_path)
        project.csv_path = Path("/data/test.csv")
        project.csv_encoding = "latin-1"
        project.text_column = "review"
        project.language = "pl"
        project.lexicon_tokens = ["word1", "word2"]
        project.selected_embedding = "glove_l2.ssdembed"
        project.analysis_type = "pca_ols"
        project.sweep_k_min = 15

        ProjectIO.create_project_structure(project.project_path)
        ProjectIO.save_project(project)

        loaded = ProjectIO.load_project(project.project_path)
        assert loaded.csv_path == Path("/data/test.csv")
        assert loaded.csv_encoding == "latin-1"
        assert loaded.text_column == "review"
        assert loaded.language == "pl"
        assert loaded.lexicon_tokens == ["word1", "word2"]
        assert loaded.selected_embedding == "glove_l2.ssdembed"
        assert loaded.analysis_type == "pca_ols"
        assert loaded.sweep_k_min == 15

    def test_project_with_result(self, tmp_path):
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)

        result = _make_result(project, status="complete")
        # Create a mock pickle that can be loaded
        pkl_path = result.result_path / "results.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"mock": True}, f)
        project.results.append(result)

        ProjectIO.save_project(project)

        loaded = ProjectIO.load_project(project.project_path)
        assert len(loaded.results) == 1
        r = loaded.results[0]
        assert r.result_id == "20260414_143000"
        assert r.status == "complete"
        assert r.config_snapshot["analysis_type"] == "pls"
        assert r.timestamp == datetime(2026, 4, 14, 14, 30, 0)

    def test_missing_result_skips(self, tmp_path):
        """project.json references nonexistent result → keeps a 'missing' marker."""
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)

        # Manually write project.json with a nonexistent result
        project_dict = project.to_dict()
        project_dict["results"] = ["nonexistent_result_id"]
        with open(project.project_path / "project.json", "w") as f:
            json.dump(project_dict, f)

        loaded = ProjectIO.load_project(project.project_path)
        assert len(loaded.results) == 1
        marker = loaded.results[0]
        assert marker.result_id == "nonexistent_result_id"
        assert marker.status == "missing"
        assert marker.result_path is None


# ===================================================================
# Result save/load
# ===================================================================

class TestResultSaveLoad:
    def test_save_load_result_config(self, tmp_path):
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        result = _make_result(project)

        ProjectIO.save_result_config(result)

        config_file = result.result_path / "config.json"
        assert config_file.exists()

        with open(config_file, "r") as f:
            data = json.load(f)
        assert data["result_id"] == "20260414_143000"
        assert data["status"] == "pending"
        assert data["config_snapshot"]["analysis_type"] == "pls"
        assert data["config_snapshot"]["csv_path"] == "/data/test.csv"
        assert data["config_snapshot"]["pls_n_components"] == 1

    def test_complete_without_pkl_sets_error(self, tmp_path):
        """complete status + no results.pkl → error status with message."""
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        result = _make_result(project, status="complete")

        # Save config only — no results.pkl
        ProjectIO.save_result_config(result)
        assert not (result.result_path / "results.pkl").exists()

        loaded = ProjectIO.load_result(project.project_path, result.result_id)
        assert loaded.status == "error"
        assert loaded.error_message is not None
        assert "not found" in loaded.error_message

    def test_load_result_with_pkl(self, tmp_path):
        """Round-trip with a mock pickle — verify data survives."""
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        result = _make_result(project, status="complete")

        # Save config
        ProjectIO.save_result_config(result)

        # Create mock results.pkl
        pkl_data = {"r2": 0.42, "mock": True}
        pkl_path = result.result_path / "results.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(pkl_data, f)

        loaded = ProjectIO.load_result(project.project_path, result.result_id)
        assert loaded.status == "complete"
        assert loaded._result is not None
        assert loaded._result["r2"] == 0.42
        assert loaded._result["mock"] is True




# ===================================================================
# Corpus save/load
# ===================================================================

class _MockCorpus:
    """Module-level class so pickle can serialize it."""
    def __init__(self):
        self.docs = [["hello"], ["world"]]
        self.pre_docs = None


class TestCorpusSaveLoad:
    def test_corpus_exists_false(self, tmp_path):
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        assert ProjectIO.corpus_exists(project) is False

    def test_save_load_corpus(self, tmp_path):
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)

        corpus = _MockCorpus()
        pre_docs = [{"text": "hello"}, {"text": "world"}]

        ProjectIO.save_corpus(project, corpus, pre_docs)
        assert ProjectIO.corpus_exists(project) is True

        loaded = ProjectIO.load_corpus(project)
        assert loaded.docs == [["hello"], ["world"]]
        assert loaded.pre_docs == pre_docs
        assert len(loaded.pre_docs) == 2
        assert loaded.pre_docs[0] == {"text": "hello"}

    def test_load_corpus_nonexistent(self, tmp_path):
        """load_corpus returns None when no corpus.pkl exists."""
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        assert ProjectIO.load_corpus(project) is None


# ===================================================================
# NumpyEncoder
# ===================================================================

class TestNumpyEncoder:
    def test_int64(self):
        val = np.int64(42)
        result = json.dumps({"x": val}, cls=_NumpyEncoder)
        assert json.loads(result)["x"] == 42

    def test_float64(self):
        val = np.float64(3.14)
        result = json.dumps({"x": val}, cls=_NumpyEncoder)
        assert abs(json.loads(result)["x"] - 3.14) < 1e-10

    def test_ndarray(self):
        val = np.array([1, 2, 3])
        result = json.dumps({"x": val}, cls=_NumpyEncoder)
        assert json.loads(result)["x"] == [1, 2, 3]

    def test_bool_(self):
        val = np.bool_(True)
        result = json.dumps({"x": val}, cls=_NumpyEncoder)
        assert json.loads(result)["x"] is True

    def test_regular_types_pass_through(self):
        data = {"a": 1, "b": "hello", "c": [1, 2], "d": None}
        result = json.dumps(data, cls=_NumpyEncoder)
        assert json.loads(result) == data

    def test_nested_numpy(self):
        data = {"scores": np.array([np.float64(0.1), np.float64(0.2)])}
        result = json.dumps(data, cls=_NumpyEncoder)
        parsed = json.loads(result)
        assert len(parsed["scores"]) == 2
        assert abs(parsed["scores"][0] - 0.1) < 1e-10


# ===================================================================
# Embedding hash + dedup
# ===================================================================

class TestEmbeddingHash:
    def test_consistent_hash(self, tmp_path):
        f = tmp_path / "test.ssdembed"
        f.write_bytes(b"some embedding data " * 100)
        h1 = ProjectIO.compute_embedding_hash(f)
        h2 = ProjectIO.compute_embedding_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.ssdembed"
        f2 = tmp_path / "b.ssdembed"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert ProjectIO.compute_embedding_hash(f1) != ProjectIO.compute_embedding_hash(f2)

    def test_find_duplicate(self, tmp_path, shared_emb_dir):
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)

        existing = shared_emb_dir / "glove.ssdembed"
        existing.write_bytes(b"embedding content")

        new_file = tmp_path / "new.ssdembed"
        new_file.write_bytes(b"embedding content")
        new_hash = ProjectIO.compute_embedding_hash(new_file)

        result = ProjectIO.find_duplicate_embedding(project, new_hash)
        assert result == "glove.ssdembed"

    def test_find_no_duplicate(self, tmp_path, shared_emb_dir):
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)

        existing = shared_emb_dir / "glove.ssdembed"
        existing.write_bytes(b"embedding content A")

        new_file = tmp_path / "new.ssdembed"
        new_file.write_bytes(b"different content B")
        new_hash = ProjectIO.compute_embedding_hash(new_file)

        result = ProjectIO.find_duplicate_embedding(project, new_hash)
        assert result is None


# ===================================================================
# list_prepared_embeddings
# ===================================================================

class TestListPreparedEmbeddings:
    def test_empty_dir(self, tmp_path):
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        result = ProjectIO.list_prepared_embeddings(project)
        assert result == []

    def test_no_embeddings_dir(self, tmp_path):
        """Returns [] if embeddings dir doesn't exist."""
        project = _make_project(tmp_path)
        # Don't call create_project_structure
        result = ProjectIO.list_prepared_embeddings(project)
        assert result == []

    def test_lists_ssdembed_files(self, tmp_path, shared_emb_dir):
        """Picks up .ssdembed files and returns metadata dicts."""
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)

        # Create a mock .ssdembed pickle (will fail unpickling in
        # list_prepared_embeddings and hit the except branch)
        (shared_emb_dir / "glove.ssdembed").write_bytes(b"not a real pickle")
        (shared_emb_dir / "nkjp.ssdembed").write_bytes(b"also not a pickle")
        # Non-.ssdembed files should be ignored
        (shared_emb_dir / "readme.txt").write_text("ignore me")

        result = ProjectIO.list_prepared_embeddings(project)
        assert len(result) == 2
        filenames = {r["filename"] for r in result}
        assert filenames == {"glove.ssdembed", "nkjp.ssdembed"}
        # Each dict has the expected keys
        for r in result:
            assert "filename" in r
            assert "vocab_size" in r
            assert "embedding_dim" in r
            assert "l2_normalized" in r
            assert "abtt" in r
            assert "file_size_mb" in r
            assert r["file_size_mb"] > 0

    def test_ignores_vectors_npy_sidecar(self, tmp_path, shared_emb_dir):
        """*.vectors.npy sidecars must not appear as separate entries."""
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)

        (shared_emb_dir / "glove.ssdembed").write_bytes(b"fake")
        (shared_emb_dir / "glove.ssdembed.vectors.npy").write_bytes(b"sidecar data")

        result = ProjectIO.list_prepared_embeddings(project)
        assert len(result) == 1
        assert result[0]["filename"] == "glove.ssdembed"


# ===================================================================
# Error handling
# ===================================================================

class TestErrorHandling:
    def test_load_project_corrupted_json(self, tmp_path):
        """Corrupted project.json → exception (not silent)."""
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        (project.project_path / "project.json").write_text("{bad json!!!")

        import pytest
        with pytest.raises(json.JSONDecodeError):
            ProjectIO.load_project(project.project_path)

    def test_load_result_corrupted_pkl(self, tmp_path):
        """Corrupted results.pkl → exception (not silent)."""
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        result = _make_result(project, status="complete")

        ProjectIO.save_result_config(result)
        (result.result_path / "results.pkl").write_bytes(b"corrupted data!!!")

        import pytest
        with pytest.raises(pickle.UnpicklingError):
            ProjectIO.load_result(project.project_path, result.result_id)
