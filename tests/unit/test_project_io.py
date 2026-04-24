"""ProjectIO: create/save/load project, result, corpus, embedding hash.

Replaces test_file_io.py with:
  - Happy-path test_lists_ssdembed_files that uses a real Embeddings pickle
    (old version wrote b'not a real pickle' and relied on the except branch)
  - All documented regression guards kept verbatim:
      test_complete_without_pkl_sets_error
      test_missing_result_skips
      test_load_project_corrupted_json
      test_load_result_corrupted_pkl
      test_ignores_vectors_npy_sidecar
      TestEmbeddingHash (3 tests)
  - NumpyEncoder: 1 end-to-end test replaces 6 variants
"""

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from ssdiff_gui.models.project import Project, Result
from ssdiff_gui.utils.file_io import ProjectIO, _NumpyEncoder


@pytest.fixture
def shared_emb_dir(tmp_path, monkeypatch):
    emb_dir = tmp_path / "shared_embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("ssdiff_gui.utils.paths.embeddings_dir", lambda: emb_dir)
    return emb_dir


def _make_project(tmp_path, name="test_project") -> Project:
    project_path = tmp_path / name
    project_path.mkdir(parents=True, exist_ok=True)
    return Project(
        project_path=project_path,
        name=name,
        created_date=datetime(2026, 1, 15, 10, 0, 0),
        modified_date=datetime(2026, 4, 14, 14, 30, 0),
    )


def _make_result(project: Project, result_id="20260414_143000", status="pending",
                 config_snapshot=None) -> Result:
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


class TestProjectStructure:
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


class TestProjectSaveLoad:
    def test_empty_project_round_trips(self, tmp_path):
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        ProjectIO.save_project(project)

        loaded = ProjectIO.load_project(project.project_path)
        assert loaded.name == "test_project"
        assert loaded.created_date == datetime(2026, 1, 15, 10, 0, 0)
        assert len(loaded.results) == 0

    def test_missing_result_skips(self, tmp_path):
        """VERBATIM regression guard: project.json references nonexistent result
        → kept as 'missing' marker so the user can clean it up."""
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)

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


class TestResultSaveLoad:
    def test_save_load_result_config(self, tmp_path):
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        result = _make_result(project)
        ProjectIO.save_result_config(result)

        config_file = result.result_path / "config.json"
        assert config_file.exists()
        data = json.loads(config_file.read_text())
        assert data["result_id"] == "20260414_143000"
        assert data["config_snapshot"]["analysis_type"] == "pls"

    def test_complete_without_pkl_sets_error(self, tmp_path):
        """VERBATIM regression guard: complete status + no results.pkl → error."""
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        result = _make_result(project, status="complete")
        ProjectIO.save_result_config(result)
        assert not (result.result_path / "results.pkl").exists()

        loaded = ProjectIO.load_result(project.project_path, result.result_id)
        assert loaded.status == "error"
        assert loaded.error_message is not None
        assert "not found" in loaded.error_message

    def test_load_result_with_pkl(self, tmp_path):
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        result = _make_result(project, status="complete")
        ProjectIO.save_result_config(result)

        pkl_data = {"r2": 0.42, "mock": True}
        (result.result_path / "results.pkl").write_bytes(pickle.dumps(pkl_data))

        loaded = ProjectIO.load_result(project.project_path, result.result_id)
        assert loaded.status == "complete"
        assert loaded._result["r2"] == 0.42


class _MockCorpus:
    """Module-level class so pickle can serialize it."""
    def __init__(self):
        self.docs = [["hello"], ["world"]]
        self.pre_docs = None


class TestCorpusSaveLoad:
    def test_no_corpus_returns_none(self, tmp_path):
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        assert ProjectIO.corpus_exists(project) is False
        assert ProjectIO.load_corpus(project) is None

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


def test_numpy_encoder_serializes_common_numpy_types():
    data = {
        "int": np.int64(42),
        "float": np.float64(3.14),
        "array": np.array([1, 2, 3]),
        "bool": np.bool_(True),
        "plain": [1, "text", None],
    }
    parsed = json.loads(json.dumps(data, cls=_NumpyEncoder))
    assert parsed["int"] == 42
    assert abs(parsed["float"] - 3.14) < 1e-10
    assert parsed["array"] == [1, 2, 3]
    assert parsed["bool"] is True
    assert parsed["plain"] == [1, "text", None]


class TestEmbeddingHash:
    def test_consistent_hash(self, tmp_path):
        f = tmp_path / "test.ssdembed"
        f.write_bytes(b"some embedding data " * 100)
        h1 = ProjectIO.compute_embedding_hash(f)
        h2 = ProjectIO.compute_embedding_hash(f)
        assert h1 == h2
        assert len(h1) == 64

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.ssdembed"
        f2 = tmp_path / "b.ssdembed"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert ProjectIO.compute_embedding_hash(f1) != ProjectIO.compute_embedding_hash(f2)

    def test_find_duplicate(self, tmp_path, shared_emb_dir):
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        (shared_emb_dir / "glove.ssdembed").write_bytes(b"embedding content")
        new_file = tmp_path / "new.ssdembed"
        new_file.write_bytes(b"embedding content")
        new_hash = ProjectIO.compute_embedding_hash(new_file)
        assert ProjectIO.find_duplicate_embedding(project, new_hash) == "glove.ssdembed"


class TestListPreparedEmbeddings:
    def test_empty_dir_returns_empty_list(self, tmp_path, shared_emb_dir):
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        assert ProjectIO.list_prepared_embeddings(project) == []

    def test_happy_path_real_embedding(self, tmp_path, shared_emb_dir):
        """REWRITTEN: use a real Embeddings pickle, not b'fake bytes'."""
        from ssdiff import Embeddings

        words = ["a", "b", "c", "d", "e"]
        vecs = np.random.RandomState(0).randn(5, 10).astype(np.float32)
        emb = Embeddings(words, vecs)

        out_path = shared_emb_dir / "glove.ssdembed"
        with open(out_path, "wb") as f:
            pickle.dump(emb, f)

        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)

        result = ProjectIO.list_prepared_embeddings(project)
        assert len(result) == 1
        entry = result[0]
        assert entry["filename"] == "glove.ssdembed"
        assert entry["vocab_size"] == 5
        assert entry["embedding_dim"] == 10
        assert entry["file_size_mb"] > 0

    def test_ignores_vectors_npy_sidecar(self, tmp_path, shared_emb_dir):
        """VERBATIM: *.vectors.npy sidecars must not appear as entries."""
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        (shared_emb_dir / "glove.ssdembed").write_bytes(b"fake")
        (shared_emb_dir / "glove.ssdembed.vectors.npy").write_bytes(b"sidecar")
        result = ProjectIO.list_prepared_embeddings(project)
        assert len(result) == 1
        assert result[0]["filename"] == "glove.ssdembed"


class TestCorruptedFiles:
    def test_load_project_corrupted_json(self, tmp_path):
        """VERBATIM: raise (not silent)."""
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        (project.project_path / "project.json").write_text("{bad json!!!")
        with pytest.raises(json.JSONDecodeError):
            ProjectIO.load_project(project.project_path)

    def test_load_result_corrupted_pkl(self, tmp_path):
        """VERBATIM: raise (not silent)."""
        project = _make_project(tmp_path)
        ProjectIO.create_project_structure(project.project_path)
        result = _make_result(project, status="complete")
        ProjectIO.save_result_config(result)
        (result.result_path / "results.pkl").write_bytes(b"corrupted data")
        with pytest.raises(pickle.UnpicklingError):
            ProjectIO.load_result(project.project_path, result.result_id)
