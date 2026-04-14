"""Project save/load functionality for SSD."""

import hashlib
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import numpy as np

from ..models.project import Project, Result


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _strip_embeddings(result_obj) -> None:
    """Detach heavy Embeddings reference from a deserialized ssdiff result.

    Old pickles may contain a full copy; the project's loaded Embeddings
    will be re-attached on demand.
    """
    if result_obj is not None and hasattr(result_obj, "embeddings"):
        result_obj.embeddings = None


class ProjectIO:
    """Handles project save/load operations."""

    # ------------------------------------------------------------------ #
    #  Project
    # ------------------------------------------------------------------ #

    @staticmethod
    def save_project(project: Project) -> None:
        """Save project to project.json and each result's config."""
        project.modified_date = datetime.now()

        project_dict = project.to_dict()
        project_file = project.project_path / "project.json"

        with open(project_file, "w", encoding="utf-8") as f:
            json.dump(project_dict, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

        for result in project.results:
            ProjectIO.save_result_config(result)

    @staticmethod
    def load_project(project_path: Path) -> Project:
        """Load project from project.json."""
        project_file = project_path / "project.json"

        with open(project_file, "r", encoding="utf-8") as f:
            project_dict = json.load(f)

        project = Project.from_dict(project_dict, project_path)

        # Load results
        result_ids = project_dict.get("results", [])
        for result_id in result_ids:
            try:
                result_path = project_path / "results" / result_id
                if result_path.exists():
                    config_file = result_path / "config.json"
                    if config_file.exists():
                        with open(config_file, "r", encoding="utf-8") as f:
                            result_dict = json.load(f)
                        result = Result.from_dict(result_dict, result_path)
                    else:
                        result = Result(
                            result_id=result_id,
                            timestamp=datetime.now(),
                            result_path=result_path,
                            config_snapshot={},
                            status="unknown",
                        )
                    # Load pickled result object if available
                    results_pkl = result_path / "results.pkl"
                    if results_pkl.exists() and result.status == "complete":
                        try:
                            with open(results_pkl, "rb") as f:
                                result._result = pickle.load(f)
                            _strip_embeddings(result._result)
                        except Exception as e:
                            print(f"Warning: Failed to load results for {result_id}: {e}")
                    project.results.append(result)
            except Exception as e:
                print(f"Warning: Failed to load result {result_id}: {e}")

        return project

    @staticmethod
    def create_project_structure(project_path: Path) -> None:
        """Create the project directory structure."""
        project_path.mkdir(parents=True, exist_ok=True)
        (project_path / "data").mkdir(exist_ok=True)
        (project_path / "results").mkdir(exist_ok=True)
        (project_path / "embeddings").mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Results
    # ------------------------------------------------------------------ #

    @staticmethod
    def save_result_config(result: Result) -> None:
        """Save result configuration (config snapshot + metadata) to its folder."""
        result.result_path.mkdir(parents=True, exist_ok=True)

        config_file = result.result_path / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    @staticmethod
    def save_result(result: Result) -> None:
        """Save the library result object as pickle and generate replication script."""
        if result._result is not None:
            result.result_path.mkdir(parents=True, exist_ok=True)
            results_file = result.result_path / "results.pkl"
            # Strip the heavy Embeddings reference before pickling —
            # it will be re-attached from the project on load.
            ssd_result = result._result
            saved_emb = getattr(ssd_result, "embeddings", None)
            ssd_result.embeddings = None
            try:
                with open(results_file, "wb") as f:
                    pickle.dump(ssd_result, f, protocol=pickle.HIGHEST_PROTOCOL)
            finally:
                ssd_result.embeddings = saved_emb

            # Generate human-readable report from report settings
            try:
                from .report_settings import (
                    get_report_setting,
                    KEY_TOP_WORDS, KEY_CLUSTERS,
                    KEY_EXTREME_DOCS, KEY_MISDIAGNOSED,
                )
                report_text = result._result.report(
                    top_words=get_report_setting(KEY_TOP_WORDS) or None,
                    clusters=get_report_setting(KEY_CLUSTERS) or None,
                    extreme_docs=get_report_setting(KEY_EXTREME_DOCS) or None,
                    misdiagnosed=get_report_setting(KEY_MISDIAGNOSED) or None,
                )
                report_path = result.result_path / "results.txt"
                report_path.write_text(report_text, encoding="utf-8")
            except Exception:
                pass  # Non-critical — don't fail the save

            # Generate PCA sweep plot for PCA+OLS results
            try:
                if hasattr(result._result, "plot_sweep") and result._result.sweep_result is not None:
                    sweep_path = result.result_path / "sweep_plot.png"
                    result._result.plot_sweep(path=str(sweep_path))
            except Exception:
                pass  # Non-critical — don't fail the save

            # Generate replication script from config snapshot
            try:
                script = result.to_replication_script()
                script_path = result.result_path / "replication_script.py"
                script_path.write_text(script, encoding="utf-8")
            except Exception:
                pass  # Non-critical — don't fail the save

    @staticmethod
    def load_result(project_path: Path, result_id: str) -> Result:
        """Load a specific result by ID."""
        result_path = project_path / "results" / result_id
        config_file = result_path / "config.json"

        with open(config_file, "r", encoding="utf-8") as f:
            result_dict = json.load(f)

        result = Result.from_dict(result_dict, result_path)

        # Load pickled result object if available
        results_pkl = result_path / "results.pkl"
        if result.status == "complete":
            if not results_pkl.exists():
                result.status = "error"
                result.error_message = f"Results file not found: {results_pkl}"
            else:
                try:
                    with open(results_pkl, "rb") as f:
                        result._result = pickle.load(f)
                    _strip_embeddings(result._result)
                except Exception as e:
                    result.status = "error"
                    result.error_message = f"Failed to load results: {e}"
                    result._result = None

        return result

    # ------------------------------------------------------------------ #
    #  Corpus
    # ------------------------------------------------------------------ #

    @staticmethod
    def save_corpus(project: Project, corpus, pre_docs: list = None) -> None:
        """Save a library Corpus object as corpus.pkl.

        If the corpus was built with pretokenized=True (so its own
        pre_docs is None), pass the app's pre_docs list and it
        will be attached before pickling.
        """
        if pre_docs is not None and getattr(corpus, 'pre_docs', None) is None:
            corpus.pre_docs = pre_docs

        data_dir = project.project_path / "data"
        data_dir.mkdir(exist_ok=True)
        with open(data_dir / "corpus.pkl", "wb") as f:
            pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_corpus(project: Project):
        """Load a library Corpus from corpus.pkl.

        Returns the Corpus object, or None if the file doesn't exist.
        """
        corpus_file = project.project_path / "data" / "corpus.pkl"
        if not corpus_file.exists():
            return None
        with open(corpus_file, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def corpus_exists(project: Project) -> bool:
        """Check whether preprocessed corpus exists on disk."""
        return (project.project_path / "data" / "corpus.pkl").exists()

    # ------------------------------------------------------------------ #
    #  Embeddings
    # ------------------------------------------------------------------ #

    @staticmethod
    def list_prepared_embeddings(project: Project) -> List[dict]:
        """List .ssdembed files in the project's embeddings/ directory.

        Returns a list of dicts with keys: filename, vocab_size, embedding_dim,
        l2_normalized, abtt_m, file_size.
        """
        emb_dir = project.project_path / "embeddings"
        if not emb_dir.exists():
            return []

        results = []
        for path in sorted(emb_dir.glob("*.ssdembed")):
            # Total size = pickle + sidecar .vectors.npy
            total_bytes = path.stat().st_size
            sidecar = Path(str(path) + ".vectors.npy")
            if sidecar.exists():
                total_bytes += sidecar.stat().st_size
            meta = {"filename": path.name, "file_size_mb": total_bytes / (1024 * 1024)}

            # Read metadata from the pickle without loading vectors.
            # The pickle stores an Embeddings with an empty vectors stub;
            # the heavy data lives in the sidecar .npy file.
            try:
                from ssdiff.embeddings import _GensimUnpickler
                with open(str(path), "rb") as f:
                    obj = _GensimUnpickler(f).load()
                meta["vocab_size"] = len(getattr(obj, "index_to_key", []))
                meta["embedding_dim"] = getattr(obj, "vector_size", 0)
                meta["l2_normalized"] = getattr(obj, "l2_normalized", False)
                meta["abtt_m"] = getattr(obj, "abtt_m", 0)
            except Exception:
                # Fallback: read vectors.npy shape for vocab/dim
                if sidecar.exists():
                    try:
                        shape = np.load(str(sidecar), mmap_mode='r').shape
                        meta["vocab_size"] = shape[0]
                        meta["embedding_dim"] = shape[1] if len(shape) > 1 else 0
                    except Exception:
                        meta["vocab_size"] = 0
                        meta["embedding_dim"] = 0
                else:
                    meta["vocab_size"] = 0
                    meta["embedding_dim"] = 0
                meta["l2_normalized"] = "l2" in path.name.lower()
                meta["abtt_m"] = 0
            results.append(meta)
        return results

    @staticmethod
    def compute_embedding_hash(path: Path) -> str:
        """Compute a content hash for an embedding file (for dedup)."""
        h = hashlib.sha256()
        npy_path = Path(str(path) + ".vectors.npy")
        target = npy_path if npy_path.exists() else path
        with open(target, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def find_duplicate_embedding(project: Project, file_hash: str) -> Optional[str]:
        """Check if an embedding with the same hash already exists.

        Returns the filename of the duplicate, or None.
        """
        emb_dir = project.project_path / "embeddings"
        if not emb_dir.exists():
            return None
        for path in emb_dir.glob("*.ssdembed"):
            existing_hash = ProjectIO.compute_embedding_hash(path)
            if existing_hash == file_hash:
                return path.name
        return None

    # ------------------------------------------------------------------ #
    #  Legacy helpers (kept for backward compat with old projects)
    # ------------------------------------------------------------------ #

    @staticmethod
    def save_preprocessed_docs(
        project: Project,
        pre_docs: list,
        docs: list,
        id_row_indices: list = None,
    ) -> None:
        """Save preprocessed documents to cache (legacy format)."""
        data_dir = project.project_path / "data"
        data_dir.mkdir(exist_ok=True)

        cache_file = data_dir / "preprocessed_docs.pkl"
        payload = {"pre_docs": pre_docs, "docs": docs}
        if id_row_indices is not None:
            payload["id_row_indices"] = id_row_indices
        with open(cache_file, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load_preprocessed_docs(project: Project) -> Optional[tuple]:
        """Load preprocessed documents from cache (legacy format).

        Returns (pre_docs, docs, id_row_indices) where id_row_indices may
        be None for projects that were preprocessed without grouping.
        """
        cache_file = project.project_path / "data" / "preprocessed_docs.pkl"
        if not cache_file.exists():
            return None

        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return (
            data.get("pre_docs"),
            data.get("docs"),
            data.get("id_row_indices"),
        )
