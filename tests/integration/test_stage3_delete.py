"""Tests for Stage3Widget._delete_result_by_index and folder-collision handling.

Why this file exists:
  Before this file, no test covered the X-button delete flow or the folder-
  name collision logic. Two bugs slipped through as a result:
    1. Deleting a missing entry only updated project.json if the subsequent
       trash step succeeded — a crash between remove() and save_project()
       left project.json stale.
    2. _resolve_folder_collision only checked on-disk folders, not names
       already tracked in project.results, so saving a new result whose
       folder had been trashed produced duplicate folder_name entries.
"""

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pytest

from ssdiff_gui.models.project import Project, Result
from ssdiff_gui.utils.file_io import ProjectIO


# ---------------------------------------------------------------------------
# Helpers: mock the Qt bits that _delete_result_by_index touches
# ---------------------------------------------------------------------------

class _FakeSelector:
    """Stands in for the real QComboBox inside Stage3Widget."""
    def __init__(self, items: list[Result]):
        self._items = list(items)

    def itemData(self, row: int) -> Result:
        return self._items[row]

    def hidePopup(self):
        pass


@pytest.fixture
def patched_widget_module(monkeypatch):
    """Install a QMessageBox stub on the widget module that always returns Yes."""
    import ssdiff_gui.views.stage3.widget as w_mod

    class _Flag(int):
        def __or__(self, other):  # the code does `Yes | No`
            return _Flag(int(self) | int(other))

    class _MsgBox:
        Yes = _Flag(1)
        No = _Flag(2)
        Ok = _Flag(4)

        @staticmethod
        def question(*_args, **_kwargs):
            return _MsgBox.Yes

        @staticmethod
        def critical(*_args, **_kwargs):
            return _MsgBox.Ok

        @staticmethod
        def warning(*_args, **_kwargs):
            return _MsgBox.Ok

    monkeypatch.setattr(w_mod, "QMessageBox", _MsgBox)
    return w_mod


def _make_widget_with(project: Project, w_mod):
    """Build a bare Stage3Widget instance bound to `project`, skipping Qt UI."""
    w = w_mod.Stage3Widget.__new__(w_mod.Stage3Widget)
    w.project = project
    w._unsaved_result = None
    w._is_viewing_unsaved = False
    w.current_result = None
    # Dropdown mirrors the widget's own ordering: reversed(project.results).
    w.result_selector = _FakeSelector(list(reversed(project.results)))
    # Skip UI refresh — it would try to render result views.
    w._populate_result_selector = lambda: None
    return w


def _write_project_with_results(project_path: Path, tracked: list[str]):
    """Write a project.json that tracks the given folder names."""
    now = datetime.now().isoformat()
    project_dict = {
        "name": "T",
        "created_date": now,
        "modified_date": now,
        "results": tracked,
    }
    (project_path / "project.json").write_text(json.dumps(project_dict))


def _materialize_complete_result(project_path: Path, folder_name: str, name: str):
    """Create an on-disk results/<folder_name> with valid config + pickle."""
    result_path = project_path / "results" / folder_name
    result_path.mkdir(parents=True)
    result = Result(
        result_id=folder_name,
        timestamp=datetime(2026, 4, 21, 19, 5, 5),
        result_path=result_path,
        folder_name=folder_name,
        config_snapshot={},
        status="complete",
        name=name,
    )
    ProjectIO.save_result_config(result)
    (result_path / "results.pkl").write_bytes(pickle.dumps({"mock": True}))
    return result


# ---------------------------------------------------------------------------
# Delete flow
# ---------------------------------------------------------------------------

class TestDeleteMissingEntry:
    def test_removes_from_project_json(self, tmp_path, patched_widget_module):
        """X on a missing entry should drop it from tracked results on disk."""
        proj_dir = tmp_path / "p"
        ProjectIO.create_project_structure(proj_dir)
        _write_project_with_results(proj_dir, ["nonexistent_folder"])

        project = ProjectIO.load_project(proj_dir)
        assert project.results[0].status == "missing"

        w = _make_widget_with(project, patched_widget_module)
        w._delete_result_by_index(0)

        assert project.results == []
        on_disk = json.loads((proj_dir / "project.json").read_text())
        assert on_disk["results"] == []

        reloaded = ProjectIO.load_project(proj_dir)
        assert reloaded.results == []


class TestDeleteExistingEntry:
    def test_trash_success_updates_project_json(
        self, tmp_path, patched_widget_module, monkeypatch,
    ):
        """Happy path: folder trashed AND project.json no longer tracks it."""
        proj_dir = tmp_path / "p"
        trash_dir = tmp_path / "trash"
        trash_dir.mkdir()
        ProjectIO.create_project_structure(proj_dir)

        _materialize_complete_result(proj_dir, "1", name="1")
        _write_project_with_results(proj_dir, ["1"])

        # Redirect send2trash to a local dir so the test is hermetic.
        import send2trash
        def fake_trash(path):
            import shutil
            shutil.move(str(path), str(trash_dir / Path(path).name))
        monkeypatch.setattr(send2trash, "send2trash", fake_trash)

        project = ProjectIO.load_project(proj_dir)
        assert project.results[0].status == "complete"

        w = _make_widget_with(project, patched_widget_module)
        w._delete_result_by_index(0)

        assert project.results == []
        on_disk = json.loads((proj_dir / "project.json").read_text())
        assert on_disk["results"] == []
        assert not (proj_dir / "results" / "1").exists()
        assert (trash_dir / "1").exists()

        reloaded = ProjectIO.load_project(proj_dir)
        assert reloaded.results == []

    def test_trash_failure_still_updates_project_json(
        self, tmp_path, patched_widget_module, monkeypatch,
    ):
        """If send2trash raises, project.json is still updated.

        This is the key regression guard: the previous order (remove → trash
        → save) skipped the save when trash failed, leaving the user with
        a ghost entry in project.json.
        """
        proj_dir = tmp_path / "p"
        ProjectIO.create_project_structure(proj_dir)

        _materialize_complete_result(proj_dir, "1", name="1")
        _write_project_with_results(proj_dir, ["1"])

        import send2trash
        def boom(path):
            raise OSError("simulated trash failure")
        monkeypatch.setattr(send2trash, "send2trash", boom)

        project = ProjectIO.load_project(proj_dir)
        w = _make_widget_with(project, patched_widget_module)
        w._delete_result_by_index(0)

        # Project tracks the result no more, even though the folder is still on disk.
        on_disk = json.loads((proj_dir / "project.json").read_text())
        assert on_disk["results"] == []
        assert (proj_dir / "results" / "1").exists()  # folder untouched

        # On reload, the still-on-disk folder shows up as an orphan, not a tracked entry.
        reloaded = ProjectIO.load_project(proj_dir)
        assert len(reloaded.results) == 1
        assert reloaded.results[0].is_orphan is True


# ---------------------------------------------------------------------------
# Folder collision
# ---------------------------------------------------------------------------

class TestResolveFolderCollision:
    def test_returns_base_when_free(self, tmp_path):
        from ssdiff_gui.views.stage3.widget import _resolve_folder_collision
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        assert _resolve_folder_collision(results_dir, "1") == "1"

    def test_skips_existing_on_disk(self, tmp_path):
        from ssdiff_gui.views.stage3.widget import _resolve_folder_collision
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "1").mkdir()
        assert _resolve_folder_collision(results_dir, "1") == "1_2"

    def test_skips_reserved_name(self, tmp_path):
        """A name tracked in project.results (even with no on-disk folder)
        must not be reused — otherwise project.json gets duplicates."""
        from ssdiff_gui.views.stage3.widget import _resolve_folder_collision
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        # "1" is not on disk but is tracked (e.g. a [missing] entry).
        out = _resolve_folder_collision(results_dir, "1", reserved={"1"})
        assert out == "1_2"

    def test_skips_combined_disk_and_reserved(self, tmp_path):
        from ssdiff_gui.views.stage3.widget import _resolve_folder_collision
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "1").mkdir()
        out = _resolve_folder_collision(results_dir, "1", reserved={"1_2"})
        assert out == "1_3"
