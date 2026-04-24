"""Result dataclass serialization and status recovery.

Keeps verbatim:
  - test_status_recovery_running (documented regression guard)
  - test_status_running_with_pkl_stays_running (same)
"""

from datetime import datetime


from ssdiff_gui.models.project import Result


def test_round_trip(make_result):
    r = make_result(name="my run", status="complete")
    d = r.to_dict()
    restored = Result.from_dict(d, r.result_path)
    assert restored.result_id == r.result_id
    assert restored.name == "my run"
    assert restored.status == "complete"
    assert restored.timestamp == r.timestamp
    assert restored.error_message is None


def test_error_message_survives_round_trip(make_result):
    r = make_result(status="error", error_message="OOM during fit")
    d = r.to_dict()
    restored = Result.from_dict(d, r.result_path)
    assert restored.status == "error"
    assert restored.error_message == "OOM during fit"


def test_status_running_without_pkl_becomes_interrupted(make_result):
    """VERBATIM regression guard: 'running' + no results.pkl → 'interrupted'."""
    r = make_result(status="running")
    d = r.to_dict()
    restored = Result.from_dict(d, r.result_path)
    assert restored.status == "interrupted"


def test_status_running_with_pkl_stays_running(make_result):
    """VERBATIM regression guard: 'running' + results.pkl exists → stays 'running'."""
    r = make_result(status="running")
    (r.result_path / "results.pkl").write_bytes(b"fake")
    d = r.to_dict()
    restored = Result.from_dict(d, r.result_path)
    assert restored.status == "running"


def test_config_snapshot_preserved(make_result):
    snap = {
        "analysis_type": "groups",
        "groups_n_perm": 5000,
        "lexicon_tokens": ["a", "b"],
    }
    r = make_result(config_snapshot=snap)
    d = r.to_dict()
    restored = Result.from_dict(d, r.result_path)
    assert restored.config_snapshot == snap
