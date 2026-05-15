"""Smoke test for the multipls fit path through SSDRunner.

Marked slow: needs real embeddings to run. Skipped by default; pytest's
"slow" marker filter at the project level excludes it unless explicitly
asked for.
"""

import pytest


@pytest.mark.slow
def test_run_multipls_smoke(tiny_project_multipls):
    from ssdiff_gui.controllers.ssd_runner import SSDRunner

    runner = SSDRunner(tiny_project_multipls)
    received = []
    runner.finished.connect(lambda r: received.append(r))
    runner.run()
    assert received
    assert received[0].status == "complete"
    res = received[0]._result
    assert hasattr(res, "_leaves")
    assert "combined" in res._leaves
