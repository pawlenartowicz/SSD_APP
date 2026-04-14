"""Tests for SSDRunner._resolve_random_state() — pure function, fast."""

from ssdiff_gui.controllers.ssd_runner import SSDRunner


class TestResolveRandomState:
    def setup_method(self):
        self.runner = SSDRunner.__new__(SSDRunner)

    def test_default(self):
        assert self.runner._resolve_random_state("default") == 2137

    def test_numeric_string(self):
        assert self.runner._resolve_random_state("42") == 42

    def test_invalid_fallback(self):
        assert self.runner._resolve_random_state("garbage") == 2137
