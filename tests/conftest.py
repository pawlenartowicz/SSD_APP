"""Shared fixtures for the whole suite."""

import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session", autouse=True)
def _qt_app():
    """Headless QApplication + isolated QSettings.

    QStandardPaths test mode reroutes QSettings to ``~/.qttest/`` so tests
    never touch the developer's real config.
    """
    from PySide6.QtCore import QSettings, QStandardPaths
    QStandardPaths.setTestModeEnabled(True)

    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    QSettings("SSD", "SSD").clear()
    yield app


def pytest_configure(config):
    config.addinivalue_line("markers", "spacy: needs spaCy + en_core_web_sm model")


_VOCAB_WORDS = [
    "happy", "sad", "angry", "love", "hate",
    "joy", "fear", "trust", "surprise", "disgust",
    "good", "bad", "great", "terrible", "wonderful",
    "hope", "despair", "calm", "fury", "peace",
    "bright", "dark", "warm", "cold", "gentle",
    "kind", "cruel", "brave", "weak", "strong",
]


class FakeSettings:
    """Minimal in-memory stand-in for QSettings."""

    def __init__(self, initial=None):
        self._store = dict(initial or {})
        self.removed: list[str] = []

    def value(self, key, default=None):
        return self._store.get(key, default)

    def setValue(self, key, value):
        self._store[key] = value

    def remove(self, key):
        self.removed.append(key)
        self._store.pop(key, None)


class StringFakeSettings(FakeSettings):
    """FakeSettings that stores everything as strings — simulates QSettings text backend."""

    def setValue(self, key, value):
        self._store[key] = str(value)


@pytest.fixture
def fake_settings():
    return FakeSettings()


@pytest.fixture
def string_fake_settings():
    return StringFakeSettings()


@pytest.fixture
def synthetic_df():
    """50-row DataFrame with text, score, group, participant_id columns."""
    import random
    rng = random.Random(42)
    texts = []
    for _ in range(50):
        n_words = rng.randint(5, 12)
        words = [rng.choice(_VOCAB_WORDS) for _ in range(n_words)]
        texts.append(" ".join(words))
    groups = ["A"] * 25 + ["B"] * 25
    rng.shuffle(groups)
    pids = [f"P{i:03d}" for i in range(45)] + [f"P{i:03d}" for i in range(5)]
    return pd.DataFrame({
        "text": texts,
        "score": [rng.uniform(1.0, 5.0) for _ in range(50)],
        "group": groups,
        "participant_id": pids,
    })


@pytest.fixture
def mock_kv():
    """Mock object with key_to_index dict for 30 common words."""
    kv = MagicMock()
    kv.key_to_index = {w: i for i, w in enumerate(_VOCAB_WORDS)}
    return kv


@pytest.fixture
def mock_docs():
    """List of 50 token lists built from the same vocab."""
    import random
    rng = random.Random(42)
    return [
        [rng.choice(_VOCAB_WORDS) for _ in range(rng.randint(3, 10))]
        for _ in range(50)
    ]


@pytest.fixture
def make_project(tmp_path):
    """Factory that creates a Project with project_path set."""
    from ssdiff_gui.models.project import Project

    def _factory(**overrides):
        defaults = dict(
            project_path=tmp_path / "test_project",
            name="test",
            created_date=datetime(2026, 1, 1),
            modified_date=datetime(2026, 4, 14),
        )
        defaults.update(overrides)
        defaults["project_path"].mkdir(parents=True, exist_ok=True)
        return Project(**defaults)

    return _factory


@pytest.fixture
def make_result(tmp_path):
    """Factory that creates a Result under tmp_path/results/<id>."""
    from ssdiff_gui.models.project import Result

    def _factory(result_id="20260414_120000", **overrides):
        result_path = tmp_path / "results" / result_id
        result_path.mkdir(parents=True, exist_ok=True)
        defaults = dict(
            result_id=result_id,
            timestamp=datetime(2026, 4, 14, 12, 0, 0),
            result_path=result_path,
            config_snapshot={"analysis_type": "pls", "pls_n_components": 1},
        )
        defaults.update(overrides)
        return Result(**defaults)

    return _factory
