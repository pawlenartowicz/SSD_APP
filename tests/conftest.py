"""Shared test configuration.

Real PySide6 is installed in the test venv. We just need a headless
QApplication so QPainter/QPixmap operations (used by the sweep-plot
exporter) work without a display.
"""

import os
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

# Headless Qt — must be set before any QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session", autouse=True)
def _qt_app():
    """Headless QApplication + isolated QSettings.

    QStandardPaths test mode reroutes QSettings to ``~/.qttest/`` so tests
    never touch the developer's real config (which would otherwise leak
    e.g. the configured embeddings directory into file-listing tests).
    """
    from PySide6.QtCore import QSettings, QStandardPaths
    QStandardPaths.setTestModeEnabled(True)

    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    QSettings("SSD", "SSD").clear()
    yield app

# ── Ensure ssdiff_gui is importable ─────────────────────────────────
_APP_ROOT = Path(__file__).resolve().parent.parent
_SSD_APP = _APP_ROOT / "SSD_APP"
if _SSD_APP.exists() and str(_SSD_APP) not in sys.path:
    sys.path.insert(0, str(_SSD_APP))


# ── Pytest marker registration ──────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs spaCy model + ssdiff computation")
    config.addinivalue_line("markers", "spacy: needs spaCy + en_core_web_sm model")

    # Try to ensure en_core_web_sm is available for @pytest.mark.spacy tests.
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        try:
            import subprocess
            import sys
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
        except Exception:
            pass  # mark will skipif below


# ── Shared fixtures ─────────────────────────────────────────────────

_VOCAB_WORDS = [
    "happy", "sad", "angry", "love", "hate",
    "joy", "fear", "trust", "surprise", "disgust",
    "good", "bad", "great", "terrible", "wonderful",
    "hope", "despair", "calm", "fury", "peace",
    "bright", "dark", "warm", "cold", "gentle",
    "kind", "cruel", "brave", "weak", "strong",
]


@pytest.fixture
def synthetic_df():
    """50-row DataFrame with text, score, group, participant_id columns."""
    import random
    rng = random.Random(42)

    texts = []
    for i in range(50):
        n_words = rng.randint(5, 12)
        words = [rng.choice(_VOCAB_WORDS) for _ in range(n_words)]
        texts.append(" ".join(words))

    groups = ["A"] * 25 + ["B"] * 25
    rng.shuffle(groups)

    # participant IDs with some duplicates
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
    docs = []
    for _ in range(50):
        n_tokens = rng.randint(3, 10)
        doc = [rng.choice(_VOCAB_WORDS) for _ in range(n_tokens)]
        docs.append(doc)
    return docs


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
