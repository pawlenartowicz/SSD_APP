"""Platform-specific path behavior for ssdiff_gui.utils.paths.

Guards against:
  - XDG_DATA_HOME / LOCALAPPDATA env handling drifting
  - QSettings string-backend bool coercion regressions
  - embeddings_dir() silently changing when location_mode setting is absent
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from ssdiff_gui.utils.paths import (
    _qsetting_bool,
    embeddings_autoload_enabled,
    embeddings_dir,
    get_app_data_dir,
    projects_dir,
)


class TestGetAppDataDir:
    def test_linux_default(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        assert get_app_data_dir() == Path.home() / ".local" / "share" / "SSD"

    def test_linux_respects_xdg(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        assert get_app_data_dir() == tmp_path / "SSD"

    def test_macos(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        assert get_app_data_dir() == (
            Path.home() / "Library" / "Application Support" / "SSD"
        )

    def test_windows_with_localappdata(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        assert get_app_data_dir() == tmp_path / "SSD"

    def test_windows_fallback_without_env(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        assert get_app_data_dir() == Path.home() / "AppData" / "Local" / "SSD"


class TestQSettingBool:
    @pytest.mark.parametrize("value,expected", [
        ("true", True), ("True", True), ("TRUE", True),
        ("1", True), ("yes", True), ("on", True),
        ("false", False), ("0", False), ("no", False), ("off", False),
        (" true ", True),
        (True, True), (False, False),
        (1, True), (0, False),
    ])
    def test_coercions(self, value, expected):
        assert _qsetting_bool(value, default=not expected) is expected

    def test_none_returns_default(self):
        assert _qsetting_bool(None, default=True) is True
        assert _qsetting_bool(None, default=False) is False


_APP_SETTINGS = "ssdiff_gui.utils.settings.app_settings"


class TestProjectsDir:
    def test_default_is_home_subdir(self, fake_settings):
        with patch(_APP_SETTINGS, return_value=fake_settings):
            assert projects_dir() == Path.home() / "SSD-Projects"

    def test_override_from_settings(self, fake_settings, tmp_path):
        fake_settings.setValue("projects_directory", str(tmp_path))
        with patch(_APP_SETTINGS, return_value=fake_settings):
            assert projects_dir() == tmp_path


class TestEmbeddingsDir:
    def test_shared_mode_default(self, fake_settings):
        with patch(_APP_SETTINGS, return_value=fake_settings):
            result = embeddings_dir()
        assert result == Path.home() / "SSD-Projects" / "SSD-Embeddings"

    def test_custom_mode(self, fake_settings, tmp_path):
        fake_settings.setValue("embeddings/location_mode", "custom")
        fake_settings.setValue("embeddings/custom_path", str(tmp_path))
        with patch(_APP_SETTINGS, return_value=fake_settings):
            assert embeddings_dir() == tmp_path

    def test_custom_mode_empty_path_falls_back_to_shared(self, fake_settings):
        fake_settings.setValue("embeddings/location_mode", "custom")
        fake_settings.setValue("embeddings/custom_path", "")
        with patch(_APP_SETTINGS, return_value=fake_settings):
            result = embeddings_dir()
        assert result == Path.home() / "SSD-Projects" / "SSD-Embeddings"


class TestEmbeddingsAutoload:
    def test_default_true(self, fake_settings):
        with patch(_APP_SETTINGS, return_value=fake_settings):
            assert embeddings_autoload_enabled() is True

    def test_string_false_coerced(self, string_fake_settings):
        string_fake_settings.setValue("embeddings/autoload_on_open", False)
        with patch(_APP_SETTINGS, return_value=string_fake_settings):
            assert embeddings_autoload_enabled() is False
