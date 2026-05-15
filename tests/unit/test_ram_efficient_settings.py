from PySide6.QtCore import QSettings
from ssdiff_gui.utils.paths import ram_efficient_enabled, set_ram_efficient_enabled


def test_default_off(qsettings_clean):
    assert ram_efficient_enabled() is False


def test_toggle_persists(qsettings_clean):
    set_ram_efficient_enabled(True)
    assert ram_efficient_enabled() is True
    set_ram_efficient_enabled(False)
    assert ram_efficient_enabled() is False
