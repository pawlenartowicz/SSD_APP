"""Smoke: every ssdiff_gui.* submodule imports cleanly.

This is the primary CI guardrail for cross-platform safety. Catches:
  - Missing platform-specific deps (e.g. Linux-only import in a Qt module)
  - Broken relative imports after a refactor
  - Syntax errors in rarely-loaded files (settings dialogs, install helpers)

Every new .py file under ssdiff_gui/ is tested automatically — no
enumeration needed.
"""

import importlib
import pkgutil
from pathlib import Path

import pytest

import ssdiff_gui


def _all_submodule_names():
    root = Path(ssdiff_gui.__file__).parent
    names = []
    for mod_info in pkgutil.walk_packages([str(root)], prefix="ssdiff_gui."):
        names.append(mod_info.name)
    return sorted(names)


@pytest.mark.parametrize("module_name", _all_submodule_names())
def test_module_imports(module_name):
    """Every ssdiff_gui.* module must import without error."""
    importlib.import_module(module_name)
