"""Drift detection: Project dataclass defaults must match SSDLite fit_* signatures."""

from inspect import signature

import pytest

from ssdiff import SSD
from ssdiff_gui.models.project import Project

_CASES = [
    ("pls_k",            SSD.fit_pls, "k"),
    ("pls_k_max",        SSD.fit_pls, "k_max"),
    ("pls_n_splits",     SSD.fit_pls, "n_splits"),
    ("pcaols_n_components", SSD.fit_ols, "fixed_k"),
    ("sweep_k_min",      SSD.fit_ols, "k_min"),
    ("sweep_k_max",      SSD.fit_ols, "k_max"),
    ("sweep_k_step",     SSD.fit_ols, "k_step"),
    ("groups_n_perm",     SSD.fit_groups, "n_perm"),
    ("groups_correction", SSD.fit_groups, "correction"),
    ("multipls_k",               SSD.fit_multipls, "k"),
    ("multipls_k_max",           SSD.fit_multipls, "k_max"),
    ("multipls_rotate",          SSD.fit_multipls, "rotate"),
    ("multipls_rotation_vocab",  SSD.fit_multipls, "rotation_vocab"),
    ("multipls_n_splits",        SSD.fit_multipls, "n_splits"),
]


@pytest.mark.parametrize("project_field,method,param_name", _CASES)
def test_project_defaults_match_ssdlite(project_field, method, param_name):
    expected = signature(method).parameters[param_name].default
    actual = getattr(Project.__dataclass_fields__[project_field], "default")
    assert actual == expected, (
        f"{project_field}={actual!r} diverged from "
        f"SSD.{method.__name__}({param_name}={expected!r}); "
        f"re-derive from signature or update SSDLite in lockstep."
    )
