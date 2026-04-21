"""Drift detection: Project dataclass defaults must match SSDLite fit_* signatures."""

from inspect import signature

import pytest

from ssdiff import SSD
from ssdiff_gui.models.project import Project

_CASES = [
    ("pls_n_components", SSD.fit_pls, "n_components"),
    ("pls_p_method",     SSD.fit_pls, "p_method"),
    ("pls_n_perm",       SSD.fit_pls, "n_perm"),
    ("pls_n_splits",     SSD.fit_pls, "n_splits"),
    ("pls_split_ratio",  SSD.fit_pls, "split_ratio"),
    ("sweep_k_min",      SSD.fit_ols, "k_min"),
    ("sweep_k_max",      SSD.fit_ols, "k_max"),
    ("sweep_k_step",     SSD.fit_ols, "k_step"),
    ("groups_n_perm",     SSD.fit_groups, "n_perm"),
    ("groups_correction", SSD.fit_groups, "correction"),
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
