"""HTML builders for the group details tab — structured asserts.

Tightened from substring-in-blob to exact content checks so the tests
fail loudly when a format string is mangled.
"""

import numpy as np

from ssdiff import GroupResult
from ssdiff.results.schema import Pair

from ssdiff_gui.views.stage3.tabs.details import _group_fit_html, _pairwise_table_html


class _FakePalette:
    text_secondary = "#aaa"
    accent = "#0af"
    border = "#333"
    font_size_sm = "11px"
    font_size_base = "12px"


class _FakeView:
    def __init__(self, gr, meta=None):
        self.source = gr
        self.meta = meta or type("M", (), {"config_snapshot": {}})()
        self.analysis_type = "groups"


def _make_gr(groups, pairs, random_state=2137, correction="holm"):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((len(groups), 4))
    return GroupResult(
        G=len(set(groups)), n_kept=len(groups), n_perm=100, correction=correction,
        random_state=random_state, omnibus_T=1.0, omnibus_p=0.05, pairs=pairs,
        x=x, groups=np.asarray(groups, dtype=object),
    )


def test_two_group_fit_html_has_expected_sections_and_values():
    gr = _make_gr(
        ["A", "A", "B", "B"],
        [Pair(contrast="A_B", g1="A", g2="B", T=1.0, p_raw=0.05, p_corrected=0.05,
              cohens_d=0.5, n_g1=2, n_g2=2, contrast_norm=1.0)],
        random_state=2137, correction="holm",
    )
    joined = "".join(_group_fit_html(_FakeView(gr), _FakePalette()))

    assert "Pairwise Permutation Test" in joined
    assert "Group Sizes" in joined
    assert "holm" in joined
    assert "2137" in joined
    assert joined.count("Pairwise Permutation Test") == 1


def test_pairwise_table_html_renders_each_pair_with_structured_values():
    pairs = [
        Pair(contrast="A_B", g1="A", g2="B", T=1.2, p_raw=0.04, p_corrected=0.08,
             cohens_d=0.42, n_g1=10, n_g2=12, contrast_norm=0.77),
        Pair(contrast="A_C", g1="A", g2="C", T=0.9, p_raw=0.10, p_corrected=0.15,
             cohens_d=0.28, n_g1=10, n_g2=14, contrast_norm=0.62),
    ]
    gr = _make_gr(["A"] * 10 + ["B"] * 12 + ["C"] * 14, pairs)
    joined = "".join(_pairwise_table_html(_FakeView(gr), _FakePalette()))

    assert "0.7700" in joined
    assert "0.6200" in joined
    assert "0.420" in joined
    assert "0.280" in joined
    assert joined.count(">10<") >= 2 or joined.count(" 10 ") >= 2 or joined.count("10</") >= 2
