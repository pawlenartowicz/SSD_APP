"""HTML builders for the group details tab — pure-function tests."""

import numpy as np

from ssdiff import GroupResult
from ssdiff.results.schema import Pair

from ssdiff_gui.views.stage3_results import Stage3Widget as Stage3Results


class _FakePalette:
    text_secondary = "#aaa"
    accent = "#0af"
    border = "#333"
    font_size_sm = "11px"
    font_size_base = "12px"


def _make_gr(groups, pairs):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((len(groups), 4))
    return GroupResult(
        G=len(set(groups)), n_kept=len(groups), n_perm=100, correction="holm",
        random_state=2137, omnibus_T=1.0, omnibus_p=0.05, pairs=pairs,
        x=x, groups=np.asarray(groups, dtype=object),
    )


def test_build_group_detail_html_two_group():
    gr = _make_gr(
        ["A", "A", "B", "B"],
        [Pair(contrast="A_B", g1="A", g2="B", T=1.0, p_raw=0.05, p_corrected=0.05,
              cohens_d=0.5, n_g1=2, n_g2=2, contrast_norm=1.0)],
    )
    html = Stage3Results._build_group_detail_html(gr, "s", "l", "v", _FakePalette())
    joined = "".join(html)
    assert "Pairwise Permutation Test" in joined
    assert "Group Sizes" in joined
    assert "holm" in joined
    assert "random" in joined.lower() or "Random" in joined


def test_build_group_detail_html_n_group_shows_omnibus():
    pairs = [
        Pair(contrast="A_B", g1="A", g2="B", T=1.0, p_raw=0.05, p_corrected=0.10,
             cohens_d=0.3, n_g1=2, n_g2=2, contrast_norm=1.0),
        Pair(contrast="A_C", g1="A", g2="C", T=1.0, p_raw=0.05, p_corrected=0.10,
             cohens_d=0.3, n_g1=2, n_g2=2, contrast_norm=1.0),
        Pair(contrast="B_C", g1="B", g2="C", T=1.0, p_raw=0.05, p_corrected=0.10,
             cohens_d=0.3, n_g1=2, n_g2=2, contrast_norm=1.0),
    ]
    gr = _make_gr(["A", "A", "B", "B", "C", "C"], pairs)
    html = Stage3Results._build_group_detail_html(gr, "s", "l", "v", _FakePalette())
    joined = "".join(html)
    assert "Omnibus Permutation Test" in joined


def test_build_pairwise_html_iterates_pairs():
    pairs = [
        Pair(contrast="A_B", g1="A", g2="B", T=1.2, p_raw=0.04, p_corrected=0.08,
             cohens_d=0.42, n_g1=10, n_g2=12, contrast_norm=0.77),
    ]
    gr = _make_gr(["A"] * 10 + ["B"] * 12, pairs)
    html = Stage3Results._build_pairwise_html(gr, "s", "l", _FakePalette())
    joined = "".join(html)
    assert "0.7700" in joined  # contrast_norm
    assert "0.420" in joined   # cohens_d (3-decimal)
    assert "10" in joined and "12" in joined
