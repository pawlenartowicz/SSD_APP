"""_label_spec_for produces the expected 7-card layout per analysis shape."""

from ssdiff_gui.views.stage3.stats_strip import _label_spec_for


class _View:
    def __init__(self, analysis_type, is_multi_pair=False):
        self.analysis_type = analysis_type
        self.is_multi_pair = is_multi_pair


def _labels(view):
    return [card["label"] for card in _label_spec_for(view)]


def test_pls_layout():
    assert _labels(_View("pls")) == [
        "R²", "p", "‖β‖", "Δy / +0.10 cos", "Docs", "IQR Effect", "Corr(y, ŷ)",
    ]


def test_pca_ols_layout():
    assert _labels(_View("pca_ols")) == [
        "R²", "Adj. R²", "p", "‖β‖", "Docs", "Selected K", "Corr(y, ŷ)",
    ]


def test_groups_single_layout():
    assert _labels(_View("groups", is_multi_pair=False)) == [
        "p", "Cohen's d", "‖Contrast‖", "Permutations", "Docs", "n(g1)", "n(g2)",
    ]


def test_groups_multi_layout():
    assert _labels(_View("groups", is_multi_pair=True)) == [
        "Omnibus p", "p(corrected)", "Cohen's d", "‖Contrast‖", "Docs", "n(g1)", "n(g2)",
    ]


def test_all_layouts_have_seven_cards():
    for at, multi in [("pls", False), ("pca_ols", False), ("groups", False), ("groups", True)]:
        assert len(_label_spec_for(_View(at, multi))) == 7
