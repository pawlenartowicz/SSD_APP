"""Tab visibility matrix — design doc §6."""

from ssdiff_gui.views.stage3.tabs.cluster_overview import ClusterOverviewTab
from ssdiff_gui.views.stage3.tabs.pca_sweep import PcaSweepTab
from ssdiff_gui.views.stage3.tabs.snippets import SnippetsTab
from ssdiff_gui.views.stage3.tabs.poles import PolesTab
from ssdiff_gui.views.stage3.tabs.scores import ScoresTab
from ssdiff_gui.views.stage3.tabs.extreme_docs import ExtremeDocsTab
from ssdiff_gui.views.stage3.tabs.misdiagnosed import MisdiagnosedTab
# DetailsTab added in Task 10


class _View:
    def __init__(self, analysis_type, is_multi_pair=False):
        self.analysis_type = analysis_type
        self.is_multi_pair = is_multi_pair
        self.is_group = analysis_type == "groups"


SHAPES = {
    "pls": _View("pls"),
    "pca_ols": _View("pca_ols"),
    "groups_single": _View("groups", is_multi_pair=False),
    "groups_multi": _View("groups", is_multi_pair=True),
}


_STUBS = {
    ClusterOverviewTab: lambda: ClusterOverviewTab(lambda: None, lambda _id: ""),
    PcaSweepTab:        lambda: PcaSweepTab(lambda: None),
    ScoresTab:          lambda: ScoresTab(lambda _id: ""),
    ExtremeDocsTab:     lambda: ExtremeDocsTab(lambda _id: ""),
    MisdiagnosedTab:    lambda: MisdiagnosedTab(lambda _id: ""),
    SnippetsTab:        lambda: SnippetsTab(),
    PolesTab:           lambda: PolesTab(),
}


def _mk(cls):
    try:
        return _STUBS[cls]()
    except KeyError:
        raise TypeError(f"_mk: no stub registered for {cls.__name__}; add one to _STUBS")


MATRIX = {
    ClusterOverviewTab: {"pls": True,  "pca_ols": True,  "groups_single": True,  "groups_multi": True},
    PcaSweepTab:        {"pls": False, "pca_ols": True,  "groups_single": False, "groups_multi": False},
    SnippetsTab:        {"pls": True,  "pca_ols": True,  "groups_single": True,  "groups_multi": True},
    PolesTab:           {"pls": True,  "pca_ols": True,  "groups_single": True,  "groups_multi": True},
    ScoresTab:          {"pls": True,  "pca_ols": True,  "groups_single": True,  "groups_multi": True},
    ExtremeDocsTab:     {"pls": True,  "pca_ols": True,  "groups_single": True,  "groups_multi": True},
    MisdiagnosedTab:    {"pls": True,  "pca_ols": True,  "groups_single": False, "groups_multi": False},
}


def test_matrix():
    for cls, expectations in MATRIX.items():
        tab = _mk(cls)
        for shape_name, expected in expectations.items():
            view = SHAPES[shape_name]
            actual = tab.is_visible_for(view)
            assert actual == expected, (
                f"{cls.__name__}.is_visible_for({shape_name!r}) "
                f"returned {actual!r}, expected {expected!r}"
            )
