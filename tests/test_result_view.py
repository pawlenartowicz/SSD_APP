"""ResultView.build across all 4 result shapes."""

import numpy as np
import pytest

from ssdiff import GroupResult, PLSResult, PCAOLSResult
from ssdiff.results.schema import Pair

from ssdiff_gui.views.stage3.result_view import ResultView


def _bare_pls():
    return object.__new__(PLSResult)


def _bare_pca_ols():
    return object.__new__(PCAOLSResult)


def _make_gr(groups_array, pairs):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((len(groups_array), 4))
    return GroupResult(
        G=len(set(groups_array)),
        n_kept=len(groups_array),
        n_perm=100,
        correction="holm",
        random_state=2137,
        omnibus_T=1.0,
        omnibus_p=0.05,
        pairs=pairs,
        x=x,
        groups=np.asarray(groups_array, dtype=object),
    )


def _pair(g1, g2):
    return Pair(
        contrast=f"{g1}_{g2}", g1=g1, g2=g2,
        T=1.0, p_raw=0.05, p_corrected=0.05,
        cohens_d=0.3, n_g1=2, n_g2=2, contrast_norm=1.0,
    )


def test_pls_result_working_is_source():
    src = _bare_pls()
    view = ResultView.build(src)
    assert view.source is src
    assert view.working is src
    assert view.analysis_type == "pls"
    assert view.pairs == []
    assert view.current_pair is None
    assert view.is_group is False
    assert view.is_multi_pair is False


def test_pca_ols_result_working_is_source():
    src = _bare_pca_ols()
    view = ResultView.build(src)
    assert view.analysis_type == "pca_ols"
    assert view.working is src
    assert view.is_group is False


def test_two_group_result_working_is_pair_leaf():
    gr = _make_gr(["A", "A", "B", "B"], [_pair("A", "B")])
    view = ResultView.build(gr)
    assert view.analysis_type == "groups"
    assert view.source is gr
    assert view.is_group is True
    assert view.is_multi_pair is False
    assert view.pairs == [("A", "B")]
    assert view.current_pair == ("A", "B")
    assert (view.current_pair_row.g1, view.current_pair_row.g2) == ("A", "B")


def test_n_group_result_slices_to_current_pair():
    pairs = [_pair("A", "B"), _pair("A", "C"), _pair("B", "C")]
    gr = _make_gr(["A", "A", "B", "B", "C", "C"], pairs)
    view = ResultView.build(gr, current_pair=("A", "C"))
    assert view.is_group is True
    assert view.is_multi_pair is True
    assert view.current_pair == ("A", "C")
    assert view.source is gr
    assert view.working is not gr
    assert (view.current_pair_row.g1, view.current_pair_row.g2) == ("A", "C")


def test_n_group_result_defaults_to_first_pair():
    pairs = [_pair("A", "B"), _pair("A", "C"), _pair("B", "C")]
    gr = _make_gr(["A", "A", "B", "B", "C", "C"], pairs)
    view = ResultView.build(gr)
    assert view.current_pair == ("A", "B")
    assert (view.current_pair_row.g1, view.current_pair_row.g2) == ("A", "B")


def test_n_group_unknown_pair_raises():
    pairs = [_pair("A", "B"), _pair("A", "C"), _pair("B", "C")]
    gr = _make_gr(["A", "A", "B", "B", "C", "C"], pairs)
    with pytest.raises(KeyError):
        ResultView.build(gr, current_pair=("X", "Y"))


def test_pls_with_current_pair_is_ignored():
    src = _bare_pls()
    view = ResultView.build(src, current_pair=("A", "B"))
    assert view.current_pair is None
    assert view.pairs == []
