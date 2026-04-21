"""Tests for PairView / resolve_pair_data."""

import numpy as np
import pytest

from ssdiff import GroupResult
from ssdiff.results.schema import Pair

from ssdiff_gui.views._pair_view import PairView, resolve_pair_data


def _make_gr(groups_array, pairs):
    """Build a minimal GroupResult from raw arrays + Pair rows."""
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


def test_single_pair_returns_pair_key_none():
    gr = _make_gr(
        ["A", "A", "B", "B"],
        [Pair(contrast="A_B", g1="A", g2="B",
              T=1.0, p_raw=0.05, p_corrected=0.05,
              cohens_d=0.5, n_g1=2, n_g2=2, contrast_norm=1.0)],
    )
    view = resolve_pair_data(gr, pair_key=None)
    assert isinstance(view, PairView)
    assert view.pair_key is None
    assert view.alignment_scores.shape == (4,)
    assert view.beta.shape == (4,)
    assert np.isclose(np.linalg.norm(view.gradient), 1.0)


def test_n_pair_requires_canonical_key():
    pairs = [
        Pair(contrast="A_B", g1="A", g2="B", T=1.0, p_raw=0.05, p_corrected=0.1,
             cohens_d=0.3, n_g1=2, n_g2=2, contrast_norm=1.0),
        Pair(contrast="A_C", g1="A", g2="C", T=1.2, p_raw=0.03, p_corrected=0.09,
             cohens_d=0.4, n_g1=2, n_g2=2, contrast_norm=1.1),
        Pair(contrast="B_C", g1="B", g2="C", T=0.8, p_raw=0.10, p_corrected=0.20,
             cohens_d=0.2, n_g1=2, n_g2=2, contrast_norm=0.9),
    ]
    gr = _make_gr(["A", "A", "B", "B", "C", "C"], pairs)

    canonical_keys = [(p.g1, p.g2) for p in gr.pairs]
    view = resolve_pair_data(gr, pair_key=canonical_keys[0])
    assert view.pair_key == canonical_keys[0]
    assert view.alignment_scores.shape == (6,)
    assert view.beta.shape == (4,)
    assert np.isclose(np.linalg.norm(view.gradient), 1.0)


def test_n_pair_without_key_raises():
    pairs = [
        Pair(contrast="A_B", g1="A", g2="B", T=1.0, p_raw=0.05, p_corrected=0.05,
             cohens_d=0.3, n_g1=2, n_g2=2, contrast_norm=1.0),
        Pair(contrast="A_C", g1="A", g2="C", T=1.0, p_raw=0.05, p_corrected=0.05,
             cohens_d=0.3, n_g1=2, n_g2=2, contrast_norm=1.0),
    ]
    gr = _make_gr(["A", "A", "B", "B", "C", "C"], pairs)
    with pytest.raises(ValueError, match="pair_key is required"):
        resolve_pair_data(gr, pair_key=None)


def test_unknown_pair_raises_keyerror():
    pairs = [
        Pair(contrast="A_B", g1="A", g2="B", T=1.0, p_raw=0.05, p_corrected=0.05,
             cohens_d=0.3, n_g1=2, n_g2=2, contrast_norm=1.0),
        Pair(contrast="A_C", g1="A", g2="C", T=1.0, p_raw=0.05, p_corrected=0.05,
             cohens_d=0.3, n_g1=2, n_g2=2, contrast_norm=1.0),
    ]
    gr = _make_gr(["A", "A", "B", "B", "C", "C"], pairs)
    with pytest.raises(KeyError, match="unknown pair"):
        resolve_pair_data(gr, pair_key=("g99", "g100"))


def test_non_groupresult_rejected():
    with pytest.raises(TypeError, match="GroupResult"):
        resolve_pair_data(object(), pair_key=None)
