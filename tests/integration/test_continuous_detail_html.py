"""Renderer smoke tests for PLS / PCA+OLS / MultiPLS detail tabs.

These hit `_pls_fit_html`, `_pca_ols_fit_html`, and `_multipls_fit_html`
with real result objects from `ssd.fit_*`, so an attribute-name drift on
the SSDLite side (e.g. `n_components` → `pca_k`) trips the renderer here
instead of in production.
"""

import numpy as np
import pytest

from ssdiff_gui.views.stage3.tabs.details import (
    _pls_fit_html,
    _pca_ols_fit_html,
    _multipls_fit_html,
)

pytestmark = pytest.mark.spacy


class _FakePalette:
    text_secondary = "#aaa"
    accent = "#0af"
    border = "#333"
    font_size_sm = "11px"
    font_size_base = "12px"


class _FakeView:
    def __init__(self, source, analysis_type):
        self.source = source
        self.meta = type("M", (), {"config_snapshot": {}})()
        self.analysis_type = analysis_type


def _fit_inputs(tiny_embeddings, synthetic_corpus):
    rng = np.random.RandomState(42)
    y = rng.randn(len(synthetic_corpus.docs))
    lexicon = ["happy", "sad", "angry", "love", "hate"]
    return y, lexicon


def test_pls_fit_html_renders(tiny_embeddings, synthetic_corpus):
    from ssdiff import SSD
    y, lexicon = _fit_inputs(tiny_embeddings, synthetic_corpus)
    ssd = SSD(tiny_embeddings, synthetic_corpus, y, lexicon, window=3)
    res = ssd.fit_pls(k=1)

    joined = "".join(_pls_fit_html(_FakeView(res, "pls"), _FakePalette()))

    assert "Model Fit" in joined
    assert "Effect Size" in joined
    assert "Data Retention" in joined
    assert ">PLS<" in joined
    assert ">Components<" in joined
    assert f">{res.n_components}<" in joined


def test_pca_ols_fit_html_renders(tiny_embeddings, synthetic_corpus):
    from ssdiff import SSD
    y, lexicon = _fit_inputs(tiny_embeddings, synthetic_corpus)
    ssd = SSD(tiny_embeddings, synthetic_corpus, y, lexicon, window=3)
    res = ssd.fit_ols(fixed_k=5)

    joined = "".join(_pca_ols_fit_html(_FakeView(res, "pca_ols"), _FakePalette()))

    assert "Model Fit" in joined
    assert "Adjusted R" in joined
    assert ">PCA<" in joined
    assert "Components (K)" in joined
    assert f">{res.pca_k}<" in joined


def test_multipls_fit_html_renders(tiny_embeddings, synthetic_corpus):
    from ssdiff import SSD
    y, lexicon = _fit_inputs(tiny_embeddings, synthetic_corpus)
    ssd = SSD(tiny_embeddings, synthetic_corpus, y, lexicon, window=3)
    res = ssd.fit_multipls(k=2, rotate="varimax")

    joined = "".join(_multipls_fit_html(_FakeView(res, "multipls"), _FakePalette()))

    assert "Model Fit" in joined
    assert ">MultiPLS<" in joined
    assert ">Components<" in joined
    assert f">{res.n_components}<" in joined
    assert "Rotation" in joined
