"""Stats strip — 7-card value display driven by ResultView."""

from __future__ import annotations

import numpy as np

_LABEL_SETS: dict[tuple[str, bool], list[str]] = {
    ("pls", False): ["R²", "p", "‖β‖", "Δy / +0.10 cos", "Docs", "IQR Effect", "Corr(y, ŷ)"],
    ("pls", True):  ["R²", "p", "‖β‖", "Δy / +0.10 cos", "Docs", "IQR Effect", "Corr(y, ŷ)"],
    ("pca_ols", False): ["R²", "Adj. R²", "p", "‖β‖", "Docs", "Selected K", "Corr(y, ŷ)"],
    ("pca_ols", True):  ["R²", "Adj. R²", "p", "‖β‖", "Docs", "Selected K", "Corr(y, ŷ)"],
    ("groups", False): ["p", "Cohen's d", "‖Contrast‖", "Permutations", "Docs", "n(g1)", "n(g2)"],
    ("groups", True):  ["Omnibus p", "p(corrected)", "Cohen's d", "‖Contrast‖", "Docs", "n(g1)", "n(g2)"],
}


def _label_spec_for(view) -> list[dict]:
    """Return [{"label": str, "value": str|None}].

    When view has no `working` attribute (e.g. test shims), only labels are
    returned with value=None.  When `working` is present, values are filled.
    """
    labels = _LABEL_SETS[(view.analysis_type, view.is_multi_pair)]
    working = getattr(view, "working", None)
    if working is None:
        return [{"label": lb, "value": None} for lb in labels]
    return _fill_values(view, labels)


def _fmt_float(x, digits=3, default="—"):
    if x is None:
        return default
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(x):
        return default
    return f"{x:.{digits}f}"


def _fmt_p(p):
    if p is None:
        return "—"
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _fmt_int(n):
    if n is None:
        return "—"
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "—"


def _pls_values(view) -> list[str]:
    r = view.working
    s = r.stats
    return [
        _fmt_float(s.r2, 4),
        _fmt_p(s.pvalue),
        _fmt_float(s.beta_norm, 4),
        _fmt_float(s.delta, 4),
        _fmt_int(s.n_kept),
        _fmt_float(s.iqr_effect, 4),
        _fmt_float(s.y_corr_pred, 4),
    ]


def _pca_ols_values(view) -> list[str]:
    r = view.working
    s = r.stats
    k = getattr(r, "pca_k", None)
    k_text = _fmt_int(k) if k else "—"
    return [
        _fmt_float(s.r2, 4),
        _fmt_float(s.r2_adj, 4),
        _fmt_p(s.pvalue),
        _fmt_float(s.beta_norm, 4),
        _fmt_int(s.n_kept),
        k_text,
        _fmt_float(s.y_corr_pred, 4),
    ]


def _groups_single_values(view) -> list[str]:
    cr = view.current_pair_row
    if cr is None:
        return ["—"] * 7
    src = view.source
    return [
        _fmt_p(cr.p_raw),
        _fmt_float(cr.cohens_d, 3),
        _fmt_float(cr.contrast_norm, 4),
        _fmt_int(getattr(src, "n_perm", None)),
        _fmt_int(getattr(src, "n_kept", None)),
        _fmt_int(cr.n_g1),
        _fmt_int(cr.n_g2),
    ]


def _groups_multi_values(view) -> list[str]:
    cr = view.current_pair_row
    if cr is None:
        return ["—"] * 7
    src = view.source
    omnibus_p = getattr(getattr(src, "test", None), "omnibus_p", None)
    return [
        _fmt_p(omnibus_p),
        _fmt_p(cr.p_corrected),
        _fmt_float(cr.cohens_d, 3),
        _fmt_float(cr.contrast_norm, 4),
        _fmt_int(getattr(src, "n_kept", None)),
        _fmt_int(cr.n_g1),
        _fmt_int(cr.n_g2),
    ]


def _fill_values(view, labels) -> list[dict]:
    at = view.analysis_type
    multi = view.is_multi_pair

    if at == "pls":
        values = _pls_values(view)
    elif at == "pca_ols":
        values = _pca_ols_values(view)
    elif at == "groups" and not multi:
        values = _groups_single_values(view)
    elif at == "groups" and multi:
        values = _groups_multi_values(view)
    else:
        values = ["—"] * len(labels)

    return [{"label": lb, "value": v} for lb, v in zip(labels, values)]


def apply(card_widgets, view) -> None:
    """Fill the 7 (label_widget, value_widget) tuples from view."""
    spec = _label_spec_for(view)
    for (label_widget, value_widget), card in zip(card_widgets, spec):
        label_widget.setText(card["label"])
        value_widget.setText(card["value"] if card["value"] is not None else "—")
