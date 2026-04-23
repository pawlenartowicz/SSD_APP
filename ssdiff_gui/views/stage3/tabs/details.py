"""Details tab — single-panel HTML summary of a completed SSD result."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ssdiff import GroupResult, PCAOLSResult, PLSResult

from .. import html_helpers


def _styles(p):
    label_style = (
        f"color: {p.text_secondary}; font-size: {p.font_size_sm}; text-transform: uppercase;"
    )
    value_style = f"font-size: {p.font_size_base}; padding-left: 8px;"
    section_style = (
        f"color: {p.accent}; font-size: 12px; font-weight: 600; "
        f"border-bottom: 1px solid {p.border}; padding-bottom: 4px; margin: 12px 0 8px 0;"
    )
    return label_style, value_style, section_style


# ---------------------------------------------------------------------------
# HTML builder pure functions
# ---------------------------------------------------------------------------


def _common_header_html(view, p) -> list[str]:
    """Result ID, Timestamp, Status, Analysis Type, Mode, Outcome/Group column, Dataset."""
    label_style, value_style, section_style = _styles(p)
    meta = view.meta
    s = meta.config_snapshot if meta is not None else {}
    res = view.source
    is_crossgroup = view.analysis_type == "groups"
    is_fulldoc = s.get("concept_mode", "lexicon") == "fulldoc"

    html: list[str] = []
    html.append('<table cellspacing="6" style="width: 100%;">')

    result_id = meta.result_id if meta is not None else "N/A"
    html.append(
        f'<tr><td style="{label_style}">Result ID</td>'
        f'<td style="{value_style}">{result_id}</td></tr>'
    )

    if meta is not None:
        ts = meta.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        html.append(
            f'<tr><td style="{label_style}">Timestamp</td>'
            f'<td style="{value_style}">{ts}</td></tr>'
        )
        html.append(
            f'<tr><td style="{label_style}">Status</td>'
            f'<td style="{value_style}">{meta.status}</td></tr>'
        )

    if isinstance(res, PLSResult):
        atype_label = "PLS"
    elif isinstance(res, PCAOLSResult):
        atype_label = "PCA+OLS"
    elif isinstance(res, GroupResult):
        atype_label = "Group Comparison"
    else:
        atype_label = type(res).__name__

    html.append(
        f'<tr><td style="{label_style}">Analysis Type</td>'
        f'<td style="{value_style}">{atype_label}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">Mode</td>'
        f'<td style="{value_style}">{"Full Document" if is_fulldoc else "Lexicon"}</td></tr>'
    )

    if is_crossgroup and s.get("group_column", ""):
        html.append(
            f'<tr><td style="{label_style}">Group Column</td>'
            f'<td style="{value_style}">{s.get("group_column", "")}</td></tr>'
        )
    elif not is_crossgroup and s.get("outcome_column", ""):
        html.append(
            f'<tr><td style="{label_style}">Outcome Column</td>'
            f'<td style="{value_style}">{s.get("outcome_column", "")}</td></tr>'
        )

    html.append("</table>")

    # Dataset sub-section
    html.append(f'<div style="{section_style}">Dataset</div>')
    html.append('<table cellspacing="6" style="width: 100%;">')
    if s.get("csv_path", ""):
        html.append(
            f'<tr><td colspan="2" style="{label_style}">File</td></tr>'
            f'<tr><td colspan="2" style="{value_style}; word-break: break-all;">{s.get("csv_path", "")}</td></tr>'
        )
    if s.get("text_column", ""):
        html.append(
            f'<tr><td style="{label_style}">Text Column</td>'
            f'<td style="{value_style}">{s.get("text_column", "")}</td></tr>'
        )
    if s.get("id_column"):
        html.append(
            f'<tr><td style="{label_style}">ID Column</td>'
            f'<td style="{value_style}">{s.get("id_column")}</td></tr>'
        )
    html.append(
        f'<tr><td style="{label_style}">Documents</td>'
        f'<td style="{value_style}">{s.get("n_docs_processed", 0):,}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">Valid Samples</td>'
        f'<td style="{value_style}">{s.get("n_valid", 0):,}</td></tr>'
    )
    if s.get("id_column"):
        html.append(
            f'<tr><td style="{label_style}">Personal Concept Vectors</td>'
            f'<td style="{value_style}">{s.get("n_docs_processed", 0):,}</td></tr>'
        )
    if s.get("mean_words_before_stopwords", 0) > 0:
        html.append(
            f'<tr><td style="{label_style}">Mean Words / Doc (pre-stopword)</td>'
            f'<td style="{value_style}">{s.get("mean_words_before_stopwords", 0):.1f}</td></tr>'
        )
    if s.get("n_docs_processed", 0):
        avg_tokens_post = s.get("total_tokens", 0) / s.get("n_docs_processed", 1)
        html.append(
            f'<tr><td style="{label_style}">Mean Tokens / Doc (post-stopword)</td>'
            f'<td style="{value_style}">{avg_tokens_post:.1f}</td></tr>'
        )
    html.append("</table>")
    return html


def _run_config_html(view, p) -> list[str]:
    """Embeddings, spaCy, Hyperparameters, and Clustering sections."""
    label_style, value_style, section_style = _styles(p)
    meta = view.meta
    s = meta.config_snapshot if meta is not None else {}
    is_fulldoc = s.get("concept_mode", "lexicon") == "fulldoc"

    html: list[str] = []

    # Embeddings section
    html.append(f'<div style="{section_style}">Embeddings</div>')
    html.append('<table cellspacing="6" style="width: 100%;">')
    html.append(
        f'<tr><td style="{label_style}">File</td>'
        f'<td style="{value_style}; word-break: break-all;">{s.get("selected_embedding", "") or "N/A"}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">Vocabulary</td>'
        f'<td style="{value_style}">{s.get("vocab_size", 0):,} words</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">Dimensions</td>'
        f'<td style="{value_style}">{s.get("embedding_dim", 0)}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">L2 Normalize</td>'
        f'<td style="{value_style}">{"Yes" if s.get("l2_normalized", False) else "No"}</td></tr>'
    )
    _abtt = s.get("abtt", 0)
    html.append(
        f'<tr><td style="{label_style}">ABTT</td>'
        f'<td style="{value_style}">{"Yes" if _abtt > 0 else "No"} (m={_abtt})</td></tr>'
    )
    if s.get("coverage_pct", 0) > 0:
        cov_str = f"{s.get('coverage_pct', 0):.1f}%"
        if s.get("n_oov", 0) > 0:
            cov_str += f" ({s.get('n_oov', 0):,} OOV)"
        html.append(
            f'<tr><td style="{label_style}">Coverage</td>'
            f'<td style="{value_style}">{cov_str}</td></tr>'
        )
    html.append("</table>")

    # spaCy section
    html.append(f'<div style="{section_style}">spaCy</div>')
    html.append('<table cellspacing="6" style="width: 100%;">')
    _input_mode = s.get("input_mode", "language")
    _spacy_model = s.get("spacy_model", "")
    _language = s.get("language", "en")
    if _input_mode == "custom" and _spacy_model:
        display_model = _spacy_model
    else:
        try:
            from ssdiff.lang_config import lang_to_model
            display_model = lang_to_model(_language)
        except (ImportError, KeyError):
            display_model = f"{_language}_core_news_lg"
    html.append(
        f'<tr><td style="{label_style}">Model</td>'
        f'<td style="{value_style}">{display_model}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">Language</td>'
        f'<td style="{value_style}">{_language}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">Documents</td>'
        f'<td style="{value_style}">{s.get("n_docs_processed", 0):,}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">Input Mode</td>'
        f'<td style="{value_style}">{_input_mode}</td></tr>'
    )
    stopword_labels = {"default": "Default", "none": "Disabled", "custom": "Custom file"}
    _stopword_mode = s.get("stopword_mode", "default")
    html.append(
        f'<tr><td style="{label_style}">Stopwords</td>'
        f'<td style="{value_style}">{stopword_labels.get(_stopword_mode, _stopword_mode)}</td></tr>'
    )
    html.append("</table>")

    atype = view.analysis_type
    html.append(f'<div style="{section_style}">Hyperparameters</div>')
    html.append('<table cellspacing="6" style="width: 100%;">')

    if not is_fulldoc:
        html.append(
            f'<tr><td style="{label_style}">Context Window</td>'
            f'<td style="{value_style}">\u00b1{s.get("context_window_size", 5)}</td></tr>'
        )
    html.append(
        f'<tr><td style="{label_style}">SIF Parameter</td>'
        f'<td style="{value_style}">{s.get("sif_a", 1e-3)}</td></tr>'
    )

    if atype == "pls":
        _pls_n_comp = s.get("pls_n_components", 0)
        n_comp_display = "auto" if _pls_n_comp == 0 else _pls_n_comp
        html.append(
            f'<tr><td style="{label_style}">PLS Components</td>'
            f'<td style="{value_style}">{n_comp_display}</td></tr>'
        )
        _pls_p_method = s.get("pls_p_method", "auto")
        html.append(
            f'<tr><td style="{label_style}">p-value Method</td>'
            f'<td style="{value_style}">{_pls_p_method}</td></tr>'
        )
        if _pls_p_method in ("perm", "auto"):
            html.append(
                f'<tr><td style="{label_style}">n Permutations</td>'
                f'<td style="{value_style}">{s.get("pls_n_perm", 1000):,}</td></tr>'
            )
        if _pls_p_method in ("split", "split_cal", "auto"):
            from ssdiff_gui.models.project import Project
            html.append(
                f'<tr><td style="{label_style}">n Splits</td>'
                f'<td style="{value_style}">{s.get("pls_n_splits", Project.__dataclass_fields__["pls_n_splits"].default)}</td></tr>'
            )
            html.append(
                f'<tr><td style="{label_style}">Split Ratio</td>'
                f'<td style="{value_style}">{s.get("pls_split_ratio", 0.5)}</td></tr>'
            )
        if s.get("pls_pca_preprocess") is not None:
            html.append(
                f'<tr><td style="{label_style}">PCA Preprocess</td>'
                f'<td style="{value_style}">{s.get("pls_pca_preprocess")}</td></tr>'
            )
        html.append(
            f'<tr><td style="{label_style}">Random State</td>'
            f'<td style="{value_style}">{s.get("pls_random_state", 42)}</td></tr>'
        )

    elif atype == "pca_ols":
        from ssdiff_gui.models.project import Project
        _pcaols_n_comp = s.get("pcaols_n_components")
        n_comp_display = "sweep" if _pcaols_n_comp is None else _pcaols_n_comp
        html.append(
            f'<tr><td style="{label_style}">PCA Components</td>'
            f'<td style="{value_style}">{n_comp_display}</td></tr>'
        )
        if _pcaols_n_comp is None:
            html.append(
                f'<tr><td style="{label_style}">K Range</td>'
                f'<td style="{value_style}">'
                f'{s.get("sweep_k_min", Project.__dataclass_fields__["sweep_k_min"].default)} \u2013 '
                f'{s.get("sweep_k_max", Project.__dataclass_fields__["sweep_k_max"].default)} '
                f'(step {s.get("sweep_k_step", Project.__dataclass_fields__["sweep_k_step"].default)})'
                f'</td></tr>'
            )

    elif atype == "groups":
        html.append(
            f'<tr><td style="{label_style}">Permutations</td>'
            f'<td style="{value_style}">{s.get("groups_n_perm", 5000):,}</td></tr>'
        )
        html.append(
            f'<tr><td style="{label_style}">Correction</td>'
            f'<td style="{value_style}">{s.get("groups_correction", "holm")}</td></tr>'
        )
        if s.get("groups_median_split", False):
            html.append(
                f'<tr><td style="{label_style}">Median Split</td>'
                f'<td style="{value_style}">Yes</td></tr>'
            )
        html.append(
            f'<tr><td style="{label_style}">Random State</td>'
            f'<td style="{value_style}">{s.get("groups_random_state", 42)}</td></tr>'
        )

    html.append("</table>")

    # Clustering section
    html.append(f'<div style="{section_style}">Clustering</div>')
    html.append('<table cellspacing="6" style="width: 100%;">')
    html.append(
        f'<tr><td style="{label_style}">Top N</td>'
        f'<td style="{value_style}">{s.get("clustering_topn", 100)}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">Auto K</td>'
        f'<td style="{value_style}">{"Yes (silhouette)" if s.get("clustering_k_auto", True) else "No"}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">K Range</td>'
        f'<td style="{value_style}">{s.get("clustering_k_min", 2)} - {s.get("clustering_k_max", 10)}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">Top Words</td>'
        f'<td style="{value_style}">{s.get("clustering_top_words", 5)}</td></tr>'
    )
    html.append("</table>")
    return html


def _pls_fit_html(view, p) -> list[str]:
    """Model Fit, Effect Size, Data Retention, and PLS section for PLS results."""
    label_style, value_style, section_style = _styles(p)
    res = view.source

    html: list[str] = []

    html.append(f'<div style="{section_style}">Model Fit</div>')
    html.append('<table cellspacing="6" style="width: 100%;">')
    html.append(
        f'<tr><td style="{label_style}">R\u00b2</td>'
        f'<td style="{value_style}">{res.stats.r2:.4f}</td></tr>'
    )
    pval_str = f"{res.stats.pvalue:.2e}" if res.stats.pvalue < 0.001 else f"{res.stats.pvalue:.4f}"
    html.append(
        f'<tr><td style="{label_style}">p-value</td>'
        f'<td style="{value_style}">{pval_str}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">Corr(y, \u0177)</td>'
        f'<td style="{value_style}">{res.stats.y_corr_pred:.4f}</td></tr>'
    )
    html.append("</table>")

    html.append(f'<div style="{section_style}">Effect Size</div>')
    html.append('<table cellspacing="6" style="width: 100%;">')
    html.append(
        f'<tr><td style="{label_style}">\u2016\u03b2\u2016 (SD(y) / cosine)</td>'
        f'<td style="{value_style}">{res.stats.beta_norm:.4f}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">\u0394y per +0.10 cosine</td>'
        f'<td style="{value_style}">{res.stats.delta:.4f}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">IQR(cos) effect</td>'
        f'<td style="{value_style}">{res.stats.iqr_effect:.4f}</td></tr>'
    )
    html.append("</table>")

    html.append(f'<div style="{section_style}">Data Retention</div>')
    html.append('<table cellspacing="6" style="width: 100%;">')
    html.append(
        f'<tr><td style="{label_style}">Input documents</td>'
        f'<td style="{value_style}">{res.stats.n_raw:,}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">Kept</td>'
        f'<td style="{value_style}">{res.stats.n_kept:,}</td></tr>'
    )
    if res.stats.n_raw:
        drop_pct = res.stats.n_dropped / res.stats.n_raw * 100
        html.append(
            f'<tr><td style="{label_style}">Dropped</td>'
            f'<td style="{value_style}">{res.stats.n_dropped:,} ({drop_pct:.1f}%)</td></tr>'
        )
    html.append("</table>")

    html.append(f'<div style="{section_style}">PLS</div>')
    html.append('<table cellspacing="6" style="width: 100%;">')
    html.append(
        f'<tr><td style="{label_style}">Components</td>'
        f'<td style="{value_style}">{res.n_components}</td></tr>'
    )
    if res.fit_info and res.fit_info.p_method:
        html.append(
            f'<tr><td style="{label_style}">p-value Method</td>'
            f'<td style="{value_style}">{res.fit_info.p_method}</td></tr>'
        )
    html.append("</table>")
    return html


def _pca_ols_fit_html(view, p) -> list[str]:
    """Model Fit, Effect Size, Data Retention, and PCA section for PCA+OLS results."""
    label_style, value_style, section_style = _styles(p)
    res = view.source

    html: list[str] = []

    html.append(f'<div style="{section_style}">Model Fit</div>')
    html.append('<table cellspacing="6" style="width: 100%;">')
    html.append(
        f'<tr><td style="{label_style}">R\u00b2</td>'
        f'<td style="{value_style}">{res.stats.r2:.4f}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">Adjusted R\u00b2</td>'
        f'<td style="{value_style}">{res.stats.r2_adj:.4f}</td></tr>'
    )
    pval_str = f"{res.stats.pvalue:.2e}" if res.stats.pvalue < 0.001 else f"{res.stats.pvalue:.4f}"
    html.append(
        f'<tr><td style="{label_style}">p-value</td>'
        f'<td style="{value_style}">{pval_str}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">Corr(y, \u0177)</td>'
        f'<td style="{value_style}">{res.stats.y_corr_pred:.4f}</td></tr>'
    )
    html.append("</table>")

    html.append(f'<div style="{section_style}">Effect Size</div>')
    html.append('<table cellspacing="6" style="width: 100%;">')
    html.append(
        f'<tr><td style="{label_style}">\u2016\u03b2\u2016 (SD(y) / cosine)</td>'
        f'<td style="{value_style}">{res.stats.beta_norm:.4f}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">\u0394y per +0.10 cosine</td>'
        f'<td style="{value_style}">{res.stats.delta:.4f}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">IQR(cos) effect</td>'
        f'<td style="{value_style}">{res.stats.iqr_effect:.4f}</td></tr>'
    )
    html.append("</table>")

    html.append(f'<div style="{section_style}">Data Retention</div>')
    html.append('<table cellspacing="6" style="width: 100%;">')
    html.append(
        f'<tr><td style="{label_style}">Input documents</td>'
        f'<td style="{value_style}">{res.stats.n_raw:,}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">Kept</td>'
        f'<td style="{value_style}">{res.stats.n_kept:,}</td></tr>'
    )
    if res.stats.n_raw:
        drop_pct = res.stats.n_dropped / res.stats.n_raw * 100
        html.append(
            f'<tr><td style="{label_style}">Dropped</td>'
            f'<td style="{value_style}">{res.stats.n_dropped:,} ({drop_pct:.1f}%)</td></tr>'
        )
    html.append("</table>")

    html.append(f'<div style="{section_style}">PCA</div>')
    html.append('<table cellspacing="6" style="width: 100%;">')
    html.append(
        f'<tr><td style="{label_style}">Components (K)</td>'
        f'<td style="{value_style}">{res.n_components}</td></tr>'
    )
    html.append("</table>")
    return html


def _group_fit_html(view, p) -> list[str]:
    """Omnibus/Pairwise permutation test + Group Sizes sections for GroupResult."""
    label_style, value_style, section_style = _styles(p)
    gr = view.source

    html: list[str] = []
    n_groups = len(gr.group_labels) if gr.group_labels else gr.G
    test_label = "Omnibus Permutation Test" if n_groups > 2 else "Pairwise Permutation Test"
    html.append(f'<div style="{section_style}">{test_label}</div>')
    html.append('<table cellspacing="6" style="width: 100%;">')
    if gr.test.omnibus_T is not None:
        html.append(
            f'<tr><td style="{label_style}">Test Statistic (T)</td>'
            f'<td style="{value_style}">{gr.test.omnibus_T:.4f}</td></tr>'
        )
    if gr.test.omnibus_p is not None:
        p_str = f"{gr.test.omnibus_p:.2e}" if gr.test.omnibus_p < 0.001 else f"{gr.test.omnibus_p:.4f}"
        html.append(
            f'<tr><td style="{label_style}">p-value</td>'
            f'<td style="{value_style}">{p_str}</td></tr>'
        )
    html.append(
        f'<tr><td style="{label_style}">Permutations</td>'
        f'<td style="{value_style}">{gr.n_perm:,}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">Correction</td>'
        f'<td style="{value_style}">{gr.correction}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">Random State</td>'
        f'<td style="{value_style}">{gr.random_state}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">Groups</td>'
        f'<td style="{value_style}">{n_groups}</td></tr>'
    )
    html.append(
        f'<tr><td style="{label_style}">Documents Used</td>'
        f'<td style="{value_style}">{gr.n_kept:,}</td></tr>'
    )
    html.append("</table>")

    groups = getattr(gr, "groups", None)
    if groups is not None:
        unique, counts = np.unique(groups, return_counts=True)
        labels_map = gr.group_labels or {}
        html.append(f'<div style="{section_style}">Group Sizes</div>')
        html.append('<table cellspacing="6" style="width: 100%;">')
        for g, cnt in zip(unique, counts):
            display = labels_map.get(g, g)
            html.append(
                f'<tr><td style="{label_style}">{g}</td>'
                f'<td style="{value_style}">{display}</td>'
                f'<td style="{value_style}">{cnt:,}</td></tr>'
            )
        html.append("</table>")

    return html


def _pairwise_table_html(view, p) -> list[str]:
    """Pairwise comparison results table for GroupResult."""
    label_style, value_style, section_style = _styles(p)
    gr = view.source
    labels_map = gr.group_labels or {}

    html: list[str] = []
    html.append(f'<div style="{section_style}">Pairwise Comparison Results</div>')
    html.append(
        '<table cellspacing="4" style="width: 100%; font-size: 12px;">'
        f'<tr style="{label_style}">'
        "<td>Group A</td><td>Group B</td><td>n_A</td><td>n_B</td>"
        "<td>Contrast norm</td><td>p</td><td>p (corr)</td><td>Cohen's d</td></tr>"
    )
    for pr in gr.pairs:
        p_raw_str = f"{pr.p_raw:.2e}" if pr.p_raw < 0.001 else f"{pr.p_raw:.4f}"
        p_corr_str = (
            f"{pr.p_corrected:.2e}" if pr.p_corrected < 0.001
            else f"{pr.p_corrected:.4f}"
        )
        g1_label = labels_map.get(pr.g1, pr.g1)
        g2_label = labels_map.get(pr.g2, pr.g2)
        html.append(
            f'<tr>'
            f'<td>{g1_label} ({pr.g1})</td>'
            f'<td>{g2_label} ({pr.g2})</td>'
            f'<td>{pr.n_g1:,}</td>'
            f'<td>{pr.n_g2:,}</td>'
            f'<td>{pr.contrast_norm:.4f}</td>'
            f'<td>{p_raw_str}</td>'
            f'<td>{p_corr_str}</td>'
            f'<td>{pr.cohens_d:.3f}</td>'
            f'</tr>'
        )
    html.append("</table>")
    return html


def _concept_config_html(view, p) -> list[str]:
    """Concept configuration panel: mode, lexicon tokens, coverage summary, per-token breakdown."""
    label_style, value_style, section_style = _styles(p)
    meta = view.meta
    s = meta.config_snapshot if meta is not None else {}
    is_crossgroup = view.analysis_type == "groups"

    html: list[str] = []
    html.append('<table cellspacing="6" style="width: 100%;">')
    html.append(
        f'<tr><td style="{label_style}">Mode</td>'
        f'<td style="{value_style}">{s.get("concept_mode", "lexicon")}</td></tr>'
    )
    if is_crossgroup and s.get("group_column", ""):
        html.append(
            f'<tr><td style="{label_style}">Group Column</td>'
            f'<td style="{value_style}">{s.get("group_column", "")}</td></tr>'
        )
    elif not is_crossgroup and s.get("outcome_column", ""):
        html.append(
            f'<tr><td style="{label_style}">Outcome Column</td>'
            f'<td style="{value_style}">{s.get("outcome_column", "")}</td></tr>'
        )
    html.append("</table>")

    if s.get("lexicon_tokens", []):
        tokens = sorted(s.get("lexicon_tokens", []))
        html.append(f'<div style="{section_style}">Lexicon ({len(tokens)} tokens)</div>')
        token_display = ", ".join(tokens[:50])
        if len(tokens) > 50:
            token_display += f" <span style='color: {p.text_muted};'>... and {len(tokens) - 50} more</span>"
        html.append(
            f'<div style="font-size: 12px; line-height: 1.6; padding: 4px 0;">{token_display}</div>'
        )

    if s.get("lexicon_coverage_summary"):
        cov = s["lexicon_coverage_summary"]
        html.append(f'<div style="{section_style}">Lexicon Coverage</div>')
        html.append('<table cellspacing="6" style="width: 100%;">')
        html.append(
            f'<tr><td style="{label_style}">Documents with hits</td>'
            f'<td style="{value_style}">{cov.get("docs_any", 0):,} '
            f'({cov.get("cov_all", 0) * 100:.1f}%)</td></tr>'
        )
        if is_crossgroup:
            cramers_v = cov.get("cramers_v", cov.get("corr_any", 0))
            html.append(
                f'<tr><td style="{label_style}">Cram\u00e9r\'s V</td>'
                f'<td style="{value_style}">{cramers_v:.4f}</td></tr>'
            )
            group_cov = cov.get("group_cov", {})
            if group_cov:
                for g, cv in sorted(group_cov.items()):
                    html.append(
                        f'<tr><td style="{label_style}">Coverage ({g})</td>'
                        f'<td style="{value_style}">{cv * 100:.1f}%</td></tr>'
                    )
        else:
            html.append(
                f'<tr><td style="{label_style}">Q1 / Q4 Coverage</td>'
                f'<td style="{value_style}">{cov.get("q1", 0) * 100:.1f}% / '
                f'{cov.get("q4", 0) * 100:.1f}%</td></tr>'
            )
            html.append(
                f'<tr><td style="{label_style}">Correlation</td>'
                f'<td style="{value_style}">{cov.get("corr_any", 0):.4f}</td></tr>'
            )
        html.append(
            f'<tr><td style="{label_style}">Hits per doc</td>'
            f'<td style="{value_style}">mean: {cov.get("hits_mean", 0):.2f}, '
            f'median: {cov.get("hits_median", 0):.1f}</td></tr>'
        )
        html.append("</table>")

    if s.get("lexicon_coverage_per_token"):
        html.append(f'<div style="{section_style}">Per-Token Breakdown</div>')
        html.append(
            '<table cellspacing="4" style="width: 100%; font-size: 12px;">'
            f'<tr style="{label_style}">'
        )
        if is_crossgroup:
            html.append("<td>Word</td><td>Docs</td><td>Coverage</td><td>Cram\u00e9r's V</td></tr>")
        else:
            html.append("<td>Word</td><td>Docs</td><td>Coverage</td><td>Correlation</td></tr>")

        for t in s["lexicon_coverage_per_token"]:
            corr = t.get("corr", 0)
            corr_color = p.success if corr > 0 else p.error if corr < 0 else p.text_secondary
            html.append(
                f'<tr><td style="font-weight: 500;">{t.get("token", "")}</td>'
                f'<td>{t.get("freq", 0):,}</td>'
                f'<td>{t.get("cov_all", 0) * 100:.1f}%</td>'
                f'<td style="color: {corr_color};">{corr:+.4f}</td></tr>'
            )
        html.append("</table>")

    return html


# ---------------------------------------------------------------------------
# DetailsTab class
# ---------------------------------------------------------------------------


class DetailsTab:
    def __init__(self):
        self._result_info_text: QTextEdit | None = None
        self._concept_config_text: QTextEdit | None = None
        self._model_config_text: QTextEdit | None = None
        self._concept_group: QGroupBox | None = None

    def create(self, parent) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        result_group = QGroupBox("Result Information")
        result_layout = QVBoxLayout()
        self._result_info_text = QTextEdit()
        self._result_info_text.setReadOnly(True)
        result_layout.addWidget(self._result_info_text)
        result_group.setLayout(result_layout)
        splitter.addWidget(result_group)

        self._concept_group = QGroupBox("Concept Configuration")
        concept_layout = QVBoxLayout()
        self._concept_config_text = QTextEdit()
        self._concept_config_text.setReadOnly(True)
        concept_layout.addWidget(self._concept_config_text)
        self._concept_group.setLayout(concept_layout)
        splitter.addWidget(self._concept_group)

        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()
        self._model_config_text = QTextEdit()
        self._model_config_text.setReadOnly(True)
        model_layout.addWidget(self._model_config_text)
        model_group.setLayout(model_layout)
        splitter.addWidget(model_group)

        splitter.setSizes([250, 350, 400])
        layout.addWidget(splitter, stretch=1)
        return tab

    def load(self, view) -> None:
        p = html_helpers.html_palette()
        meta = view.meta
        is_fulldoc = False
        if meta is not None:
            is_fulldoc = meta.config_snapshot.get("concept_mode", "lexicon") == "fulldoc"

        result_html: list[str] = []
        result_html += _common_header_html(view, p)

        if view.analysis_type == "pls":
            result_html += _pls_fit_html(view, p)
        elif view.analysis_type == "pca_ols":
            result_html += _pca_ols_fit_html(view, p)
        else:
            result_html += _group_fit_html(view, p)
            if is_fulldoc:
                result_html += _pairwise_table_html(view, p)

        self._result_info_text.setHtml("".join(result_html))

        if is_fulldoc:
            self._concept_group.hide()
            self._concept_config_text.clear()
        else:
            self._concept_group.show()
            concept_html = _concept_config_html(view, p)
            if view.analysis_type == "groups":
                concept_html += _pairwise_table_html(view, p)
            self._concept_config_text.setHtml("".join(concept_html))

        self._model_config_text.setHtml("".join(_run_config_html(view, p)))

    def is_visible_for(self, view) -> bool:
        return True
