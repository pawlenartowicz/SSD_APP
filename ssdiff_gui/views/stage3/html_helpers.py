"""HTML rendering primitives shared across stage-3 tabs."""

from __future__ import annotations

from PySide6.QtWidgets import QTextEdit


def html_palette():
    """Return the current theme palette for HTML content styling."""
    from ...theme import build_current_palette
    return build_current_palette()


def escape_html(text: str) -> str:
    """Escape HTML special characters and preserve line breaks."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def show_snippet_detail(snip: dict, text_edit: QTextEdit) -> None:
    """Render a snippet dict into a QTextEdit with rich HTML."""
    p = html_palette()
    doc_id = snip.get("doc_id", "N/A")

    html = []
    html.append(
        '<table cellspacing="8" style="margin-bottom: 12px;"><tr>'
    )

    html.append(
        f'<td style="padding-right: 20px;">'
        f'<span style="color: {p.text_secondary}; font-size: {p.font_size_sm};">DOCUMENT</span><br/>'
        f'<span style="font-size: {p.font_size_lg}; font-weight: 600;">{doc_id}</span>'
        f'</td>'
    )
    html.append(
        f'<td style="padding-right: 20px;">'
        f'<span style="color: {p.text_secondary}; font-size: {p.font_size_sm};">SEED WORD</span><br/>'
        f'<span style="font-size: {p.font_size_lg}; font-weight: 600;">{snip.get("seed", "N/A")}</span>'
        f'</td>'
    )
    html.append(
        f'<td style="padding-right: 20px;">'
        f'<span style="color: {p.text_secondary}; font-size: {p.font_size_sm};">COSINE</span><br/>'
        f'<span style="font-size: {p.font_size_lg}; font-weight: 600;">{snip.get("cosine", 0):.4f}</span>'
        f'</td>'
    )
    cluster = snip.get("cluster_id")
    if cluster is not None:
        html.append(
            f'<td style="padding-right: 20px;">'
            f'<span style="color: {p.text_secondary}; font-size: {p.font_size_sm};">CLUSTER</span><br/>'
            f'<span style="font-size: {p.font_size_lg}; font-weight: 600;">{cluster}</span>'
            f'</td>'
        )
    html.append('</tr></table>')

    anchor = snip.get("text_window", "")
    if anchor:
        html.append(
            f'<div style="border-top: 1px solid {p.border}; padding-top: 12px;">'
            f'<span style="color: {p.text_secondary}; font-size: {p.font_size_sm}; text-transform: uppercase;">Snippet Context</span>'
            f'</div>'
            f'<div style="margin-top: 8px; line-height: 1.5;">{escape_html(anchor)}</div>'
        )

    surface = snip.get("text_surface", "")
    if surface:
        html.append(
            f'<div style="border-top: 1px solid {p.border}; padding-top: 12px; margin-top: 12px;">'
            f'<span style="color: {p.text_secondary}; font-size: {p.font_size_sm}; text-transform: uppercase;">Full Document Text</span>'
            f'</div>'
            f'<div style="margin-top: 8px; line-height: 1.5;">{escape_html(surface)}</div>'
        )

    text_edit.setHtml("".join(html))
