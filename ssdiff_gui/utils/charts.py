"""Lightweight chart rendering using QPainter — no matplotlib needed."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (
    QPixmap, QPainter, QPen, QColor, QFont, QFontMetrics,
)


def _rolling_median(x: np.ndarray, window: int = 7) -> np.ndarray:
    """Rolling median with NaN-awareness (matches ssdiff.results)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        w = x[lo:hi]
        w = w[np.isfinite(w)]
        if len(w):
            out[i] = float(np.median(w))
    return out


def _nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Generate human-friendly tick values spanning [lo, hi]."""
    if hi - lo < 1e-12:
        return [lo]
    raw_step = (hi - lo) / max(target, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for nice in (1, 2, 2.5, 5, 10):
        step = nice * magnitude
        if (hi - lo) / step <= target + 1:
            break
    first = math.floor(lo / step) * step
    ticks = []
    v = first
    while v <= hi + step * 0.01:
        if v >= lo - step * 0.01:
            ticks.append(round(v, 10))
        v += step
    return ticks


def render_sweep_plot(
    rows: list[dict],
    best_k: int,
    *,
    width: int = 1800,
    height: int = 1000,
) -> QPixmap:
    """Render the PCA-K sweep dual-axis chart and return a QPixmap.

    Parameters
    ----------
    rows : list[dict]
        ``sweep_result.df_joined`` — each dict has at least
        ``PCA_K``, ``interp_resid_z``, ``beta_delta_1_minus_cos``.
    best_k : int
        The selected optimal K value.
    width, height : int
        Pixel dimensions of the output image.
    """
    # -- extract data --------------------------------------------------
    def _val(r, key):
        v = r.get(key)
        return float(v) if v is not None else np.nan

    x = np.array([r["PCA_K"] for r in rows], dtype=float)
    y_left = np.array([_val(r, "interp_resid_z") for r in rows], dtype=float)
    y_right_raw = np.array([_val(r, "beta_delta_1_minus_cos") for r in rows], dtype=float)
    y_right = _rolling_median(y_right_raw, window=7)

    # -- colours -------------------------------------------------------
    col_blue = QColor(31, 119, 180)     # tab:blue
    col_orange = QColor(255, 127, 14)   # tab:orange
    col_red = QColor(220, 40, 40)
    col_grid = QColor(225, 225, 225)
    col_text = QColor(50, 50, 50)
    col_bg = QColor(255, 255, 255)

    # -- fonts ---------------------------------------------------------
    font_label = QFont("Sans Serif", 18)
    font_tick = QFont("Sans Serif", 14)
    font_legend = QFont("Sans Serif", 14)
    fm_label = QFontMetrics(font_label)
    fm_tick = QFontMetrics(font_tick)

    # -- margins (pixels) ----------------------------------------------
    margin_top = 50
    margin_bottom = 85
    margin_left = 140
    margin_right = 160

    plot_x0 = margin_left
    plot_y0 = margin_top
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    # -- axis ranges ---------------------------------------------------
    finite_yl = y_left[np.isfinite(y_left)]
    finite_yr = y_right[np.isfinite(y_right)]

    x_min, x_max = float(np.min(x)), float(np.max(x))

    if len(finite_yl):
        yl_min, yl_max = float(np.min(finite_yl)), float(np.max(finite_yl))
    else:
        yl_min, yl_max = -1.0, 1.0
    if len(finite_yr):
        yr_min, yr_max = float(np.min(finite_yr)), float(np.max(finite_yr))
    else:
        yr_min, yr_max = 0.0, 1.0

    # add 5 % padding
    def _pad(lo, hi):
        span = hi - lo if hi - lo > 1e-12 else 1.0
        return lo - span * 0.05, hi + span * 0.05

    yl_min, yl_max = _pad(yl_min, yl_max)
    yr_min, yr_max = _pad(yr_min, yr_max)
    x_min_pad, x_max_pad = _pad(x_min, x_max)

    # -- coordinate mappers --------------------------------------------
    def map_x(v):
        return plot_x0 + (v - x_min_pad) / (x_max_pad - x_min_pad) * plot_w

    def map_yl(v):
        return plot_y0 + plot_h - (v - yl_min) / (yl_max - yl_min) * plot_h

    def map_yr(v):
        return plot_y0 + plot_h - (v - yr_min) / (yr_max - yr_min) * plot_h

    # -- paint ---------------------------------------------------------
    pixmap = QPixmap(width, height)
    pixmap.fill(col_bg)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.Antialiasing)

    # grid + ticks — left Y
    p.setFont(font_tick)
    grid_pen = QPen(col_grid, 1, Qt.DotLine)
    ticks_yl = _nice_ticks(yl_min, yl_max, 6)
    for v in ticks_yl:
        py = map_yl(v)
        p.setPen(grid_pen)
        p.drawLine(QPointF(plot_x0, py), QPointF(plot_x0 + plot_w, py))
        p.setPen(QPen(col_blue, 1))
        label = f"{v:.2g}"
        lw = fm_tick.horizontalAdvance(label)
        p.drawText(QPointF(plot_x0 - lw - 10, py + fm_tick.ascent() / 2 - 1), label)

    # grid + ticks — right Y
    ticks_yr = _nice_ticks(yr_min, yr_max, 6)
    for v in ticks_yr:
        py = map_yr(v)
        p.setPen(QPen(col_orange, 1))
        label = f"{v:.2g}"
        p.drawText(QPointF(plot_x0 + plot_w + 10, py + fm_tick.ascent() / 2 - 1), label)

    # grid + ticks — X
    ticks_x = _nice_ticks(x_min, x_max, 8)
    for v in ticks_x:
        px = map_x(v)
        p.setPen(grid_pen)
        p.drawLine(QPointF(px, plot_y0), QPointF(px, plot_y0 + plot_h))
        p.setPen(QPen(col_text, 1))
        label = str(int(v)) if v == int(v) else f"{v:.1f}"
        lw = fm_tick.horizontalAdvance(label)
        p.drawText(QPointF(px - lw / 2, plot_y0 + plot_h + fm_tick.height() + 4), label)

    # plot border
    p.setPen(QPen(QColor(180, 180, 180), 1))
    p.drawRect(QRectF(plot_x0, plot_y0, plot_w, plot_h))

    # horizontal reference at y=0 (left axis)
    if yl_min <= 0 <= yl_max:
        py0 = map_yl(0)
        p.setPen(QPen(QColor(120, 120, 120), 1.5, Qt.DashLine))
        p.drawLine(QPointF(plot_x0, py0), QPointF(plot_x0 + plot_w, py0))

    # -- left series: interpretability (blue, markers) -----------------
    pen_blue = QPen(col_blue, 2.5)
    p.setPen(pen_blue)
    pts_left = []
    for i in range(len(x)):
        if np.isfinite(y_left[i]):
            pts_left.append(QPointF(map_x(x[i]), map_yl(y_left[i])))
    for i in range(len(pts_left) - 1):
        p.drawLine(pts_left[i], pts_left[i + 1])
    # markers
    p.setBrush(col_blue)
    for pt in pts_left:
        p.drawEllipse(pt, 5, 5)
    p.setBrush(Qt.NoBrush)

    # -- right series: beta change (orange, thick, no markers) ---------
    pen_orange = QPen(col_orange, 3)
    p.setPen(pen_orange)
    pts_right = []
    for i in range(len(x)):
        if np.isfinite(y_right[i]):
            pts_right.append(QPointF(map_x(x[i]), map_yr(y_right[i])))
    for i in range(len(pts_right) - 1):
        p.drawLine(pts_right[i], pts_right[i + 1])

    # -- best K vertical line (red, dashed) ----------------------------
    bk_px = map_x(best_k)
    pen_red = QPen(col_red, 2.5, Qt.DashLine)
    p.setPen(pen_red)
    p.drawLine(QPointF(bk_px, plot_y0), QPointF(bk_px, plot_y0 + plot_h))

    # -- axis labels ---------------------------------------------------
    p.setFont(font_label)
    p.setPen(QPen(col_text, 1))
    # X label
    x_label = "PCA_K"
    xlw = fm_label.horizontalAdvance(x_label)
    p.drawText(QPointF(plot_x0 + plot_w / 2 - xlw / 2,
                        height - 10), x_label)

    # Left Y label (rotated)
    p.save()
    p.setPen(QPen(col_blue, 1))
    p.translate(24, plot_y0 + plot_h / 2)
    p.rotate(-90)
    yl_label = "Detrended interpretability (z)"
    p.drawText(QPointF(-fm_label.horizontalAdvance(yl_label) / 2, 0), yl_label)
    p.restore()

    # Right Y label (rotated)
    p.save()
    p.setPen(QPen(col_orange, 1))
    p.translate(width - 20, plot_y0 + plot_h / 2)
    p.rotate(90)
    yr_label = "Beta change (smoothed 1 \u2212 cosine)"
    p.drawText(QPointF(-fm_label.horizontalAdvance(yr_label) / 2, 0), yr_label)
    p.restore()

    # -- legend (with background box, top-right) -----------------------
    p.setFont(font_legend)
    fm_leg = QFontMetrics(font_legend)

    leg_labels = [
        "detrended interpretability (z)",
        "beta change (smoothed 1\u2212cos)",
        f"best K = {best_k}",
    ]
    line_h = fm_leg.height() + 6
    swatch_w = 28
    gap = 8
    max_text_w = max(fm_leg.horizontalAdvance(t) for t in leg_labels)
    box_w = 12 + swatch_w + gap + max_text_w + 12
    box_h = 10 + line_h * 3 + 6

    leg_bx = plot_x0 + plot_w - box_w - 14
    leg_by = plot_y0 + 14

    # legend background
    leg_bg = QColor(255, 255, 255, 220)
    p.setBrush(leg_bg)
    p.setPen(QPen(QColor(180, 180, 180), 1))
    p.drawRoundedRect(QRectF(leg_bx, leg_by, box_w, box_h), 4, 4)
    p.setBrush(Qt.NoBrush)

    leg_x = leg_bx + 12
    leg_y = leg_by + 10 + fm_leg.ascent() / 2

    # blue line + marker
    p.setPen(QPen(col_blue, 2.5))
    p.drawLine(QPointF(leg_x, leg_y), QPointF(leg_x + swatch_w, leg_y))
    p.setBrush(col_blue)
    p.drawEllipse(QPointF(leg_x + swatch_w / 2, leg_y), 4, 4)
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(col_text, 1))
    p.drawText(QPointF(leg_x + swatch_w + gap, leg_y + fm_leg.ascent() / 2 - 1),
               leg_labels[0])

    leg_y += line_h

    # orange line
    p.setPen(QPen(col_orange, 3))
    p.drawLine(QPointF(leg_x, leg_y), QPointF(leg_x + swatch_w, leg_y))
    p.setPen(QPen(col_text, 1))
    p.drawText(QPointF(leg_x + swatch_w + gap, leg_y + fm_leg.ascent() / 2 - 1),
               leg_labels[1])

    leg_y += line_h

    # red line
    p.setPen(QPen(col_red, 2.5, Qt.DashLine))
    p.drawLine(QPointF(leg_x, leg_y), QPointF(leg_x + swatch_w, leg_y))
    p.setPen(QPen(col_text, 1))
    p.drawText(QPointF(leg_x + swatch_w + gap, leg_y + fm_leg.ascent() / 2 - 1),
               leg_labels[2])

    p.end()
    return pixmap
