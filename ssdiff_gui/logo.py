"""Logo generation for SSD.

Generates a theme-aware logo: a translucent 3D-shaded sphere
(embedding space) with an SSD gradient line piercing through it.

Two rendering paths:
  - generate_logo_svg()  — full-fidelity SVG for file export / browsers
  - paint_logo()         — QPainter-native for Qt widgets & icons
                           (QSvgRenderer cannot handle stop-opacity or filters)
"""
from __future__ import annotations

import colorsys
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .theme import ThemePalette


# ------------------------------------------------------------------
#  Colour helpers
# ------------------------------------------------------------------

def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"


def _blend(a: str, b: str, t: float) -> str:
    """Blend two hex colours.  t=0 -> a, t=1 -> b."""
    ra, ga, ba = _hex_to_rgb(a)
    rb, gb, bb = _hex_to_rgb(b)
    return _rgb_to_hex(
        int(ra + (rb - ra) * t),
        int(ga + (gb - ga) * t),
        int(ba + (bb - ba) * t),
    )


def _gradient_colors(accent: str) -> tuple[str, str, str]:
    """Derive gradient-line colours from the accent: hue-shifted +-15 degrees."""
    r, g, b = _hex_to_rgb(accent)
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    start_h = (h - 15 / 360) % 1.0
    end_h = (h + 15 / 360) % 1.0
    rs, gs, bs = colorsys.hsv_to_rgb(start_h, s, v)
    re, ge, be = colorsys.hsv_to_rgb(end_h, s, v)
    return (
        _rgb_to_hex(int(rs * 255), int(gs * 255), int(bs * 255)),
        accent,
        _rgb_to_hex(int(re * 255), int(ge * 255), int(be * 255)),
    )


def _sphere_stops(accent: str, bg: str) -> list[str]:
    """Derive four radial-gradient stop colours for the sphere body."""
    return [
        _blend(accent, bg, 0.30),   # 0%   highlight
        _blend(accent, bg, 0.52),   # 35%  mid-tone
        _blend(accent, bg, 0.70),   # 70%  shadow
        _blend(accent, bg, 0.84),   # 100% edge (almost bg)
    ]


# ------------------------------------------------------------------
#  Geometry (shared by both rendering paths)
# ------------------------------------------------------------------

def _logo_geometry(size: int, content_scale: float = 1.0):
    """Compute all coordinates needed for the logo at *size* px."""
    cx, cy = size / 2, size / 2
    r = size * 0.35 * content_scale

    angle = math.radians(32)
    extend = r * 1.30

    lx1 = cx - extend * math.cos(angle)
    ly1 = cy + extend * math.sin(angle)
    lx2 = cx + extend * math.cos(angle)
    ly2 = cy - extend * math.sin(angle)

    dx, dy = lx2 - lx1, ly2 - ly1
    fx, fy = lx1 - cx, ly1 - cy
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    sd = math.sqrt(max(0, b * b - 4 * a * c))
    t_front = (-b + sd) / (2 * a)
    front_x = lx1 + t_front * dx
    front_y = ly1 + t_front * dy

    return dict(
        cx=cx, cy=cy, r=r,
        lx1=lx1, ly1=ly1, lx2=lx2, ly2=ly2,
        front_x=front_x, front_y=front_y,
        scale=size / 512,
    )


# ------------------------------------------------------------------
#  SVG generation  (for file export / browser preview)
# ------------------------------------------------------------------

def generate_logo_svg(
    palette: ThemePalette,
    theme_name: str = "Midnight",
    size: int = 512,
) -> str:
    """Return the full SVG markup for the SSD logo."""
    g = _logo_geometry(size)
    cx, cy, r = g["cx"], g["cy"], g["r"]
    lx1, ly1 = g["lx1"], g["ly1"]
    lx2, ly2 = g["lx2"], g["ly2"]
    fix, fiy = g["front_x"], g["front_y"]

    bg = palette.bg_base
    c_start, c_mid, c_end = _gradient_colors(palette.accent)
    s0, s1, s2, s3 = _sphere_stops(palette.accent, bg)
    glow = _blend(palette.accent, "#ffffff", 0.20)

    return f"""\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 {size} {size}" width="{size}" height="{size}">
  <defs>
    <radialGradient id="sphere-body"
                    cx="42%" cy="38%" r="58%" fx="35%" fy="30%">
      <stop offset="0%"   stop-color="{s0}" stop-opacity="0.82"/>
      <stop offset="35%"  stop-color="{s1}" stop-opacity="0.78"/>
      <stop offset="70%"  stop-color="{s2}" stop-opacity="0.75"/>
      <stop offset="100%" stop-color="{s3}" stop-opacity="0.72"/>
    </radialGradient>
    <radialGradient id="specular"
                    cx="36%" cy="30%" r="18%" fx="36%" fy="30%">
      <stop offset="0%"   stop-color="#ffffff" stop-opacity="0.22"/>
      <stop offset="100%" stop-color="#ffffff" stop-opacity="0.0"/>
    </radialGradient>
    <linearGradient id="ssd-grad"
                    x1="{lx1:.1f}" y1="{ly1:.1f}"
                    x2="{lx2:.1f}" y2="{ly2:.1f}"
                    gradientUnits="userSpaceOnUse">
      <stop offset="0%"   stop-color="{c_start}"/>
      <stop offset="50%"  stop-color="{c_mid}"/>
      <stop offset="100%" stop-color="{c_end}"/>
    </linearGradient>
    <filter id="line-glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur in="SourceGraphic" stdDeviation="6"/>
    </filter>
    <filter id="sphere-glow" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur in="SourceGraphic" stdDeviation="12"/>
    </filter>
    <filter id="dot-glow" x="-200%" y="-200%" width="500%" height="500%">
      <feGaussianBlur in="SourceGraphic" stdDeviation="3"/>
    </filter>
  </defs>

  <rect width="{size}" height="{size}" fill="{bg}"/>

  <!-- Full line behind sphere (interior ghosts through) -->
  <line x1="{lx1:.1f}" y1="{ly1:.1f}" x2="{lx2:.1f}" y2="{ly2:.1f}"
        stroke="url(#ssd-grad)" stroke-width="28" stroke-linecap="round"
        stroke-opacity="0.20" filter="url(#line-glow)"/>
  <line x1="{lx1:.1f}" y1="{ly1:.1f}" x2="{lx2:.1f}" y2="{ly2:.1f}"
        stroke="url(#ssd-grad)" stroke-width="9.0" stroke-linecap="round"
        stroke-opacity="0.80"/>

  <!-- Back endpoint -->
  <circle cx="{lx1:.1f}" cy="{ly1:.1f}" r="4.5"
          fill="{c_start}" fill-opacity="0.50" filter="url(#dot-glow)"/>
  <circle cx="{lx1:.1f}" cy="{ly1:.1f}" r="2.5"
          fill="{c_start}" fill-opacity="0.85"/>

  <!-- Sphere -->
  <circle cx="{cx}" cy="{cy}" r="{r * 1.01:.1f}"
          fill="none" stroke="{glow}" stroke-width="3"
          stroke-opacity="0.06" filter="url(#sphere-glow)"/>
  <circle cx="{cx}" cy="{cy}" r="{r:.1f}" fill="url(#sphere-body)"/>
  <circle cx="{cx}" cy="{cy}" r="{r:.1f}" fill="url(#specular)"/>

  <!-- Front segment (on top of sphere) -->
  <line x1="{fix:.1f}" y1="{fiy:.1f}" x2="{lx2:.1f}" y2="{ly2:.1f}"
        stroke="url(#ssd-grad)" stroke-width="32" stroke-linecap="round"
        stroke-opacity="0.25" filter="url(#line-glow)"/>
  <line x1="{fix:.1f}" y1="{fiy:.1f}" x2="{lx2:.1f}" y2="{ly2:.1f}"
        stroke="url(#ssd-grad)" stroke-width="10.0" stroke-linecap="round"
        stroke-opacity="0.90"/>

  <!-- Front endpoint -->
  <circle cx="{lx2:.1f}" cy="{ly2:.1f}" r="5"
          fill="{c_end}" fill-opacity="0.60" filter="url(#dot-glow)"/>
  <circle cx="{lx2:.1f}" cy="{ly2:.1f}" r="3"
          fill="{c_end}" fill-opacity="0.90"/>
</svg>"""


# ------------------------------------------------------------------
#  QPainter rendering  (for Qt widgets / icons)
# ------------------------------------------------------------------

def paint_logo(painter, size: int, palette: ThemePalette, theme_name: str = "Midnight", *, draw_bg: bool = True, content_scale: float = 1.0):
    """Paint the logo directly with QPainter.

    This uses Qt-native gradients with full alpha-stop support,
    side-stepping QSvgRenderer's limitations.
    Set *draw_bg* to False for transparent-background icons.
    *content_scale* > 1 enlarges the sphere/line within the canvas.
    """
    from PySide6.QtCore import QPointF, Qt, QRectF
    from PySide6.QtGui import (
        QColor, QPen, QBrush, QRadialGradient, QLinearGradient, QPainter,
    )

    painter.setRenderHint(QPainter.Antialiasing)

    g = _logo_geometry(size, content_scale)
    cx, cy, r = g["cx"], g["cy"], g["r"]
    lx1, ly1 = g["lx1"], g["ly1"]
    lx2, ly2 = g["lx2"], g["ly2"]
    fix, fiy = g["front_x"], g["front_y"]
    sc = g["scale"]

    bg = palette.bg_base
    c_start, c_mid, c_end = _gradient_colors(palette.accent)
    s0, s1, s2, s3 = _sphere_stops(palette.accent, bg)

    def qc(hex_color: str, alpha: int = 255) -> QColor:
        rv, gv, bv = _hex_to_rgb(hex_color)
        return QColor(rv, gv, bv, alpha)

    # ── background ────────────────────────────────────────────────
    if draw_bg:
        painter.fillRect(QRectF(0, 0, size, size), qc(bg))

    # ── SSD gradient (reused for all line segments) ───────────────
    line_grad = QLinearGradient(QPointF(lx1, ly1), QPointF(lx2, ly2))
    line_grad.setColorAt(0.0, qc(c_start))
    line_grad.setColorAt(0.5, qc(c_mid))
    line_grad.setColorAt(1.0, qc(c_end))

    # ── full line behind sphere ───────────────────────────────────
    # glow
    pen = QPen(QBrush(line_grad), 28 * sc)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    painter.setOpacity(0.20)
    painter.drawLine(QPointF(lx1, ly1), QPointF(lx2, ly2))
    # crisp
    pen.setWidthF(9.0 * sc)
    painter.setPen(pen)
    painter.setOpacity(0.80)
    painter.drawLine(QPointF(lx1, ly1), QPointF(lx2, ly2))

    painter.setOpacity(1.0)
    painter.setPen(Qt.NoPen)

    # ── back endpoint ─────────────────────────────────────────────
    painter.setBrush(qc(c_start))
    painter.setOpacity(0.50)
    painter.drawEllipse(QPointF(lx1, ly1), 4.5 * sc, 4.5 * sc)
    painter.setOpacity(0.85)
    painter.drawEllipse(QPointF(lx1, ly1), 2.5 * sc, 2.5 * sc)
    painter.setOpacity(1.0)

    # ── sphere body ───────────────────────────────────────────────
    # Map SVG objectBoundingBox percentages to absolute coords.
    # BBox is (cx-r, cy-r, 2r, 2r).
    gcx = cx + r * (2 * 0.42 - 1)       # gradient center
    gcy = cy + r * (2 * 0.38 - 1)
    gfx = cx + r * (2 * 0.35 - 1)       # focal point
    gfy = cy + r * (2 * 0.30 - 1)
    gr = 0.58 * 2 * r                    # gradient radius

    sphere_grad = QRadialGradient(QPointF(gcx, gcy), gr, QPointF(gfx, gfy))
    sphere_grad.setColorAt(0.00, qc(s0, 209))   # 0.82 * 255
    sphere_grad.setColorAt(0.35, qc(s1, 199))   # 0.78 * 255
    sphere_grad.setColorAt(0.70, qc(s2, 191))   # 0.75 * 255
    sphere_grad.setColorAt(1.00, qc(s3, 184))   # 0.72 * 255

    painter.setBrush(QBrush(sphere_grad))
    painter.drawEllipse(QPointF(cx, cy), r, r)

    # ── specular highlight ────────────────────────────────────────
    scx = cx + r * (2 * 0.36 - 1)
    scy = cy + r * (2 * 0.30 - 1)
    sr = 0.18 * 2 * r

    spec_grad = QRadialGradient(QPointF(scx, scy), sr, QPointF(scx, scy))
    spec_grad.setColorAt(0.0, QColor(255, 255, 255, 56))   # 0.22 * 255
    spec_grad.setColorAt(1.0, QColor(255, 255, 255, 0))

    painter.setBrush(QBrush(spec_grad))
    painter.drawEllipse(QPointF(cx, cy), r, r)

    # ── front segment (on top of sphere) ──────────────────────────
    # glow
    pen = QPen(QBrush(line_grad), 32 * sc)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.setOpacity(0.25)
    painter.drawLine(QPointF(fix, fiy), QPointF(lx2, ly2))
    # crisp
    pen.setWidthF(10.0 * sc)
    painter.setPen(pen)
    painter.setOpacity(0.90)
    painter.drawLine(QPointF(fix, fiy), QPointF(lx2, ly2))

    painter.setOpacity(1.0)
    painter.setPen(Qt.NoPen)

    # ── front endpoint ────────────────────────────────────────────
    painter.setBrush(qc(c_end))
    painter.setOpacity(0.60)
    painter.drawEllipse(QPointF(lx2, ly2), 5 * sc, 5 * sc)
    painter.setOpacity(0.90)
    painter.drawEllipse(QPointF(lx2, ly2), 3 * sc, 3 * sc)
    painter.setOpacity(1.0)


# ------------------------------------------------------------------
#  QIcon / QPixmap creation  (use QPainter path)
# ------------------------------------------------------------------

def _render_to_pixmap(palette: ThemePalette, theme_name: str, size: int, *, draw_bg: bool = True, content_scale: float = 1.0):
    """Render the logo to a QPixmap using the QPainter path."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap, QPainter

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    paint_logo(painter, size, palette, theme_name, draw_bg=draw_bg, content_scale=content_scale)
    painter.end()
    return pixmap


def create_app_icon(
    palette: ThemePalette,
    theme_name: str = "Midnight",
):
    """Create a multi-resolution QIcon from the logo."""
    from PySide6.QtGui import QIcon

    icon = QIcon()
    for sz in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(_render_to_pixmap(palette, theme_name, sz, draw_bg=False, content_scale=1.2))
    return icon


def create_logo_pixmap(
    palette: ThemePalette,
    theme_name: str = "Midnight",
    size: int = 128,
):
    """Render the logo to a single QPixmap at *size* px."""
    return _render_to_pixmap(palette, theme_name, size)


# ------------------------------------------------------------------
#  ICNS generation  (macOS app icon)
# ------------------------------------------------------------------

def generate_icns(
    out_path: str | None = None,
    palette: "ThemePalette | None" = None,
    theme_name: str = "Midnight",
) -> str:
    """Render the logo at standard ICNS sizes and write a .icns file.

    Uses the QPainter path (no external tools required).
    Returns the path to the written file.

    Must be called with a running QApplication.
    """
    import struct
    from pathlib import Path as _Path
    from PySide6.QtCore import Qt, QBuffer, QIODevice
    from PySide6.QtGui import QPixmap, QPainter

    if palette is None:
        from .theme import ThemePalette
        palette = ThemePalette()  # default Midnight palette

    if out_path is None:
        out_path = str(
            _Path(__file__).resolve().parent / "resources" / "icon.icns"
        )

    # ICNS icon types: (ostype, pixel_size)
    icon_types = [
        (b"icp4", 16),
        (b"icp5", 32),
        (b"ic07", 128),
        (b"ic08", 256),
        (b"ic09", 512),
    ]

    entries: list[bytes] = []
    for ostype, sz in icon_types:
        pixmap = QPixmap(sz, sz)
        pixmap.fill(Qt.transparent)
        p = QPainter(pixmap)
        paint_logo(p, sz, palette, theme_name, draw_bg=False, content_scale=1.2)
        p.end()

        # Convert to PNG bytes
        buf = QBuffer()
        buf.open(QIODevice.WriteOnly)
        pixmap.save(buf, "PNG")
        png_data = bytes(buf.data())
        buf.close()

        # ICNS entry: 4-byte type + 4-byte length (includes header) + data
        entry_len = 8 + len(png_data)
        entries.append(ostype + struct.pack(">I", entry_len) + png_data)

    # ICNS file: 4-byte magic + 4-byte total length + entries
    body = b"".join(entries)
    total_len = 8 + len(body)
    icns_data = b"icns" + struct.pack(">I", total_len) + body

    with open(out_path, "wb") as f:
        f.write(icns_data)

    return out_path
