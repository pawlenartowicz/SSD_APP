"""Mixin for overlaying InfoButtons on QGroupBoxes."""

from PySide6.QtCore import QEvent, QTimer

from .info_button import InfoButton


class OverlayInfoMixin:
    """Mixin that pins InfoButtons to the top-right corner of widgets.

    Subclasses can set ``_info_margin_right`` and ``_info_margin_top``
    to adjust positioning (defaults: 6px right, 20px top).
    """

    _info_margin_right: int = 6
    _info_margin_top: int = 20

    def _init_overlay_info(self):
        """Call from __init__ to initialise the overlay tracking list."""
        self._overlay_info_buttons: list = []

    def _add_overlay_info(self, widget, tooltip_html: str):
        """Pin an InfoButton to the top-right corner of *widget*."""
        btn = InfoButton(tooltip_html, parent=widget)
        widget.installEventFilter(self)
        self._overlay_info_buttons.append((widget, btn))
        QTimer.singleShot(0, lambda w=widget, b=btn: self._reposition_info(w, b))

    def _overlay_info_event_filter(self, obj, event):
        """Handle resize/layout events for overlay info buttons.

        Call from ``eventFilter`` and return True if handled.
        """
        if event.type() in (QEvent.Resize, QEvent.LayoutRequest):
            for widget, btn in self._overlay_info_buttons:
                if obj is widget:
                    self._reposition_info(widget, btn)
                    return True
        return False

    def _reposition_info(self, widget, btn):
        x = widget.width() - btn.width() - self._info_margin_right
        btn.move(x, self._info_margin_top)
        btn.raise_()
