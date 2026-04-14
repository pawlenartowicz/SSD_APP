"""Reusable combobox item delegate that paints a hover × removal button."""

from typing import Callable, Optional

from PySide6.QtWidgets import QStyledItemDelegate, QComboBox
from PySide6.QtCore import Qt, QRect, QSize, QEvent
from PySide6.QtGui import QPen, QColor


class RemovableItemDelegate(QStyledItemDelegate):
    """Delegate that paints an × button on hovered combobox items.

    Args:
        combo:         The QComboBox to install on.
        on_remove:     Called with the row index when × is clicked.
        is_removable:  Optional callable(row) -> bool.  When supplied, only
                       rows for which it returns True will show the × button.
                       Defaults to allowing all enabled rows.
    """

    _X_SIZE = 16  # clickable region width/height

    def __init__(
        self,
        combo: QComboBox,
        on_remove: Callable[[int], None],
        is_removable: Optional[Callable[[int], bool]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._combo = combo
        self._on_remove = on_remove
        self._is_removable = is_removable or (lambda row: True)
        self._hovered_row = -1

        view = combo.view()
        view.setMouseTracking(True)
        view.viewport().setMouseTracking(True)
        view.viewport().installEventFilter(self)

    # -- painting ----------------------------------------------------------

    def paint(self, painter, option, index):
        super().paint(painter, option, index)

        if index.row() != self._hovered_row:
            return
        if not (index.flags() & Qt.ItemIsEnabled):
            return
        if not self._is_removable(index.row()):
            return

        painter.save()
        x_rect = self._x_rect(option.rect)
        painter.setPen(QPen(QColor("#8e8ea0"), 1.5))
        margin = 4
        painter.drawLine(
            x_rect.left() + margin, x_rect.top() + margin,
            x_rect.right() - margin, x_rect.bottom() - margin,
        )
        painter.drawLine(
            x_rect.right() - margin, x_rect.top() + margin,
            x_rect.left() + margin, x_rect.bottom() - margin,
        )
        painter.restore()

    def sizeHint(self, option, index):
        hint = super().sizeHint(option, index)
        return QSize(hint.width() + self._X_SIZE + 4, max(hint.height(), self._X_SIZE + 4))

    # -- event filter for hover & click ------------------------------------

    def eventFilter(self, obj, event):
        try:
            return self._handle_event(obj, event)
        except RuntimeError:
            return False

    def _handle_event(self, obj, event):
        if obj is self._combo.view().viewport():
            if event.type() == QEvent.MouseMove:
                idx = self._combo.view().indexAt(event.pos())
                row = idx.row() if idx.isValid() else -1
                if row != self._hovered_row:
                    self._hovered_row = row
                    self._combo.view().viewport().update()
            elif event.type() == QEvent.Leave:
                self._hovered_row = -1
                self._combo.view().viewport().update()
            elif event.type() == QEvent.MouseButtonRelease:
                idx = self._combo.view().indexAt(event.pos())
                if idx.isValid() and idx.row() == self._hovered_row:
                    if not self._is_removable(idx.row()):
                        return super().eventFilter(obj, event)
                    vis_rect = self._combo.view().visualRect(idx)
                    x_rect = self._x_rect(vis_rect)
                    if x_rect.contains(event.pos()):
                        self._on_remove(idx.row())
                        return True
        return super().eventFilter(obj, event)

    # -- helpers -----------------------------------------------------------

    @classmethod
    def _x_rect(cls, item_rect: QRect) -> QRect:
        """Return the clickable × region on the right side of the row."""
        return QRect(
            item_rect.right() - cls._X_SIZE - 4,
            item_rect.center().y() - cls._X_SIZE // 2,
            cls._X_SIZE,
            cls._X_SIZE,
        )
