"""Tab Protocol — uniform interface for every stage-3 tab."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget


class Tab(Protocol):
    def create(self, parent: QWidget) -> QWidget: ...
    def load(self, view) -> None: ...
    def is_visible_for(self, view) -> bool: ...
