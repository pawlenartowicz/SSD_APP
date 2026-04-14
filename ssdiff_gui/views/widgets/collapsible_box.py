"""Collapsible group box widget."""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QToolButton,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QPropertyAnimation


class CollapsibleBox(QWidget):
    """A collapsible/expandable group box widget."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._is_collapsed = True

        # Toggle button
        self.toggle_button = QToolButton()
        # Styling handled by the centralized theme (QToolButton rules in QSS)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.clicked.connect(self._on_toggle)

        # Content area
        self.content_area = QWidget()
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Content layout (users add widgets to this)
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(15, 5, 5, 5)
        self.content_area.setLayout(self.content_layout)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)

        self.setLayout(main_layout)

        # Animation
        self.animation = QPropertyAnimation(self.content_area, b"maximumHeight")
        self.animation.setDuration(150)

    def _on_toggle(self, checked: bool):
        """Handle toggle button click."""
        if checked:
            self.expand()
        else:
            self.collapse()

    def expand(self):
        """Expand the content area."""
        self._is_collapsed = False
        self.toggle_button.setArrowType(Qt.DownArrow)
        self.toggle_button.setChecked(True)

        # Animate to estimated height, then uncap so layout can size naturally
        content_height = self.content_layout.sizeHint().height() + 20

        self.animation.setStartValue(0)
        self.animation.setEndValue(content_height)
        self.animation.finished.connect(self._uncap_height)
        self.animation.start()

    def _uncap_height(self):
        """Remove the maximum height constraint after expand animation."""
        self.animation.finished.disconnect(self._uncap_height)
        self.content_area.setMaximumHeight(16777215)

    def collapse(self):
        """Collapse the content area."""
        self._is_collapsed = True
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.setChecked(False)

        self.animation.setStartValue(self.content_area.maximumHeight())
        self.animation.setEndValue(0)
        self.animation.start()

    def is_collapsed(self) -> bool:
        """Check if the box is collapsed."""
        return self._is_collapsed

    def set_title(self, title: str):
        """Set the title text."""
        self.toggle_button.setText(title)

    def add_widget(self, widget: QWidget):
        """Add a widget to the content area."""
        self.content_layout.addWidget(widget)

    def add_layout(self, layout):
        """Add a layout to the content area."""
        self.content_layout.addLayout(layout)
