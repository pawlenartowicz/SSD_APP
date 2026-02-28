"""Progress dialog widget for long-running operations."""

import json
import math
import random
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QHBoxLayout,
    QFrame,
    QPlainTextEdit,
)
from PySide6.QtCore import Qt, QTimer, QElapsedTimer


def _load_quotes():
    """Load quotes from the bundled quotes.json file."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # PyInstaller extracts 'ssdiff_gui/resources' → '_MEIPASS/resources'
        quotes_path = Path(sys._MEIPASS) / "resources" / "quotes.json"
    else:
        quotes_path = Path(__file__).resolve().parents[2] / "resources" / "quotes.json"
    try:
        with open(quotes_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _reading_time_ms(text: str) -> int:
    """Estimate comfortable reading time for *text* in milliseconds.

    Uses ~180 WPM (a relaxed pace for reflective quotes) plus a 1.5-second
    buffer so the reader never feels rushed.
    """
    words = len(text.split())
    ms = int((words / 180) * 60 * 1000) + 2500
    return max(ms, 3000)  # at least 3 seconds per quote


_QUOTES = _load_quotes()


class ProgressDialog(QDialog):
    """A dialog showing progress for long-running operations."""

    def __init__(self, title: str = "Processing", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(540)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint
        )

        self._cancelled = False
        self._errored = False
        self._pending_accept = False
        self._pending_reject = False

        # Quote cycling state
        self._quote_timer = QTimer(self)
        self._quote_timer.setSingleShot(True)
        self._quote_timer.timeout.connect(self._show_next_quote)
        self._elapsed = QElapsedTimer()
        self._current_reading_ms = 0

        from ssdiff_gui.theme import build_current_palette
        self._palette = build_current_palette()

        self._setup_ui()
        self._compute_max_quote_height()
        self._show_next_quote()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(24, 20, 24, 20)

        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(22)
        layout.addWidget(self.progress_bar)

        # Detail label
        self.detail_label = QLabel("")
        self.detail_label.setObjectName("label_muted")
        layout.addWidget(self.detail_label)

        # ── Quote area ──────────────────────────────────────────
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setObjectName("quote_separator")
        separator.setStyleSheet(
            f"QFrame#quote_separator {{ color: {self._palette.border}; }}"
        )
        layout.addSpacing(4)
        layout.addWidget(separator)
        layout.addSpacing(2)

        self.quote_label = QLabel("")
        self.quote_label.setWordWrap(True)
        self.quote_label.setAlignment(Qt.AlignCenter)
        self.quote_label.setMinimumHeight(60)
        self.quote_label.setSizePolicy(
            self.quote_label.sizePolicy().horizontalPolicy(),
            self.quote_label.sizePolicy().verticalPolicy(),
        )
        self.quote_label.setStyleSheet(
            f"QLabel {{"
            f"  font-style: italic;"
            f"  color: {self._palette.text_secondary};"
            f"  padding: 6px 16px 2px 16px;"
            f"  font-size: {self._palette.font_size_md};"
            f"}}"
        )
        layout.addWidget(self.quote_label)

        self.author_label = QLabel("")
        self.author_label.setAlignment(Qt.AlignCenter)
        self.author_label.setStyleSheet(
            f"QLabel {{"
            f"  color: {self._palette.text_muted};"
            f"  padding: 0 16px 4px 16px;"
            f"  font-size: {self._palette.font_size_base};"
            f"}}"
        )
        layout.addWidget(self.author_label)

        layout.addSpacing(4)

        # Buttons
        self._button_layout = QHBoxLayout()
        self._button_layout.addStretch()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._on_cancel)
        self._button_layout.addWidget(self.cancel_button)

        layout.addLayout(self._button_layout)
        self.setLayout(layout)

    def _compute_max_quote_height(self):
        """Pre-compute max quote height so the dialog never resizes."""
        if not _QUOTES:
            return
        fm = self.quote_label.fontMetrics()
        # Available width = dialog min width - dialog margins - label padding
        available_width = 540 - 48 - 32  # 24*2 margins, 16*2 padding
        max_h = 0
        for q in _QUOTES:
            text = f"\u201c{q.get('quote', '')}\u201d"
            rect = fm.boundingRect(
                0, 0, available_width, 10000,
                Qt.TextWordWrap | Qt.AlignCenter, text,
            )
            max_h = max(max_h, rect.height())
        # Add breathing room
        self.quote_label.setFixedHeight(max_h + 12)

    # ── Quote cycling ───────────────────────────────────────────

    def _show_next_quote(self):
        """Pick a random quote, display it, and schedule the next one."""
        if not _QUOTES:
            return

        quote_data = random.choice(_QUOTES)
        text = quote_data.get("quote", "")
        author = quote_data.get("author", "")

        self.quote_label.setText(f"\u201c{text}\u201d")
        self.author_label.setText(f"\u2014  {author}" if author else "")

        self._current_reading_ms = _reading_time_ms(text)
        self._elapsed.start()
        self._quote_timer.start(self._current_reading_ms)

    def _remaining_reading_ms(self) -> int:
        """Milliseconds left for the reader to finish the current quote."""
        if not self._elapsed.isValid():
            return 0
        return max(0, self._current_reading_ms - int(self._elapsed.elapsed()))

    # ── Deferred close logic ────────────────────────────────────

    def _try_deferred_close(self):
        """Close the dialog now if the quote has been fully shown."""
        remaining = self._remaining_reading_ms()
        if remaining <= 0:
            self._quote_timer.stop()
            if self._pending_accept:
                self._pending_accept = False
                super().accept()
            elif self._pending_reject:
                self._pending_reject = False
                super().reject()
        else:
            # Wait until the current quote's time is up, then close.
            self._quote_timer.stop()
            QTimer.singleShot(remaining, self._try_deferred_close)

    def _on_proceed(self):
        """User clicked Proceed to skip remaining quote time."""
        self._quote_timer.stop()
        self._pending_accept = False
        self._pending_reject = False
        super().accept()

    def accept(self):
        """Accept (close) the dialog, deferring if mid-quote."""
        remaining = self._remaining_reading_ms()
        if remaining <= 0:
            self._quote_timer.stop()
            super().accept()
        else:
            self._pending_accept = True
            # Let the user skip the wait
            self.cancel_button.setText("Proceed")
            self.cancel_button.setEnabled(True)
            try:
                self.cancel_button.clicked.disconnect()
            except RuntimeError:
                pass
            self.cancel_button.clicked.connect(self._on_proceed)
            self._try_deferred_close()

    def reject(self):
        """Reject (close) the dialog — immediate if cancelled/errored, deferred otherwise."""
        if self._cancelled or self._errored:
            self._quote_timer.stop()
            super().reject()
            return
        remaining = self._remaining_reading_ms()
        if remaining <= 0:
            self._quote_timer.stop()
            super().reject()
        else:
            self._pending_reject = True
            self._try_deferred_close()

    # ── Faux (simulated) progress ──────────────────────────────

    def start_faux_progress(self, start_pct: int, end_pct: int, message: str):
        """Start a slow fake progress from *start_pct* toward *end_pct*.

        Uses an asymptotic curve so it never actually reaches *end_pct*.
        Automatically stops when :meth:`update_progress` is called with
        real data.
        """
        self._faux_start = start_pct
        self._faux_end = end_pct
        self._faux_message = message
        self._faux_elapsed = QElapsedTimer()
        self._faux_elapsed.start()
        if not hasattr(self, "_faux_timer") or self._faux_timer is None:
            self._faux_timer = QTimer(self)
            self._faux_timer.timeout.connect(self._tick_faux)
        self._faux_timer.start(500)  # tick every 500ms

    def _tick_faux(self):
        """Advance the faux progress bar asymptotically."""
        elapsed_s = self._faux_elapsed.elapsed() / 1000.0
        # Covers ~63% of range in 45s, ~86% in 90s — never reaches end_pct
        tau = 45.0
        fraction = 1.0 - math.exp(-elapsed_s / tau)
        pct = self._faux_start + fraction * (self._faux_end - self._faux_start - 1)
        self.progress_bar.setValue(int(pct))

    def _stop_faux_progress(self):
        """Stop the faux progress timer if running."""
        if hasattr(self, "_faux_timer") and self._faux_timer is not None and self._faux_timer.isActive():
            self._faux_timer.stop()

    # ── Public API (unchanged signatures) ───────────────────────

    def update_progress(self, percent: int, message: str):
        """Update the progress display."""
        self._stop_faux_progress()
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def set_detail(self, detail: str):
        """Set the detail text."""
        self.detail_label.setText(detail)

    def _on_cancel(self):
        """Handle cancel button click."""
        self._cancelled = True
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling...")
        self.reject()

    def is_cancelled(self) -> bool:
        """Check if operation was cancelled."""
        return self._cancelled

    def set_complete(self, message: str = "Complete!"):
        """Mark the operation as complete."""
        self.progress_bar.setValue(100)
        self.status_label.setText(message)
        self.cancel_button.setText("Close")
        self.cancel_button.setEnabled(True)
        self.cancel_button.clicked.disconnect()
        self.cancel_button.clicked.connect(self.accept)

    def set_error(self, message: str):
        """Show an error state with a scrollable traceback area."""
        # Stop quote timer so Close dismisses immediately
        self._quote_timer.stop()
        self._errored = True
        self._pending_accept = False
        self._pending_reject = False

        # Split into short summary (first line) and full detail
        first_line = message.split("\n", 1)[0]
        has_detail = "\n" in message

        self.status_label.setText(f"Error: {first_line}")
        self.status_label.setObjectName("label_status_error")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

        # Contact info
        contact_label = QLabel(
            "If this error persists, please copy the details below and send "
            "them to <b>hplisiecki@gmail.com</b> along with a brief description "
            "of what you were doing when it occurred."
        )
        contact_label.setWordWrap(True)
        contact_label.setObjectName("label_muted")
        contact_label.setTextFormat(Qt.RichText)
        self.layout().insertWidget(2, contact_label)

        # Hide progress bar and quote area
        self.progress_bar.hide()
        self.detail_label.hide()
        self.quote_label.hide()
        self.author_label.hide()

        if has_detail:
            # Insert a scrollable traceback viewer above the buttons
            error_text = QPlainTextEdit(message)
            error_text.setReadOnly(True)
            error_text.setMaximumHeight(200)
            p = self._palette
            error_text.setStyleSheet(
                f"QPlainTextEdit {{"
                f"  background: {p.bg_input};"
                f"  color: {p.text_primary};"
                f"  font-family: 'Consolas', 'Courier New', monospace;"
                f"  font-size: 12px;"
                f"  border: 1px solid {p.border};"
                f"  border-radius: 4px;"
                f"  padding: 6px;"
                f"}}"
            )
            # Insert before the button layout (last item in the main layout)
            self.layout().insertWidget(self.layout().count() - 1, error_text)
            self._error_text = error_text

        # Add Copy button next to Close
        copy_btn = QPushButton("Copy Error")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(message))
        self._button_layout.insertWidget(self._button_layout.count() - 1, copy_btn)

        self.cancel_button.setText("Close")
        self.cancel_button.setEnabled(True)
        self.cancel_button.clicked.disconnect()
        self.cancel_button.clicked.connect(self.reject)
