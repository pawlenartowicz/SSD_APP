"""PCA Sweep tab — variance / R² / p-value figure across PCA K values."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ssdiff import PCAOLSResult


class PcaSweepTab:
    def __init__(self, get_current_result):
        self._get_current_result = get_current_result
        self._widget: QWidget | None = None
        self._info: QLabel | None = None
        self._image: QLabel | None = None
        self._scroll: QScrollArea | None = None
        self._zoom_label: QLabel | None = None

        self._pixmap: Optional[QPixmap] = None
        self._zoom_pct: int = 0

    def create(self, parent) -> QWidget:
        tab = QWidget()
        self._widget = tab
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        header.setContentsMargins(4, 4, 4, 0)

        self._info = QLabel()
        self._info.setWordWrap(True)
        header.addWidget(self._info, stretch=1)

        zoom_out_btn = QPushButton("\u2212")
        zoom_out_btn.setFixedSize(28, 28)
        zoom_out_btn.setToolTip("Zoom out")
        zoom_out_btn.clicked.connect(lambda: self._zoom(-10))
        header.addWidget(zoom_out_btn)

        self._zoom_label = QLabel("Fit")
        self._zoom_label.setFixedWidth(44)
        self._zoom_label.setAlignment(Qt.AlignCenter)
        header.addWidget(self._zoom_label)

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedSize(28, 28)
        zoom_in_btn.setToolTip("Zoom in")
        zoom_in_btn.clicked.connect(lambda: self._zoom(10))
        header.addWidget(zoom_in_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.setFixedWidth(48)
        reset_btn.setToolTip("Reset to fit window")
        reset_btn.clicked.connect(self._zoom_reset)
        header.addWidget(reset_btn)

        layout.addLayout(header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)

        self._image = QLabel()
        self._image.setAlignment(Qt.AlignCenter)
        self._scroll.setWidget(self._image)
        layout.addWidget(self._scroll, stretch=1)

        return tab

    def load(self, view) -> None:
        from ....utils.settings import app_settings
        result = self._get_current_result()
        ssd_result = view.source

        selected_k = ssd_result.n_components if isinstance(ssd_result, PCAOLSResult) else None
        if selected_k is not None:
            self._info.setText(f"Selected PCA K: {selected_k}")
        else:
            self._info.setText("PCA K was set manually (no sweep performed).")

        pixmap = self._render_pixmap(ssd_result)
        if pixmap is None and result is not None and result.result_path is not None:
            sweep_png = result.result_path / "sweep_plot.png"
            if sweep_png.exists():
                candidate = QPixmap(str(sweep_png))
                if not candidate.isNull():
                    pixmap = candidate

        if pixmap is not None:
            self._pixmap = pixmap
            self._zoom_pct = app_settings().value("pca_sweep_zoom_pct", 0, type=int)
            self._apply_zoom()
        else:
            self._pixmap = None
            self._image.setText(
                "No sweep plot available.\n"
                "This run may have used manual PCA K selection."
            )

    def is_visible_for(self, view) -> bool:
        return view.analysis_type == "pca_ols"

    @property
    def pixmap(self) -> Optional[QPixmap]:
        return self._pixmap

    def on_container_resized(self) -> None:
        """Re-apply fit zoom when the outer container resizes."""
        if self._zoom_pct == 0 and self._pixmap is not None:
            self._apply_zoom()

    @staticmethod
    def _render_pixmap(ssd_result):
        from ....utils.charts import render_sweep_plot
        sweep = getattr(ssd_result, "sweep_result", None) if ssd_result is not None else None
        if sweep is not None:
            return render_sweep_plot(sweep.df_joined, sweep.best_k)
        return None

    def _apply_zoom(self) -> None:
        if self._pixmap is None:
            return

        if self._zoom_pct == 0:
            available = self._scroll.viewport().width()
            if available < 50:
                available = 800
            scaled = self._pixmap.scaledToWidth(available, Qt.SmoothTransformation)
            self._zoom_label.setText("Fit")
        else:
            w = int(self._pixmap.width() * self._zoom_pct / 100)
            w = max(w, 100)
            scaled = self._pixmap.scaledToWidth(w, Qt.SmoothTransformation)
            self._zoom_label.setText(f"{self._zoom_pct}%")

        self._image.setPixmap(scaled)

    def _zoom(self, delta: int) -> None:
        from ....utils.settings import app_settings
        if self._pixmap is None:
            return

        if self._zoom_pct == 0:
            available = self._scroll.viewport().width()
            if available < 50:
                available = 800
            self._zoom_pct = round(available / self._pixmap.width() * 100)

        self._zoom_pct = max(10, self._zoom_pct + delta)
        self._apply_zoom()
        app_settings().setValue("pca_sweep_zoom_pct", self._zoom_pct)

    def _zoom_reset(self) -> None:
        from ....utils.settings import app_settings
        self._zoom_pct = 0
        self._apply_zoom()
        app_settings().setValue("pca_sweep_zoom_pct", self._zoom_pct)
