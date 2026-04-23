"""Pair-selector helpers: build the per-tab combo and populate all synchronized combos."""

from __future__ import annotations

from typing import Callable, List

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget


def make_pair_selector(
    on_changed: Callable[[int], None],
    combos: List[QComboBox],
    frames: List[QWidget],
) -> QWidget:
    """Build one 'Choose pair:' combo+frame and register it for synchronized updates.

    The returned frame starts hidden — visibility is driven by populate_pair_combos.
    """
    frame = QWidget()
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(QLabel("Choose pair:"))
    combo = QComboBox()
    combo.setMinimumWidth(250)
    combo.currentIndexChanged.connect(on_changed)
    layout.addWidget(combo)
    frame.hide()
    combos.append(combo)
    frames.append(frame)
    return frame


def populate_pair_combos(view, combos: List[QComboBox], frames: List[QWidget]) -> None:
    """Populate all synchronized pair combos from a ResultView.

    For non-group views (pls, pca_ols), hides every frame.
    For group views, fills each combo with one item per pair (text + userData tuple),
    then shows the frames only when multiple pairs exist.
    """
    if not view.is_group or not view.pairs:
        for frame in frames:
            frame.hide()
        return

    items = []
    for key in view.pairs:
        g1, g2 = key
        text = f"{g1} vs {g2}"
        items.append((text, key))

    for combo in combos:
        combo.blockSignals(True)
        combo.clear()
        for text, key in items:
            combo.addItem(text, userData=key)
        if view.current_pair is not None:
            for i, (_, key) in enumerate(items):
                if key == view.current_pair:
                    combo.setCurrentIndex(i)
                    break
            else:
                combo.setCurrentIndex(0)
        else:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    show_selectors = view.is_multi_pair
    for frame in frames:
        frame.setVisible(show_selectors)
