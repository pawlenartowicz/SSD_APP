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


def _pair_label(key: tuple[str, str]) -> str:
    g1, g2 = key
    return f"{g1} vs {g2}"


def _leaf_label(key: str) -> str:
    if key == "combined":
        return "Combined β"
    if key.startswith("dim-"):
        return f"Dim {key.split('-', 1)[1]}"
    return key


def populate_pair_combos(view, combos: List[QComboBox], frames: List[QWidget]) -> None:
    """Populate all synchronized pair / leaf combos from a ResultView.

    For pls / pca_ols views, hides every frame.
    For group views, fills each combo with one item per pair (text + userData tuple).
    For multipls views, fills each combo with one item per leaf key (text + userData str).
    Frames are shown only when there are multiple pairs or multiple leaves.
    """
    is_group = view.is_group and bool(view.pairs)
    is_multipls = view.analysis_type == "multipls" and bool(view.leaves)
    if not (is_group or is_multipls):
        for frame in frames:
            frame.hide()
        return

    if is_group:
        items = [(_pair_label(key), key) for key in view.pairs]
        current = view.current_pair
    else:
        items = [(_leaf_label(key), key) for key in view.leaves]
        current = view.current_leaf

    for combo in combos:
        combo.blockSignals(True)
        combo.clear()
        for text, key in items:
            combo.addItem(text, userData=key)
        if current is not None:
            for i, (_, key) in enumerate(items):
                if key == current:
                    combo.setCurrentIndex(i)
                    break
            else:
                combo.setCurrentIndex(0)
        else:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    show_selectors = view.is_multi_pair or view.is_multi_leaf
    for frame in frames:
        frame.setVisible(show_selectors)
