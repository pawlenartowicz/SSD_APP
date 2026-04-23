"""Group/continuous label application.

Pure functions — caller passes a dict of widget references. This is a 1:1
port of the Stage3Widget._update_crossgroup_labels / _reset_continuous_labels
methods. No behavior changes beyond reading through `ResultView`.
"""

from __future__ import annotations


def apply_group_labels(view, widgets: dict) -> None:
    """Set group-contrast labels for the currently selected pair.

    `view` is a ResultView; `widgets` maps widget-name → Qt widget. Expected keys:
        - ov_pos_group, ov_neg_group
        - pos_pole_group, neg_pole_group
        - pos_pole_desc, neg_pole_desc
        - snippet_side_combo
        - snippet_tab_desc
    """
    cr = view.current_pair_row
    if cr is not None:
        g1_name = cr.g1
        g2_name = cr.g2
    else:
        g1_name, g2_name = "Group A", "Group B"

    widgets["ov_pos_group"].setTitle(f"Positive Direction  \u2192  {g1_name}")
    widgets["ov_neg_group"].setTitle(f"Negative Direction  \u2192  {g2_name}")

    widgets["pos_pole_group"].setTitle(f"{g1_name} Direction")
    widgets["neg_pole_group"].setTitle(f"{g2_name} Direction")
    widgets["pos_pole_desc"].setText(f"Words most aligned with {g1_name}:")
    widgets["neg_pole_desc"].setText(f"Words most aligned with {g2_name}:")

    combo = widgets["snippet_side_combo"]
    combo.blockSignals(True)
    combo.clear()
    combo.addItems([
        f"{g1_name} direction",
        f"{g2_name} direction",
    ])
    combo.blockSignals(False)

    widgets["snippet_tab_desc"].setText(
        "Snippets ranked by alignment with the contrast direction (not clustered)"
    )


def apply_continuous_labels(widgets: dict) -> None:
    """Reset labels to continuous (\u03b2-driven) defaults."""
    widgets["ov_pos_group"].setTitle("Positive Clusters  (+\u03b2  \u2192  higher outcome)")
    widgets["ov_neg_group"].setTitle("Negative Clusters  (\u2212\u03b2  \u2192  lower outcome)")

    widgets["pos_pole_group"].setTitle("Positive End (+\u03b2) \u2014 Higher Outcome")
    widgets["neg_pole_group"].setTitle("Negative End (\u2212\u03b2) \u2014 Lower Outcome")
    widgets["pos_pole_desc"].setText("Words most aligned with higher outcome values:")
    widgets["neg_pole_desc"].setText("Words most aligned with lower outcome values:")

    combo = widgets["snippet_side_combo"]
    combo.blockSignals(True)
    combo.clear()
    combo.addItems([
        "Positive (+\u03b2)",
        "Negative (\u2212\u03b2)",
    ])
    combo.blockSignals(False)

    widgets["snippet_tab_desc"].setText(
        "Snippets ranked by alignment with the \u03b2 direction (not clustered)"
    )
