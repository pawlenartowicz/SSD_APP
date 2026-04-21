"""Pair-scoped data resolution for stage-3 results display.

Localizes every paired-dict lookup (gr.words[(g1,g2)] / gr.alignment_scores[...]),
every gr.cluster_snippets(pair=..., side=...) call, and every 2-group
pair=None fallback to a single place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ssdiff import GroupResult


@dataclass
class PairView:
    """Resolved per-pair slice of a GroupResult (or the whole view for 2-group fits).

    For a 2-group GroupResult, ``pair_key is None`` and the views / arrays are the
    single-pair top-level accessors. For an N-group GroupResult, ``pair_key`` is
    the canonical tuple and the views / arrays are indexed into that pair.
    """

    pair_key: Optional[Tuple[str, str]]
    words: object
    clusters: object
    beta: np.ndarray
    gradient: np.ndarray
    beta_norm: float
    alignment_scores: np.ndarray

    def cluster_snippets(self, *, side: str, top_per_cluster: int = 100):
        """Centroid-based cluster snippets for this pair + side."""
        return self._result.cluster_snippets(
            pair=self.pair_key,
            side=side,
            top_per_cluster=top_per_cluster,
        )

    _result: GroupResult = None  # type: ignore[assignment]


def resolve_pair_data(
    result: GroupResult,
    pair_key: Optional[Tuple[str, str]],
) -> PairView:
    """Build a PairView for one pair of a GroupResult."""
    if not isinstance(result, GroupResult):
        raise TypeError(f"resolve_pair_data expects GroupResult, got {type(result).__name__}")

    pair_count = len(result.pairs)
    if pair_count == 1:
        if pair_key is not None:
            expected = (result.pairs[0].g1, result.pairs[0].g2)
            if pair_key != expected:
                raise KeyError(
                    f"resolve_pair_data: got pair_key={pair_key!r} but result has single pair {expected!r}"
                )
        view = PairView(
            pair_key=None,
            words=result.words,
            clusters=result.clusters,
            beta=result.beta,
            gradient=result.gradient,
            beta_norm=result.beta_norm,
            alignment_scores=result.alignment_scores,
        )
        view._result = result
        return view

    if pair_key is None:
        raise ValueError(
            f"resolve_pair_data: pair_key is required for GroupResult with "
            f"{pair_count} pairs; known: {[(p.g1, p.g2) for p in result.pairs]!r}"
        )
    known = [(p.g1, p.g2) for p in result.pairs]
    if pair_key not in known:
        raise KeyError(f"unknown pair {pair_key!r}; known: {known!r}")

    view = PairView(
        pair_key=pair_key,
        words=result.words[pair_key],
        clusters=result.clusters[pair_key],
        beta=result.beta[pair_key],
        gradient=result.gradient[pair_key],
        beta_norm=result.beta_norm[pair_key],
        alignment_scores=result.alignment_scores[pair_key],
    )
    view._result = result
    return view
