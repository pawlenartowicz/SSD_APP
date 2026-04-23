"""ResultView — uniform single-pair adapter for stage-3 rendering.

`working` is either the raw continuous result or a `PairResult` leaf obtained
via `source[(g1, g2)]` for groups. `current_pair_row` exposes the `Pair`
dataclass for the selected pair with original (user-visible) group labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dc_replace
from typing import Optional

from ssdiff import GroupResult, PCAOLSResult, PLSResult


@dataclass(frozen=True)
class ResultView:
    source: object
    working: object
    analysis_type: str
    pairs: list[tuple[str, str]] = field(default_factory=list)
    current_pair: Optional[tuple[str, str]] = None
    group_labels: Optional[dict[str, str]] = None
    current_pair_row: object = None
    meta: object = field(default=None)

    @classmethod
    def build(cls, source, current_pair: Optional[tuple[str, str]] = None, meta=None) -> "ResultView":
        if isinstance(source, PLSResult):
            return cls(source=source, working=source, analysis_type="pls", meta=meta)
        if isinstance(source, PCAOLSResult):
            return cls(source=source, working=source, analysis_type="pca_ols", meta=meta)
        if isinstance(source, GroupResult):
            return cls._build_group(source, current_pair, meta=meta)
        raise TypeError(f"ResultView.build: unsupported result type {type(source).__name__}")

    @classmethod
    def _build_group(cls, source: GroupResult, current_pair, meta=None) -> "ResultView":
        labels: dict[str, str] = getattr(source, "group_labels", None) or {}

        def _orig(canonical: str) -> str:
            return labels.get(canonical, canonical)

        canonical_pairs = [(p.g1, p.g2) for p in source.pairs]
        orig_pairs = [(_orig(g1), _orig(g2)) for (g1, g2) in canonical_pairs]

        target = tuple(current_pair) if current_pair is not None else orig_pairs[0]
        if target not in orig_pairs:
            raise KeyError(f"unknown pair {target!r}; known: {orig_pairs!r}")

        idx = orig_pairs.index(target)
        canonical_key = canonical_pairs[idx]
        canonical_row = source.pairs[idx]
        orig_row = dc_replace(
            canonical_row,
            g1=target[0], g2=target[1],
            contrast=f"{target[0]}_{target[1]}",
        )
        working = source[canonical_key]

        return cls(
            source=source,
            working=working,
            analysis_type="groups",
            pairs=orig_pairs,
            current_pair=target,
            group_labels=labels or None,
            current_pair_row=orig_row,
            meta=meta,
        )

    @property
    def is_group(self) -> bool:
        return self.analysis_type == "groups"

    @property
    def is_multi_pair(self) -> bool:
        return len(self.pairs) > 1
