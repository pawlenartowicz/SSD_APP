"""Write a saved result's artifacts to disk based on SaveConfig.

Pure: no QSettings, no Qt. The caller owns config loading and
directory lifecycle. ``export_result`` is one code path for first-save
and re-save — it re-syncs ``report.*`` and ``tables/`` on every call.
"""

from __future__ import annotations

import pickle
import shutil
import warnings
from pathlib import Path

from .save_config import SaveConfig


# Map from concrete ssdiff result class → report-section keys it accepts.
# Sections are filtered against this before calling result.report(**kwargs);
# passing an unknown kwarg would TypeError.
def _report_sections_for(result) -> tuple[str, ...]:
    try:
        from ssdiff import GroupResult, PCAOLSResult, PLSResult
        from ssdiff.results.lexicon_result import LexiconResult
        from ssdiff.results.multi_pls_result import MultiPLSResult
    except Exception:
        return ()
    if isinstance(result, (PLSResult, PCAOLSResult)):
        return ("top_words", "clusters", "extreme_docs", "misdiagnosed")
    if isinstance(result, GroupResult):
        return ("top_words", "clusters")
    if isinstance(result, MultiPLSResult):
        return ("top_words",)
    if isinstance(result, LexiconResult):
        return ("top",)
    return ()


def export_result(result, result_path: Path, cfg: SaveConfig) -> None:
    """Write all ticked artifacts for ``result`` into ``result_path``."""
    result_path = Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True)

    _write_pickle(result, result_path)
    _write_replication_script(result, result_path)
    _resync_report(result, result_path, cfg)
    _resync_tables(result, result_path, cfg)


def _write_pickle(result, result_path: Path) -> None:
    if result._result is None:
        return
    ssd_result = result._result
    saved_emb = getattr(ssd_result, "embeddings", None)
    ssd_result.embeddings = None
    try:
        with open(result_path / "results.pkl", "wb") as f:
            pickle.dump(ssd_result, f, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        ssd_result.embeddings = saved_emb


def _write_replication_script(result, result_path: Path) -> None:
    try:
        script = result.to_replication_script()
    except Exception:
        return
    (result_path / "replication_script.py").write_text(script, encoding="utf-8")


def _resync_report(result, result_path: Path, cfg: SaveConfig) -> None:
    for stale in result_path.glob("report.*"):
        stale.unlink()
    if not cfg.report_enabled or result._result is None:
        return
    kwargs = cfg.report_kwargs(_report_sections_for(result._result))
    try:
        report = result._result.report(**kwargs)
    except Exception:
        return
    report.save(str(result_path / f"report.{cfg.report_format}"))


def _item_enabled(cfg: SaveConfig, key: str) -> bool:
    """Return True if the named artifact is enabled in cfg."""
    item = cfg.items.get(key)
    return item is not None and item.enabled


def _save_kwargs(cfg: SaveConfig, key: str) -> dict:
    """Omits keys where the user kept the library default — so library default
    behavior remains the source of truth, not a frozen snapshot.
    """
    item = cfg.items.get(key)
    if item is None:
        return {}
    kwargs: dict = {}
    if item.cols is not None:
        kwargs["cols"] = list(item.cols)
    if item.k is not None:
        kwargs["k"] = item.k
    return kwargs


def _resync_tables(result, result_path: Path, cfg: SaveConfig) -> None:
    tables_dir = result_path / "tables"
    any_tabular = any(_item_enabled(cfg, k) for k in cfg.items)
    if tables_dir.exists():
        shutil.rmtree(tables_dir)
    if not any_tabular or result._result is None:
        return

    tables_dir.mkdir(parents=True, exist_ok=True)
    ext = cfg.tables_format
    r = result._result

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        if _item_enabled(cfg, "sweep_plot"):
            sweep = getattr(r, "sweep_result", None)
            if sweep is not None:
                from .charts import render_sweep_plot
                pixmap = render_sweep_plot(sweep.df_joined, sweep.best_k)
                pixmap.save(str(tables_dir / "sweep_plot.png"), "PNG")

        if _item_enabled(cfg, "sweep") and hasattr(r, "sweep"):
            r.sweep.save(str(tables_dir / f"sweep.{ext}"), **_save_kwargs(cfg, "sweep"))

        if _item_enabled(cfg, "words") and hasattr(r, "words"):
            r.words.save(str(tables_dir / f"words.{ext}"), **_save_kwargs(cfg, "words"))

        if _item_enabled(cfg, "clusters") and hasattr(r, "clusters"):
            clusters_view = r.clusters
            kw = _save_kwargs(cfg, "clusters")
            if hasattr(clusters_view, "pos"):  # single-result sided API
                clusters_view.pos.save(str(tables_dir / f"clusters_pos.{ext}"), **kw)
                clusters_view.neg.save(str(tables_dir / f"clusters_neg.{ext}"), **kw)
            else:  # multi-pair _ShimView — fan out per pair via shim.save()
                clusters_view.save(str(tables_dir / f"clusters.{ext}"), **kw)

        if _item_enabled(cfg, "cluster_words") and hasattr(r, "clusters"):
            clusters_view = r.clusters
            kw = _save_kwargs(cfg, "cluster_words")
            if hasattr(clusters_view, "pos"):
                clusters_view.pos.words.save(str(tables_dir / f"cluster_words_pos.{ext}"), **kw)
                clusters_view.neg.words.save(str(tables_dir / f"cluster_words_neg.{ext}"), **kw)
            else:  # multi-pair — iterate leaves per pair / per side
                for pair_key in clusters_view.keys():
                    pair_str = r._key_to_str(pair_key)
                    leaf = clusters_view[pair_key]
                    leaf.pos.words.save(
                        str(tables_dir / f"cluster_words_{pair_str}_pos.{ext}"), **kw,
                    )
                    leaf.neg.words.save(
                        str(tables_dir / f"cluster_words_{pair_str}_neg.{ext}"), **kw,
                    )

        if _item_enabled(cfg, "snippets") and hasattr(r, "snippets"):
            r.snippets.save(str(tables_dir / f"snippets.{ext}"), **_save_kwargs(cfg, "snippets"))

        if _item_enabled(cfg, "docs_extreme") and hasattr(r, "docs"):
            r.docs.pos().save(str(tables_dir / f"docs_pos.{ext}"), **_save_kwargs(cfg, "docs_extreme"))
            r.docs.neg().save(str(tables_dir / f"docs_neg.{ext}"), **_save_kwargs(cfg, "docs_extreme"))

        if _item_enabled(cfg, "docs_misdiagnosed") and hasattr(r, "docs"):
            r.docs.misdiagnosed().save(str(tables_dir / f"docs_misdiagnosed.{ext}"), **_save_kwargs(cfg, "docs_misdiagnosed"))

        if _item_enabled(cfg, "pairs") and hasattr(r, "pairs"):
            r.pairs.save(str(tables_dir / f"pairs.{ext}"), **_save_kwargs(cfg, "pairs"))
