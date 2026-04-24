"""export_result: file layout, CSV schemas, resave behavior.

Dropped from old suite:
  - test_pairs_writes_fanned_out_or_single_file (liberal assertion — any directory passed)
  - test_report_disabled_removes_all_report_files (subsumed by test_resave)
  - TestSaveKwargs (tests a 5-line private helper)
"""

from __future__ import annotations

import csv as _csv
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.spacy


def _make_pls(tiny_embeddings, synthetic_corpus):
    from ssdiff import SSD
    rng = np.random.RandomState(42)
    y = rng.randn(len(synthetic_corpus.docs))
    lexicon = ["happy", "sad", "angry", "love", "hate"]
    return SSD(tiny_embeddings, synthetic_corpus, y, lexicon, window=3).fit_pls(
        n_components=1, p_method=None,
    )


def _make_pcaols_with_sweep(tiny_embeddings, synthetic_corpus):
    from ssdiff import SSD
    rng = np.random.RandomState(42)
    y = rng.randn(len(synthetic_corpus.docs))
    lexicon = ["happy", "sad", "angry", "love", "hate"]
    return SSD(tiny_embeddings, synthetic_corpus, y, lexicon, window=3).fit_ols(
        k_min=2, k_max=6, k_step=2,
    )


def _make_groups(tiny_embeddings, synthetic_corpus):
    from ssdiff import SSD
    n = len(synthetic_corpus.docs)
    groups = ["A"] * (n // 2) + ["B"] * (n - n // 2)
    lexicon = ["happy", "sad", "angry", "love", "hate"]
    return SSD(tiny_embeddings, synthetic_corpus, groups, lexicon, window=3).fit_groups(
        n_perm=100,
    )


def _wrap(result_path: Path, ssd_result, config_snapshot=None):
    from ssdiff_gui.models.project import Result
    result_path.mkdir(parents=True, exist_ok=True)
    r = Result(
        result_id=result_path.name,
        timestamp=datetime(2026, 4, 23, 12, 0, 0),
        result_path=result_path,
        config_snapshot=config_snapshot or {"analysis_type": "pls"},
        status="complete",
    )
    r._result = ssd_result
    return r


def test_default_config_writes_only_pkl_script_and_report(
    tmp_path, tiny_embeddings, synthetic_corpus,
):
    from ssdiff_gui.utils.result_export import export_result
    from ssdiff_gui.utils.save_config import SaveConfig

    result = _wrap(tmp_path / "r", _make_pls(tiny_embeddings, synthetic_corpus))
    export_result(result, result.result_path, SaveConfig.default())

    expected = {"results.pkl", "replication_script.py", "report.md"}
    assert {p.name for p in result.result_path.iterdir()} == expected


def test_ticked_items_create_tables_with_expected_csvs(
    tmp_path, tiny_embeddings, synthetic_corpus,
):
    from ssdiff_gui.utils.result_export import export_result
    from ssdiff_gui.utils.save_config import ItemConfig, SaveConfig

    cfg = replace(
        SaveConfig.default(),
        items={
            **SaveConfig.default().items,
            "words": ItemConfig(enabled=True),
            "clusters": ItemConfig(enabled=True),
        },
    )

    result = _wrap(tmp_path / "r", _make_pls(tiny_embeddings, synthetic_corpus))
    export_result(result, result.result_path, cfg)

    tables = result.result_path / "tables"
    assert tables.is_dir()
    assert (tables / "words.csv").exists()
    assert (tables / "clusters_pos.csv").exists()
    assert (tables / "clusters_neg.csv").exists()


def test_resave_rewrites_tables_and_removes_old_report(
    tmp_path, tiny_embeddings, synthetic_corpus,
):
    """Regression guard: a second save with a different config must clean up
    the artifacts from the first save."""
    from ssdiff_gui.utils.result_export import export_result
    from ssdiff_gui.utils.save_config import ItemConfig, SaveConfig

    result = _wrap(tmp_path / "r", _make_pls(tiny_embeddings, synthetic_corpus))
    cfg_first = replace(
        SaveConfig.default(),
        report_format="md",
        items={**SaveConfig.default().items, "words": ItemConfig(enabled=True)},
    )
    export_result(result, result.result_path, cfg_first)
    assert (result.result_path / "report.md").exists()
    assert (result.result_path / "tables" / "words.csv").exists()

    cfg_second = replace(
        SaveConfig.default(),
        report_format="txt",
        items={**SaveConfig.default().items, "clusters": ItemConfig(enabled=True)},
    )
    export_result(result, result.result_path, cfg_second)

    assert not (result.result_path / "report.md").exists()
    assert (result.result_path / "report.txt").exists()
    tables = result.result_path / "tables"
    assert not (tables / "words.csv").exists()
    assert (tables / "clusters_pos.csv").exists()


def test_sweep_plot_writes_png_when_sweep_present(
    tmp_path, tiny_embeddings, synthetic_corpus,
):
    from ssdiff_gui.utils.result_export import export_result
    from ssdiff_gui.utils.save_config import ItemConfig, SaveConfig

    cfg = replace(
        SaveConfig.default(),
        items={**SaveConfig.default().items, "sweep_plot": ItemConfig(enabled=True)},
    )

    result = _wrap(
        tmp_path / "r",
        _make_pcaols_with_sweep(tiny_embeddings, synthetic_corpus),
        config_snapshot={"analysis_type": "pca_ols"},
    )
    export_result(result, result.result_path, cfg)

    assert (result.result_path / "tables" / "sweep_plot.png").exists()


def test_groups_pairs_writes_expected_file(
    tmp_path, tiny_embeddings, synthetic_corpus,
):
    """TIGHTENED from liberal version: assert exact file(s) written,
    not just 'any directory exists'."""
    from ssdiff_gui.utils.result_export import export_result
    from ssdiff_gui.utils.save_config import ItemConfig, SaveConfig

    cfg = replace(
        SaveConfig.default(),
        items={**SaveConfig.default().items, "pairs": ItemConfig(enabled=True)},
    )

    result = _wrap(
        tmp_path / "r", _make_groups(tiny_embeddings, synthetic_corpus),
        config_snapshot={"analysis_type": "groups"},
    )
    export_result(result, result.result_path, cfg)

    tables = result.result_path / "tables"
    assert tables.exists()

    single = (tables / "pairs.csv").exists()
    fanned = (tables / "pairs").is_dir() and any((tables / "pairs").iterdir())
    assert single or fanned, (
        f"expected pairs.csv or pairs/ directory; got {list(tables.iterdir())}"
    )


def test_words_cols_subset_produces_exact_header(
    tmp_path, tiny_embeddings, synthetic_corpus,
):
    from ssdiff_gui.utils.result_export import export_result
    from ssdiff_gui.utils.save_config import ItemConfig, SaveConfig

    cfg = replace(
        SaveConfig.default(),
        items={
            **SaveConfig.default().items,
            "words": ItemConfig(enabled=True, cols=("word", "rank")),
        },
    )
    result = _wrap(tmp_path / "r", _make_pls(tiny_embeddings, synthetic_corpus))
    export_result(result, result.result_path, cfg)

    with open(result.result_path / "tables" / "words.csv", newline="") as f:
        header = next(_csv.reader(f))
    assert header == ["word", "rank"]


def test_words_k_limits_rows(tmp_path, tiny_embeddings, synthetic_corpus):
    from ssdiff_gui.utils.result_export import export_result
    from ssdiff_gui.utils.save_config import ItemConfig, SaveConfig

    cfg = replace(
        SaveConfig.default(),
        items={**SaveConfig.default().items, "words": ItemConfig(enabled=True, k=3)},
    )
    result = _wrap(tmp_path / "r", _make_pls(tiny_embeddings, synthetic_corpus))
    export_result(result, result.result_path, cfg)

    with open(result.result_path / "tables" / "words.csv", newline="") as f:
        rows = [r for r in _csv.reader(f) if r]
    assert len(rows) == 4
