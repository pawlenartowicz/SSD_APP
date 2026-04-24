"""Replication script generation — must produce valid Python that
mirrors the analysis config."""

import pytest


_BASE_CONFIG = {
    "csv_path": "/data/test.csv",
    "csv_encoding": "utf-8-sig",
    "text_column": "text",
    "language": "en",
    "selected_embedding": "glove.ssdembed",
    "concept_mode": "lexicon",
    "lexicon_tokens": ["happy", "sad", "angry"],
    "outcome_column": "score",
    "group_column": "group",
    "context_window_size": 3,
    "sif_a": 1e-3,
}


@pytest.mark.parametrize("config,must_contain,must_not_contain", [
    (
        {
            **_BASE_CONFIG,
            "analysis_type": "pls",
            "pls_n_components": 2,
            "pls_p_method": "perm",
            "pls_n_perm": 1000,
            "pls_n_splits": 50,
            "pls_split_ratio": 0.5,
            "pls_random_state": "default",
        },
        ["fit_pls(", "n_components=2", '"perm"', "n_perm=1000",
         "n_splits=50", "split_ratio=0.5", "random_state=2137"],
        ["fit_ols(", "fit_groups("],
    ),
    (
        {
            **_BASE_CONFIG,
            "analysis_type": "pca_ols",
            "pcaols_n_components": 40,
            "sweep_k_min": 20,
            "sweep_k_max": 120,
            "sweep_k_step": 2,
        },
        ["fit_ols(", "fixed_k=40", "k_min=20", "k_max=120", "k_step=2"],
        ["fit_pls(", "fit_groups("],
    ),
    (
        {
            **_BASE_CONFIG,
            "analysis_type": "groups",
            "groups_n_perm": 5000,
            "groups_correction": "holm",
            "groups_median_split": True,
            "groups_random_state": "42",
        },
        ["fit_groups(", '"holm"', "n_perm=5000", "median_split=True",
         "random_state=42"],
        ["fit_pls(", "fit_ols("],
    ),
    (
        {
            **_BASE_CONFIG,
            "analysis_type": "pls",
            "concept_mode": "fulldoc",
            "pls_n_components": 1,
            "pls_p_method": "auto",
            "pls_random_state": "default",
        },
        ["use_full_doc=True", "set(t for doc"],
        ["lexicon = ['happy'"],
    ),
])
def test_script_contains_expected_args(make_result, config, must_contain, must_not_contain):
    r = make_result(config_snapshot=config)
    script = r.to_replication_script()
    for s in must_contain:
        assert s in script, f"expected {s!r} in script"
    for s in must_not_contain:
        assert s not in script, f"unexpected {s!r} in script"


def test_all_generated_scripts_compile(make_result):
    """VERBATIM regression guard: generated Python must parse."""
    configs = [
        {**_BASE_CONFIG, "analysis_type": "pls",
         "pls_n_components": 1, "pls_p_method": "auto",
         "pls_random_state": "default"},
        {**_BASE_CONFIG, "analysis_type": "pca_ols",
         "pcaols_n_components": None, "sweep_k_min": 20,
         "sweep_k_max": 120, "sweep_k_step": 2},
        {**_BASE_CONFIG, "analysis_type": "groups",
         "groups_n_perm": 5000, "groups_correction": "holm",
         "groups_median_split": False, "groups_random_state": "default"},
    ]
    for cfg in configs:
        r = make_result(config_snapshot=cfg)
        script = r.to_replication_script()
        compile(script, "<replication_script>", "exec")
