"""Flat project data model for SSD.

Two classes: Project (all config + computed readiness + validation)
and Result (config snapshot + results reference).
"""

from dataclasses import dataclass, field
from inspect import signature
from pathlib import Path
from typing import Optional, List, Any, Tuple
from datetime import datetime

from ssdiff import SSD

_PLS    = signature(SSD.fit_pls).parameters
_PCAOLS = signature(SSD.fit_ols).parameters
_GROUPS = signature(SSD.fit_groups).parameters

DEFAULT_RANDOM_SEED = 2137


@dataclass
class Result:
    """A single SSD analysis result with a flat config snapshot."""
    result_id: str                   # YYYYMMDD_HHMMSS — stable unique id
    timestamp: datetime
    config_snapshot: dict            # flat copy of all config at run time

    # Set at save time (None for an unsaved, in-memory result).
    result_path: Optional[Path] = None
    folder_name: Optional[str] = None

    name: Optional[str] = None      # user-assigned name for archiving
    status: str = "pending"         # pending, running, complete, error, interrupted
    error_message: Optional[str] = None

    # Runtime only (not serialized) — reset every session
    _result: Optional[Any] = field(default=None, repr=False)   # ssdiff PLSResult/PCAOLSResult/GroupResult
    is_orphan: bool = field(default=False, repr=False)         # on-disk but not in project.json tracked list
    load_error: Optional[str] = field(default=None, repr=False)  # set when the folder exists but can't be loaded

    def to_dict(self) -> dict:
        """Serialize result metadata + config snapshot to JSON-friendly dict."""
        return {
            "result_id": self.result_id,
            "name": self.name,
            "folder_name": self.folder_name,
            "timestamp": self.timestamp.isoformat(),
            "result_path": str(self.result_path) if self.result_path else None,
            "config_snapshot": self.config_snapshot,
            "status": self.status,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, d: dict, result_path: Path) -> "Result":
        """Reconstruct Result from saved dict."""
        result = cls(
            result_id=d.get("result_id") or d["run_id"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            result_path=result_path,
            folder_name=d.get("folder_name"),
            config_snapshot=d.get("config_snapshot", {}),
            name=d.get("name"),
            status=d.get("status", "pending"),
            error_message=d.get("error_message"),
        )
        # Status recovery: running but no results file → interrupted
        if result.status == "running":
            results_pkl = result_path / "results.pkl"
            if not results_pkl.exists():
                result.status = "interrupted"
        return result

    def to_replication_script(self) -> str:
        """Generate a standalone Python script to reproduce this run."""
        s = self.config_snapshot
        atype = s.get("analysis_type", "pls")

        lines = [
            "from ssdiff import SSD, Corpus, Embeddings",
            "import pandas as pd",
            "",
            f'df = pd.read_csv("{s.get("csv_path", "data.csv")}", '
            f'encoding="{s.get("csv_encoding", "utf-8-sig")}")',
            f'corpus = Corpus.from_dataframe(df, text_column="{s.get("text_column", "text")}", '
            f'language="{s.get("language", "en")}")',
            f'emb = Embeddings.load("{s.get("selected_embedding", "embeddings.ssdembed")}")',
            "",
        ]

        # Lexicon
        lexicon = s.get("lexicon_tokens", [])
        concept_mode = s.get("concept_mode", "lexicon")
        use_full_doc = concept_mode != "lexicon"

        if concept_mode == "lexicon":
            lines.append(f"lexicon = {lexicon!r}")
        else:
            lines.append("lexicon = list(set(t for doc in corpus.docs for t in doc))")

        # Outcome / groups
        if atype in ("pls", "pca_ols"):
            col = s.get("outcome_column", "outcome")
            lines.append(f'y = df["{col}"]')
        else:
            col = s.get("group_column", "group")
            lines.append(f'y = df["{col}"]')

        lines.append("")
        lines.append("ssd = SSD(")
        lines.append("    embeddings=emb,")
        lines.append("    corpus=corpus,")
        lines.append("    y=y,")
        lines.append("    lexicon=lexicon,")
        lines.append(f"    window={s.get('context_window_size', 3)},")
        lines.append(f"    sif_a={s.get('sif_a', 1e-3)},")
        if use_full_doc:
            lines.append("    use_full_doc=True,")
        lines.append(")")
        lines.append("")

        # Fit call
        if atype == "pls":
            n_comp = s.get("pls_n_components", 1)
            n_comp_str = f'"{n_comp}"' if n_comp == "auto" else str(n_comp)
            p_method = s.get("pls_p_method", "auto")
            p_method_str = f'"{p_method}"' if p_method else "None"
            rs = s.get("pls_random_state", "default")
            rs_val = DEFAULT_RANDOM_SEED if rs == "default" else int(rs)
            lines.append("result = ssd.fit_pls(")
            lines.append(f"    n_components={n_comp_str},")
            pca_pre = s.get("pls_pca_preprocess")
            if pca_pre is not None:
                lines.append(f"    pca_preprocess={pca_pre},")
            lines.append(f"    p_method={p_method_str},")
            lines.append(f"    n_perm={s.get('pls_n_perm', Project.__dataclass_fields__['pls_n_perm'].default)},")
            lines.append(f"    n_splits={s.get('pls_n_splits', Project.__dataclass_fields__['pls_n_splits'].default)},")
            lines.append(f"    split_ratio={s.get('pls_split_ratio', Project.__dataclass_fields__['pls_split_ratio'].default)},")
            lines.append(f"    random_state={rs_val},")
            lines.append(")")
        elif atype == "pca_ols":
            lines.append("result = ssd.fit_ols(")
            n_comp = s.get("pcaols_n_components")
            lines.append(f"    fixed_k={n_comp},")
            lines.append(f"    k_min={s.get('sweep_k_min', Project.__dataclass_fields__['sweep_k_min'].default)},")
            lines.append(f"    k_max={s.get('sweep_k_max', Project.__dataclass_fields__['sweep_k_max'].default)},")
            lines.append(f"    k_step={s.get('sweep_k_step', Project.__dataclass_fields__['sweep_k_step'].default)},")
            lines.append(")")
        elif atype == "groups":
            rs = s.get("groups_random_state", "default")
            rs_val = DEFAULT_RANDOM_SEED if rs == "default" else int(rs)
            lines.append("result = ssd.fit_groups(")
            lines.append(f"    median_split={s.get('groups_median_split', False)},")
            lines.append(f"    n_perm={s.get('groups_n_perm', Project.__dataclass_fields__['groups_n_perm'].default)},")
            lines.append(f"    correction=\"{s.get('groups_correction', Project.__dataclass_fields__['groups_correction'].default)}\",")
            lines.append(f"    random_state={rs_val},")
            lines.append(")")

        lines.append("")
        lines.append("print(result.summary())")
        lines.append("")
        lines.append("# --- Or load saved results directly: ---")
        lines.append("# import pickle")
        lines.append('# with open("results.pkl", "rb") as f:')
        lines.append("#     result = pickle.load(f)")
        lines.append("# print(result.summary())")

        return "\n".join(lines) + "\n"


# Fields serialized to project.json (excludes runtime-only and run list)
_SERIALIZED_FIELDS = [
    "name", "created_date", "modified_date",
    "csv_path", "csv_encoding", "text_column", "id_column", "n_rows", "n_valid",
    "language", "spacy_model", "input_mode", "stopword_mode", "custom_stopwords",
    "preprocessed_text_column", "preprocessed_language",
    "n_docs_processed", "total_tokens",
    "mean_words_before_stopwords",
    "selected_embedding", "vocab_size", "embedding_dim", "l2_normalized", "abtt",
    "emb_coverage_pct", "emb_n_oov",
    "analysis_type", "concept_mode", "outcome_column", "group_column", "lexicon_tokens",
    "min_hits_per_doc", "drop_no_hits", "fulldoc_stoplist",
    "concept_coverage_pct", "concept_n_docs_with_hits",
    "concept_median_hits", "concept_mean_hits",
    "context_window_size", "sif_a",
    "clustering_topn", "clustering_k_auto", "clustering_k_min",
    "clustering_k_max", "clustering_top_words",
    "pls_n_components", "pls_pca_preprocess", "pls_p_method",
    "pls_n_perm", "pls_n_splits", "pls_split_ratio", "pls_random_state",
    "pcaols_n_components", "sweep_k_min", "sweep_k_max", "sweep_k_step",
    "groups_n_perm", "groups_correction", "groups_median_split", "groups_random_state",
]

# Fields included in the per-run config snapshot
_SNAPSHOT_COMMON = [
    "csv_path", "csv_encoding", "text_column", "id_column",
    "outcome_column", "group_column",
    "n_rows", "n_valid",
    "n_docs_processed", "total_tokens", "mean_words_before_stopwords",
    "language", "spacy_model", "input_mode", "stopword_mode",
    "analysis_type", "concept_mode", "lexicon_tokens",
    "min_hits_per_doc", "drop_no_hits", "fulldoc_stoplist",
    "context_window_size", "sif_a",
    "selected_embedding", "l2_normalized", "abtt",
    "clustering_topn", "clustering_k_auto", "clustering_k_min",
    "clustering_k_max", "clustering_top_words",
]
_SNAPSHOT_PLS = [
    "pls_n_components", "pls_pca_preprocess", "pls_p_method",
    "pls_n_perm", "pls_n_splits", "pls_split_ratio", "pls_random_state",
]
_SNAPSHOT_PCA_OLS = [
    "pcaols_n_components", "sweep_k_min", "sweep_k_max", "sweep_k_step",
]
_SNAPSHOT_GROUPS = [
    "groups_n_perm", "groups_correction", "groups_median_split", "groups_random_state",
]


@dataclass
class Project:
    """Complete project state — flat, no nested config objects."""

    # -- Identity --
    project_path: Path
    name: str
    created_date: datetime
    modified_date: datetime

    # -- Dataset --
    csv_path: Optional[Path] = None
    csv_encoding: str = "utf-8-sig"
    text_column: Optional[str] = None
    id_column: Optional[str] = None
    n_rows: int = 0
    n_valid: int = 0

    # -- Text Processing --
    language: str = "en"
    spacy_model: str = ""
    input_mode: str = "language"          # "language" or "custom"
    stopword_mode: str = "default"        # "default", "none", "custom"
    custom_stopwords: List[str] = field(default_factory=list)
    preprocessed_text_column: Optional[str] = None
    preprocessed_language: str = ""
    n_docs_processed: int = 0
    total_tokens: int = 0
    mean_words_before_stopwords: float = 0.0

    # -- Embeddings --
    selected_embedding: Optional[str] = None
    vocab_size: int = 0
    embedding_dim: int = 0
    l2_normalized: bool = False
    abtt: int = 0
    emb_coverage_pct: float = 0.0
    emb_n_oov: int = 0

    # -- Analysis --
    analysis_type: str = "pls"            # "pls", "pca_ols", "groups"
    concept_mode: str = "lexicon"         # "lexicon", "fulldoc"
    outcome_column: Optional[str] = None
    group_column: Optional[str] = None
    lexicon_tokens: List[str] = field(default_factory=list)
    min_hits_per_doc: Optional[int] = None
    drop_no_hits: bool = True
    fulldoc_stoplist: List[str] = field(default_factory=list)

    # -- Concept stats (computed, not user-set) --
    concept_coverage_pct: float = 0.0
    concept_n_docs_with_hits: int = 0
    concept_median_hits: float = 0.0
    concept_mean_hits: float = 0.0

    # -- Hyperparameters: common --
    context_window_size: int = 3
    sif_a: float = 1e-3

    # -- Hyperparameters: clustering --
    clustering_topn: int = 100
    clustering_k_auto: bool = True
    clustering_k_min: int = 2
    clustering_k_max: int = 10
    clustering_top_words: int = 10

    # -- Hyperparameters: PLS --
    pls_n_components: int   = _PLS["n_components"].default
    pls_pca_preprocess: Optional[int] = None
    pls_p_method:     str   = _PLS["p_method"].default
    pls_n_perm:       int   = _PLS["n_perm"].default
    pls_n_splits:     int   = _PLS["n_splits"].default
    pls_split_ratio:  float = _PLS["split_ratio"].default
    pls_random_state: str   = "default"

    # -- Hyperparameters: PCA+OLS --
    pcaols_n_components: Optional[int] = None
    sweep_k_min:  int = _PCAOLS["k_min"].default
    sweep_k_max:  int = _PCAOLS["k_max"].default
    sweep_k_step: int = _PCAOLS["k_step"].default

    # -- Hyperparameters: Groups --
    groups_n_perm:       int  = _GROUPS["n_perm"].default
    groups_correction:   str  = _GROUPS["correction"].default
    groups_median_split: bool = False
    groups_random_state: str  = "default"

    # -- Results --
    results: List[Result] = field(default_factory=list)

    # -- Runtime only (not serialized) --
    _dirty: bool = field(default=False, repr=False)
    _df: Optional[Any] = field(default=None, repr=False)
    _corpus: Optional[Any] = field(default=None, repr=False)
    _emb: Optional[Any] = field(default=None, repr=False)
    _kv: Optional[Any] = field(default=None, repr=False)
    _nlp: Optional[Any] = field(default=None, repr=False)
    _stopwords: Optional[List] = field(default=None, repr=False)
    _pre_docs: Optional[List] = field(default=None, repr=False)
    _docs: Optional[List] = field(default=None, repr=False)
    _y: Optional[Any] = field(default=None, repr=False)
    _groups: Optional[Any] = field(default=None, repr=False)
    _id_row_indices: Optional[List] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    #  Dirty tracking
    # ------------------------------------------------------------------ #

    def mark_dirty(self):
        self._dirty = True

    def mark_clean(self):
        self._dirty = False

    # ------------------------------------------------------------------ #
    #  Computed readiness (no boolean flags)
    # ------------------------------------------------------------------ #

    @property
    def text_ready(self) -> bool:
        """Dataset loaded and text column selected."""
        return self._df is not None and self.text_column is not None

    @property
    def preprocessing_ready(self) -> bool:
        """Corpus in RAM matches the current text column AND chosen language."""
        return (self._corpus is not None
                and self.preprocessed_text_column == self.text_column
                and self.language == self.preprocessed_language)

    @property
    def embeddings_ready(self) -> bool:
        """Embeddings loaded into RAM."""
        return self._emb is not None

    @property
    def stage1_ready(self) -> bool:
        """All Stage 1 prerequisites satisfied."""
        return self.text_ready and self.preprocessing_ready and self.embeddings_ready

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #

    def validate_text(self) -> Tuple[List[str], List[str], Optional[dict]]:
        """Validate text column and optional ID column.

        Returns (errors, warnings, id_stats).
        """

        errors: List[str] = []
        warnings: List[str] = []
        id_stats = None

        if self._df is None:
            errors.append("No dataset loaded")
            return errors, warnings, id_stats

        df = self._df
        text_col = self.text_column

        if not text_col or text_col not in df.columns:
            errors.append(f"Text column '{text_col}' not found in dataset")
            return errors, warnings, id_stats

        # Empty texts
        n_empty = df[text_col].isna().sum() + (df[text_col].astype(str).str.strip() == "").sum()
        if n_empty > 0:
            pct = n_empty / len(df) * 100
            if pct > 50:
                errors.append(f"{pct:.1f}% of texts are empty or missing")
            elif pct > 10:
                warnings.append(f"{pct:.1f}% of texts are empty or missing")

        # Meaningful text (7+ chars)
        meaningful = df[text_col].astype(str).str.strip().str.len() >= 7
        n_meaningful = int(meaningful.sum())
        n_total = len(df)

        if n_meaningful < 30:
            errors.append(
                f"Only {n_meaningful} texts have 7+ characters"
                " \u2014 not enough for SSD analysis"
            )
        else:
            pct_meaningful = n_meaningful / n_total * 100
            if pct_meaningful < 50:
                warnings.append(
                    f"Only {pct_meaningful:.0f}% of texts have 7+ characters"
                    " \u2014 results may be unreliable"
                )

        # Sample size
        if n_total < 30:
            errors.append(f"Only {n_total} rows (need at least 30)")
        elif n_total < 100:
            warnings.append(f"Small sample size ({n_total} documents)")

        # ID stats
        id_col = self.id_column
        if id_col and id_col in df.columns:
            n_unique = df[id_col].nunique(dropna=True)
            has_duplicates = n_unique < (~df[id_col].isna()).sum()
            avg_texts = len(df) / n_unique if n_unique > 0 else 0.0
            id_stats = {
                "n_unique_ids": n_unique,
                "has_duplicates": has_duplicates,
                "avg_texts_per_id": avg_texts,
            }

        return errors, warnings, id_stats

    def validate_outcome(self) -> Tuple[List[str], List[str]]:
        """Validate outcome/group column before a run.

        For pls/pca_ols: numeric, >50% valid, sufficient variance, >=30 samples.
        For groups: 2+ groups with >=10 members each.
        """
        import pandas as pd

        errors: List[str] = []
        warnings: List[str] = []

        if self._df is None:
            errors.append("No dataset loaded")
            return errors, warnings

        df = self._df

        if self.analysis_type in ("pls", "pca_ols"):
            col = self.outcome_column
            if not col or col not in df.columns:
                errors.append(f"Outcome column '{col}' not found in dataset")
                return errors, warnings

            outcome = pd.to_numeric(df[col], errors="coerce")
            n_invalid = outcome.isna().sum()
            n_original_na = df[col].isna().sum()
            n_non_numeric = n_invalid - n_original_na

            if n_non_numeric > 0:
                pct = n_non_numeric / len(df) * 100
                if pct > 50:
                    errors.append(f"{pct:.1f}% of outcome values are non-numeric")
                else:
                    warnings.append(f"{n_non_numeric} outcome values are non-numeric")

            valid_outcome = outcome.dropna()
            if len(valid_outcome) > 0:
                outcome_std = valid_outcome.std()
                if outcome_std < 0.01:
                    errors.append("Outcome has near-zero variance")
                elif outcome_std < 0.1:
                    warnings.append("Outcome has low variance")

            n_valid = len(valid_outcome)
            if n_valid < 30:
                errors.append(f"Only {n_valid} valid samples (need at least 30)")
            elif n_valid < 100:
                warnings.append(f"Small sample size ({n_valid} documents)")

        elif self.analysis_type == "groups":
            col = self.group_column
            if not col or col not in df.columns:
                errors.append(f"Group column '{col}' not found in dataset")
                return errors, warnings

            groups = df[col].dropna()
            counts = groups.value_counts()
            if len(counts) < 2:
                errors.append("Need at least 2 groups")
            else:
                small = counts[counts < 10]
                if len(small) > 0:
                    labels = ", ".join(str(lbl) for lbl in small.index[:3])
                    errors.append(f"Groups with <10 members: {labels}")

        return errors, warnings

    def validate_lexicon(self) -> Tuple[List[str], List[str]]:
        """Validate lexicon tokens against loaded embedding vocab.

        Requires self._emb and self._corpus to be loaded.
        """
        errors: List[str] = []
        warnings: List[str] = []

        lexicon = set(self.lexicon_tokens) if self.lexicon_tokens else set()
        if not lexicon:
            errors.append("Lexicon is empty")
            return errors, warnings

        if self._kv is None:
            errors.append("No embeddings loaded")
            return errors, warnings

        vocab = set(self._kv.key_to_index.keys()) if hasattr(self._kv, 'key_to_index') else set()

        # OOV tokens
        oov_tokens = lexicon - vocab
        if oov_tokens == lexicon:
            errors.append("None of the lexicon tokens are in the embedding vocabulary")
        elif len(oov_tokens) > 0:
            pct = len(oov_tokens) / len(lexicon) * 100
            if pct > 50:
                warnings.append(f"{pct:.1f}% of lexicon tokens not in vocabulary: {list(oov_tokens)[:5]}...")
            else:
                warnings.append(f"{len(oov_tokens)} tokens not in vocabulary: {list(oov_tokens)[:5]}")

        # Coverage
        valid_tokens = lexicon & vocab
        if valid_tokens and self._docs:
            n_docs_with_hit = sum(1 for doc in self._docs if any(t in valid_tokens for t in doc))
            coverage = n_docs_with_hit / len(self._docs) * 100 if self._docs else 0
            if coverage < 10:
                errors.append(f"Very low coverage: only {coverage:.1f}% of documents contain lexicon terms")
            elif coverage < 30:
                warnings.append(f"Low coverage: {coverage:.1f}% of documents contain lexicon terms")

        # Lexicon size
        if len(valid_tokens) < 3:
            warnings.append(f"Very small lexicon ({len(valid_tokens)} tokens)")
        elif len(valid_tokens) < 5:
            warnings.append(f"Small lexicon ({len(valid_tokens)} tokens)")

        return errors, warnings

    # ------------------------------------------------------------------ #
    #  Config snapshot (for Result)
    # ------------------------------------------------------------------ #

    def snapshot_config(self) -> dict:
        """Flat dict of all config fields needed to reproduce a run."""
        snap = {}
        for key in _SNAPSHOT_COMMON:
            val = getattr(self, key)
            if isinstance(val, Path):
                val = str(val)
            elif isinstance(val, list):
                val = list(val)
            snap[key] = val

        if self.analysis_type == "pls":
            for key in _SNAPSHOT_PLS:
                snap[key] = getattr(self, key)
        elif self.analysis_type == "pca_ols":
            for key in _SNAPSHOT_PCA_OLS:
                snap[key] = getattr(self, key)
        elif self.analysis_type == "groups":
            for key in _SNAPSHOT_GROUPS:
                snap[key] = getattr(self, key)

        return snap

    # ------------------------------------------------------------------ #
    #  Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Serialize to JSON-friendly dict."""
        d = {}
        for key in _SERIALIZED_FIELDS:
            val = getattr(self, key)
            if isinstance(val, Path):
                val = str(val)
            elif isinstance(val, datetime):
                val = val.isoformat()
            d[key] = val
        # Only tracked results are persisted; orphans are discovered from
        # the filesystem each session and don't enter project.json until
        # the user explicitly opens/registers them.
        d["results"] = [
            (r.folder_name or r.result_id)
            for r in self.results
            if not r.is_orphan and (r.folder_name or r.result_id)
        ]
        if self._id_row_indices is not None:
            d["id_row_indices"] = self._id_row_indices
        return d

    @classmethod
    def from_dict(cls, d: dict, project_path: Path) -> "Project":
        """Reconstruct Project from saved dict."""
        proj = cls(
            project_path=project_path,
            name=d["name"],
            created_date=datetime.fromisoformat(d["created_date"]),
            modified_date=datetime.fromisoformat(d["modified_date"]),
        )
        # Restore all serialized fields
        for key in _SERIALIZED_FIELDS:
            if key in ("name", "created_date", "modified_date"):
                continue  # already set
            if key not in d:
                continue
            val = d[key]
            # Type coercions
            if key == "csv_path" and val is not None:
                val = Path(val)
            setattr(proj, key, val)

        # Restore id_row_indices
        if "id_row_indices" in d:
            proj._id_row_indices = d["id_row_indices"]

        return proj

