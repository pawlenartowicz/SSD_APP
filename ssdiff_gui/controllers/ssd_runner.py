"""SSD analysis runner thread — uses ssdiff v1.0.0 API."""

import sys
import traceback
from datetime import datetime

from PySide6.QtCore import QThread, Signal


def _ensure_streams() -> None:
    """In windowed PyInstaller exes sys.stdout/stderr are None, which breaks tqdm.
    Redirect them to a no-op stream so any library that writes to them won't crash."""
    import io
    if sys.stdout is None:
        sys.stdout = io.StringIO()
    if sys.stderr is None:
        sys.stderr = io.StringIO()


def _debug_log(msg: str) -> None:
    """Write a message to stderr (dev) and to a log file (frozen exe where stderr is /dev/null)."""
    try:
        print(msg, file=sys.stderr)
    except Exception:
        pass
    try:
        from ..utils.paths import get_app_data_dir
        log_dir = get_app_data_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "debug.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")
    except Exception:
        pass


from ..models.project import Project, Result, DEFAULT_RANDOM_SEED  # noqa: E402


class SSDRunner(QThread):
    """Worker thread for running SSD analysis."""

    progress = Signal(int, str)  # percent, message
    finished = Signal(object)  # Result object
    error = Signal(str)

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self._is_cancelled = False

    def cancel(self):
        """Request cancellation of the analysis."""
        self._is_cancelled = True

    def run(self):
        """Execute the SSD analysis pipeline."""
        try:
            # Build an in-memory Result — nothing is written to disk here.
            # The result folder, config.json, results.pkl, report and sweep
            # plot are only written when the user clicks Save in Stage 3.
            result_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            result = Result(
                result_id=result_id,
                timestamp=datetime.now(),
                config_snapshot=self.project.snapshot_config(),
                status="running",
            )

            if self._is_cancelled:
                return

            atype = self.project.analysis_type
            if atype == "pls":
                self._run_pls(result)
            elif atype == "pca_ols":
                self._run_pca_ols(result)
            elif atype == "groups":
                self._run_groups(result)
            else:
                raise ValueError(f"Unknown analysis_type: {atype!r}")

        except Exception as e:
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _build_ssd(self):
        """Construct the SSD object from cached project data."""
        from ssdiff import SSD, Corpus

        emb = self.project._emb
        docs = self.project._docs
        pre_docs = self.project._pre_docs

        atype = self.project.analysis_type
        if atype == "groups":
            y = self.project._groups
        else:
            y = self.project._y

        if emb is None or docs is None or y is None:
            raise ValueError("Cached data not available. Please complete Stage 1 setup.")

        p = self.project

        # Use cached Corpus if available, otherwise build from pre-tokenized docs
        corpus = self.project._corpus
        if corpus is None:
            lang = self.project.language
            corpus = Corpus(docs, pretokenized=True, lang=lang)

        # Snippet extraction reads corpus.pre_docs; Corpus(pretokenized=True)
        # leaves it as None, so attach the cached preprocessed docs here.
        if pre_docs is not None and getattr(corpus, "pre_docs", None) is None:
            corpus.pre_docs = pre_docs

        # Lexicon / full-doc mode
        lexicon = set()
        use_full_doc = True
        if self.project.concept_mode == "lexicon":
            lexicon = set(self.project.lexicon_tokens) if self.project.lexicon_tokens else set()
            use_full_doc = False
            if not lexicon:
                raise ValueError("Lexicon mode selected but no tokens provided.")
        else:
            # Full doc mode: still needs a non-empty lexicon for SSD constructor
            # Use all unique tokens from the corpus as a pseudo-lexicon
            all_tokens = set()
            for doc in docs:
                all_tokens.update(doc)
            lexicon = all_tokens

        ssd = SSD(
            emb, corpus, y, lexicon,
            window=p.context_window_size,
            sif_a=p.sif_a,
            use_full_doc=use_full_doc,
        )

        return ssd, pre_docs

    def _compute_coverage(self, var_type="continuous"):
        """Compute lexicon coverage if in lexicon mode."""
        if self.project.concept_mode != "lexicon":
            return None, None

        lexicon = set(self.project.lexicon_tokens) if self.project.lexicon_tokens else set()
        if not lexicon:
            return None, None

        corpus = self.project._corpus
        if corpus is None:
            return None, None

        if var_type == "categorical":
            y_full = getattr(self.project, "_groups_full", None)
        else:
            y_full = getattr(self.project, "_y_full", None)
        if y_full is None:
            return None, None

        try:
            cov_summary = corpus.coverage_summary(
                y_full, lexicon, var_type=var_type,
            )
            cov_per_token = corpus.token_stats(
                y_full, lexicon, var_type=var_type,
            )
            return cov_summary, cov_per_token
        except Exception as e:
            _debug_log(f"Lexicon coverage failed: {e}\n{traceback.format_exc()}")
            return None, None

    def _cache_interpretation(self, result, pre_docs, project):
        """Cache interpretation data (neighbors, clusters, snippets) on the ssdiff result.

        Called before saving so that the cached data persists in results.pkl.
        """
        clustering_topn = project.clustering_topn

        _ = result.words

        k = None if project.clustering_k_auto else project.clustering_k_min
        cluster_kwargs = dict(
            topn=clustering_topn,
            k=k, k_min=project.clustering_k_min, k_max=project.clustering_k_max,
        )

        paired_clusters = hasattr(result.clusters, "_views")
        sided_views: list = []
        try:
            if paired_clusters:
                for pair_key, pair_idx in result.clusters._views.items():
                    sided_views.append((pair_key, pair_idx.pos(**cluster_kwargs)))
                    sided_views.append((pair_key, pair_idx.neg(**cluster_kwargs)))
            else:
                sided_views.append((None, result.clusters.pos(**cluster_kwargs)))
                sided_views.append((None, result.clusters.neg(**cluster_kwargs)))
        except Exception as e:
            _debug_log(f"Clustering failed: {e}\n{traceback.format_exc()}")

        if pre_docs:
            try:
                if paired_clusters:
                    for pair_key in result.snippets._views:
                        result.snippets[pair_key](top_per_side=200)
                else:
                    result.snippets(top_per_side=200)
            except Exception as e:
                _debug_log(f"Snippets failed: {e}\n{traceback.format_exc()}")

            for _pair_key, side_view in sided_views:
                if side_view is None or getattr(side_view, "_parent", None) is None:
                    continue
                try:
                    _ = side_view.snippets
                except Exception as e:
                    _debug_log(
                        f"Cluster snippets failed: {e}\n{traceback.format_exc()}"
                    )

    def _resolve_random_state(self, rs_str: str) -> int:
        """Convert "default" or str(int) to an int seed."""
        if rs_str == "default":
            return DEFAULT_RANDOM_SEED
        try:
            return int(rs_str)
        except (ValueError, TypeError):
            return DEFAULT_RANDOM_SEED

    # ------------------------------------------------------------------ #
    #  PLS pipeline
    # ------------------------------------------------------------------ #

    def _run_pls(self, result: Result):
        _ensure_streams()
        from ssdiff import progress_hook
        from ..utils.progress import make_progress_cb
        p = self.project

        self.progress.emit(5, "Building SSD model...")
        ssd, pre_docs = self._build_ssd()

        if self._is_cancelled:
            return

        # Coverage
        self.progress.emit(10, "Computing lexicon coverage...")
        cov_summary, cov_per_token = self._compute_coverage()

        if self._is_cancelled:
            return

        # Resolve parameters
        n_comp = p.pls_n_components if p.pls_n_components != 0 else "auto"
        p_method = p.pls_p_method if p.pls_p_method != "none" else None
        random_state = self._resolve_random_state(p.pls_random_state)

        self.progress.emit(15, f"Fitting PLS (n_comp={n_comp}, p_method={p_method})...")

        cb = make_progress_cb(self.progress, 15, 55, "Fitting PLS")
        with progress_hook(cb):
            ssd_result = ssd.fit_pls(
                n_components=n_comp,
                pca_preprocess=p.pls_pca_preprocess,
                p_method=p_method,
                n_perm=p.pls_n_perm,
                n_splits=p.pls_n_splits,
                split_ratio=p.pls_split_ratio,
                random_state=random_state,
            )

        if self._is_cancelled:
            return

        self.progress.emit(55, "Caching interpretation data...")

        # Cache interpretation on the ssdiff result
        self._cache_interpretation(ssd_result, pre_docs, p)

        if self._is_cancelled:
            return

        self.progress.emit(85, "Finalizing...")
        self._finalize_result(result, ssd_result,
                              cov_summary=cov_summary, cov_per_token=cov_per_token)

    # ------------------------------------------------------------------ #
    #  PCA+OLS pipeline
    # ------------------------------------------------------------------ #

    def _run_pca_ols(self, result: Result):
        _ensure_streams()
        from ssdiff import progress_hook
        from ..utils.progress import make_progress_cb
        p = self.project

        self.progress.emit(5, "Building SSD model...")
        ssd, pre_docs = self._build_ssd()

        if self._is_cancelled:
            return

        # Coverage
        self.progress.emit(10, "Computing lexicon coverage...")
        cov_summary, cov_per_token = self._compute_coverage()

        if self._is_cancelled:
            return

        # Resolve parameters
        n_comp = p.pcaols_n_components  # None = sweep
        sweep_msg = "sweep" if n_comp is None else f"K={n_comp}"
        self.progress.emit(10, f"Fitting PCA+OLS ({sweep_msg})...")

        cb = make_progress_cb(self.progress, 10, 45, "PCA sweep")
        with progress_hook(cb):
            ssd_result = ssd.fit_ols(
                fixed_k=n_comp,
                k_min=p.sweep_k_min,
                k_max=p.sweep_k_max,
                k_step=p.sweep_k_step,
            )

        if self._is_cancelled:
            return

        self.progress.emit(50, "Caching interpretation data...")

        # Cache interpretation on the ssdiff result
        self._cache_interpretation(ssd_result, pre_docs, p)

        if self._is_cancelled:
            return

        self.progress.emit(80, "Finalizing...")
        self._finalize_result(result, ssd_result,
                              cov_summary=cov_summary, cov_per_token=cov_per_token)

    # ------------------------------------------------------------------ #
    #  Groups pipeline
    # ------------------------------------------------------------------ #

    def _run_groups(self, result: Result):
        _ensure_streams()
        from ssdiff import progress_hook
        from ..utils.progress import make_progress_cb
        p = self.project

        self.progress.emit(5, "Building SSD model...")
        ssd, pre_docs = self._build_ssd()

        if self._is_cancelled:
            return

        # Coverage
        self.progress.emit(10, "Computing lexicon coverage...")
        cov_summary, cov_per_token = self._compute_coverage(
            var_type="categorical",
        )

        if self._is_cancelled:
            return

        random_state = self._resolve_random_state(p.groups_random_state)

        self.progress.emit(10, f"Fitting group comparison (n_perm={p.groups_n_perm})...")

        cb = make_progress_cb(self.progress, 10, 55, "Permutation test")
        with progress_hook(cb):
            ssd_result = ssd.fit_groups(
                median_split=p.groups_median_split,
                n_perm=p.groups_n_perm,
                correction=p.groups_correction,
                random_state=random_state,
            )

        if self._is_cancelled:
            return

        self.progress.emit(55, "Caching interpretation data...")

        self._cache_interpretation(ssd_result, pre_docs, p)

        if self._is_cancelled:
            return

        self.progress.emit(85, "Finalizing...")
        self._finalize_result(result, ssd_result,
                              cov_summary=cov_summary, cov_per_token=cov_per_token)

    # ------------------------------------------------------------------ #
    #  Finalize
    # ------------------------------------------------------------------ #

    def _finalize_result(self, result, ssd_result,
                         cov_summary=None, cov_per_token=None):
        """Attach the ssdiff result to the in-memory Result and signal done.

        Nothing is written to disk here — persistence happens in Stage 3
        when the user clicks Save.
        """
        result._result = ssd_result
        result.status = "complete"

        if cov_summary is not None:
            result.config_snapshot["lexicon_coverage_summary"] = cov_summary
        if cov_per_token is not None:
            result.config_snapshot["lexicon_coverage_per_token"] = cov_per_token

        self.progress.emit(100, "Complete!")
        self.finished.emit(result)
