"""Background worker threads for SSD."""

from PySide6.QtCore import QThread, Signal
from typing import List, Any, Optional, Union
from pathlib import Path
import os
import ssl


def _urlopen(req, *, timeout=30):
    """urlopen wrapper that falls back to unverified SSL on cert errors.

    PyInstaller bundles can't always find system CA certificates,
    especially after OS updates.  These requests only fetch public
    spaCy models and GitHub release metadata, so falling back to
    unverified SSL is acceptable.
    """
    from urllib.request import urlopen
    from urllib.error import URLError
    try:
        return urlopen(req, timeout=timeout)
    except URLError as e:
        if "CERTIFICATE_VERIFY_FAILED" in str(e):
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return urlopen(req, timeout=timeout, context=ctx)
        raise


def get_spacy_models_dir() -> Path:
    """Return the directory for locally-downloaded spaCy models."""
    from .paths import get_app_data_dir

    models_dir = get_app_data_dir() / "spacy_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def find_local_model(model_name: str) -> Optional[Path]:
    """Check if a spaCy model has been downloaded locally.

    Returns the path to the model data directory if found, otherwise None.
    The wheel extracts into ``{model_name}-{version}/{model_name}/{model_name}-{version}``.
    We look for any version sub-folder matching this pattern.
    """
    import glob as _glob

    models_dir = get_spacy_models_dir()
    # Pattern: models_dir/{model_name}-*/{model_name}/{model_name}-*/
    pattern = str(models_dir / f"{model_name}-*" / model_name / f"{model_name}-*")
    matches = sorted(_glob.glob(pattern))
    if matches:
        return Path(matches[-1])  # latest version
    return None


class PreprocessWorker(QThread):
    """Worker thread for spaCy preprocessing."""

    progress = Signal(int, str)  # percent, message
    finished = Signal(list, list, dict)  # pre_docs, docs, stats
    error = Signal(str)

    def __init__(
        self,
        texts_raw: Union[List[str], List[List[str]]],
        language: str,
        model: str,
        model_path: Optional[Path] = None,
        stopwords_override=None,
        parent=None,
    ):
        super().__init__(parent)
        self.texts_raw = texts_raw
        self.language = language
        self.model = model
        self.model_path = model_path
        self.stopwords_override = stopwords_override  # None=default, []=none, [words]=custom
        self._is_cancelled = False

    def cancel(self):
        """Request cancellation of the preprocessing."""
        self._is_cancelled = True

    def run(self):
        """Execute preprocessing in background thread."""
        try:
            from ssdiff.utils.text import (
                load_spacy,
                load_stopwords,
                preprocess_texts,
                build_docs_from_preprocessed,
            )
            import spacy

            self.progress.emit(5, "Loading spaCy model...")
            if self.model_path:
                # Load from local path (downloaded wheel)
                nlp = spacy.load(str(self.model_path), disable=["ner"])
                if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
                    nlp.add_pipe("sentencizer")
            else:
                nlp = load_spacy(self.model)

            if self._is_cancelled:
                return

            self.progress.emit(15, "Loading stopwords...")
            if self.stopwords_override is not None:
                stopwords = self.stopwords_override
            else:
                stopwords = load_stopwords(self.language)

            if self._is_cancelled:
                return

            n_items = len(self.texts_raw)
            label = "profiles" if self.texts_raw and isinstance(self.texts_raw[0], list) else "documents"
            self.progress.emit(25, f"Preprocessing {n_items} {label}...")

            # Process in batches for progress updates
            pre_docs = preprocess_texts(self.texts_raw, nlp, stopwords)

            if self._is_cancelled:
                return

            self.progress.emit(85, "Building document vectors...")
            docs = build_docs_from_preprocessed(pre_docs)

            if self._is_cancelled:
                return

            self.progress.emit(95, "Computing statistics...")
            stats = self._compute_stats(pre_docs, docs)

            self.progress.emit(100, "Complete!")
            self.finished.emit(pre_docs, docs, stats)

        except Exception as e:
            import traceback
            self.error.emit(f"Preprocessing failed: {str(e)}\n{traceback.format_exc()}")

    def _compute_stats(self, pre_docs: list, docs: list) -> dict:
        """Compute preprocessing statistics.

        Handles both flat (PreprocessedDoc) and grouped (PreprocessedProfile)
        outputs from ssdiff.preprocess_texts().
        """
        from ssdiff.utils.text import PreprocessedProfile

        is_grouped = pre_docs and isinstance(pre_docs[0], PreprocessedProfile)

        if is_grouped:
            # docs is List[List[List[str]]] — profiles × posts × lemmas
            total_tokens = sum(
                sum(len(post) for post in profile) for profile in docs
            )
            n_profiles = len(docs)
            n_total_posts = sum(len(profile) for profile in docs)
            avg_tokens = total_tokens / n_profiles if n_profiles else 0
            empty_profiles = sum(
                1 for profile in docs if all(len(post) == 0 for post in profile)
            )

            # Mean words per profile before stopword removal
            words_per_profile = [
                sum(
                    len(s.split())
                    for post_sents in pdoc.post_sents_surface
                    for s in post_sents
                )
                for pdoc in pre_docs
            ]
            mean_words_before_stopwords = (
                sum(words_per_profile) / len(words_per_profile) if words_per_profile else 0.0
            )

            return {
                "n_docs": n_profiles,
                "n_total_rows": n_total_posts,
                "is_grouped": True,
                "total_tokens": total_tokens,
                "avg_tokens_per_doc": avg_tokens,
                "empty_docs": empty_profiles,
                "mean_words_before_stopwords": mean_words_before_stopwords,
            }
        else:
            # Flat: docs is List[List[str]]
            total_tokens = sum(len(doc) for doc in docs)
            avg_tokens = total_tokens / len(docs) if docs else 0
            empty_docs = sum(1 for doc in docs if len(doc) == 0)

            words_per_doc = [
                sum(len(s.split()) for s in pdoc.sents_surface)
                for pdoc in pre_docs
            ]
            mean_words_before_stopwords = (
                sum(words_per_doc) / len(words_per_doc) if words_per_doc else 0.0
            )

            return {
                "n_docs": len(docs),
                "is_grouped": False,
                "total_tokens": total_tokens,
                "avg_tokens_per_doc": avg_tokens,
                "empty_docs": empty_docs,
                "mean_words_before_stopwords": mean_words_before_stopwords,
            }


class EmbeddingPrepareWorker(QThread):
    """Worker thread for preparing embeddings: load -> normalize -> save .ssdembed.

    Does NOT keep the embedding in RAM — caller decides whether to load it
    after saving. Emits the path to the saved .ssdembed file.
    """

    progress = Signal(int, str)
    finished = Signal(str, dict)  # saved_path, metadata
    error = Signal(str)

    def __init__(
        self,
        source_path: Path,
        output_dir: Path,
        l2_normalize: bool = True,
        abtt_m: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self.source_path = source_path
        self.output_dir = output_dir
        self.l2_normalize = l2_normalize
        self.abtt_m = abtt_m
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            from ssdiff import Embeddings, progress_hook
            from .progress import make_progress_cb

            self.progress.emit(5, "Loading embeddings file...")
            cb = make_progress_cb(self.progress, 5, 55, "Loading embeddings")
            with progress_hook(cb):
                emb = Embeddings.load(str(self.source_path))

            if self._is_cancelled:
                return

            # Check if source already has matching normalization
            source_l2 = emb.l2_normalized
            source_abtt = emb.abtt_m
            needs_normalize = (
                (self.l2_normalize and not source_l2) or
                (self.abtt_m > source_abtt)
            )

            if needs_normalize:
                self.progress.emit(55, "Normalizing embeddings...")
                emb.normalize(l2=self.l2_normalize, abtt_m=self.abtt_m)
            else:
                self.progress.emit(55, "Normalization already applied.")

            if self._is_cancelled:
                return

            # Build output filename stem: {stem}[_l2][_abtt{m}]
            # Note: Embeddings.save() appends ".ssdembed" automatically
            stem = self.source_path.stem
            # Remove existing _l2/_abtt suffixes to avoid stacking
            import re
            stem = re.sub(r"(_l2)?(_abtt\d+)?$", "", stem)

            suffix_parts = []
            if emb.l2_normalized:
                suffix_parts.append("l2")
            abtt_val = emb.abtt_m
            if abtt_val > 0:
                suffix_parts.append(f"abtt{abtt_val}")
            suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
            out_stem = f"{stem}{suffix}"

            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_stem_path = str(self.output_dir / out_stem)

            self.progress.emit(70, "Saving prepared embedding (.ssdembed)...")
            # save() appends ".ssdembed" to the stem
            emb.save(out_stem_path)

            # The actual file is at out_stem_path + ".ssdembed"
            out_path = Path(out_stem_path + ".ssdembed")
            out_name = out_path.name

            if self._is_cancelled:
                return

            self.progress.emit(95, "Computing metadata...")
            # Collect metadata (include sidecar size)
            sidecar = Path(str(out_path) + ".vectors.npy")
            total_bytes = out_path.stat().st_size
            if sidecar.exists():
                total_bytes += sidecar.stat().st_size
            meta = {
                "filename": out_name,
                "vocab_size": len(emb),
                "embedding_dim": emb.vector_size,
                "l2_normalized": emb.l2_normalized,
                "abtt_m": emb.abtt_m,
                "file_size_mb": total_bytes / (1024 * 1024),
            }

            # Explicitly delete to free RAM
            del emb

            self.progress.emit(100, "Complete!")
            self.finished.emit(str(out_path), meta)

        except Exception as e:
            import traceback
            self.error.emit(f"Failed to prepare embeddings: {str(e)}\n{traceback.format_exc()}")


class EmbeddingLoadWorker(QThread):
    """Worker thread for loading a prepared .ssdembed file into RAM."""

    progress = Signal(int, str)
    finished = Signal(object, dict)  # Embeddings object, coverage_stats
    error = Signal(str)

    def __init__(
        self,
        ssdembed_path: Path,
        docs: Optional[List[List[str]]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.ssdembed_path = ssdembed_path
        self.docs = docs
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            from ssdiff import Embeddings

            self.progress.emit(20, "Loading prepared embedding...")
            emb = Embeddings.load(str(self.ssdembed_path))

            if self._is_cancelled:
                return

            self.progress.emit(70, "Computing coverage statistics...")
            stats = {
                "vocab_size": len(emb),
                "embedding_dim": emb.vector_size,
                "l2_normalized": emb.l2_normalized,
                "abtt_m": emb.abtt_m,
            }

            if self.docs:
                # Embeddings.key_to_index is a dict[str, int]
                vocab_set = set(emb.key_to_index.keys())
                all_tokens = set()
                is_grouped = self.docs and self.docs[0] and isinstance(self.docs[0][0], list)
                if is_grouped:
                    for profile in self.docs:
                        for post in profile:
                            all_tokens.update(post)
                else:
                    for doc in self.docs:
                        all_tokens.update(doc)

                in_vocab = all_tokens & vocab_set
                oov = all_tokens - vocab_set
                stats["coverage_pct"] = len(in_vocab) / len(all_tokens) * 100 if all_tokens else 0
                stats["n_oov"] = len(oov)

            self.progress.emit(100, "Complete!")
            self.finished.emit(emb, stats)

        except Exception as e:
            import traceback
            self.error.emit(f"Failed to load embedding: {str(e)}\n{traceback.format_exc()}")


class SpacyDownloadWorker(QThread):
    """Worker thread for downloading a spaCy model wheel from GitHub.

    Downloads the compatible .whl file, extracts it to the local
    AppData models directory, and emits the extracted model path
    on success.
    """

    progress = Signal(int, str)
    finished = Signal(str)  # model_path
    error = Signal(str)

    def __init__(self, model: str, parent=None):
        super().__init__(parent)
        self.model = model

    def run(self):
        import json
        import zipfile
        import tempfile
        import shutil
        from urllib.request import Request
        from urllib.error import URLError

        try:
            import spacy

            # 1. Resolve spaCy minor version (e.g. "3.7")
            spacy_version = spacy.__version__
            parts = spacy_version.split(".")
            minor_version = f"{parts[0]}.{parts[1]}"

            self.progress.emit(5, "Fetching model compatibility info...")

            compat_url = (
                "https://raw.githubusercontent.com/explosion/spacy-models"
                "/master/compatibility.json"
            )
            req = Request(compat_url, headers={"User-Agent": "SSD-App/1.0"})
            with _urlopen(req, timeout=30) as resp:
                compat = json.loads(resp.read().decode("utf-8"))

            # 2. Look up model version
            spacy_compat = compat.get("spacy", {})

            # Try exact minor, then try prefix match
            model_versions = None
            for key in spacy_compat:
                if key == minor_version or key.startswith(minor_version + "."):
                    if self.model in spacy_compat[key]:
                        model_versions = spacy_compat[key][self.model]
                        break

            if not model_versions:
                self.error.emit(
                    f"No compatible version of '{self.model}' found for "
                    f"spaCy {spacy_version}.\n\n"
                    f"Check https://spacy.io/models for available models."
                )
                return

            model_version = model_versions[0]  # latest compatible
            wheel_name = f"{self.model}-{model_version}-py3-none-any.whl"

            # 3. Download the wheel
            self.progress.emit(15, f"Downloading {wheel_name}...")

            download_url = (
                f"https://github.com/explosion/spacy-models/releases/download/"
                f"{self.model}-{model_version}/{wheel_name}"
            )

            req = Request(download_url, headers={"User-Agent": "SSD-App/1.0"})
            with _urlopen(req, timeout=60) as resp:
                total = resp.headers.get("Content-Length")
                total = int(total) if total else None

                tmp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".whl"
                )
                try:
                    downloaded = 0
                    chunk_size = 64 * 1024
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = 15 + int(60 * downloaded / total)
                            mb = downloaded / (1024 * 1024)
                            total_mb = total / (1024 * 1024)
                            self.progress.emit(
                                min(pct, 75),
                                f"Downloading... {mb:.1f}/{total_mb:.1f} MB",
                            )
                    tmp_file.close()

                    # 4. Extract the wheel
                    self.progress.emit(80, "Extracting model...")
                    models_dir = get_spacy_models_dir()
                    extract_dir = models_dir / f"{self.model}-{model_version}"

                    # Remove previous version if exists
                    if extract_dir.exists():
                        shutil.rmtree(extract_dir)

                    with zipfile.ZipFile(tmp_file.name, "r") as zf:
                        zf.extractall(extract_dir)

                finally:
                    # Clean up temp file
                    try:
                        os.unlink(tmp_file.name)
                    except OSError:
                        pass

            # 5. Verify the model data exists
            model_data_path = (
                extract_dir / self.model / f"{self.model}-{model_version}"
            )
            if not model_data_path.exists():
                self.error.emit(
                    f"Extraction succeeded but model data not found at "
                    f"expected path:\n{model_data_path}"
                )
                return

            self.progress.emit(100, "Download complete!")
            self.finished.emit(str(model_data_path))

        except URLError as e:
            self.error.emit(
                f"Network error downloading '{self.model}': {e}\n\n"
                f"Check your internet connection and try again."
            )
        except Exception as e:
            import traceback
            self.error.emit(
                f"Download failed: {str(e)}\n{traceback.format_exc()}"
            )


class UpdateCheckWorker(QThread):
    """Worker thread for checking GitHub for a newer app release.

    Emits ``update_available`` with (latest_version, release_html_url) when
    a newer version is found.  Errors and "up to date" results are both
    silent — this check must never disrupt the user's startup experience.
    """

    update_available = Signal(str, str)  # (latest_version, release_html_url)

    def __init__(self, current_version: str, parent=None):
        super().__init__(parent)
        self.current_version = current_version

    def run(self):
        from urllib.request import Request
        import json

        api_url = "https://api.github.com/repos/hplisiecki/SSD_APP/releases/latest"
        try:
            req = Request(api_url, headers={"User-Agent": "SSD-App/1.0"})
            with _urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            tag = data.get("tag_name", "").lstrip("v")
            url = data.get("html_url", "")

            def parse(v):
                try:
                    return tuple(int(x) for x in v.split("."))
                except ValueError:
                    return (0,)

            if parse(tag) > parse(self.current_version):
                self.update_available.emit(tag, url)
        except Exception:
            pass  # silent — no internet, rate-limited, etc.



class CoverageWorker(QThread):
    """Worker thread for computing lexicon coverage via Corpus methods."""

    progress = Signal(int, str)
    finished = Signal(dict, object)  # summary, per_token
    error = Signal(str)

    def __init__(
        self,
        corpus,
        y_full,
        lexicon: set,
        *,
        var_type: str = "continuous",
        parent=None,
    ):
        super().__init__(parent)
        self.corpus = corpus
        self.y_full = y_full
        self.lexicon = lexicon
        self.var_type = var_type

    def run(self):
        """Compute coverage statistics."""
        try:
            self.progress.emit(50, "Computing coverage...")

            summary = self.corpus.coverage_summary(
                self.y_full, self.lexicon, var_type=self.var_type,
            )
            per_token = self.corpus.token_stats(
                self.y_full, self.lexicon, var_type=self.var_type,
            )

            self.progress.emit(100, "Complete!")
            self.finished.emit(summary, per_token)

        except Exception as e:
            self.error.emit(f"Coverage computation failed: {str(e)}")
