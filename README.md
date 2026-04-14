<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/logo_light.svg">
    <img src="docs/logo_dark.svg" alt="SSD" width="256">
  </picture>
</p>

<h1 align="center">SSD</h1>

<p align="center">A desktop application for <b>Supervised Semantic Differential (SSD)</b> analysis.</p>

<p align="center">
  <a href="https://github.com/hplisiecki/SSD_APP/actions"><img src="https://github.com/hplisiecki/SSD_APP/workflows/Tests/badge.svg" alt="Tests"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3"></a>
  <a href="https://doi.org/10.31234/osf.io/gvrsb_v1"><img src="https://img.shields.io/badge/DOI-10.31234%2Fosf.io%2Fgvrsb__v1-blue" alt="DOI"></a>
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg" alt="Platform">
</p>

<p align="center">
  <a href="https://github.com/hplisiecki/SSD_APP/releases/latest">Download</a> &nbsp;|&nbsp;
  <a href="https://github.com/hplisiecki/Supervised-Semantic-Differential">ssdiff core library</a>
</p>

SSD finds interpretable semantic dimensions in text data that are associated with a continuous outcome variable or categorical group labels.

Given a corpus of texts with associated numeric scores or group memberships, SSD identifies the direction through word-embedding space that best explains variation in the outcome. The result is a semantic dimension with two interpretable poles — one associated with high outcomes, the other with low — complete with thematic clusters, example sentences, and statistical validation.

This application is the GUI frontend for the [ssdiff](https://github.com/hplisiecki/Supervised-Semantic-Differential) Python package.

---

## Download

Pre-built binaries for **Windows, Linux, and macOS** are available on the [Releases](https://github.com/hplisiecki/SSD_APP/releases/latest) page. No Python installation required — just download the binary for your platform and run it.

The first startup will be quite slow because the app has to set itself up. On Windows you will also have to click through a couple of security messages — the only way to avoid this from my side is to pay Microsoft for a license, which I am not ready to do at the moment.

spaCy language models are downloaded automatically on first use.

---

## Tutorial

The app includes a **built-in tutorial** accessible from the menu bar. It covers the full workflow with detailed explanations:

| Section | Contents |
|---------|----------|
| **What Is SSD?** | The problem SSD solves, how the method works, when to use it, and what you need |
| **Getting Started** | Launching the app, the three-stage workflow, and menu bar overview |
| **Stage 1 — Setup** | Loading datasets, choosing analysis type, preprocessing text, loading embeddings |
| **Stage 2 — Run** | Concept modes (lexicon vs. full-document), backend selection (PLS vs. PCA+OLS), advanced settings |
| **Stage 3 — Results** | Cluster overview, semantic poles, snippets, document scores, exporting |
| **Types of Analysis** | Four analysis combinations: continuous/group × full-document/lexicon |
| **Projects** | Saving, loading, and follow-up analyses |
| **Glossary** | Key terms and concepts |
| **Troubleshooting** | Common issues with datasets, embeddings, lexicons, and analysis |
| **Where to Get Embeddings** | Sources for pre-trained word embeddings by language |

---

## Supported Languages

SSD supports 23 languages via spaCy models (small, medium, and large variants available for each):

| Code | Language | Code | Language | Code | Language |
|-----|---------|------|----------|------|----------|
| `ca` | Catalan | `hr` | Croatian | `pl` | Polish |
| `da` | Danish  | `it` | Italian | `pt` | Portuguese |
| `de` | German  | `ja` | Japanese | `ro` | Romanian |
| `el` | Greek   | `ko` | Korean | `ru` | Russian |
| `en` | English | `lt` | Lithuanian | `sl` | Slovenian |
| `es` | Spanish | `mk` | Macedonian | `sv` | Swedish |
| `fr` | French  | `nb` | Norwegian | `uk` | Ukrainian |
| `nl` | Dutch | `zh` | Chinese |

---

## Word Embeddings

SSD requires pre-trained word embeddings, which are **not bundled** with the application due to their size. You need to download an embedding file separately before running an analysis.

### Recommended Sources

- **[GloVe](https://nlp.stanford.edu/projects/glove/)** (English) — Download GloVe 840B 300d (~2 GB) for the best coverage, or GloVe 6B for quick tests.
- **[fastText](https://fasttext.cc/docs/en/crawl-vectors.html)** (157 languages) — Pre-trained word vectors for most languages. Download the `.bin` format.
- **[Polish distributional models](https://dsmodels.nlp.ipipan.waw.pl)** — For Polish-language analyses.
- **Custom** — Train your own with gensim's Word2Vec or fastText and export as `.kv`.

### Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| SSD native | `.ssdembed` | Fastest to load; tracks L2/ABTT state to prevent double-normalization |
| gensim KeyedVectors | `.kv` | Fast to load |
| word2vec binary | `.bin` | Standard binary format |
| Text | `.txt`, `.vec` | One word per line + floats |
| Compressed | `.txt.gz`, `.vec.gz`, `.bin.gz` | Gzip-compressed versions of the above |

> **Tip:** On first load, the app normalizes and saves embeddings as `.ssdembed` for faster loading in subsequent sessions.

---

## Use Cases

- **Clinical psychology** — linking patient narratives to symptom severity or treatment outcomes
- **Computational social science** — analyzing survey responses across demographic groups
- **Political communication** — comparing rhetorical framing across party lines
- **Psycholinguistics** — discovering latent semantic dimensions in language production

---

## Features

- **Three-stage guided workflow** — Setup, Run, and Results, with validation at each step
- **Two analysis backends** — PLS (single-pass, no dimensionality choice) or PCA+OLS (automated PCA sweep with manual override)
- **Two analysis modes** — continuous outcome regression or categorical group comparison with permutation tests
- **Two concept modes** — full-document analysis or lexicon-focused with context windows
- **Interactive lexicon builder** — token suggestions ranked by correlation, per-token coverage statistics with quartile breakdowns
- **Cluster interpretation** — K-means clustering of pole neighbors with coherence scores and representative snippets
- **Snippet browser** — real sentences from the data, organized by cluster or beta alignment, with full document context
- **Per-result export** — human-readable report (`results.txt`), configuration snapshot (`config.json`), full result object (`results.pkl`), and a standalone replication script (`replication_script.py`)
- **Configurable report** — control which sections and how many items appear in the saved report via Report Settings
- **Project system** — save, reload, and delete analyses; run multiple analyses with different lexicons or settings
- **Automatic update notifications** — silent startup check against GitHub releases with a dismissible in-app banner
- **Customizable appearance** — eight color themes (six dark, two light) and four font size levels
- **In-app tutorial** — navigable guide with table of contents covering the full workflow
- **Single-instance guard** — prevents multiple windows from opening simultaneously

---

## Installation (from source)

### Prerequisites

- Python 3.10+
- Pre-trained word embeddings (see [Word Embeddings](#word-embeddings) above)

### Setup

```bash
# Clone the repository
git clone https://github.com/hplisiecki/SSD_APP.git
cd SSD_APP

# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py
```

### Building the Executable

```bash
pip install pyinstaller
pyinstaller SSD.spec --clean --noconfirm
```

---

## Workflow

### Stage 1: Setup

Configure the data, text processing, and embedding settings.

1. **Load dataset** — import a CSV, TSV, or Excel file and select the text, ID, and outcome/group columns
2. **Preprocess** — tokenize, lemmatize, and sentence-split texts using spaCy
3. **Load embeddings** — load a pre-trained word-embedding file; the app normalizes (L2 + ABTT) and saves as `.ssdembed` for faster reuse
4. **Choose analysis type** — continuous outcome (regression) or group comparison (permutation test)
5. **Set hyperparameters** — context window size, SIF weighting, random seed, and optional PCA preprocessing

A ready indicator shows which sections are complete before proceeding.

### Stage 2: Run

Define the concept and execute the analysis.

- **Lexicon mode** — build a keyword list using the interactive lexicon builder with automated suggestions, coverage statistics, and per-token diagnostics
- **Full-document mode** — analyze entire texts with an optional custom stoplist
- **Backend selection** — choose PLS (recommended, single-pass) or PCA+OLS (sweep across PCA dimensionalities)
- **Pre-flight review** — a read-only summary of the full configuration with sanity checks (outcome variance, sample size, OOV rate)
- **Run** — executes the SSD pipeline: document embedding, dimensionality reduction, beta estimation, pole extraction, clustering, and snippet collection

### Stage 3: Results

Explore and export the results across multiple tabs.

| Tab | Contents |
|-----|----------|
| **Cluster Overview** | Side-by-side positive/negative cluster tables with size, coherence, and top words |
| **Details** | R², p-value, effect sizes, sample counts, dataset info, and full configuration snapshot |
| **Semantic Poles** | Ranked word lists for each pole with cosine similarities |
| **Snippets** | Real sentences organized by beta alignment with document context |
| **Document Scores** | Per-document table with cosine scores, predicted values, and true outcomes |
| **Extreme Documents** | Top/bottom documents by predicted or observed outcome |
| **Misdiagnosed** | Documents where model predictions diverge most from actual outcomes |
| **PCA Sweep** | Plot of fit criterion across K values with selected elbow (PCA+OLS only) |

Multiple runs can be saved and compared using the run selector. Each saved result includes:

- **results.txt** — human-readable report (sections configurable via Report Settings)
- **config.json** — complete configuration snapshot for reproducibility
- **results.pkl** — full result object for further analysis in Python
- **replication_script.py** — standalone script to reproduce the analysis using the `ssdiff` library

---

## Project Structure

```
SSD_APP/
├── run.py                          # Application entry point
├── pyproject.toml                  # Project metadata and dependencies
├── requirements.txt                # pip dependencies
├── SSD.spec                        # PyInstaller build configuration
│
├── ssdiff_gui/                     # Main package
│   ├── main.py                     # App initialization and single-instance guard
│   ├── theme.py                    # Centralized theme system (8 themes)
│   ├── logo.py                     # Theme-aware icon generation
│   ├── models/
│   │   └── project.py              # Data models, config dataclasses, replication script generation
│   ├── controllers/
│   │   └── ssd_runner.py           # SSD analysis execution (background thread)
│   ├── views/
│   │   ├── main_window.py          # Main application window and project management
│   │   ├── stage1_setup.py         # Stage 1: Data, preprocessing, embeddings
│   │   ├── stage2_concept.py       # Stage 2: Concept definition, backend, run
│   │   ├── stage3_results.py       # Stage 3: Results viewer and export
│   │   ├── appearance_dialog.py    # Theme and font customization
│   │   ├── settings_dialog.py      # Application settings
│   │   ├── report_settings_dialog.py  # Report output configuration
│   │   ├── tutorial_dialog.py      # In-app tutorial
│   │   └── widgets/                # Reusable UI components
│   ├── utils/
│   │   ├── file_io.py              # Project save/load (JSON + pickle)
│   │   ├── validators.py           # Input validation
│   │   ├── worker_threads.py       # Background workers (preprocessing, embedding loading)
│   │   ├── settings.py             # QSettings wrapper
│   │   ├── report_settings.py      # Report configuration persistence
│   │   ├── paths.py                # Cross-platform path management
│   │   └── linux_install.py        # Desktop integration on Linux
│   └── resources/
│       ├── styles.qss              # Application stylesheet
│       └── quotes.json             # Loading screen quotes
│
├── tests/                          # Test suite
├── docs/                           # Logo assets (SVG)
└── .github/workflows/              # CI (tests) and CD (cross-platform builds)
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| [ssdiff](https://github.com/hplisiecki/Supervised-Semantic-Differential) | Core SSD analysis engine |
| [PySide6](https://doc.qt.io/qtforpython-6/) | GUI framework |
| [pandas](https://pandas.pydata.org/) | Data loading and manipulation |
| [numpy](https://numpy.org/) | Numerical computation |
| [openpyxl](https://openpyxl.readthedocs.io/) | Excel file support |

The `ssdiff` package brings in its own dependencies (spaCy, numpy, matplotlib).

---

## Citation

If you use SSD in your research, please cite:

> Plisiecki, H., Lenartowicz, P., Pokropek, A., Malyska, K., & Flakus, M. (2025). Measuring Individual Differences in Meaning: The Supervised Semantic Differential. *PsyArXiv*. https://doi.org/10.31234/osf.io/gvrsb_v1

```bibtex
@article{plisiecki2025ssd,
  title     = {Measuring Individual Differences in Meaning: The Supervised Semantic Differential},
  author    = {Plisiecki, Hubert and Lenartowicz, Pawe{\l} and Pokropek, Artur and Ma{\l}yska, Kinga and Flakus, Maria},
  year      = {2025},
  journal   = {PsyArXiv},
  doi       = {10.31234/osf.io/gvrsb_v1},
  url       = {https://doi.org/10.31234/osf.io/gvrsb_v1}
}
```

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

## Questions / Contributions
- File issues and feature requests on the repo's Issues page.
- Pull requests welcome — especially for documentation improvements.

Contact: hplisiecki@gmail.com

Project was funded by the National Science Centre, Poland (grant no. 2020/38/E/HS6/00302).
