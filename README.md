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
  <a href="https://github.com/hplisiecki/SSD_APP/releases/latest">Download</a> &nbsp;|&nbsp;
  <a href="https://github.com/hplisiecki/Supervised-Semantic-Differential">ssdiff core library</a>
</p>

SSD finds interpretable semantic dimensions in text data that are associated with a continuous outcome variable or categorical group labels.

Given a corpus of texts with associated numeric scores or group memberships, SSD identifies the direction through word-embedding space that best explains variation in the outcome. The result is a semantic dimension with two interpretable poles — one associated with high outcomes, the other with low — complete with thematic clusters, example sentences, and statistical validation.

This application is the GUI frontend for the [ssdiff](https://github.com/hplisiecki/Supervised-Semantic-Differential) Python package.

---

## Download

A pre-built Windows executable is available on the [Releases](https://github.com/hplisiecki/SSD_APP/releases/latest) page. No Python installation required — just download **SSD.exe** and run it.

The first startup will be quite slow because the APP has to set itself up, and you will have to click through a couple Windows security messages - the only way to avoid this from my side is to pay Microsoft for a license, which I am not ready to do at the moment.

spaCy language models are downloaded automatically on first use.

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
| gensim KeyedVectors | `.kv` | Fastest to load |
| word2vec binary | `.bin` | Standard binary format |
| Text | `.txt`, `.vec` | One word per line + floats |
| Compressed text | `.txt.gz`, `.vec.gz` | Gzip-compressed text |

> **Tip:** The app offers to convert text-format embeddings to `.kv` on first load, which makes subsequent loads much faster.

---

## Use Cases

- **Clinical psychology** — linking patient narratives to symptom severity or treatment outcomes
- **Computational social science** — analyzing survey responses across demographic groups
- **Political communication** — comparing rhetorical framing across party lines
- **Psycholinguistics** — discovering latent semantic dimensions in language production

---

## Features

- **Three-stage guided workflow** — Setup, Run, and Results, with validation at each step
- **Two analysis modes** — continuous outcome regression or categorical group comparison with permutation tests
- **Two concept modes** — full-document analysis or lexicon-focused with context windows
- **Interactive lexicon builder** — token suggestions ranked by correlation, per-token coverage statistics with quartile breakdowns
- **Automated PCA sweep** — elbow detection for optimal dimensionality, with manual override
- **Cluster interpretation** — K-means clustering of pole neighbors with coherence scores and representative snippets
- **Snippet browser** — real sentences from the data, organized by cluster or beta alignment, with full document context
- **APA-formatted export** — regression tables, pairwise comparisons, and cluster summaries as Word documents
- **Comprehensive export** — CSV scores, pole neighbors, PCA plots, configuration JSON, and a human-readable hyperparameters file
- **Project system** — save and reload analyses; run multiple analyses with different lexicons or settings
- **Customizable appearance** — multiple color themes and font size scaling
- **In-app tutorial** — navigable guide with table of contents

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

# Asian language support (Chinese, Japanese, Korean)
pip install spacy-pkuseg sudachipy sudachidict-core

# Run the application
python run.py
```

### Building the Executable

```bash
pyinstaller SSD.spec --clean --noconfirm
```

---

## Workflow

### Stage 1: Setup

Configure the data, text processing, and embedding settings.

1. **Load dataset** — import a CSV, TSV, or Excel file and select the text, ID, and outcome/group columns
2. **Validate** — check for missing values and confirm the dataset is ready
3. **Preprocess** — tokenize, lemmatize, and sentence-split texts using spaCy
4. **Load embeddings** — load a pre-trained word-embedding file with optional L2 normalization and ABTT (All-But-The-Top) denoising; text-format files (`.txt`, `.vec`) can be auto-converted to `.kv` for faster future loading
5. **Choose analysis type** — continuous outcome (regression) or group comparison (permutation test)
6. **Set hyperparameters** — PCA sweep range, context window size, SIF weighting, clustering parameters, and more

A ready indicator shows which sections are complete before proceeding.

### Stage 2: Run

Define the concept and execute the analysis.

- **Lexicon mode** — build a keyword list using the interactive lexicon builder with automated suggestions, coverage statistics, and per-token diagnostics
- **Full-document mode** — analyze entire texts with an optional custom stoplist
- **Pre-flight review** — a read-only summary of the full configuration with sanity checks (outcome variance, sample size, OOV rate)
- **Run** — executes the SSD pipeline: document embedding, PCA, beta estimation, pole extraction, clustering, and snippet collection

### Stage 3: Results

Explore and export the results across multiple tabs.

| Tab | Contents |
|-----|----------|
| **Summary** | R², F-statistic, p-value, standardized beta, effect sizes, sample counts |
| **Clusters** | Side-by-side positive/negative cluster tables with size, coherence, and top words |
| **Poles** | Ranked word lists for each pole with cosine similarities |
| **Themes** | Detailed cluster view with full member lists |
| **Snippets** | Real sentences organized by cluster or beta alignment with document context |
| **Scores** | Per-document table with cosine scores, predicted values, and true outcomes |
| **PCA Sweep** | Plot of fit criterion across K values with selected elbow (auto mode) |
| **Config** | Read-only snapshot of all settings used for the run |

Multiple runs can be saved and compared using the run selector. Results can be exported as:

- **CSV** — per-document scores, pole neighbors
- **Word (.docx)** — APA-formatted regression/comparison tables, cluster summaries, snippet tables
- **PNG** — PCA sweep plot
- **JSON** — full configuration snapshot
- **TXT** — human-readable hyperparameters file

---

## Project Structure

```
SSD_APP/
├── run.py                          # Application entry point
├── requirements.txt                # Python dependencies
├── SSD.spec                        # PyInstaller build configuration
│
└── ssdiff_gui/                     # Main package
    ├── main.py                     # App initialization
    ├── models/
    │   └── project.py              # Data models and configuration dataclasses
    ├── controllers/
    │   ├── ssd_runner.py           # SSD analysis execution
    │   └── export_controller.py    # Result export (DOCX, CSV, PNG, JSON, TXT)
    ├── views/
    │   ├── main_window.py          # Main application window
    │   ├── stage1_setup.py         # Stage 1: Setup
    │   ├── stage2_concept.py       # Stage 2: Concept definition & run
    │   ├── stage3_results.py       # Stage 3: Results viewer
    │   ├── appearance_dialog.py    # Theme and font customization
    │   ├── settings_dialog.py      # Application settings
    │   ├── tutorial_dialog.py      # In-app tutorial
    │   └── widgets/                # Reusable UI components
    ├── utils/
    │   ├── file_io.py              # Project save/load
    │   ├── validators.py           # Input validation
    │   └── worker_threads.py       # Background workers
    └── resources/
        ├── styles.qss              # Application stylesheet
        └── quotes.json             # Loading screen quotes
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| [PySide6](https://doc.qt.io/qtforpython-6/) | GUI framework |
| [ssdiff](https://github.com/hplisiecki/Supervised-Semantic-Differential) | Core SSD analysis engine |
| [spaCy](https://spacy.io/) | Text preprocessing and lemmatization |
| [gensim](https://radimrehurek.com/gensim/) | Word embedding loading and management |
| [scikit-learn](https://scikit-learn.org/) | PCA, K-means clustering, metrics |
| [pandas](https://pandas.pydata.org/) | Data manipulation |
| [numpy](https://numpy.org/) / [scipy](https://scipy.org/) | Numerical computation |
| [python-docx](https://python-docx.readthedocs.io/) | Word document generation |
| [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) | Visualization |

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

This project is licensed under the [MIT License](LICENSE).

Some dependencies are distributed under the LGPL — see the [LICENSES/](LICENSES/) directory for details.

## Questions / Contributions
- File issues and feature requests on the repo’s Issues page.
- Pull requests welcome — especially for:
  - Documentation improvements

Contact: hplisiecki@gmail.com

Project was funded by the National Science Centre, Poland (grant no. 2020/38/E/HS6/00302).

