# SSDiff Tutorial

## Table of Contents

- [What Is SSD?](#what-is-ssd)
  - [The Problem](#the-problem)
  - [What SSD Does](#what-ssd-does)
  - [When to Use It](#when-to-use-it)
  - [What You Need](#what-you-need)
- [Getting Started](#getting-started)
  - [Launching the App](#launching-the-app)
  - [The Three-Stage Workflow](#the-three-stage-workflow)
  - [The Menu Bar](#the-menu-bar)
- [Stage 1 — Setup](#stage-1--setup)
  - [Loading Your Dataset](#loading-your-dataset)
  - [Choosing an Analysis Type](#choosing-an-analysis-type)
  - [Preprocessing Text](#preprocessing-text)
  - [Loading Word Embeddings](#loading-word-embeddings)
  - [The "Project Ready" Indicator](#the-project-ready-indicator)
- [Stage 2 — Run](#stage-2--run)
  - [Choosing a Concept Mode](#choosing-a-concept-mode)
  - [Lexicon Mode](#lexicon-mode)
  - [Full-Document Mode](#full-document-mode)
  - [Choosing an Analysis Backend](#choosing-an-analysis-backend)
  - [PLS Settings (Advanced)](#pls-settings-advanced)
  - [Text Processing Settings (Advanced)](#text-processing-settings-advanced)
  - [PCA+OLS Settings (Advanced)](#pcaols-settings-advanced)
  - [Group Settings (Advanced)](#group-settings-advanced)
  - [Clustering Settings (Advanced)](#clustering-settings-advanced)
  - [Run Details](#run-details)
  - [Running the Analysis](#running-the-analysis)
- [Stage 3 — Results](#stage-3--results)
  - [Summary Cards](#summary-cards)
  - [Cluster Overview](#cluster-overview)
  - [Details](#details)
  - [PCA Sweep](#pca-sweep)
  - [Beta Snippets / Contrast Snippets](#beta-snippets--contrast-snippets)
  - [Semantic Poles](#semantic-poles)
  - [Document Scores / Contrast Scores](#document-scores--contrast-scores)
  - [Extreme Documents](#extreme-documents)
  - [Misdiagnosed Documents](#misdiagnosed-documents)
  - [Saving Results](#saving-results)
  - [Exporting Results](#exporting-results)
- [Types of Analysis](#types-of-analysis)
  - [Continuous Outcome + Full Document](#continuous-outcome--full-document)
  - [Continuous Outcome + Lexicon](#continuous-outcome--lexicon)
  - [Group Comparison + Full Document](#group-comparison--full-document)
  - [Group Comparison + Lexicon](#group-comparison--lexicon)
- [Saving & Managing Projects](#saving--managing-projects)
  - [What Is a Project?](#what-is-a-project)
  - [Saving](#saving)
  - [Follow-Up Analyses](#follow-up-analyses)
- [Glossary](#glossary)
- [Troubleshooting](#troubleshooting)
  - [Dataset Issues](#dataset-issues)
  - [Embedding Issues](#embedding-issues)
  - [Lexicon Issues](#lexicon-issues)
  - [Analysis Issues](#analysis-issues)
- [Where to Get Embeddings](#where-to-get-embeddings)

---

## What Is SSD?

### The Problem

You have a collection of open-ended texts — therapy transcripts, survey responses, social-media posts, interview excerpts — and alongside each text you have a **numeric variable** (a well-being score, a severity rating) or a **categorical group label** (a diagnosis, an experimental condition).

You want to know: *what is it about the language itself that is associated with that variable?*

Traditional approaches count specific words or run sentiment analysis, but these only scratch the surface. They cannot tell you which *meanings* — not individual words but coherent semantic themes — co-vary with your outcome.

### What SSD Does

Supervised Semantic Differential (SSD) finds the **single direction through meaning space (word embeddings vector space)** most strongly associated with your variable of interest.

Think of word embeddings as a vast map where every word has a location, and words with similar meanings are close together. SSD draws a line through that map — a **semantic dimension** — such that language at one end tends to co-occur with high values and language at the other end with low values.

The result is an **interpretable semantic axis** with two poles:

- **Positive pole** — words and themes associated with higher values of the outcome (or one side of the group comparison).
- **Negative pole** — words and themes associated with lower values (or the other side).

You can inspect the top words at each pole, see how they cluster into themes, and read real example sentences from your data. The primary goal of the analysis is **explanation**, secondary to prediction.

> For a detailed methodological description, see the preprint: <https://osf.io/preprints/psyarxiv/gvrsb_v1>

### When to Use It

SSD is designed for researchers who want to move beyond word-frequency approaches to understand the deeper semantic structure of language. Typical use cases:

- **Psycholinguistics** — What semantic content in patient narratives is associated with symptom severity?
- **Computational social science** — How does the language of survey responses differ across groups?
- **Political communication** — What dimensions of meaning distinguish party rhetoric?
- **Clinical psychology** — What themes in therapy transcripts are linked to treatment outcomes?
- **Media studies** — What semantic frames characterize coverage of different events?

### What You Need

| Requirement | Description |
|---|---|
| **A dataset** | A spreadsheet (CSV, TSV, or Excel) where each row is a document with a text column and either a numeric outcome or a categorical group column. |
| **Word embeddings** | A pre-trained embedding file (GloVe, fastText, gensim KeyedVectors, or SSDiff `.ssdembed`). See [Where to Get Embeddings](#where-to-get-embeddings). |
| **A language** | Select a language in the app and the required spaCy model is downloaded automatically. 19 languages supported. |

---

## Getting Started

### Launching the App

On first launch you see a **welcome screen** with options to create a new project or open an existing one. A project in SSDiff is a self-contained folder that stores your configuration, cached data, and all analysis runs.

### The Three-Stage Workflow

The application is organised around three stages, visible in the navigation bar at the top of the window:

> **Stage 1: Setup** → **Stage 2: Run** → **Stage 3: Results**

- **Stage 1 (Setup)** — Load data, preprocess text, load embeddings, choose analysis type.
- **Stage 2 (Run)** — Define your concept and launch the analysis.
- **Stage 3 (Results)** — Inspect, interpret, save, and export.

You progress in order but can always return to earlier stages to adjust settings.

### The Menu Bar

- **File** — Create/open projects, save, settings, appearance, exit.
- **View** — Jump directly to any unlocked stage.
- **Help** — Tutorial and application info.

---

## Stage 1 — Setup

### Loading Your Dataset

1. Click **Browse** next to the dataset field and select your CSV, TSV, or Excel file.
2. Select the **Text column** — the column containing raw text.
3. Optionally select an **ID column** for document identifiers. When multiple rows share the same ID, they will be grouped into a single profile (personal concept vector) during preprocessing.
4. Click **Validate Dataset**.

Validation checks for enough rows (minimum 30; warning below 100) and empty text cells. The status message also shows ID statistics when an ID column is selected — the number of unique IDs and the average texts per ID.

### Choosing an Analysis Type

| Mode | When to use it | Needs |
|---|---|---|
| **Continuous Outcome** | You have a numeric variable (e.g. a 1–7 well-being scale) and want to find the semantic dimension most associated with it. | An outcome column with numeric values. |
| **Group Comparison** | You have a categorical variable (e.g. diagnosis groups) and want to find the semantic dimensions that best separate the groups. | A group column with two or more category labels. |

### Preprocessing Text

1. Select a **language** from the 19 supported languages (including English, Polish, German, French, Spanish, and more). The required spaCy model is downloaded automatically on first use.
2. Leave **Lemmatize** and **Remove stopwords** enabled (recommended).
3. Click **Preprocess Texts**.

**Lemmatization** collapses inflected forms (runs, running, ran → run) so the analysis focuses on meaning rather than surface variation. **Stopword removal** filters out frequent function words (the, is, at) that carry little semantic content.

### Loading Word Embeddings

Word embeddings map every word to a point in high-dimensional space. SSD uses these to compute document-level semantic representations.

**Supported formats:** `.ssdembed` (SSDiff native), `.kv` (gensim KeyedVectors), `.bin` (word2vec / fastText binary), `.txt`, `.vec`, and `.gz` (text formats).

The `.ssdembed` format is SSDiff's native format. It loads as fast as `.kv` but also tracks whether L2 normalisation and ABTT have already been applied, preventing accidental double-normalisation. Use it when moving embeddings between projects. `.ssdembed` files are created within the app — they are not available for download online.

1. Click **Browse** and select your embedding file.
2. **L2 normalisation** (default on) — scales vectors to unit length.
3. **ABTT** (default on) — removes dominant principal components that reflect word frequency rather than meaning.
4. Click **Load Embeddings**.

After loading you will see vocabulary size, embedding dimension, coverage (percentage of your words found), and OOV count.

> **Tip:** If coverage is low, try a larger embedding file or one closer to your domain.

### The "Project Ready" Indicator

At the bottom of Stage 1, a status indicator shows whether all inputs are configured:

- **Green** — all sections complete. Click **Continue to Run**.
- **Red / grey** — something is missing. The checklist shows which sections still need attention.

---

## Stage 2 — Run

### Choosing a Concept Mode

| Mode | Description | Best for |
|---|---|---|
| **Lexicon** | You provide keywords. SSD analyses text *near* those keywords (within the context window). | A specific concept: self-reference (I, me, my), agency (choose, decide, control), domain terms. |
| **Full Document** | SSD uses the entire text. No keyword filtering. | Exploratory analysis, or short texts (e.g. survey responses). |

### Lexicon Mode

The lexicon builder lets you define your keyword set:

- Type a word and press Enter, or use **Paste Token List** for bulk entry.
- The app shows **coverage statistics** (see below).
- **Lexicon suggestions** — tokens from your data ranked by a composite score that balances coverage and correlation with the outcome. Double-click to add.

*Aim for:* coverage above 30% of documents, at least 3–5 tokens, and tokens present in the embedding vocabulary.

#### Coverage Statistics

After defining your lexicon, coverage statistics tell you how well your keywords connect to the data:

| Metric | Meaning |
|---|---|
| **Overall coverage** | Percentage of documents containing at least one lexicon word. |
| **Hits mean / median** | Average and median number of keyword occurrences per document. Higher means the concept is discussed more extensively. |
| **Types mean / median** | Average and median number of unique lexicon types per document. Higher means more variety in how the concept is expressed. |
| **Per-token details** | For each keyword: its individual coverage, and its correlation with the outcome (Pearson r for continuous, Cramer's V for groups). |

### Full-Document Mode

No keywords needed — SSD computes a semantic representation for each entire document, taking all of the words into account.

### Choosing an Analysis Backend

For continuous outcomes, the app offers two backends. For categorical groups, use the Groups backend.

| Backend | Description | Key difference |
|---|---|---|
| **PLS** | Added in v1.0.0. Fewer tuning parameters. Includes significance testing methods (permutation and novel split-half). | Finds the optimal direction directly via the NIPALS algorithm. No separate PCA step. |
| **PCA+OLS** | The original backend described in the paper. Offers the PCA sweep plot for exploring how the number of PCA components affects results. | Two-step: PCA dimensionality reduction first, then OLS regression. More hyperparameters to tune. |
| **Groups** | For categorical group comparison. | Permutation-based pairwise contrasts between group centroids. |

Select your backend in the toolbar at the top of Stage 2 before running the analysis.

### PLS Settings (Advanced)

These settings appear in the collapsible Advanced Settings panel when PLS is selected.

| Parameter | Default | What it controls |
|---|---|---|
| Components | 1 (0 = auto via CV) | Number of PLS components. Set to 0 to select automatically via cross-validation. |
| p-value method | auto | How significance is tested. See below. |
| Permutations | 1000 | Number of permutations for the permutation test. |
| Splits | 50 | Number of random splits for split-half tests. |
| Split ratio | 0.5 | Fraction of data in each split half. |
| Random state | default (2137) | Seed for reproducibility. |

**p-value methods:**

- **perm** (permutation) — standard, well-established method. Shuffles the outcome variable many times to build a null distribution of cross-validated R². Use this when you want a safe, conventional significance estimate.
- **split** (split-half with overlap correction) — novel method (Lenartowicz, 2026, in preparation). Repeatedly splits the data in half, fits on one half, tests on the other. Uses an overlap-corrected standard error to account for dependence between splits. Returns mean cross-half correlation as an effect size.
- **split_cal** (permutation-calibrated split-half) — novel method (Lenartowicz, 2026, in preparation). Runs the full split-half procedure on permuted y to build an exact null distribution. Guarantees correct false-positive rate control regardless of split-overlap structure. Computationally expensive.
- **auto** — selects `split` when `n_components = 1`, `perm` when `n_components > 1`.
- **none** — skip p-value computation entirely.

### Text Processing Settings (Advanced)

These settings apply to all backends and appear in the collapsible Advanced Settings panel.

| Parameter | Default | What it controls |
|---|---|---|
| Context window | ±3 tokens | Tokens on each side of a keyword in Lexicon mode. Only visible when Lexicon mode is selected. |
| SIF parameter (a) | 1e-3 | Smooth Inverse Frequency weighting. Smaller → rarer words matter more. |

### PCA+OLS Settings (Advanced)

These settings appear in the collapsible Advanced Settings panel when PCA+OLS is selected.

| Parameter | Default | What it controls |
|---|---|---|
| K sweep range | 20 to 120, step 2 | Range and step size of K values tested. The best K is selected by R² on the outcome. |

### Group Settings (Advanced)

These settings appear in the collapsible Advanced Settings panel when Groups is selected.

| Parameter | Default | What it controls |
|---|---|---|
| Median split | Off | When on, splits a numeric outcome at the median to create two groups. Useful when you have a continuous variable but want to compare "high" vs "low" groups. |
| Permutations | 5000 | Number of permutations for the null distribution. |
| Correction | Holm | Multiple-comparison correction method: Holm, Bonferroni, or FDR-BH. |

### Clustering Settings (Advanced)

These settings apply to all backends.

| Parameter | Default | What it controls |
|---|---|---|
| Top N neighbours | 100 | How many top pole words to cluster at each pole. |
| Auto-select K | On (silhouette) | Automatically pick the number of clusters using silhouette scores. |
| Clustering K range | 2–10 | Range of cluster counts to try when auto-selecting. |

### Run Details

The left panel displays a read-only summary of your entire configuration. Below the summary, **sanity checks** verify outcome variance, sample size, and OOV levels. Each check shows a green, yellow, or red indicator.

Review these before running to catch potential issues early.

### Running the Analysis

Click **Run SSD Analysis**. The analysis computes document embeddings, performs dimensionality reduction (PCA or PLS depending on the selected backend), finds the beta vector (or group contrasts), extracts pole words, clusters them into themes, and extracts illustrative snippets.

A progress dialog shows the current step. The process typically takes a few seconds to a minute depending on dataset size.

---

## Stage 3 — Results

### Summary Cards

A row of cards at the top of the results view presents the key statistics at a glance. The cards adapt to the analysis type.

#### PLS Analysis

| Card | What it means |
|---|---|
| **R²** | Proportion of outcome variance explained by the semantic dimension. Even modest values (0.05–0.20) can be meaningful in text-based research. |
| **p-value** | Significance of the association. The method used (permutation, split-half, etc.) is shown alongside the value. |
| **Components** | Number of PLS components used. |
| **Documents Used** | Number of documents (or profiles) kept after filtering. |

#### PCA+OLS Analysis

| Card | What it means |
|---|---|
| **R²** | Proportion of outcome variance explained by the semantic dimension. Even modest values (0.05–0.20) can be meaningful in text-based research. |
| **Adj. R²** | R² adjusted for the number of predictors (degrees of freedom). More conservative, penalises over-fitting. |
| **F-statistic** | Overall model significance test. Larger → stronger association. |
| **p-value** | Probability of observing this association by chance. |
| **Documents Used** | Number of documents (or profiles) kept after filtering. |
| **PCA K** | Number of principal components selected (auto or manual). |
| **PCA Variance Explained** | Percentage of total embedding variance captured by the K components. |

#### Group Comparison — Two Groups

| Card | What it means |
|---|---|
| **p-value** | Permutation-based p-value for the pairwise comparison. |
| **Cohen's d** | Standardised effect size of the semantic separation between groups. Values around 0.2, 0.5, and 0.8 are conventionally considered small, medium, and large effects. |
| **Cos Distance** | 1 minus the cosine similarity between group centroids. Higher → more semantic separation. |
| **‖Contrast‖** | Magnitude of the raw contrast vector (centroid difference) before normalisation. |
| **Documents Used** | Total documents kept after filtering. |
| **n (Group A) / n (Group B)** | Sample size for each group. |

#### Group Comparison — Three or More Groups

| Card | What it means |
|---|---|
| **Omnibus p** | Permutation p-value for the omnibus test (mean pairwise cosine distance across all group centroids). Tests whether *any* groups differ semantically. |
| **p (corrected)** | Corrected p-value for the currently viewed pairwise contrast (Holm by default; correction method is configurable). |
| **Cohen's d** | Standardised effect size for the viewed contrast. |
| **Cos Distance** | Cosine distance between the viewed pair of group centroids. |
| **Documents Used** | Total documents kept. |
| **n (Group A) / n (Group B)** | Sample sizes for the viewed pair. |

Use the contrast selector dropdown to switch between pairwise comparisons. The summary cards, clusters, snippets, and poles all update to reflect the selected contrast.

### Cluster Overview

This is often the most important tab. Two side-by-side tables — one for **positive clusters**, one for **negative clusters**.

Each row shows: cluster number, side, size, coherence, and top words.

**Click a cluster** to see the full word list and a snippet preview — real sentences from your data that contain words from that cluster.

Try to name each cluster by its top words to interpret the semantic dimension. For instance, a positive cluster with words like *happy, grateful, enjoy* and a negative cluster with *worried, anxious, tense* tells a clear story.

### Details

A read-only snapshot of every setting in effect for this particular run: dataset paths, column selections, preprocessing statistics, embeddings, hyperparameters, and concept configuration.

For **continuous** analyses, this tab also shows the full set of effect-size statistics:

| Statistic | Meaning |
|---|---|
| **Beta norm (std CN)** | Magnitude of the beta vector in standardised cosine-norm units. A one-unit increase in cosine similarity to the positive pole corresponds to this many standard deviations of the outcome. |
| **Delta per 0.10** | Expected change in the raw outcome for a 0.10 increase in cosine similarity to the positive pole. Provides a concrete, interpretable effect size in original units. |
| **IQR effect (raw)** | Outcome difference between the 75th and 25th percentile of semantic scores. Shows the practical range of the effect across your sample. |
| **Corr(y, pred)** | Correlation between observed and predicted outcome values. Another perspective on model fit. |

For **group comparison** analyses, the pairwise results table is shown with cosine distance, permutation p-values (raw and corrected), and Cohen's d for each pair.

Important for **reproducibility** — when you have several results you can see exactly what produced each one.

### PCA Sweep

*Only available for continuous analyses with PCA mode set to Auto.*

A plot of the PCA sweep results across different values of K (number of principal components). The algorithm evaluates each K on two criteria:

- **Interpretability** — cluster coherence and alignment with beta, combined into a joint score.
- **Stability** — how consistent the beta direction is compared to neighbouring K values.

The final selection uses the joint interpretability-stability score. The chosen K is marked on the plot. If unsatisfied, return to Stage 1 and set PCA mode to Manual.

### Beta Snippets / Contrast Snippets

Real sentences from your dataset that illustrate the semantic dimension.

*This tab is called **Beta Snippets** in continuous analyses and **Contrast Snippets** in group comparisons.*

**Controls:**

- **Mode toggle** — *Cluster centroids* (organised by cluster) or *Beta-aligned terms* (by proximity to the beta vector).
- **Cluster filter** — view only one cluster's snippets.
- **Navigation** — Previous / Next buttons to step through.

Each snippet shows the anchor term, cosine score, cluster assignment, and the full sentence with the term highlighted.

Snippets are essential for **qualitative validation** — they let you verify whether the clusters represent what you think they do.

### Semantic Poles

Two ranked word lists side by side:

- **Positive pole** — words aligned with higher outcome values (or one side of the group contrast).
- **Negative pole** — words aligned with lower values (or the other side).

Each word is shown with its **cosine similarity** to the beta vector. This is the raw, unstructured ranking — the Cluster Overview groups these same words into themes.

### Document Scores / Contrast Scores

*This tab is called **Document Scores** in continuous analyses and **Contrast Scores** in group comparisons.*

A table of per-document results with the document text visible on the right when a row is selected:

| Column | Meaning |
|---|---|
| **doc_index** | Row index from the original dataset. |
| **kept** | Whether the document was included in the analysis. |
| **cos** | Cosine similarity to the beta vector — the document's position on the semantic dimension (-1 to +1). |
| **yhat_std / yhat_raw** | Predicted outcome (standardised / original units). *Continuous only.* |
| **y_true_std / y_true_raw** | Actual outcome. *Continuous only.* |

Sort by any column to find highest/lowest scoring documents or outliers.

### Extreme Documents

*Available for continuous analyses only.*

A table of the most extreme documents — those with the highest and lowest predicted (or observed) outcome values. Use the **By** toggle to switch between ranking by predicted value or by observed value.

This helps identify which documents are most strongly associated with each pole of the semantic dimension. Reading extreme documents is a quick way to get an intuitive sense of what the dimension captures.

### Misdiagnosed Documents

*Available for continuous analyses only.*

Shows documents where the model's prediction diverges most from the actual observed value — the largest residuals. These are texts that "should" score high based on their language but actually have low observed values (over-predicted), or vice versa (under-predicted).

Use the **Side** filter to view over-predicted, under-predicted, or both. Misdiagnosed documents can reveal interesting edge cases, data quality issues, or texts that use language in unexpected ways relative to the semantic dimension.

### Saving Results

After an analysis completes, the result initially exists only in memory. To save it permanently:

1. Enter a descriptive name in the text field at the top of the results view (e.g. "Self-reference lexicon, GloVe 300d").
2. Click **Save Result**.

Saved results appear in the result selector dropdown and persist across sessions. You can switch between saved results to compare.

If you leave Stage 3 without saving, the result is lost.

### Exporting Results

When you save a result, the app writes the following files into the result folder:

| File | Format | Contents |
|---|---|---|
| `results.txt` | TXT | Human-readable report. Contents controlled by Report Settings. |
| `results.pkl` | PKL | Complete result object (scores, poles, clusters, snippets). Load in Python for further analysis. |
| `config.json` | JSON | Complete configuration snapshot for reproducibility. |
| `replication_script.py` | Python | Standalone script to reproduce this result from scratch using the `ssdiff` library. |

The replication script hardcodes all parameters and file paths, making it easy to share exact methodology with collaborators or re-run the analysis outside the GUI.

Use **Report Settings** (in the results toolbar) to control which sections appear in the `results.txt` file.

---

## Types of Analysis

### Continuous Outcome + Full Document

The most exploratory mode. Use this when you have a numeric variable and want to discover what the language *as a whole* says about it — without presupposing specific keywords.

**Typical scenario:** 500 survey responses, each with a well-being score (1–10). You want to know what themes in the response text co-vary with well-being.

**Steps:**

1. Stage 1 — Load the dataset, select Continuous Outcome, preprocess, load embeddings.
2. Stage 2 — Choose **PLS** or **PCA+OLS** as the backend. Select **Full Document** mode. Review the run details panel. Run.
3. Stage 3 — The positive pole will show themes associated with *higher* scores; the negative pole with *lower* scores.

**Interpreting:** Start with the Cluster Overview. Name each cluster by its top words, then confirm with Snippets. For example, you might find a *gratitude* cluster on the positive pole and an *anxiety* cluster on the negative pole.

### Continuous Outcome + Lexicon

Use this when you have a specific semantic concept in mind and want to know how language *around that concept* relates to the outcome.

**Typical scenario:** Therapy transcripts with symptom-severity ratings. You hypothesise that *self-referential* language matters, so your lexicon is: *I, me, my, myself, mine*. SSD analyses text within the context window around each pronoun.

**Steps:**

1. Stage 1 — Load data, select Continuous Outcome, preprocess, load embeddings.
2. Stage 2 — Choose **PLS** or **PCA+OLS** as the backend. Select **Lexicon** mode. Build your keyword set. Check coverage statistics. Use suggestions to discover additional relevant tokens.
3. Stage 3 — The dimension now reflects *how* the concept is talked about, not whether it appears. Positive-pole themes show the *style* of self-reference associated with lower severity; negative-pole themes show the style associated with higher severity.

**When to prefer this over Full Document:** When the texts are long and the concept of interest is localised (e.g. emotional language in otherwise factual reports).

### Group Comparison + Full Document

Use this when you have categorical groups and want to know what semantic content distinguishes them.

**Typical scenario:** Social-media posts labelled by political affiliation (Left / Centre / Right). You want to find the dimensions of meaning that separate the groups.

**Steps:**

1. Stage 1 — Load the dataset, select Group Comparison, choose the group column, preprocess, load embeddings.
2. Stage 2 — Select **Full Document**. Run.
3. Stage 3 — With two groups you get one dimension. With three or more you get pairwise comparisons, each with its own poles and clusters. Use the contrast selector to navigate between pairs.

**Interpreting:** The summary shows cosine distance between group centroids, permutation-based p-values, and Cohen's d. Cluster Overview shows what each group "sounds like."

### Group Comparison + Lexicon

Combines group comparison with keyword focusing. Use this when you want to compare how different groups talk *about a specific concept*.

**Typical scenario:** Interview transcripts from patients with different diagnoses. Your lexicon targets *treatment* language (therapy, medication, doctor, session). SSD finds how treatment-talk differs across diagnostic groups.

**Steps:**

1. Stage 1 — Load data, select Group Comparison, preprocess, load embeddings.
2. Stage 2 — Select **Lexicon** mode. Build the keyword set. Run.
3. Stage 3 — The pairwise comparisons now reflect differences in how each group uses language *around the concept*, not differences in overall text.

---

## Saving & Managing Projects

### What Is a Project?

A project is a folder that stores everything related to an analysis:

```
my_project/
  project.json              # Configuration and metadata
  data/
    corpus.pkl              # Cached preprocessed text
  results/
    20250601_143015/        # One folder per saved result
      config.json
      results.pkl
      results.txt
      replication_script.py
```

### Saving

Use **File > Save Project** (or the keyboard shortcut) at any time. The project file stores your current Stage 1 configuration, cached data references, and metadata for all saved results.

Analysis results must be saved separately. After a run completes in Stage 3, enter a name and click **Save Result** to persist it. Unsaved results are lost when you close the project or leave Stage 3.

### Follow-Up Analyses

Go back to Stage 2, change your lexicon or concept mode, and run again. Each result is stored independently. Compare results by switching results in the Stage 3 dropdown.

All results within a project share the same preprocessed data and embeddings, so you only pay the loading cost once.

---

## Glossary

| Term | Definition |
|---|---|
| **ABTT** | All-But-The-Top. Removes dominant principal components from embeddings to reduce frequency bias. |
| **Beta vector (β̂)** | The direction in embedding space most associated with the outcome. |
| **Cluster coherence** | How similar the words within a cluster are to each other (mean pairwise cosine similarity). |
| **Cohen's d** | Standardised effect size for group differences. Computed from projections onto the contrast vector, divided by pooled SD. |
| **Context window** | Tokens on each side of a keyword in Lexicon mode (±3 by default). |
| **Cosine distance** | 1 minus cosine similarity between two vectors. Used for measuring semantic separation between groups. |
| **Cosine similarity** | Angle between two vectors, -1 to +1. Measures alignment with the beta vector. |
| **Embedding** | A vector representation of a word in high-dimensional space. |
| **FDR** | False Discovery Rate — controls expected proportion of false positives among rejections. |
| **Holm correction** | Step-down multiple-comparison correction for group analyses. Less conservative than Bonferroni. |
| **IQR effect** | Outcome difference between the 75th and 25th percentile of semantic scores. |
| **Lemma** | Base form of a word (running → run). |
| **Median split** | Converts a continuous variable into two groups by splitting at the median value. |
| **NIPALS** | Non-linear Iterative Partial Least Squares — the algorithm used by PLS to extract components. |
| **OOV** | Out of vocabulary — word not in the embedding file. |
| **PCA** | Principal Component Analysis — reduces embedding dimensionality before regression. |
| **PCV** | Personal Concept Vector — a single semantic representation aggregated from all texts belonging to one ID. |
| **Permutation test** | Non-parametric significance test that shuffles group labels to build a null distribution. Used for group comparisons. |
| **PLS** | Partial Least Squares — a regression backend that finds the optimal direction via the NIPALS algorithm. Added in v1.0.0. |
| **Pole** | One end of the semantic dimension (positive = higher outcome, negative = lower). |
| **R²** | Proportion of outcome variance explained by the dimension (0 to 1). |
| **Replication script** | Auto-generated Python script that reproduces a saved result from scratch. |
| **SIF** | Smooth Inverse Frequency — weighting that down-weights common words when building document embeddings. |
| **Snippet** | A real sentence from the data illustrating a pole or cluster. |
| **Split-half test** | Novel significance test (Lenartowicz, 2026, in preparation) that repeatedly splits the data in half, fits on one half, and tests on the other. Reports mean cross-half correlation. Use permutation test for a standard, well-established alternative. |
| **Stopwords** | Frequent function words (the, is, and) removed during preprocessing. |
| **Theme / Cluster** | A group of semantically similar pole words identified by K-means clustering. |

---

## Troubleshooting

### Dataset Issues

| Problem | Cause | Solution |
|---|---|---|
| "Not enough rows" | Fewer than 30 documents. | Add more data or check the correct file was loaded. |
| "Near-zero variance" | Outcome column nearly constant. | Verify the correct column is selected. |
| "Too many empty texts" | >10% blank text cells. | Clean the dataset before loading. |

### Embedding Issues

| Problem | Cause | Solution |
|---|---|---|
| Low coverage (<60%) | Embedding vocabulary doesn't match data. | Try GloVe 840B or language-specific fastText. |
| High OOV count | Jargon, misspellings, unusual tokenisation. | Clean text before loading. |
| Loading takes very long | Large file (>2 GB). | Normal. Loaded once and cached. |

### Lexicon Issues

| Problem | Cause | Solution |
|---|---|---|
| Coverage below 10% | Tokens rarely appear. | Add broader terms or use suggestions. |
| OOV tokens flagged | Words missing from embeddings. | Replace with in-vocabulary synonyms. |

### Analysis Issues

| Problem | Cause | Solution |
|---|---|---|
| Very low R² | Weak language–outcome link. | Try a different concept, embeddings, or accept the finding. |
| Non-significant p-value | Small sample or weak effect. | Increase sample size. |
| Incoherent clusters | Poor K selection or noisy data. | Set clustering K manually, try different PCA K. |

---

## Where to Get Embeddings

### GloVe (Global Vectors) — English

Recommended for most English-language analyses. Download from: <https://nlp.stanford.edu/projects/glove/>

- **GloVe 840B 300d** (recommended) — 2.2M vocabulary, 300 dimensions, ~2 GB. Best coverage for general-purpose analyses.
- **GloVe 6B** — Smaller, faster, lower coverage. Good for quick tests.

### Polish Embeddings

For Polish-language analyses, download distributional semantic models from: <https://dsmodels.nlp.ipipan.waw.pl>

### fastText

Recommended for other non-English languages or when you need subword information. Pre-trained models for 157 languages are available from the fastText website. Download the `.bin` format.

### Custom Embeddings

Train your own with gensim's Word2Vec or fastText and export as `.kv`.

### Format Compatibility

| Format | Extension | Notes |
|---|---|---|
| SSDiff native | `.ssdembed` | As fast as `.kv`. Tracks L2/ABTT state, preventing double-normalisation. Created within the app. |
| gensim KeyedVectors | `.kv` | Fast to load. |
| word2vec binary | `.bin` | Standard binary format. |
| Text | `.txt`, `.vec` | One word per line + floats. |
| Compressed text | `.gz` | Gzip-compressed text. |

> **Tip:** After loading embeddings in the app, save them as `.ssdembed` for faster loading across projects and to preserve normalisation state.
