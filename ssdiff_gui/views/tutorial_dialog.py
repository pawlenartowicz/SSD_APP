"""Tutorial dialog for SSD.

Displays a navigable, styled tutorial inside a non-modal window with a
collapsible table-of-contents sidebar and HTML content area.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QTextBrowser,
    QPushButton,
    QWidget,
)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices


# ------------------------------------------------------------------
#  Tutorial content — structured as (anchor, title, html_body)
# ------------------------------------------------------------------

_SECTIONS: list[tuple[str, str, str | None, list[tuple[str, str, str]]]] = [
    # (anchor, title, intro_html | None, [(sub_anchor, sub_title, sub_html), ...])

    ("overview", "What Is SSD?", None, [
        ("overview-problem", "The Problem", """
<p>You have a collection of open-ended texts &mdash; therapy transcripts, survey responses,
social-media posts, interview excerpts &mdash; and alongside each text you have a
<b>numeric variable</b> (a well-being score, a severity rating) or a
<b>categorical group label</b> (a diagnosis, an experimental condition).</p>
<p>You want to know: <em>what is it about the language itself that is
associated with that variable?</em></p>
<p>Traditional approaches count specific words or run sentiment analysis, but these only
scratch the surface. They cannot tell you which <em>meanings</em> &mdash; not individual
words but coherent semantic themes &mdash; co-vary with your outcome.</p>"""),

        ("overview-what", "What SSD Does", """
<p>Supervised Semantic Differential (SSD) finds the <b>single direction through
meaning space (word embeddings vector space)</b> most strongly associated with your variable of interest.</p>
<p>Think of word embeddings as a vast map where every word has a location,
and words with similar meanings are close together. SSD draws a line through
that map &mdash; a <b>semantic dimension</b> &mdash; such that language at one
end tends to co-occur with high values and language at the other end with low
values.</p>
<p>The result is an <b>interpretable semantic axis</b> with two poles:</p>
<ul>
  <li><b>Positive pole</b> &mdash; words and themes associated with higher values
      of the outcome (or one side of the group comparison).</li>
  <li><b>Negative pole</b> &mdash; words and themes associated with lower values
      (or the other side).</li>
</ul>
<p>You can inspect the top words at each pole, see how they cluster into themes,
and read real example sentences from your data. The primary goal of the analysis is
<b>explanation</b>, secondary to prediction.</p>
<p class="muted">For a detailed methodological description, see the preprint:
<a href="https://osf.io/preprints/psyarxiv/gvrsb_v1">https://osf.io/preprints/psyarxiv/gvrsb_v1</a></p>"""),

        ("overview-when", "When to Use It", """
<p>SSD is designed for researchers who want to move beyond word-frequency
approaches to understand the deeper semantic structure of language. Typical
use cases:</p>
<ul>
  <li><b>Psycholinguistics</b> &mdash; What semantic content in patient
      narratives is associated with symptom severity?</li>
  <li><b>Computational social science</b> &mdash; How does the language of
      survey responses differ across groups?</li>
  <li><b>Political communication</b> &mdash; What dimensions of meaning
      distinguish party rhetoric?</li>
  <li><b>Clinical psychology</b> &mdash; What themes in therapy transcripts
      are linked to treatment outcomes?</li>
  <li><b>Media studies</b> &mdash; What semantic frames characterize coverage
      of different events?</li>
</ul>"""),

        ("overview-need", "What You Need", """
<table>
  <tr><td><b>A dataset</b></td>
      <td>A spreadsheet (CSV, TSV, or Excel) where each row is a document
          with a text column and either a numeric outcome or a categorical
          group column.</td></tr>
  <tr><td><b>Word embeddings</b></td>
      <td>A pre-trained embedding file (GloVe, fastText, gensim
          KeyedVectors, or SSDiff <code>.ssdembed</code>). See the <em>Where to Get Embeddings</em> section below.</td></tr>
  <tr><td><b>A language</b></td>
      <td>Select a language in the app and the required spaCy model is
          downloaded automatically. 19 languages supported.</td></tr>
</table>"""),
    ]),

    ("getting-started", "Getting Started", None, [
        ("gs-launch", "Launching the App", """
<p>On first launch you see a <b>welcome screen</b> with options to create a new
project or open an existing one. A project in SSD is a self-contained folder
that stores your configuration, cached data, and all analysis results.</p>"""),

        ("gs-workflow", "The Three-Stage Workflow", """
<p>The application is organised around three stages, visible in the navigation
bar at the top of the window:</p>
<p style="text-align:center; letter-spacing:2px;">
  <b>Stage 1: Setup</b> &rarr; <b>Stage 2: Run</b> &rarr; <b>Stage 3: Results</b>
</p>
<ul>
  <li><b>Stage 1 (Setup)</b> &mdash; Load data, preprocess text, load embeddings,
      choose analysis type.</li>
  <li><b>Stage 2 (Run)</b> &mdash; Define your concept and launch the analysis.</li>
  <li><b>Stage 3 (Results)</b> &mdash; Inspect, interpret, save, and export.</li>
</ul>
<p>You progress in order but can always return to earlier stages to adjust
settings.</p>"""),

        ("gs-menu", "The Menu Bar", """
<ul>
  <li><b>File</b> &mdash; Create/open projects, save, settings, appearance, exit.</li>
  <li><b>View</b> &mdash; Jump directly to any unlocked stage.</li>
  <li><b>Help</b> &mdash; This tutorial and application info.</li>
</ul>"""),
    ]),

    ("stage1", "Stage 1 &mdash; Setup", None, [
        ("s1-data", "Loading Your Dataset", """
<ol>
  <li>Click <b>Browse</b> next to the dataset field and select your CSV, TSV, or
      Excel file.</li>
  <li>Select the <b>Text column</b> &mdash; the column containing raw text.</li>
  <li>Optionally select an <b>ID column</b> for document identifiers. When multiple
      rows share the same ID, they will be grouped into a single profile
      (personal concept vector) during preprocessing.</li>
  <li>Click <b>Validate Dataset</b>.</li>
</ol>
<p>Validation checks for enough rows (minimum 30; warning below 100) and empty
text cells. The status message also shows ID statistics when an ID column is
selected &mdash; the number of unique IDs and the average texts per ID.</p>"""),

        ("s1-analysis-type", "Choosing an Analysis Type", """
<table>
  <tr><th>Mode</th><th>When to use it</th><th>Needs</th></tr>
  <tr><td><b>Continuous Outcome</b></td>
      <td>You have a numeric variable (e.g.&nbsp;a 1&ndash;7 well-being scale)
          and want to find the semantic dimension most associated with it.</td>
      <td>An outcome column with numeric values.</td></tr>
  <tr><td><b>Group Comparison</b></td>
      <td>You have a categorical variable (e.g.&nbsp;diagnosis groups) and want
          to find the semantic dimensions that best separate the groups.</td>
      <td>A group column with two or more category labels.</td></tr>
</table>"""),

        ("s1-preprocess", "Preprocessing Text", """
<ol>
  <li>Select a <b>language</b> from the 19 supported languages (including English,
      Polish, German, French, Spanish, and more). The required spaCy model is
      downloaded automatically on first use.</li>
  <li>Leave <b>Lemmatize</b> and <b>Remove stopwords</b> enabled (recommended).</li>
  <li>Click <b>Preprocess Texts</b>.</li>
</ol>
<p><b>Lemmatization</b> collapses inflected forms (runs, running, ran &rarr; run)
so the analysis focuses on meaning rather than surface variation.
<b>Stopword removal</b> filters out frequent function words
(the, is, at) that carry little semantic content.</p>"""),

        ("s1-embeddings", "Loading Word Embeddings", """
<p>Word embeddings map every word to a point in high-dimensional space.
SSD uses these to compute document-level semantic representations.</p>
<p><b>Supported formats:</b> <code>.ssdembed</code> (SSDiff native),
<code>.kv</code> (gensim KeyedVectors),
<code>.bin</code> (word2vec / fastText binary), <code>.txt</code>,
<code>.vec</code>, and <code>.gz</code> (text formats).</p>
<p>The <code>.ssdembed</code> format is SSDiff&rsquo;s native format. It loads as
fast as <code>.kv</code> but also tracks whether L2 normalisation and ABTT have
already been applied, preventing accidental double-normalisation. Use it when
moving embeddings between projects. <code>.ssdembed</code> files are created
within the app &mdash; they are not available for download online.</p>
<ol>
  <li>Click <b>Browse</b> and select your embedding file.</li>
  <li><b>L2 normalisation</b> (default on) &mdash; scales vectors to unit
      length.</li>
  <li><b>ABTT</b> (default on) &mdash; removes dominant principal
      components that reflect word frequency rather than meaning.</li>
  <li>Click <b>Load Embeddings</b>.</li>
</ol>
<p>After loading you will see vocabulary size, embedding dimension,
coverage (percentage of your words found), and OOV count.</p>
<p><em>Tip:</em> If coverage is low, try a larger embedding file or one
closer to your domain.</p>"""),

        ("s1-ready", "The &ldquo;Project Ready&rdquo; Indicator", """
<p>At the bottom of Stage 1, a status indicator shows whether all inputs are
configured:</p>
<ul>
  <li><span style="color:#4ec9a0;"><b>Green</b></span> &mdash; all sections
      complete. Click <b>Continue to Run</b>.</li>
  <li><span style="color:#e06c75;"><b>Red / grey</b></span> &mdash; something
      is missing. The checklist shows which sections still need attention.</li>
</ul>"""),
    ]),

    ("stage2", "Stage 2 &mdash; Run", None, [
        ("s2-concept", "Choosing a Concept Mode", """
<table>
  <tr><th>Mode</th><th>Description</th><th>Best for</th></tr>
  <tr><td><b>Lexicon</b></td>
      <td>You provide keywords. SSD analyses text <em>near</em> those
          keywords (within the context window).</td>
      <td>A specific concept: self-reference (I, me, my),
          agency (choose, decide, control), domain terms.</td></tr>
  <tr><td><b>Full Document</b></td>
      <td>SSD uses the entire text. No keyword filtering.</td>
      <td>Exploratory analysis, or short texts (e.g.&nbsp;survey
          responses).</td></tr>
</table>"""),

        ("s2-lexicon", "Lexicon Mode", """
<p>The lexicon builder lets you define your keyword set:</p>
<ul>
  <li>Type a word and press Enter, or use <b>Paste Token List</b> for
      bulk entry.</li>
  <li>The app shows <b>coverage statistics</b> (see below).</li>
  <li><b>Lexicon suggestions</b> &mdash; tokens from your data ranked by
      a composite score that balances coverage and correlation with the
      outcome. Double-click to add.</li>
</ul>
<p><em>Aim for:</em> coverage above 30&percnt; of documents, at least
3&ndash;5 tokens, and tokens present in the embedding vocabulary.</p>

<h3>Coverage Statistics</h3>
<p>After defining your lexicon, coverage statistics tell you how well your
keywords connect to the data:</p>
<table>
  <tr><th>Metric</th><th>Meaning</th></tr>
  <tr><td><b>Overall coverage</b></td>
      <td>Percentage of documents containing at least one lexicon word.</td></tr>
  <tr><td><b>Hits mean / median</b></td>
      <td>Average and median number of keyword occurrences per document.
          Higher means the concept is discussed more extensively.</td></tr>
  <tr><td><b>Types mean / median</b></td>
      <td>Average and median number of unique lexicon types per document.
          Higher means more variety in how the concept is expressed.</td></tr>
  <tr><td><b>Per-token details</b></td>
      <td>For each keyword: its individual coverage, and its correlation
          with the outcome (Pearson r for continuous, Cram&eacute;r&rsquo;s V
          for groups).</td></tr>
          
</table>"""),

        ("s2-fulldoc", "Full-Document Mode", """
<p>No keywords needed &mdash; SSD computes a semantic representation for
each entire document, taking all of the words into account.</p>"""),

        ("s2-backend", "Choosing an Analysis Backend", """
<p>For continuous outcomes, the app offers two backends. For categorical
groups, use the Groups backend.</p>
<table>
  <tr><th>Backend</th><th>Description</th><th>Key difference</th></tr>
  <tr><td><b>PLS</b></td>
      <td>Added in v1.0.0. Fewer tuning parameters. Includes significance
          testing methods (permutation and novel split-half).</td>
      <td>Finds the optimal direction directly via the NIPALS algorithm.
          No separate PCA step.</td></tr>
  <tr><td><b>PCA+OLS</b></td>
      <td>The original backend described in the paper. Offers the PCA sweep
          plot for exploring how the number of PCA components affects
          results.</td>
      <td>Two-step: PCA dimensionality reduction first, then OLS regression.
          More hyperparameters to tune.</td></tr>
  <tr><td><b>Groups</b></td>
      <td>For categorical group comparison.</td>
      <td>Permutation-based pairwise contrasts between group centroids.</td></tr>
</table>
<p>Select your backend in the toolbar at the top of Stage 2 before running
the analysis.</p>"""),

        ("s2-pls", "PLS Settings (Advanced)", """
<p>These settings appear in the collapsible Advanced Settings panel when PLS
is selected.</p>
<table>
  <tr><th>Parameter</th><th>Default</th><th>What it controls</th></tr>
  <tr><td>Components</td><td>1 (0 = auto via CV)</td>
      <td>Number of PLS components. Set to 0 to select automatically via
          cross-validation.</td></tr>
  <tr><td>p-value method</td><td>auto</td>
      <td>How significance is tested. See below.</td></tr>
  <tr><td>Permutations</td><td>1000</td>
      <td>Number of permutations for the permutation test.</td></tr>
  <tr><td>Splits</td><td>50</td>
      <td>Number of random splits for split-half tests.</td></tr>
  <tr><td>Split ratio</td><td>0.5</td>
      <td>Fraction of data in each split half.</td></tr>
  <tr><td>Random state</td><td>default (2137)</td>
      <td>Seed for reproducibility.</td></tr>
</table>
<h3>p-value methods</h3>
<ul>
  <li><b>perm</b> (permutation) &mdash; standard, well-established method.
      Shuffles the outcome variable many times to build a null distribution
      of cross-validated R&sup2;. Use this when you want a safe, conventional
      significance estimate.</li>
  <li><b>split</b> (split-half with overlap correction) &mdash; novel method
      (Lenartowicz, 2026, in preparation). Repeatedly splits data in half,
      fits on one half, tests on the other. Uses an overlap-corrected
      standard error. Returns mean cross-half correlation as effect size.</li>
  <li><b>split_cal</b> (permutation-calibrated split-half) &mdash; novel
      method (Lenartowicz, 2026, in preparation). Runs the full split-half
      procedure on permuted y to build an exact null distribution.
      Guarantees correct false-positive rate. Computationally expensive.</li>
  <li><b>auto</b> &mdash; selects <em>split</em> when components = 1,
      <em>perm</em> when components &gt; 1.</li>
  <li><b>none</b> &mdash; skip p-value computation entirely.</li>
</ul>"""),

        ("s2-text", "Text Processing Settings (Advanced)", """
<p>These settings apply to all backends and appear in the collapsible
Advanced Settings panel.</p>
<table>
  <tr><th>Parameter</th><th>Default</th><th>What it controls</th></tr>
  <tr><td>Context window</td><td>&plusmn;3 tokens</td>
      <td>Tokens on each side of a keyword in Lexicon mode. Only visible
          when Lexicon mode is selected.</td></tr>
  <tr><td>SIF parameter (a)</td><td>1e-3</td>
      <td>Smooth Inverse Frequency weighting. Smaller &rarr; rarer words
          matter more.</td></tr>
</table>"""),

        ("s2-pcaols", "PCA+OLS Settings (Advanced)", """
<p>These settings appear in the collapsible Advanced Settings panel when
PCA+OLS is selected.</p>
<table>
  <tr><th>Parameter</th><th>Default</th><th>What it controls</th></tr>
  <tr><td>K sweep range</td><td>20 to 120, step 2</td>
      <td>Range and step size of K values tested. The best K is selected
          by R&sup2; on the outcome.</td></tr>
</table>"""),

        ("s2-groups", "Group Settings (Advanced)", """
<p>These settings appear in the collapsible Advanced Settings panel when
Groups is selected.</p>
<table>
  <tr><th>Parameter</th><th>Default</th><th>What it controls</th></tr>
  <tr><td>Median split</td><td>Off</td>
      <td>When on, splits a numeric outcome at the median to create two
          groups. Useful when you have a continuous variable but want to
          compare &ldquo;high&rdquo; vs &ldquo;low&rdquo; groups.</td></tr>
  <tr><td>Permutations</td><td>5000</td>
      <td>Number of permutations for the null distribution.</td></tr>
  <tr><td>Correction</td><td>Holm</td>
      <td>Multiple-comparison correction method: Holm, Bonferroni,
          or FDR-BH.</td></tr>
</table>"""),

        ("s2-clustering", "Clustering Settings (Advanced)", """
<p>These settings apply to all backends.</p>
<table>
  <tr><th>Parameter</th><th>Default</th><th>What it controls</th></tr>
  <tr><td>Top N neighbours</td><td>100</td>
      <td>How many top pole words to cluster at each pole.</td></tr>
  <tr><td>Auto-select K</td><td>On (silhouette)</td>
      <td>Automatically pick the number of clusters using silhouette scores.</td></tr>
  <tr><td>Clustering K range</td><td>2&ndash;10</td>
      <td>Range of cluster counts to try when auto-selecting.</td></tr>
</table>"""),

        ("s2-preflight", "Run Details", """
<p>The left panel displays a read-only summary of your entire configuration.
Below the summary, <b>sanity checks</b> verify outcome variance, sample
size, and OOV levels. Each check shows a green, yellow, or red indicator.</p>
<p>Review these before running to catch potential issues early.</p>"""),

        ("s2-run", "Running the Analysis", """
<p>Click <b>Run SSD Analysis</b>. The analysis computes document embeddings,
performs dimensionality reduction (PCA or PLS depending on the selected backend), finds the beta vector (or group contrasts), extracts
pole words, clusters them into themes, and extracts illustrative snippets.</p>
<p>A progress dialog shows the current step. The process typically takes a
few seconds to a minute depending on dataset size.</p>"""),
    ]),

    ("stage3", "Stage 3 &mdash; Results", None, [
        ("s3-summary", "Summary Cards", """
<p>A row of cards at the top of the results view presents the key statistics
at a glance. The cards adapt to the analysis type.</p>

<h3>PLS Analysis</h3>
<table>
  <tr><th>Card</th><th>What it means</th></tr>
  <tr><td><b>R&sup2;</b></td>
      <td>Proportion of outcome variance explained by the semantic
          dimension. Even modest values (0.05&ndash;0.20) can be meaningful
          in text-based research.</td></tr>
  <tr><td><b>p-value</b></td>
      <td>Significance of the association. The method used (permutation,
          split-half, etc.) is shown alongside the value.</td></tr>
  <tr><td><b>Components</b></td>
      <td>Number of PLS components used.</td></tr>
  <tr><td><b>Documents Used</b></td>
      <td>Number of documents (or profiles) kept after filtering.</td></tr>
</table>

<h3>PCA+OLS Analysis</h3>
<table>
  <tr><th>Card</th><th>What it means</th></tr>
  <tr><td><b>R&sup2;</b></td>
      <td>Proportion of outcome variance explained by the semantic
          dimension. Even modest values (0.05&ndash;0.20) can be meaningful
          in text-based research.</td></tr>
  <tr><td><b>Adj. R&sup2;</b></td>
      <td>R&sup2; adjusted for the number of predictors (degrees of freedom).
          More conservative, penalises over-fitting.</td></tr>
  <tr><td><b>F-statistic</b></td>
      <td>Overall model significance test. Larger &rarr; stronger
          association.</td></tr>
  <tr><td><b>p-value</b></td>
      <td>Probability of observing this association by chance.</td></tr>
  <tr><td><b>Documents Used</b></td>
      <td>Number of documents (or profiles) kept after filtering.</td></tr>
  <tr><td><b>PCA K</b></td>
      <td>Number of principal components selected (auto or manual).</td></tr>
  <tr><td><b>PCA Variance Explained</b></td>
      <td>Percentage of total embedding variance captured by the K
          components.</td></tr>
</table>

<h3>Group Comparison &mdash; Two Groups</h3>
<table>
  <tr><th>Card</th><th>What it means</th></tr>
  <tr><td><b>p-value</b></td>
      <td>Permutation-based p-value for the pairwise comparison.</td></tr>
  <tr><td><b>Cohen&rsquo;s d</b></td>
      <td>Standardised effect size of the semantic separation between
          groups. Values around 0.2, 0.5, and 0.8 are conventionally
          considered small, medium, and large effects.</td></tr>
  <tr><td><b>Cos Distance</b></td>
      <td>1 minus the cosine similarity between group centroids.
          Higher &rarr; more semantic separation.</td></tr>
  <tr><td><b>||Contrast||</b></td>
      <td>Magnitude of the raw contrast vector (centroid difference)
          before normalisation.</td></tr>
  <tr><td><b>Documents Used</b></td>
      <td>Total documents kept after filtering.</td></tr>
  <tr><td><b>n (Group A) / n (Group B)</b></td>
      <td>Sample size for each group.</td></tr>
</table>

<h3>Group Comparison &mdash; Three or More Groups</h3>
<table>
  <tr><th>Card</th><th>What it means</th></tr>
  <tr><td><b>Omnibus p</b></td>
      <td>Permutation p-value for the omnibus test (mean pairwise
          cosine distance across all group centroids). Tests whether
          <em>any</em> groups differ semantically.</td></tr>
  <tr><td><b>p (corrected)</b></td>
      <td>Corrected p-value for the currently viewed pairwise contrast
          (Holm by default; correction method is configurable).</td></tr>
  <tr><td><b>Cohen&rsquo;s d</b></td>
      <td>Standardised effect size for the viewed contrast.</td></tr>
  <tr><td><b>Cos Distance</b></td>
      <td>Cosine distance between the viewed pair of group centroids.</td></tr>
  <tr><td><b>Documents Used</b></td>
      <td>Total documents kept.</td></tr>
  <tr><td><b>n (Group A) / n (Group B)</b></td>
      <td>Sample sizes for the viewed pair.</td></tr>
</table>
<p>Use the contrast selector dropdown to switch between pairwise comparisons.
The summary cards, clusters, snippets, and poles all update to reflect the
selected contrast.</p>"""),

        ("s3-clusters", "Cluster Overview", """
<p>This is often the most important tab. Two side-by-side tables &mdash; one for
<b>positive clusters</b>, one for <b>negative clusters</b>.</p>
<p>Each row shows: cluster number, side, size, coherence, and top words.</p>
<p><b>Click a cluster</b> to see the full word list and a snippet preview &mdash;
real sentences from your data that contain words from that cluster.</p>
<p>Try to name each cluster by its top words to interpret the semantic
dimension. For instance, a positive cluster with words like
<em>happy, grateful, enjoy</em> and a negative cluster with
<em>worried, anxious, tense</em> tells a clear story.</p>"""),

        ("s3-details", "Details", """
<p>A read-only snapshot of every setting in effect for this particular run:
dataset paths, column selections, preprocessing statistics, embeddings,
hyperparameters, and concept configuration.</p>
<p>For <b>continuous</b> analyses, this tab also shows the full set of effect-size
statistics:</p>
<table>
  <tr><th>Statistic</th><th>Meaning</th></tr>
  <tr><td><b>Beta norm (std CN)</b></td>
      <td>Magnitude of the beta vector in standardised cosine-norm units.
          A one-unit increase in cosine similarity to the positive pole
          corresponds to this many standard deviations of the outcome.</td></tr>
  <tr><td><b>Delta per 0.10</b></td>
      <td>Expected change in the raw outcome for a 0.10 increase in
          cosine similarity to the positive pole. Provides a concrete,
          interpretable effect size in original units.</td></tr>
  <tr><td><b>IQR effect (raw)</b></td>
      <td>Outcome difference between the 75th and 25th percentile of
          semantic scores. Shows the practical range of the effect
          across your sample.</td></tr>
  <tr><td><b>Corr(y, pred)</b></td>
      <td>Correlation between observed and predicted outcome values.
          Another perspective on model fit.</td></tr>
</table>
<p>For <b>group comparison</b> analyses, the pairwise results table is shown
with cosine distance, permutation p-values (raw and corrected),
and Cohen&rsquo;s d for each pair.</p>
<p>Important for <b>reproducibility</b> &mdash; when you have several results you
can see exactly what produced each one.</p>"""),

        ("s3-pca", "PCA Sweep", """
<p><em>Only available for continuous analyses with PCA mode set to Auto.</em></p>
<p>A plot of the PCA sweep results across different values of K (number of
principal components). The algorithm evaluates each K on two criteria:</p>
<ul>
  <li><b>Interpretability</b> &mdash; cluster coherence and alignment with
      beta, combined into a joint score.</li>
  <li><b>Stability</b> &mdash; how consistent the beta direction is compared
      to neighbouring K values.</li>
</ul>
<p>The final selection uses the joint interpretability-stability score. The
chosen K is marked on the plot. If unsatisfied, return to Stage 1 and set
PCA mode to Manual.</p>"""),

        ("s3-snippets", "Beta Snippets / Contrast Snippets", """
<p>Real sentences from your dataset that illustrate the semantic dimension.</p>
<p><em>This tab is called <b>Beta Snippets</b> in continuous analyses and
<b>Contrast Snippets</b> in group comparisons.</em></p>
<p><b>Controls:</b></p>
<ul>
  <li><b>Mode toggle</b> &mdash; <em>Cluster centroids</em> (organised by
      cluster) or <em>Beta-aligned terms</em> (by proximity to the beta
      vector).</li>
  <li><b>Cluster filter</b> &mdash; view only one cluster&rsquo;s snippets.</li>
  <li><b>Navigation</b> &mdash; Previous / Next buttons to step through.</li>
</ul>
<p>Each snippet shows the anchor term, cosine score, cluster assignment,
and the full sentence with the term highlighted.</p>
<p>Snippets are essential for <b>qualitative validation</b> &mdash; they let you
verify whether the clusters represent what you think they do.</p>"""),

        ("s3-poles", "Semantic Poles", """
<p>Two ranked word lists side by side:</p>
<ul>
  <li><b>Positive pole</b> &mdash; words aligned with higher outcome values
      (or one side of the group contrast).</li>
  <li><b>Negative pole</b> &mdash; words aligned with lower values
      (or the other side).</li>
</ul>
<p>Each word is shown with its <b>cosine similarity</b> to the beta vector.
This is the raw, unstructured ranking &mdash; the Cluster Overview groups
these same words into themes.</p>"""),

        ("s3-scores", "Document Scores / Contrast Scores", """
<p><em>This tab is called <b>Document Scores</b> in continuous analyses and
<b>Contrast Scores</b> in group comparisons.</em></p>
<p>A table of per-document results with the document text visible on the right
when a row is selected:</p>
<table>
  <tr><th>Column</th><th>Meaning</th></tr>
  <tr><td><b>doc_index</b></td><td>Row index from the original dataset.</td></tr>
  <tr><td><b>kept</b></td><td>Whether the document was included in the analysis.</td></tr>
  <tr><td><b>cos</b></td><td>Cosine similarity to the beta vector &mdash; the
      document&rsquo;s position on the semantic dimension (&minus;1 to +1).</td></tr>
  <tr><td><b>yhat_std / yhat_raw</b></td><td>Predicted outcome
      (standardised / original units). <em>Continuous only.</em></td></tr>
  <tr><td><b>y_true_std / y_true_raw</b></td><td>Actual outcome.
      <em>Continuous only.</em></td></tr>
</table>
<p>Sort by any column to find highest/lowest scoring documents or
outliers.</p>"""),

        ("s3-extreme", "Extreme Documents", """
<p><em>Available for continuous analyses only.</em></p>
<p>A table of the most extreme documents &mdash; those with the highest and
lowest predicted (or observed) outcome values. Use the <b>By</b> toggle to
switch between ranking by predicted value or by observed value.</p>
<p>This helps identify which documents are most strongly associated with
each pole of the semantic dimension. Reading extreme documents is a quick
way to get an intuitive sense of what the dimension captures.</p>"""),

        ("s3-misdiagnosed", "Misdiagnosed Documents", """
<p><em>Available for continuous analyses only.</em></p>
<p>Shows documents where the model&rsquo;s prediction diverges most from the
actual observed value &mdash; the largest residuals. These are texts that
&ldquo;should&rdquo; score high based on their language but actually have low
observed values (over-predicted), or vice versa (under-predicted).</p>
<p>Use the <b>Side</b> filter to view over-predicted, under-predicted, or
both. Misdiagnosed documents can reveal interesting edge cases, data quality
issues, or texts that use language in unexpected ways relative to the
semantic dimension.</p>"""),

        ("s3-saving", "Saving Results", """
<p>After an analysis completes, the result initially exists only in memory.
To save it permanently:</p>
<ol>
  <li>Enter a descriptive name in the text field at the top of the results
      view (e.g.&nbsp;&ldquo;Self-reference lexicon, GloVe 300d&rdquo;).</li>
  <li>Click <b>Save Result</b>.</li>
</ol>
<p>Saved results appear in the result selector dropdown and persist across sessions.
You can switch between saved results to compare.</p>
<p>If you leave Stage 3 without saving, the result is lost.</p>"""),

        ("s3-export", "Exporting Results", """
<p>When you save a result, the app writes the following files into the result
folder:</p>
<table>
  <tr><th>File</th><th>Format</th><th>Contents</th></tr>
  <tr><td>report.md</td><td>MD</td>
      <td>Human-readable report. Format and contents controlled by Save Settings.</td></tr>
  <tr><td>results.pkl</td><td>PKL</td>
      <td>Complete result object (scores, poles, clusters, snippets).
          Load in Python for further analysis.</td></tr>
  <tr><td>config.json</td><td>JSON</td>
      <td>Complete configuration snapshot for reproducibility.</td></tr>
  <tr><td>replication_script.py</td><td>Python</td>
      <td>Standalone script to reproduce this result from scratch using
          the <code>ssdiff</code> library.</td></tr>
</table>
<p>The replication script hardcodes all parameters and file paths, making it
easy to share exact methodology with collaborators or re-run the analysis
outside the GUI.</p>
<p>Use <b>Save Settings</b> (in the results toolbar) to choose which artifacts
and formats are written. Tabular outputs (words, clusters, snippets, etc.) go
under <code>tables/</code>.</p>"""),
    ]),

    ("analysis-types", "Types of Analysis", None, [
        ("at-cont-full", "Continuous Outcome + Full Document", """
<p>The most exploratory mode. Use this when you have a numeric variable and
want to discover what the language <em>as a whole</em> says about it &mdash;
without presupposing specific keywords.</p>
<p><b>Typical scenario:</b> 500 survey responses, each with a well-being
score (1&ndash;10). You want to know what themes in the response text
co-vary with well-being.</p>
<p><b>Steps:</b></p>
<ol>
  <li>Stage 1 &mdash; Load the dataset, select Continuous Outcome, preprocess,
      load embeddings.</li>
  <li>Stage 2 &mdash; Choose <b>PLS</b> or <b>PCA+OLS</b> as the backend.
      Select <b>Full Document</b> mode. Review the run details panel. Run.</li>
  <li>Stage 3 &mdash; The positive pole will show themes associated with
      <em>higher</em> scores; the negative pole with <em>lower</em>
      scores.</li>
</ol>
<p><b>Interpreting:</b> Start with the Cluster Overview. Name each cluster by
its top words, then confirm with Snippets. For example, you might find a
<em>gratitude</em> cluster on the positive pole and an <em>anxiety</em>
cluster on the negative pole.</p>"""),

        ("at-cont-lex", "Continuous Outcome + Lexicon", """
<p>Use this when you have a specific semantic concept in mind and want to know
how language <em>around that concept</em> relates to the outcome.</p>
<p><b>Typical scenario:</b> Therapy transcripts with symptom-severity ratings.
You hypothesise that <em>self-referential</em> language matters, so your
lexicon is: <em>I, me, my, myself, mine</em>. SSD analyses text within the
context window around each pronoun.</p>
<p><b>Steps:</b></p>
<ol>
  <li>Stage 1 &mdash; Load data, select Continuous Outcome, preprocess, load
      embeddings.</li>
  <li>Stage 2 &mdash; Choose <b>PLS</b> or <b>PCA+OLS</b> as the backend.
      Select <b>Lexicon</b> mode. Build your keyword set. Check coverage
      statistics. Use suggestions to discover additional relevant tokens.</li>
  <li>Stage 3 &mdash; The dimension now reflects <em>how</em> the concept is
      talked about, not whether it appears. Positive-pole themes show the
      <em>style</em> of self-reference associated with lower severity;
      negative-pole themes show the style associated with higher
      severity.</li>
</ol>
<p><b>When to prefer this over Full Document:</b> When the texts are long
and the concept of interest is localised (e.g.&nbsp;emotional language in
otherwise factual reports).</p>"""),

        ("at-group-full", "Group Comparison + Full Document", """
<p>Use this when you have categorical groups and want to know what semantic
content distinguishes them.</p>
<p><b>Typical scenario:</b> Social-media posts labelled by political
affiliation (Left / Centre / Right). You want to find the dimensions of
meaning that separate the groups.</p>
<p><b>Steps:</b></p>
<ol>
  <li>Stage 1 &mdash; Load the dataset, select Group Comparison, choose
      the group column, preprocess, load embeddings.</li>
  <li>Stage 2 &mdash; Select <b>Full Document</b>. Run.</li>
  <li>Stage 3 &mdash; With two groups you get one dimension. With three
      or more you get pairwise comparisons, each with its own poles and
      clusters. Use the contrast selector to navigate between pairs.</li>
</ol>
<p><b>Interpreting:</b> The summary shows cosine distance between group
centroids, permutation-based p-values, and Cohen&rsquo;s d. Cluster
Overview shows what each group &ldquo;sounds like.&rdquo;</p>"""),

        ("at-group-lex", "Group Comparison + Lexicon", """
<p>Combines group comparison with keyword focusing. Use this when you want to
compare how different groups talk <em>about a specific concept</em>.</p>
<p><b>Typical scenario:</b> Interview transcripts from patients with different
diagnoses. Your lexicon targets <em>treatment</em> language
(therapy, medication, doctor, session). SSD finds how
treatment-talk differs across diagnostic groups.</p>
<p><b>Steps:</b></p>
<ol>
  <li>Stage 1 &mdash; Load data, select Group Comparison, preprocess,
      load embeddings.</li>
  <li>Stage 2 &mdash; Select <b>Lexicon</b> mode. Build the keyword set.
      Run.</li>
  <li>Stage 3 &mdash; The pairwise comparisons now reflect differences
      in how each group uses language <em>around the concept</em>, not
      differences in overall text.</li>
</ol>"""),
    ]),

    ("projects", "Saving &amp; Managing Projects", None, [
        ("proj-what", "What Is a Project?", """
<p>A project is a folder that stores everything related to an analysis:</p>
<pre>my_project/
  project.json              # Configuration and metadata
  data/
    corpus.pkl              # Cached preprocessed text
  results/
    20250601_143015/        # One folder per saved result
      config.json
      results.pkl
      report.md
      replication_script.py</pre>"""),

        ("proj-save", "Saving", """
<p>Use <b>File &rarr; Save Project</b> (or the keyboard shortcut) at any time.
The project file stores your current Stage 1 configuration, cached data
references, and metadata for all saved results.</p>
<p>Analysis results must be saved separately. After a run completes in Stage 3,
enter a name and click <b>Save Result</b> to persist it. Unsaved results
are lost when you close the project or leave Stage 3.</p>"""),

        ("proj-followup", "Follow-Up Analyses", """
<p>Go back to Stage 2, change your lexicon or concept mode, and run again.
Each result is stored independently. Compare results by switching results in the
Stage 3 dropdown.</p>
<p>All results within a project share the same preprocessed data and embeddings,
so you only pay the loading cost once.</p>"""),
    ]),

    ("glossary", "Glossary", """
<table>
<tr><td><b>ABTT</b></td>
    <td>All-But-The-Top. Removes dominant principal components from
        embeddings to reduce frequency bias.</td></tr>
<tr><td><b>Beta vector (&beta;&#770;)</b></td>
    <td>The direction in embedding space most associated with the outcome.</td></tr>
<tr><td><b>Cluster coherence</b></td>
    <td>How similar the words within a cluster are to each other
        (mean pairwise cosine similarity).</td></tr>
<tr><td><b>Cohen&rsquo;s d</b></td>
    <td>Standardised effect size for group differences. Computed from
        projections onto the contrast vector, divided by pooled SD.</td></tr>
<tr><td><b>Context window</b></td>
    <td>Tokens on each side of a keyword in Lexicon mode (&plusmn;3 by
        default).</td></tr>
<tr><td><b>Cosine distance</b></td>
    <td>1 minus cosine similarity between two vectors. Used for
        measuring semantic separation between groups.</td></tr>
<tr><td><b>Cosine similarity</b></td>
    <td>Angle between two vectors, &minus;1 to +1. Measures alignment with
        the beta vector.</td></tr>
<tr><td><b>Embedding</b></td>
    <td>A vector representation of a word in high-dimensional space.</td></tr>
<tr><td><b>FDR</b></td>
    <td>False Discovery Rate &mdash; controls expected proportion of false
        positives among rejections.</td></tr>
<tr><td><b>Holm correction</b></td>
    <td>Step-down multiple-comparison correction for group analyses. Less
        conservative than Bonferroni.</td></tr>
<tr><td><b>IQR effect</b></td>
    <td>Outcome difference between the 75th and 25th percentile of
        semantic scores.</td></tr>
<tr><td><b>Lemma</b></td>
    <td>Base form of a word (running &rarr; run).</td></tr>
<tr><td><b>Median split</b></td>
    <td>Converts a continuous variable into two groups by splitting at the
        median value.</td></tr>
<tr><td><b>NIPALS</b></td>
    <td>Non-linear Iterative Partial Least Squares &mdash; the algorithm
        used by PLS to extract components.</td></tr>
<tr><td><b>OOV</b></td>
    <td>Out of vocabulary &mdash; word not in the embedding file.</td></tr>
<tr><td><b>PCA</b></td>
    <td>Principal Component Analysis &mdash; reduces embedding
        dimensionality before regression.</td></tr>
<tr><td><b>PCV</b></td>
    <td>Personal Concept Vector &mdash; a single semantic representation
        aggregated from all texts belonging to one ID.</td></tr>
<tr><td><b>Permutation test</b></td>
    <td>Non-parametric significance test that shuffles group labels to
        build a null distribution. Used for group comparisons.</td></tr>
<tr><td><b>PLS</b></td>
    <td>Partial Least Squares &mdash; a regression backend that finds the
        optimal direction via the NIPALS algorithm. Added in v1.0.0.</td></tr>
<tr><td><b>Pole</b></td>
    <td>One end of the semantic dimension (positive = higher outcome,
        negative = lower).</td></tr>
<tr><td><b>R&sup2;</b></td>
    <td>Proportion of outcome variance explained by the dimension
        (0 to 1).</td></tr>
<tr><td><b>Replication script</b></td>
    <td>Auto-generated Python script that reproduces a saved result from
        scratch.</td></tr>
<tr><td><b>SIF</b></td>
    <td>Smooth Inverse Frequency &mdash; weighting that down-weights common
        words when building document embeddings.</td></tr>
<tr><td><b>Snippet</b></td>
    <td>A real sentence from the data illustrating a pole or cluster.</td></tr>
<tr><td><b>Split-half test</b></td>
    <td>Novel significance test (Lenartowicz, 2026, in preparation) that
        repeatedly splits the data in half, fits on one half, and tests on
        the other. Reports mean cross-half correlation. Use permutation test
        for a standard, well-established alternative.</td></tr>
<tr><td><b>Stopwords</b></td>
    <td>Frequent function words (the, is, and) removed during
        preprocessing.</td></tr>
<tr><td><b>Theme / Cluster</b></td>
    <td>A group of semantically similar pole words identified by
        K-means clustering.</td></tr>
</table>""", []),

    ("troubleshooting", "Troubleshooting", None, [
        ("ts-data", "Dataset Issues", """
<table>
  <tr><th>Problem</th><th>Cause</th><th>Solution</th></tr>
  <tr><td>&ldquo;Not enough rows&rdquo;</td>
      <td>Fewer than 30 documents.</td>
      <td>Add more data or check the correct file was loaded.</td></tr>
  <tr><td>&ldquo;Near-zero variance&rdquo;</td>
      <td>Outcome column nearly constant.</td>
      <td>Verify the correct column is selected.</td></tr>
  <tr><td>&ldquo;Too many empty texts&rdquo;</td>
      <td>&gt;10&percnt; blank text cells.</td>
      <td>Clean the dataset before loading.</td></tr>
</table>"""),

        ("ts-emb", "Embedding Issues", """
<table>
  <tr><th>Problem</th><th>Cause</th><th>Solution</th></tr>
  <tr><td>Low coverage (&lt;60&percnt;)</td>
      <td>Embedding vocabulary doesn&rsquo;t match data.</td>
      <td>Try GloVe 840B or language-specific fastText.</td></tr>
  <tr><td>High OOV count</td>
      <td>Jargon, misspellings, unusual tokenisation.</td>
      <td>Clean text before loading.</td></tr>
  <tr><td>Loading takes very long</td>
      <td>Large file (&gt;2 GB).</td>
      <td>Normal. Loaded once and cached.</td></tr>
</table>"""),

        ("ts-lex", "Lexicon Issues", """
<table>
  <tr><th>Problem</th><th>Cause</th><th>Solution</th></tr>
  <tr><td>Coverage below 10&percnt;</td>
      <td>Tokens rarely appear.</td>
      <td>Add broader terms or use suggestions.</td></tr>
  <tr><td>OOV tokens flagged</td>
      <td>Words missing from embeddings.</td>
      <td>Replace with in-vocabulary synonyms.</td></tr>
</table>"""),

        ("ts-analysis", "Analysis Issues", """
<table>
  <tr><th>Problem</th><th>Cause</th><th>Solution</th></tr>
  <tr><td>Very low R&sup2;</td>
      <td>Weak language&ndash;outcome link.</td>
      <td>Try a different concept, embeddings, or accept the finding.</td></tr>
  <tr><td>Non-significant p-value</td>
      <td>Small sample or weak effect.</td>
      <td>Increase sample size.</td></tr>
  <tr><td>Incoherent clusters</td>
      <td>Poor K selection or noisy data.</td>
      <td>Set clustering K manually, try different PCA K.</td></tr>
</table>"""),
    ]),

    ("embeddings-guide", "Where to Get Embeddings", """
<h3>GloVe (Global Vectors) &mdash; English</h3>
<p>Recommended for most English-language analyses. Download from:<br>
<a href="https://nlp.stanford.edu/projects/glove/">https://nlp.stanford.edu/projects/glove/</a></p>
<ul>
  <li><b>GloVe 840B 300d</b> (recommended) &mdash; 2.2M vocabulary, 300 dimensions,
      ~2 GB. Best coverage for general-purpose analyses.</li>
  <li><b>GloVe 6B</b> &mdash; Smaller, faster, lower coverage. Good for
      quick tests.</li>
</ul>

<h3>Polish Embeddings</h3>
<p>For Polish-language analyses, download distributional semantic models from:<br>
<a href="https://dsmodels.nlp.ipipan.waw.pl">https://dsmodels.nlp.ipipan.waw.pl</a></p>

<h3>fastText</h3>
<p>Recommended for other non-English languages or when you need subword
information. Pre-trained models for 157 languages are available from the
fastText website. Download the <code>.bin</code> format.</p>

<h3>Custom Embeddings</h3>
<p>Train your own with gensim&rsquo;s Word2Vec or fastText and export as
<code>.kv</code>.</p>

<h3>Format Compatibility</h3>
<table>
  <tr><th>Format</th><th>Extension</th><th>Notes</th></tr>
  <tr><td>SSDiff native</td><td>.ssdembed</td><td>As fast as .kv. Tracks L2/ABTT state, preventing double-normalisation. Created within the app.</td></tr>
  <tr><td>gensim KeyedVectors</td><td>.kv</td><td>Fast to load.</td></tr>
  <tr><td>word2vec binary</td><td>.bin</td><td>Standard binary format.</td></tr>
  <tr><td>Text</td><td>.txt, .vec</td><td>One word per line + floats.</td></tr>
  <tr><td>Compressed text</td><td>.gz</td><td>Gzip-compressed text.</td></tr>
</table>
<p><em>Tip:</em> After loading embeddings in the app, save them as
<code>.ssdembed</code> for faster loading across projects and to preserve
normalisation state.</p>""", []),
]


def _build_html(palette) -> str:
    """Assemble the full HTML document from sections, themed to *palette*."""
    bg = palette.bg_surface
    fg = palette.text_primary
    fg2 = palette.text_secondary
    muted = palette.text_muted
    accent = palette.accent
    card_bg = palette.bg_card
    border = palette.border_subtle

    style = f"""
    body {{
        background: {bg};
        color: {fg};
        font-family: 'Segoe UI', system-ui, sans-serif;
        font-size: 14px;
        line-height: 1.6;
        margin: 0; padding: 16px 24px;
    }}
    h1 {{
        color: {accent};
        font-size: 22px;
        font-weight: 600;
        margin: 32px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid {border};
    }}
    h2 {{
        color: {fg};
        font-size: 17px;
        font-weight: 600;
        margin: 24px 0 8px 0;
    }}
    h3 {{
        color: {fg2};
        font-size: 15px;
        font-weight: 600;
        margin: 18px 0 6px 0;
    }}
    p, li {{ color: {fg}; }}
    a {{ color: {accent}; text-decoration: none; }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 10px 0;
    }}
    th {{
        text-align: left;
        padding: 6px 10px;
        border-bottom: 2px solid {border};
        color: {fg2};
        font-size: 13px;
    }}
    td {{
        padding: 5px 10px;
        border-bottom: 1px solid {border};
        vertical-align: top;
    }}
    code {{
        background: {card_bg};
        padding: 1px 5px;
        border-radius: 3px;
        font-size: 13px;
    }}
    pre {{
        background: {card_bg};
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 13px;
        overflow-x: auto;
        border: 1px solid {border};
    }}
    em {{ color: {fg2}; }}
    ul, ol {{ padding-left: 22px; }}
    li {{ margin-bottom: 3px; }}
    .muted {{ color: {muted}; font-size: 13px; }}
    """

    parts = [f"<html><head><style>{style}</style></head><body>"]

    for anchor, title, intro_html, subs in _SECTIONS:
        parts.append(f'<a name="{anchor}"></a>')
        parts.append(f"<h1>{title}</h1>")
        if intro_html:
            parts.append(intro_html)
        for sub_anchor, sub_title, sub_html in subs:
            parts.append(f'<a name="{sub_anchor}"></a>')
            parts.append(f"<h2>{sub_title}</h2>")
            parts.append(sub_html)

    parts.append("</body></html>")
    return "\n".join(parts)


def _build_toc_tree(tree: QTreeWidget):
    """Populate the tree widget with the table of contents."""
    tree.clear()
    tree.setHeaderHidden(True)
    tree.setIndentation(16)

    for anchor, title, _intro, subs in _SECTIONS:
        # Strip HTML entities for display
        display = title.replace("&mdash;", "\u2014").replace("&amp;", "&").replace("&ldquo;", "\u201c").replace("&rdquo;", "\u201d")
        item = QTreeWidgetItem([display])
        item.setData(0, Qt.UserRole, anchor)
        for sub_anchor, sub_title, _sub_html in subs:
            sub_display = sub_title.replace("&mdash;", "\u2014").replace("&amp;", "&").replace("&ldquo;", "\u201c").replace("&rdquo;", "\u201d")
            child = QTreeWidgetItem([sub_display])
            child.setData(0, Qt.UserRole, sub_anchor)
            item.addChild(child)
        tree.addTopLevelItem(item)

    # Expand the first two sections by default
    for i in range(min(2, tree.topLevelItemCount())):
        tree.topLevelItem(i).setExpanded(True)


class TutorialDialog(QDialog):
    """Navigable tutorial dialog with sidebar TOC and HTML content."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SSD Tutorial")
        self.setMinimumSize(950, 650)
        self.resize(1050, 720)
        # Allow the main window to stay interactive
        self.setModal(False)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)

        # --- sidebar ---
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(4)

        self._tree = QTreeWidget()
        self._tree.setMinimumWidth(200)
        self._tree.setMaximumWidth(280)
        _build_toc_tree(self._tree)
        self._tree.currentItemChanged.connect(self._on_toc_click)
        sidebar_layout.addWidget(self._tree)

        splitter.addWidget(sidebar)

        # --- content ---
        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(False)
        self._browser.setOpenLinks(False)
        self._browser.anchorClicked.connect(self._on_link_clicked)
        splitter.addWidget(self._browser)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([230, 800])

        layout.addWidget(splitter)

        # --- bottom bar ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setObjectName("btn_secondary")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        # Load themed content
        self._load_content()

    def _load_content(self):
        from ..theme import build_current_palette
        palette = build_current_palette()
        html = _build_html(palette)
        self._browser.setHtml(html)

    def _on_toc_click(self, current, _previous):
        if current is None:
            return
        anchor = current.data(0, Qt.UserRole)
        if anchor:
            self._browser.scrollToAnchor(anchor)

    def _on_link_clicked(self, url: QUrl):
        """Open external links in the system browser."""
        if url.scheme() in ("http", "https"):
            QDesktopServices.openUrl(url)
