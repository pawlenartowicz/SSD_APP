"""Session-scoped fixtures for integration tests (spaCy + SSD)."""

import numpy as np
import pytest

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    _HAS_SPACY = True
except (ImportError, OSError):
    _HAS_SPACY = False
    _nlp = None


pytestmark = pytest.mark.skipif(
    not _HAS_SPACY, reason="spaCy / en_core_web_sm not available",
)


@pytest.fixture(scope="session")
def spacy_nlp():
    return _nlp


@pytest.fixture(scope="session")
def synthetic_texts():
    """100 short English sentences using a controlled vocabulary."""
    import random
    rng = random.Random(42)
    vocab = [
        "happy", "sad", "angry", "love", "hate",
        "joy", "fear", "trust", "surprise", "disgust",
        "good", "bad", "great", "terrible", "wonderful",
        "hope", "despair", "calm", "fury", "peace",
    ]
    templates = [
        "The {} feeling was very {}",
        "I felt {} and {} today",
        "This is a {} {} experience",
        "They expressed {} {} emotions",
    ]
    texts = []
    for _ in range(100):
        tmpl = rng.choice(templates)
        words = [rng.choice(vocab) for _ in range(tmpl.count("{}"))]
        texts.append(tmpl.format(*words))
    return texts


@pytest.fixture(scope="session")
def tokenized_docs(spacy_nlp, synthetic_texts):
    docs = []
    for text in synthetic_texts:
        doc = spacy_nlp(text)
        tokens = [t.lemma_.lower() for t in doc
                  if not t.is_stop and not t.is_punct and len(t.text) > 1]
        docs.append(tokens)
    return docs


@pytest.fixture(scope="session")
def tiny_embeddings():
    """50-word × 10-dim ssdiff.Embeddings, seeded, L2-normalized."""
    from ssdiff import Embeddings
    rng = np.random.RandomState(42)
    words = [
        "happy", "sad", "angry", "love", "hate",
        "joy", "fear", "trust", "surprise", "disgust",
        "good", "bad", "great", "terrible", "wonderful",
        "hope", "despair", "calm", "fury", "peace",
        "bright", "dark", "warm", "cold", "gentle",
        "kind", "cruel", "brave", "weak", "strong",
        "feeling", "emotion", "experience", "express", "today",
        "the", "a", "is", "was", "very",
        "i", "they", "this", "felt", "and",
        "beautiful", "ugly", "nice", "mean", "sweet",
    ]
    vecs = rng.randn(len(words), 10).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norms > 0, norms, 1.0)
    return Embeddings(words, vecs)


@pytest.fixture(scope="session")
def synthetic_corpus(tokenized_docs):
    from ssdiff import Corpus
    return Corpus(tokenized_docs, pretokenized=True, lang="en")
