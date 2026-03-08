"""
core/embeddings.py
==================
Part 1 — Corpus preparation, embedding, and vector database setup.

All design decisions are documented inline.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & rationale
# ---------------------------------------------------------------------------

# Correction 1 — Token vs character limit (precise explanation).
# The model supports a maximum input length of 256 WordPiece tokens.
# Because token length varies depending on vocabulary density
# (short common words = 1 token; rare/long words = multiple tokens),
# this typically corresponds to roughly 800–1500 characters.
# We pre-truncate to 1500 chars to prevent invisible truncation inside
# the model while staying within the safe upper bound of that range.
MAX_CHARS = 1500

# Documents with fewer than 30 whitespace-delimited tokens are typically
# one-liners ("Me too!"), forwarding notices, or near-empty posts.
# Their embeddings are unreliable and pollute cluster centroids.
MIN_TOKENS = 30


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_20newsgroups(subset: str = "all"):
    """
    Load the 20 Newsgroups dataset via sklearn (which fetches the UCI copy).

    We remove headers, footers, and quoted reply lines because:
    - Headers contain routing metadata (e.g. NNTP-Posting-Host) that are
      highly predictive of category but carry zero semantic content.
      A model that clusters by server hostname is not doing semantic clustering.
    - Footers are email signatures and legal boilerplate — pure noise.
    - Quoted reply lines ("> ...") duplicate content from a different post;
      including them biases embeddings toward the *quoted* document.
    """
    from sklearn.datasets import fetch_20newsgroups

    dataset = fetch_20newsgroups(
        subset=subset,
        remove=("headers", "footers", "quotes"),
        shuffle=False,
    )
    logger.info(
        f"Loaded {len(dataset.data)} documents across "
        f"{len(dataset.target_names)} categories"
    )
    return dataset


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Light-touch cleaning that preserves semantic content.

    We deliberately avoid stemming/lemmatisation here because the embedding
    model handles morphological variation natively via its subword tokenizer.
    We only remove structural noise that adds no semantic signal.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        # Remove quoted reply lines
        if stripped.startswith(">"):
            continue
        # Remove attribution lines like "John Smith writes:"
        if re.match(r"^.{0,60} writes?:$", stripped, re.IGNORECASE):
            continue
        lines.append(line)
    text = "\n".join(lines)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove URL-only lines (no semantic payload)
    text = re.sub(r"^\s*https?://\S+\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Pre-truncate to MAX_CHARS (approximation of the 256-token limit)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    return text


def is_valid(text: str) -> bool:
    """
    Return True if the cleaned text is worth embedding.

    Rejects near-empty posts and binary/patch content (common in comp.* groups)
    where >60% of tokens are non-alphabetic.
    """
    tokens = text.split()
    if len(tokens) < MIN_TOKENS:
        return False
    alpha_ratio = sum(1 for t in tokens if re.search(r"[a-zA-Z]", t)) / len(tokens)
    if alpha_ratio < 0.40:
        return False
    return True


def prepare_corpus(dataset) -> tuple[list, list, list, list]:
    texts, labels, cat_names, doc_ids = [], [], [], []
    skipped = 0

    for raw, label in zip(dataset.data, dataset.target):
        cleaned = clean_text(raw)
        if not is_valid(cleaned):
            skipped += 1
            continue
        doc_id = hashlib.sha256(cleaned.encode()).hexdigest()[:16]
        texts.append(cleaned)
        labels.append(int(label))
        cat_names.append(dataset.target_names[label])
        doc_ids.append(doc_id)

    logger.info(
        f"Kept {len(texts)} documents, discarded {skipped} "
        f"({skipped / (len(texts) + skipped) * 100:.1f}%)"
    )
    return texts, labels, cat_names, doc_ids


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load a sentence-transformer model.

    Why all-MiniLM-L6-v2?
    ─────────────────────
    This model maps sentences into a 384-dimensional dense vector space
    optimised for semantic similarity tasks such as clustering and retrieval.
    It is trained using contrastive learning on large NLI and paraphrase
    datasets containing hundreds of millions to billions of sentence pairs,
    making it far more semantically aware than raw BERT or TF-IDF baselines.

    At 22M parameters it is ~10× smaller than all-mpnet-base-v2 while
    retaining ~95% of its benchmark quality — well-suited to the "lightweight"
    requirement. The 256-token sequence limit aligns with our MAX_CHARS
    pre-truncation, so no meaningful content is lost implicitly.

    Alternative considered: OpenAI text-embedding-ada-002.
    Rejected: requires API key, external latency and cost, no local inference.
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def embed_corpus(
    texts: list[str],
    model,
    batch_size: int = 64,
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Embed all documents and L2-normalise the output vectors.

    Correction 2 — Cosine similarity equation (explicit):

        Cosine similarity is defined as:
            cos(a, b) = (a · b) / (||a|| · ||b||)

        Since embeddings are L2-normalised before storage (||a|| = ||b|| = 1):
            cos(a, b) = a · b

        Therefore all similarity comparisons reduce to a simple dot product,
        making each comparison O(D) with no division — important for efficient
        cache lookup as the number of cached entries grows.
    important for fast cache lookup.
    """
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    logger.info(f"Embedding {len(texts)} documents (batch_size={batch_size}) …")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # explicit L2 normalisation
    )
    embeddings = embeddings.astype(np.float32)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info(f"Saved embeddings to {cache_path}")

    return embeddings


# ---------------------------------------------------------------------------
# Vector database
# ---------------------------------------------------------------------------

def build_vector_db(
    texts: list,
    embeddings: np.ndarray,
    labels: list,
    cat_names: list,
    doc_ids: list,
    persist_dir: str = "./data",
    **kwargs,
):
    """
    Persist embeddings and metadata using a pure-numpy VectorStore.

    We use a custom VectorStore instead of ChromaDB because ChromaDB
    requires onnxruntime which has no Python 3.14 build. The VectorStore
    saves embeddings as a .npy file and metadata as .json, then serves
    queries via brute-force dot-product search on L2-normalised vectors.
    For ~18,000 documents this is fast enough (<100ms) with no dependencies.
    """
    from core.vector_store import VectorStore

    vs = VectorStore(store_dir=persist_dir)
    vs.build(texts, embeddings, labels, cat_names, doc_ids)
    return vs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_part1(data_dir: str = "./data"):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_20newsgroups()
    texts, labels, cat_names, doc_ids = prepare_corpus(dataset)

    corpus_path = data_dir / "corpus.json"
    with open(corpus_path, "w") as f:
        json.dump(
            {
                "texts": texts,
                "labels": labels,
                "cat_names": cat_names,
                "doc_ids": doc_ids,
                "target_names": dataset.target_names,
            },
            f,
        )
    logger.info(f"Saved corpus to {corpus_path}")

    model = load_embedding_model()
    embeddings = embed_corpus(texts, model, cache_path=data_dir / "embeddings.npy")
    build_vector_db(texts, embeddings, labels, cat_names, doc_ids)

    logger.info("Part 1 complete.")
    return texts, labels, cat_names, doc_ids, embeddings, dataset.target_names


if __name__ == "__main__":
    run_part1()
