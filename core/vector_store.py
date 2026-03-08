"""
core/vector_store.py
====================
Pure-numpy vector store replacing ChromaDB.

ChromaDB requires onnxruntime which has no Python 3.14 build yet.
This implementation stores embeddings + metadata in plain .npy and .json
files and does brute-force cosine similarity search (dot product on
L2-normalised vectors). For ~18,000 documents this is fast enough
(<100ms per query) and needs zero extra dependencies.
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Lightweight vector store backed by numpy arrays on disk.

    Files written:
        data/vs_embeddings.npy   — float32 (N, D) matrix
        data/vs_metadata.json    — list of {text, category, label, doc_id}
    """

    def __init__(self, store_dir: str = "./data"):
        self.store_dir = Path(store_dir)
        self.embeddings: np.ndarray = None   # (N, D) float32, L2-normalised
        self.metadata: list = []

    # ----------------------------------------------------------------- build
    def build(
        self,
        texts: list,
        embeddings: np.ndarray,
        labels: list,
        cat_names: list,
        doc_ids: list,
    ):
        self.embeddings = embeddings.astype(np.float32)
        self.metadata = [
            {"text": texts[i], "category": cat_names[i],
             "label": labels[i], "doc_id": doc_ids[i]}
            for i in range(len(texts))
        ]
        self._save()
        logger.info(f"VectorStore built: {len(self.metadata)} documents")

    # ----------------------------------------------------------------- save
    def _save(self):
        self.store_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.store_dir / "vs_embeddings.npy", self.embeddings)
        with open(self.store_dir / "vs_metadata.json", "w") as f:
            # Save only text snippet to keep file small
            slim = [
                {"text": m["text"][:500], "category": m["category"],
                 "label": m["label"], "doc_id": m["doc_id"]}
                for m in self.metadata
            ]
            json.dump(slim, f)
        logger.info(f"VectorStore saved to {self.store_dir}")

    # ----------------------------------------------------------------- load
    def load(self) -> bool:
        emb_path = self.store_dir / "vs_embeddings.npy"
        meta_path = self.store_dir / "vs_metadata.json"
        if not emb_path.exists() or not meta_path.exists():
            return False
        self.embeddings = np.load(emb_path)
        with open(meta_path) as f:
            self.metadata = json.load(f)
        logger.info(f"VectorStore loaded: {len(self.metadata)} documents")
        return True

    # ----------------------------------------------------------------- query
    def query(self, query_embedding: np.ndarray, n_results: int = 5) -> list:
        """
        Return top-n most similar documents.

        Since all embeddings are L2-normalised, cosine similarity = dot product:
            cos(a, b) = a · b  when ||a|| = ||b|| = 1
        """
        if self.embeddings is None:
            return []

        q = query_embedding.astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            q = q / norm

        # Dot product with all corpus embeddings — shape (N,)
        scores = self.embeddings @ q

        top_idx = np.argsort(scores)[::-1][:n_results]
        results = []
        for i in top_idx:
            m = self.metadata[i]
            results.append({
                "doc_id": m["label"],
                "text": m["text"],
                "category": m["category"],
                "similarity": round(float(scores[i]), 4),
            })
        return results

    def __len__(self):
        return len(self.metadata) if self.metadata else 0
