"""
core/semantic_cache.py
======================
Part 3 — Cluster-aware semantic cache, built entirely from first principles.

No Redis, Memcached, or any caching library. All logic is in this file.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """A single cached query–result pair."""
    query: str
    embedding: np.ndarray        # L2-normalised, shape (D,)
    result: Any
    dominant_cluster: int
    membership: list             # full fuzzy membership vector
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0


@dataclass
class CacheStats:
    total_entries: int = 0
    hit_count: int = 0
    miss_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Semantic Cache
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Cluster-aware semantic cache.

    Design
    ──────
    A conventional string-equality cache fails the moment two users phrase
    the same question differently. This cache uses cosine similarity between
    L2-normalised query embeddings to decide whether two queries are
    "close enough" to share a result.

    Data structure
    ──────────────
    Entries are partitioned by cluster ID into buckets:

        _buckets: dict[int, list[CacheEntry]]

    Correction 5 — Lookup complexity with cluster partitioning:
    Partitioning cache entries by cluster reduces lookup complexity.
    Instead of scanning all N cache entries (O(N)), only entries in the
    relevant cluster bucket(s) are checked. Expected complexity becomes
    O(N/K) where K is the number of clusters. At K=15 with 1,000 cached
    entries, the average lookup touches ~67 entries instead of 1,000 —
    a 15× speedup that scales linearly with K as the cache grows.

    Correction 6 — Threshold θ behaviour:
    Similarity thresholds control the trade-off between cache hit rate
    and answer correctness.

        θ = 0.95 → near-identical queries only (high precision, low reuse)
        θ = 0.85 → paraphrases and synonym variants captured (chosen default)
        θ = 0.70 → broader topical matches — lower thresholds increase cache
                   hits but may return topically related rather than
                   semantically equivalent queries
        θ < 0.60 → effectively a cluster-level cache; too aggressive

    Higher thresholds improve precision but reduce reuse.
    Lower thresholds increase cache hits but risk returning wrong results.
    Experiments show θ≈0.85 provides the best balance.
    See data/analysis/threshold_experiment.png for the full sweep.

    Cosine similarity computation
    ──────────────────────────────
    Since all embeddings are L2-normalised before storage:

        Cosine similarity:  cos(a, b) = (a · b) / (||a|| · ||b||)
        When ||a|| = ||b|| = 1:  cos(a, b) = a · b

    All comparisons reduce to a dot product — O(D) with no division.

    Thread safety
    ─────────────
    All mutations are protected by threading.RLock to support concurrent
    FastAPI requests without corrupting in-memory state.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        n_clusters: int = 15,
        multi_cluster: bool = True,
        secondary_threshold: float = 0.25,
    ):
        self.theta = similarity_threshold
        self.n_clusters = n_clusters
        self.multi_cluster = multi_cluster
        self.secondary_threshold = secondary_threshold

        self._buckets: dict = {c: [] for c in range(n_clusters)}
        self._stats = CacheStats()
        self._lock = threading.RLock()

    # ----------------------------------------------------------------- lookup
    def lookup(
        self,
        query_embedding: np.ndarray,
        membership: np.ndarray,
    ) -> Optional[tuple]:
        """
        Search for a semantically similar cached entry.

        Returns (CacheEntry, similarity_score) if found, else None.

        Steps:
        1. Identify which cluster buckets to search (dominant + secondaries)
        2. For each candidate, compute cosine similarity as dot product
           (valid because all embeddings are L2-normalised)
        3. Return the highest-similarity entry above self.theta
        """
        query_embedding = self._normalise(query_embedding)
        dominant = int(np.argmax(membership))
        search_clusters = {dominant}

        if self.multi_cluster:
            for c, mem in enumerate(membership):
                if mem >= self.secondary_threshold and c != dominant:
                    search_clusters.add(c)

        best_entry: Optional[CacheEntry] = None
        best_sim: float = -1.0

        with self._lock:
            for c in search_clusters:
                for entry in self._buckets[c]:
                    # Cosine similarity = dot product (both L2-normalised)
                    sim = float(np.dot(query_embedding, entry.embedding))
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = entry

        if best_sim >= self.theta:
            with self._lock:
                best_entry.hit_count += 1
                self._stats.hit_count += 1
            return best_entry, best_sim

        with self._lock:
            self._stats.miss_count += 1
        return None

    # ------------------------------------------------------------------ store
    def store(
        self,
        query: str,
        query_embedding: np.ndarray,
        membership: np.ndarray,
        result: Any,
    ) -> CacheEntry:
        """Add a new entry to the dominant-cluster bucket."""
        query_embedding = self._normalise(query_embedding)
        dominant = int(np.argmax(membership))

        entry = CacheEntry(
            query=query,
            embedding=query_embedding.copy(),
            result=result,
            dominant_cluster=dominant,
            membership=membership.tolist(),
        )

        with self._lock:
            self._buckets[dominant].append(entry)
            self._stats.total_entries += 1

        return entry

    # ------------------------------------------------------------------ flush
    def flush(self):
        with self._lock:
            for c in range(self.n_clusters):
                self._buckets[c] = []
            self._stats = CacheStats()

    # ------------------------------------------------------------------ stats
    @property
    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                total_entries=self._stats.total_entries,
                hit_count=self._stats.hit_count,
                miss_count=self._stats.miss_count,
            )

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-10 else v

    def __len__(self):
        return self._stats.total_entries

    def __repr__(self):
        s = self.stats
        return (
            f"SemanticCache(θ={self.theta}, K={self.n_clusters}, "
            f"entries={s.total_entries}, hit_rate={s.hit_rate:.2%})"
        )
