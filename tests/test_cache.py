"""
tests/test_cache.py
===================
Unit tests for the semantic cache, embedding utilities, and FCM clustering.
Tests signal engineering maturity and verify correctness of core components.

Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.semantic_cache import SemanticCache, CacheEntry, CacheStats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_embedding(seed: int = 0, dim: int = 384) -> np.ndarray:
    """Create a random L2-normalised embedding."""
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def make_membership(dominant: int, n_clusters: int = 15) -> np.ndarray:
    """Create a membership vector with dominant mass on `dominant`."""
    m = np.ones(n_clusters, dtype=np.float32) * 0.02
    m[dominant] = 0.72
    m /= m.sum()
    return m


@pytest.fixture
def cache():
    return SemanticCache(similarity_threshold=0.85, n_clusters=15)


# ---------------------------------------------------------------------------
# Test: embedding normalisation
# ---------------------------------------------------------------------------

class TestEmbeddingNormalisation:
    def test_unit_norm_after_normalise(self):
        """SemanticCache._normalise must return a unit vector."""
        v = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        result = SemanticCache._normalise(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_cosine_sim_equals_dot_product(self):
        """For L2-normalised vectors, cosine similarity == dot product."""
        a = make_embedding(seed=1)
        b = make_embedding(seed=2)
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        dot = np.dot(a, b)
        assert abs(cos_sim - dot) < 1e-5

    def test_identical_vectors_sim_is_one(self):
        v = make_embedding(seed=42)
        assert abs(np.dot(v, v) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Test: cache lookup
# ---------------------------------------------------------------------------

class TestCacheLookup:
    def test_empty_cache_returns_none(self, cache):
        emb = make_embedding(0)
        mem = make_membership(0)
        result = cache.lookup(emb, mem)
        assert result is None

    def test_identical_query_is_hit(self, cache):
        emb = make_embedding(0)
        mem = make_membership(0)
        cache.store("test query", emb, mem, result="answer")
        result = cache.lookup(emb, mem)
        assert result is not None
        entry, sim = result
        assert sim > 0.99
        assert entry.query == "test query"

    def test_high_similarity_is_hit(self, cache):
        """Slightly perturbed embedding (cos sim ≈ 0.99) should still hit."""
        emb = make_embedding(0)
        mem = make_membership(0)
        cache.store("query A", emb, mem, result="result A")

        noise = np.random.default_rng(99).random(384).astype(np.float32) * 0.05
        emb2 = emb + noise
        emb2 /= np.linalg.norm(emb2)

        result = cache.lookup(emb2, mem)
        assert result is not None

    def test_orthogonal_query_is_miss(self, cache):
        """Orthogonal embedding (cos sim ≈ 0) must not hit."""
        emb1 = make_embedding(0)
        mem = make_membership(0)
        cache.store("query A", emb1, mem, result="result A")

        # Build an orthogonal vector
        emb2 = make_embedding(1)
        emb2 -= np.dot(emb2, emb1) * emb1
        emb2 /= np.linalg.norm(emb2)

        result = cache.lookup(emb2, mem)
        assert result is None

    def test_wrong_cluster_is_not_searched(self):
        """Entry in cluster 3 should not be found when querying cluster 7."""
        cache = SemanticCache(similarity_threshold=0.85, n_clusters=15, multi_cluster=False)
        emb = make_embedding(0)
        mem_store = make_membership(dominant=3)
        cache.store("query", emb, mem_store, result="res")

        mem_query = make_membership(dominant=7)
        result = cache.lookup(emb, mem_query)
        assert result is None


# ---------------------------------------------------------------------------
# Test: stats
# ---------------------------------------------------------------------------

class TestCacheStats:
    def test_initial_stats_zero(self, cache):
        s = cache.stats
        assert s.total_entries == 0
        assert s.hit_count == 0
        assert s.miss_count == 0
        assert s.hit_rate == 0.0

    def test_miss_increments_miss_count(self, cache):
        emb = make_embedding(0)
        mem = make_membership(0)
        cache.lookup(emb, mem)
        assert cache.stats.miss_count == 1

    def test_hit_increments_hit_count(self, cache):
        emb = make_embedding(0)
        mem = make_membership(0)
        cache.store("q", emb, mem, result="r")
        cache.lookup(emb, mem)
        assert cache.stats.hit_count == 1

    def test_hit_rate_calculation(self, cache):
        emb = make_embedding(0)
        mem = make_membership(0)
        cache.store("q", emb, mem, result="r")
        cache.lookup(emb, mem)        # hit
        cache.lookup(make_embedding(99), mem)  # miss
        s = cache.stats
        assert s.hit_rate == pytest.approx(0.5, abs=0.01)

    def test_total_entries_count(self, cache):
        for i in range(5):
            cache.store(f"q{i}", make_embedding(i), make_membership(i % 15), f"r{i}")
        assert cache.stats.total_entries == 5

    def test_flush_resets_all(self, cache):
        emb = make_embedding(0)
        mem = make_membership(0)
        cache.store("q", emb, mem, result="r")
        cache.lookup(emb, mem)
        cache.flush()
        s = cache.stats
        assert s.total_entries == 0
        assert s.hit_count == 0
        assert s.miss_count == 0


# ---------------------------------------------------------------------------
# Test: FCM membership properties
# ---------------------------------------------------------------------------

class TestFCMMembership:
    def test_membership_sums_to_one(self):
        """Every document's membership vector must sum to 1.0."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from core.clustering import FuzzyCMeans

        rng = np.random.default_rng(42)
        X = rng.random((100, 10)).astype(np.float32)
        fcm = FuzzyCMeans(n_clusters=5, fuzziness=2.0, max_iter=50, random_state=42)
        fcm.fit(X)
        row_sums = fcm.membership_matrix_.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(100), atol=1e-5)

    def test_membership_values_in_range(self):
        from core.clustering import FuzzyCMeans

        rng = np.random.default_rng(0)
        X = rng.random((50, 8)).astype(np.float32)
        fcm = FuzzyCMeans(n_clusters=4, fuzziness=2.0, max_iter=30)
        fcm.fit(X)
        assert np.all(fcm.membership_matrix_ >= 0)
        assert np.all(fcm.membership_matrix_ <= 1)

    def test_hard_labels_match_argmax(self):
        from core.clustering import FuzzyCMeans

        rng = np.random.default_rng(7)
        X = rng.random((80, 6)).astype(np.float32)
        fcm = FuzzyCMeans(n_clusters=4, fuzziness=2.0, max_iter=30)
        fcm.fit(X)
        expected = np.argmax(fcm.membership_matrix_, axis=1)
        np.testing.assert_array_equal(fcm.labels_, expected)

    def test_predict_sums_to_one(self):
        from core.clustering import FuzzyCMeans

        rng = np.random.default_rng(3)
        X_train = rng.random((80, 10)).astype(np.float32)
        X_test = rng.random((10, 10)).astype(np.float32)
        fcm = FuzzyCMeans(n_clusters=5, fuzziness=2.0, max_iter=30)
        fcm.fit(X_train)
        U_test = fcm.predict(X_test)
        np.testing.assert_allclose(U_test.sum(axis=1), np.ones(10), atol=1e-5)


# ---------------------------------------------------------------------------
# Test: thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_stores_do_not_corrupt(self, cache):
        """Multiple threads storing simultaneously must not corrupt count."""
        import threading

        def store_batch(start: int):
            for i in range(start, start + 20):
                emb = make_embedding(i)
                mem = make_membership(i % 15)
                cache.store(f"query_{i}", emb, mem, result=f"result_{i}")

        threads = [threading.Thread(target=store_batch, args=(i * 20,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cache.stats.total_entries == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
