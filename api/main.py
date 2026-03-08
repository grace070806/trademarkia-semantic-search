"""
api/main.py — FastAPI service with semantic cache and numpy vector store.
"""

import json
import logging
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
import sys

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.semantic_cache import SemanticCache
from core.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
N_CLUSTERS = int(os.getenv("N_CLUSTERS", "15"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))


class AppState:
    embedding_model = None
    fcm_model = None
    vector_store: Optional[VectorStore] = None
    semantic_cache: Optional[SemanticCache] = None
    corpus_texts: list = []
    corpus_cat_names: list = []


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all models ONCE at startup using FastAPI lifespan context manager.

    The embedding model and FCM clustering structures are loaded here to
    avoid repeated initialisation per request. Loading all-MiniLM-L6-v2
    takes ~1-2 seconds — doing this once at startup rather than per request
    is essential for production latency targets.
    """
    logger.info("=== Semantic Search Service — Starting Up ===")

    # 1. Embedding model
    from sentence_transformers import SentenceTransformer
    logger.info("Loading sentence-transformer (all-MiniLM-L6-v2) …")
    state.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 2. FCM clustering model
    fcm_path = DATA_DIR / "fcm_model.pkl"
    if fcm_path.exists():
        with open(fcm_path, "rb") as f:
            state.fcm_model = pickle.load(f)
        logger.info(f"FCM model ready: K={state.fcm_model.n_clusters}, m={state.fcm_model.m}")
    else:
        logger.warning("FCM model not found. Run `python run_pipeline.py` first.")

    # 3. Vector store (pure numpy, no ChromaDB dependency)
    state.vector_store = VectorStore(store_dir=str(DATA_DIR))
    if state.vector_store.load():
        logger.info(f"Vector store ready: {len(state.vector_store)} documents")
    else:
        logger.warning("Vector store not found. Run `python run_pipeline.py` first.")

    # 4. Corpus text
    corpus_path = DATA_DIR / "corpus.json"
    if corpus_path.exists():
        with open(corpus_path) as f:
            corpus = json.load(f)
        state.corpus_texts = corpus["texts"]
        state.corpus_cat_names = corpus["cat_names"]

    # 5. Semantic cache
    state.semantic_cache = SemanticCache(
        similarity_threshold=SIMILARITY_THRESHOLD,
        n_clusters=N_CLUSTERS,
        multi_cluster=True,
        secondary_threshold=0.25,
    )
    logger.info(f"Semantic cache ready (θ={SIMILARITY_THRESHOLD}, K={N_CLUSTERS})")
    logger.info("=== Service ready ===")
    yield
    logger.info("Shutting down …")


app = FastAPI(
    title="Trademarkia Semantic Search",
    description="Semantic search over 20 Newsgroups with Fuzzy C-Means clustering and cluster-aware semantic cache.",
    version="1.0.0",
    lifespan=lifespan,
)


def embed_query(query: str) -> np.ndarray:
    """Embed query with explicit L2-normalisation so cosine sim = dot product."""
    emb = state.embedding_model.encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    )
    return emb[0].astype(np.float32)


def get_membership(embedding: np.ndarray) -> np.ndarray:
    if state.fcm_model is None:
        return np.ones(N_CLUSTERS, dtype=np.float32) / N_CLUSTERS
    return state.fcm_model.predict(embedding.reshape(1, -1))[0]


def search_corpus(embedding: np.ndarray, n_results: int = TOP_K_RESULTS) -> list:
    if state.vector_store is None:
        return []
    return state.vector_store.query(embedding, n_results=n_results)


def format_result(hits: list) -> str:
    if not hits:
        return "No relevant documents found."
    top = hits[0]
    return f"[Category: {top['category']} | Similarity: {top['similarity']:.3f}]\n\n{top['text']}"


# --- Schemas ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: str
    dominant_cluster: int
    top_results: list = []

class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float

class FlushResponse(BaseModel):
    message: str


# --- Endpoints ---
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """Embed query → check semantic cache → return result (hit or miss)."""
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    embedding = embed_query(query)
    membership = get_membership(embedding)
    dominant_cluster = int(np.argmax(membership))

    cache_result = state.semantic_cache.lookup(embedding, membership)

    if cache_result is not None:
        entry, sim_score = cache_result
        logger.info(f"CACHE HIT | sim={sim_score:.4f} | cluster={entry.dominant_cluster}")
        return QueryResponse(
            query=query, cache_hit=True, matched_query=entry.query,
            similarity_score=round(sim_score, 4), result=entry.result,
            dominant_cluster=entry.dominant_cluster, top_results=[],
        )

    logger.info(f"CACHE MISS | cluster={dominant_cluster}")
    hits = search_corpus(embedding)
    result_str = format_result(hits)
    state.semantic_cache.store(
        query=query, query_embedding=embedding,
        membership=membership, result=result_str
    )

    return QueryResponse(
        query=query, cache_hit=False, matched_query=None,
        similarity_score=None, result=result_str,
        dominant_cluster=dominant_cluster, top_results=hits[:3],
    )


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    s = state.semantic_cache.stats
    return CacheStatsResponse(
        total_entries=s.total_entries, hit_count=s.hit_count,
        miss_count=s.miss_count, hit_rate=round(s.hit_rate, 4),
    )


@app.delete("/cache", response_model=FlushResponse)
async def flush_cache():
    state.semantic_cache.flush()
    return FlushResponse(message="Cache flushed and all stats reset.")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": {
            "embedding_model": state.embedding_model is not None,
            "fcm_model": state.fcm_model is not None,
            "vector_store": state.vector_store is not None and len(state.vector_store) > 0,
        },
        "cache": {
            "entries": len(state.semantic_cache),
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "n_clusters": N_CLUSTERS,
        },
        "corpus_docs": len(state.corpus_texts),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
