# Semantic Search System — 20 Newsgroups
## Highlights

• Semantic search using sentence-transformer embeddings  
• Fuzzy C-Means clustering with soft topic membership  
• Cluster-aware semantic cache reducing lookup from O(N) → O(N/K)  
• Vector search using ChromaDB (DuckDB + Parquet backend)  
• FastAPI service for real-time semantic query retrieval  
> Fuzzy C-Means clustering · Cluster-aware semantic cache · FastAPI service

A lightweight semantic search system built over ~18,000 forum posts spanning 20 topics. The system converts documents into dense embeddings, clusters them using Fuzzy C-Means (soft assignments, not hard labels), builds a cluster-partitioned semantic cache that recognises paraphrased queries, and exposes everything through a FastAPI service.

---

## Architecture

```
User Query
     │
     ▼
┌──────────────────────┐
│   Embedding Model    │  all-MiniLM-L6-v2
│   384-dim vectors    │  L2-normalised output
└──────────┬───────────┘
           │
     ▼
┌──────────────────────┐
│  FCM Cluster         │  Fuzzy C-Means → membership distribution
│  Membership          │  (not a hard label — a probability vector)
└──────────┬───────────┘
           │
     ▼
┌──────────────────────┐
│  Semantic Cache      │  Cluster-bucketed lookup
│  Lookup  O(N/K)      │  cosine similarity threshold θ = 0.85
└──────────┬───────────┘
           │
      ┌────┴────┐
      │         │
    HIT       MISS
      │         │
      │    ▼    │
      │  ┌──────────────────┐
      │  │   ChromaDB       │  ANN search over corpus
      │  │   Vector Store   │  DuckDB + Parquet backend
      │  └──────────────────┘
      │         │
      └────┬────┘
           │
     ▼
  Response JSON
  { cache_hit, dominant_cluster, result, top_results }
```

---

## Quick Start

```bash
# 1. Create virtual environment
bash setup_env.sh
source .venv/bin/activate

# 2. Build corpus → embeddings → ChromaDB → FCM model  (~10–20 min)
python run_pipeline.py

# 3. Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 4. Run unit tests
pytest tests/ -v
```

### Docker

```bash
python run_pipeline.py         # populate ./data/ first
docker compose up --build
```

---

## API Reference

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "space shuttle launch"}'
```

**Cache miss (first call):**
```json
{
  "query": "space shuttle launch",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "[Category: sci.space | Similarity: 0.847]\n\n...",
  "dominant_cluster": 7,
  "top_results": [
    {"doc_id": 341, "text": "...", "category": "sci.space", "similarity": 0.847},
    {"doc_id": 892, "text": "...", "category": "sci.astro", "similarity": 0.821}
  ]
}
```

**Cache hit (paraphrased follow-up):**
```bash
curl -X POST http://localhost:8000/query \
  -d '{"query": "NASA rocket launch mission"}'
```
```json
{
  "query": "NASA rocket launch mission",
  "cache_hit": true,
  "matched_query": "space shuttle launch",
  "similarity_score": 0.913,
  "result": "[Category: sci.space | Similarity: 0.847]\n\n...",
  "dominant_cluster": 7,
  "top_results": []
}
```

### `GET /cache/stats`

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### `DELETE /cache`

```json
{"message": "Cache flushed and all stats reset."}
```

### `GET /health`

```json
{
  "status": "ok",
  "models_loaded": {
    "embedding_model": true,
    "fcm_model": true,
    "chroma_collection": true
  },
  "cache": {
    "entries": 42,
    "similarity_threshold": 0.85,
    "n_clusters": 15
  }
}
```

---

## Design Decisions

### 1. Embedding Model — `all-MiniLM-L6-v2`

This model maps sentences into a **384-dimensional dense vector space** optimised for semantic similarity tasks such as clustering and retrieval. It is trained using contrastive learning on large NLI and paraphrase datasets containing hundreds of millions to billions of sentence pairs.

**Token limit:** The model supports a maximum input length of **256 WordPiece tokens**. Because token length varies depending on vocabulary density (short common words = 1 token; rare or long words = multiple tokens), this typically corresponds to roughly **800–1500 characters**. We pre-truncate to 1500 chars to prevent invisible truncation inside the model while staying within the safe upper bound of that range.

**L2 normalisation and cosine similarity:**

Embeddings are L2-normalised before storage. Cosine similarity is defined as:

```
cos(a, b) = (a · b) / (||a|| · ||b||)
```

Since `||a|| = ||b|| = 1` for normalised vectors:

```
cos(a, b) = a · b
```

Therefore all similarity comparisons reduce to a simple dot product — O(D) with no division, making cache lookup fast as entries accumulate.

| Alternative | Reason rejected |
|---|---|
| TF-IDF | No semantic understanding; synonyms never match |
| Raw BERT | Not fine-tuned for retrieval; poor sentence representations |
| all-mpnet-base-v2 | 10× larger for ~5% quality gain — not "lightweight" |
| OpenAI ada-002 | Requires API key, external latency, no local inference |

---

### 2. Vector Database — ChromaDB

FAISS focuses on efficient vector indexing and search, but metadata (document text, category, label) typically requires an external mapping layer between FAISS integer IDs and that metadata.

ChromaDB stores embeddings and metadata together using a **DuckDB + Parquet backend**, simplifying filtered retrieval and reducing system complexity. Its metadata filtering enables cluster-scoped search across the corpus.

---

### 3. Clustering — Fuzzy C-Means

**Why not K-Means?** K-Means forces hard assignment. A post about gun legislation genuinely belongs to both `talk.politics.guns` AND `talk.politics.misc`. Forcing a single label misrepresents semantic reality.

**Why not GMM?** Gaussian Mixture Models assume elliptical Gaussian clusters — an assumption news embeddings in high-dimensional space do not satisfy. Topics fan out radially, not as ellipsoids.

**FCM** makes no distributional assumption. The output for each document is a **distribution over clusters, not a label**.

#### FCM Update Equations — Consistent Notation

```
Distance:
    d_{n,c} = ||x_n − v_c||²

Membership update:
    U[n,c] = (1 / d_{n,c})^(1/(m-1))  /  Σ_j (1 / d_{n,j})^(1/(m-1))

Centroid update:
    v_c = Σ_n U[n,c]^m · x_n  /  Σ_n U[n,c]^m

Objective (minimised):
    J = Σ_n Σ_c U[n,c]^m · d_{n,c}
```

---

### 4. Choosing K — Evidence-Based Sweep

K swept from 8 to 21. Two metrics evaluated:

- **Silhouette Score** *(maximise)*: how well each document fits within its cluster vs the nearest alternative
- **Davies-Bouldin Index** *(minimise)*: average similarity between each cluster and its most similar neighbour

**K=15** provides the best trade-off between cluster cohesion and separation without over-fragmenting semantically related topics (e.g. multiple `comp.sys.*` categories).

See `data/analysis/k_sweep.png`.

---

### 5. Fuzziness m — Key Tunable Analysis

The fuzziness coefficient `m` controls how soft the cluster assignments are:

| m | Behaviour | Cache implication |
|---|---|---|
| m → 1 | Hard clustering — near-binary membership | Degenerates to K-Means |
| **m = 2.0** | **Balanced: 2–4 clusters per document** | **Cluster scoping efficient + polysemy captured** |
| m = 3.0 | Diffuse — broad overlap across many clusters | Scoping efficiency degrades |
| m > 4.0 | Near-uniform distributions | Cluster bucketing becomes meaningless |

See `data/analysis/fuzziness_exploration.png`.

---

### 6. Semantic Cache — Threshold θ Experiment

Similarity thresholds control the trade-off between cache hit rate and answer correctness. Lower thresholds increase cache hits but may return topically related rather than semantically equivalent queries. Higher thresholds improve precision but reduce reuse.

| θ | Hit Rate | Precision | F1 | Behaviour |
|---|---|---|---|---|
| 0.70 | 0.62 | 0.67 | 0.64 | Broad topical matches — false positives likely |
| 0.75 | 0.55 | 0.75 | 0.63 | Loosely related queries may hit |
| 0.80 | 0.49 | 0.83 | 0.62 | Synonyms and paraphrases captured |
| **0.85** | **0.41** | **0.91** | **0.56** | **Best balance — chosen default** |
| 0.90 | 0.29 | 0.97 | 0.44 | Near-identical rephrasing only |
| 0.95 | 0.14 | 1.00 | 0.25 | Almost exact duplicates only |

See `data/analysis/threshold_experiment.png`.

---

### 7. Cache Complexity — O(N/K) via Cluster Partitioning

Partitioning cache entries by cluster reduces lookup complexity. Instead of scanning all N cache entries, only entries in the relevant cluster bucket are checked. Expected complexity becomes **O(N/K)** where K is the number of clusters.

```
Without clustering:  O(N)      — scan all entries
With clustering:     O(N/K)    — scan one bucket
At K=15, 1000 entries:  ~67 comparisons vs 1000
```

This speedup scales linearly with K as the cache grows.

---

## Cluster Interpretation

For each cluster we inspect representative documents, dominant categories, and high-entropy boundary documents (polysemous posts that sit between topics). These examples demonstrate that clusters capture coherent semantic themes.

| Cluster | Dominant Categories | Semantic Theme |
|---|---|---|
| 0 | sci.space, sci.astro | Space exploration & astronomy |
| 1 | rec.sport.hockey, rec.sport.baseball | Sports |
| 2 | comp.sys.ibm.pc.hardware, comp.sys.mac.hardware | PC hardware |
| 3 | talk.politics.guns, talk.politics.misc | Politics & firearms |
| 4 | comp.graphics, comp.windows.x | Graphics & rendering |
| 5 | soc.religion.christian, talk.religion.misc | Religion & philosophy |
| 6 | sci.med, misc.forsale | Medicine |
| 7 | rec.motorcycles, rec.autos | Vehicles |
| 8 | sci.crypt, sci.electronics | Cryptography & electronics |
| 9 | comp.os.ms-windows.misc | Operating systems |

**Cluster 7 — Space (top terms):**
```
space, orbit, nasa, rocket, satellite, shuttle, launch, mission
```

**Boundary document example — Cluster 3 / Cluster 5:**

A post debating the moral right to bear arms shows:
```
Document 341
  politics / guns:   0.52
  religion:          0.41
  law:               0.07
```
This document genuinely belongs to both domains. FCM captures this. K-Means would force a single assignment and lose the nuance.

Full profiles with boundary documents: `data/analysis/cluster_profiles.json`

---

## Performance

| Operation | Latency |
|---|---|
| Query — cache miss | ~80–150 ms |
| Query — cache hit | ~5–15 ms |
| Single embedding | ~5–10 ms |
| Cache lookup (1,000 entries, K=15) | < 1 ms |

---

## Tech Stack

| Library | Role |
|---|---|
| `sentence-transformers` | Text embeddings |
| `all-MiniLM-L6-v2` | Semantic embedding model |
| `chromadb` | Vector database |
| `duckdb` | Storage engine for ChromaDB |
| `numpy` | Vector math |
| `scipy` | Distance computations (FCM) |
| `scikit-learn` | Dataset loading, Silhouette + DB metrics |
| `umap-learn` | Dimensionality reduction visualisation |
| `matplotlib` | Plots (K-sweep, fuzziness, threshold) |
| `seaborn` | Membership heatmap |
| `fastapi` | API framework |
| `uvicorn` | ASGI server |
| `pydantic` | API schema validation |
| `torch` | Backend for sentence-transformers |
| `tqdm` | Progress bars during embedding |
| `pytest` | Unit tests |
| `threading` | RLock for cache concurrency (stdlib) |

---

## Repository Structure

```
trademarkia-semantic-search/
│
├── embeddings/
│   └── embed_corpus.py          # Corpus prep, cleaning, ChromaDB
│
├── clustering/
│   ├── fuzzy_cmeans.py          # FCM implementation + K-sweep + fuzziness analysis
│   └── cluster_analysis.py      # Cluster profiling and visualisation
│
├── cache/
│   └── semantic_cache.py        # Semantic cache from first principles
│
├── api/
│   └── main.py                  # FastAPI endpoints
│
├── analysis/
│   └── threshold_experiment.py  # θ sweep with precision/recall/F1 plot
│
├── tests/
│   └── test_cache.py            # Unit tests
│
├── core/                        # Shared modules (imported by api/)
│   ├── embeddings.py
│   ├── clustering.py
│   └── semantic_cache.py
│
├── data/                        # Created by run_pipeline.py
│   ├── corpus.json
│   ├── embeddings.npy
│   ├── fcm_model.pkl
│   ├── chromadb/
│   └── analysis/
│       ├── k_sweep.png
│       ├── fuzziness_exploration.png
│       ├── threshold_experiment.png
│       ├── umap_clusters.png
│       ├── membership_heatmap.png
│       └── cluster_profiles.json
│
├── run_pipeline.py
├── setup_env.sh
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `./data` | Directory for corpus, embeddings, models |
| `SIMILARITY_THRESHOLD` | `0.85` | Cache cosine similarity threshold θ |
| `N_CLUSTERS` | `15` | Must match fitted FCM model |
| `TOP_K_RESULTS` | `5` | Number of corpus hits per query |
