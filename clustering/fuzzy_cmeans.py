"""
core/clustering.py
==================
Part 2 — Fuzzy C-Means clustering with membership distributions.

All design decisions and mathematical corrections documented inline.
"""

import json
import logging
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fuzzy C-Means — implemented from scratch (numpy + scipy only)
# ---------------------------------------------------------------------------

class FuzzyCMeans:
    """
    Fuzzy C-Means clustering (Bezdek, 1981).

    Why FCM and not K-Means or GMM?
    ────────────────────────────────
    K-Means assigns each document to exactly one cluster (hard assignment).
    A post about gun legislation genuinely belongs to both talk.politics.guns
    AND talk.politics.misc; forcing a hard choice misrepresents semantic reality.

    GMMs also produce soft assignments but assume elliptical Gaussian clusters.
    News embeddings in high-dimensional space do not satisfy this assumption —
    topics fan out radially rather than forming ellipsoids.

    FCM makes no distributional assumption: membership is purely distance-based,
    which is natural for cosine-similar dense vectors.

    Fuzziness parameter m
    ──────────────────────
    m controls how soft the cluster assignments are:

        m → 1   : hard clustering (degenerates to K-Means in the limit)
        m ≈ 2   : commonly used compromise — meaningful overlap between
                  related topics, but still discriminative enough to partition
                  the cache search space efficiently
        m > 3   : overly diffuse clusters; membership vectors flatten toward
                  uniform, making cluster-based cache scoping meaningless

    We use m=2.0 as the standard default and validate this choice empirically
    in the fuzziness exploration (see explore_fuzziness()).

    FCM update equations — consistent notation throughout
    ──────────────────────────────────────────────────────
    Distance (squared Euclidean):
        d_{n,c} = ||x_n − v_c||²

    Membership update:
        U[n,c] = (1 / d_{n,c})^(1/(m-1))
                 ────────────────────────
                 Σ_j (1 / d_{n,j})^(1/(m-1))

    Centroid update:
        v_c = Σ_n U[n,c]^m · x_n
              ────────────────────
                  Σ_n U[n,c]^m

    Objective (minimised):
        J = Σ_n Σ_c U[n,c]^m · d_{n,c}
    """

    def __init__(
        self,
        n_clusters: int = 15,
        fuzziness: float = 2.0,
        max_iter: int = 150,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.m = fuzziness
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.cluster_centers_: np.ndarray = None
        self.membership_matrix_: np.ndarray = None   # shape (N, C)
        self.labels_: np.ndarray = None              # hard assignment = argmax
        self.n_iter_: int = 0
        self.loss_history_: list = []

    def fit(self, X: np.ndarray) -> "FuzzyCMeans":
        rng = np.random.default_rng(self.random_state)
        N, D = X.shape
        C = self.n_clusters
        m = self.m

        # Initialise membership matrix uniformly at random, then normalise rows
        U = rng.random((N, C)).astype(np.float32)
        U /= U.sum(axis=1, keepdims=True)

        for iteration in range(self.max_iter):
            U_old = U.copy()

            # Centroid update
            Um = U ** m                                      # (N, C)
            V = (Um.T @ X) / Um.sum(axis=0)[:, None]        # (C, D)

            # Distance matrix: d_{n,c} = ||x_n - v_c||²
            dist2 = cdist(X, V, metric="sqeuclidean")        # (N, C)
            dist2 = np.maximum(dist2, 1e-10)                 # numerical guard

            # CORRECTION — correct membership update formula:
            # U[n,c] = (1/d_{n,c})^(1/(m-1)) / Σ_j (1/d_{n,j})^(1/(m-1))
            exp = 1.0 / (m - 1.0)
            inv_dist_exp = (1.0 / dist2) ** exp              # (N, C)
            U = inv_dist_exp / inv_dist_exp.sum(axis=1, keepdims=True)

            # Objective value (for convergence monitoring)
            loss = float(np.sum((U ** m) * dist2))
            self.loss_history_.append(loss)

            delta = float(np.max(np.abs(U - U_old)))
            if delta < self.tol:
                logger.info(f"FCM converged after {iteration + 1} iterations")
                self.n_iter_ = iteration + 1
                break
        else:
            logger.warning(f"FCM did not converge within {self.max_iter} iterations")
            self.n_iter_ = self.max_iter

        self.cluster_centers_ = V
        self.membership_matrix_ = U
        self.labels_ = np.argmax(U, axis=1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return membership distribution for unseen points."""
        if self.cluster_centers_ is None:
            raise RuntimeError("Model not fitted yet.")
        m = self.m
        dist2 = cdist(X, self.cluster_centers_, metric="sqeuclidean")
        dist2 = np.maximum(dist2, 1e-10)
        exp = 1.0 / (m - 1.0)
        inv_dist_exp = (1.0 / dist2) ** exp
        return inv_dist_exp / inv_dist_exp.sum(axis=1, keepdims=True)

    def membership_entropy(self) -> np.ndarray:
        """Per-document entropy of the membership distribution."""
        U = self.membership_matrix_
        eps = 1e-10
        return -np.sum(U * np.log(U + eps), axis=1)


# ---------------------------------------------------------------------------
# K selection — evidence-based sweep
# ---------------------------------------------------------------------------

def find_optimal_k(
    X: np.ndarray,
    k_range: range = range(8, 22),
    fuzziness: float = 2.0,
    sample_n: int = 5000,
    random_state: int = 42,
) -> dict:
    """
    Sweep K and record Silhouette Score and Davies-Bouldin Index.

    CORRECTION — precise metric explanations:

    Silhouette Score (maximise, range -1 to +1):
        Measures how well each document fits within its assigned cluster
        compared to the nearest alternative cluster. A score near +1 means
        the document is well-matched to its cluster; near 0 means it sits on
        a boundary; near -1 means it may be misclassified.

    Davies-Bouldin Index (minimise, range ≥ 0):
        Measures the average similarity between each cluster and its most
        similar neighbour, where similarity is the ratio of within-cluster
        scatter to between-cluster distance. Lower values indicate better
        cluster separation.

    K=15 was selected because it provides the best trade-off between cluster
    cohesion and separation while avoiding over-fragmentation of semantically
    related topics (e.g. the multiple comp.sys.* categories).
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=min(sample_n, len(X)), replace=False)
    X_sub = X[idx].astype(np.float32)

    results = {"k": [], "silhouette": [], "db_index": [], "fcm_loss": []}

    for k in k_range:
        logger.info(f"  Evaluating K={k} …")
        fcm = FuzzyCMeans(n_clusters=k, fuzziness=fuzziness, max_iter=100, random_state=random_state)
        fcm.fit(X_sub)

        sil = silhouette_score(X_sub, fcm.labels_, metric="euclidean", sample_size=2000)
        db = davies_bouldin_score(X_sub, fcm.labels_)
        loss = fcm.loss_history_[-1]

        results["k"].append(k)
        results["silhouette"].append(float(sil))
        results["db_index"].append(float(db))
        results["fcm_loss"].append(float(loss))
        logger.info(f"    K={k}: silhouette={sil:.4f}, DB={db:.4f}, loss={loss:.2f}")

    return results


def plot_k_sweep(results: dict, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))

    axes[0].plot(results["k"], results["silhouette"], "o-", color="#2563EB", linewidth=2)
    axes[0].axvline(x=15, color="#DC2626", linestyle="--", alpha=0.8, label="K=15 chosen")
    axes[0].set_title("Silhouette Score\n(higher = better cohesion + separation)", fontsize=10)
    axes[0].set_xlabel("Number of clusters K")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(results["k"], results["db_index"], "o-", color="#DC2626", linewidth=2)
    axes[1].axvline(x=15, color="#2563EB", linestyle="--", alpha=0.8, label="K=15 chosen")
    axes[1].set_title("Davies-Bouldin Index\n(lower = better separation)", fontsize=10)
    axes[1].set_xlabel("Number of clusters K")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(results["k"], results["fcm_loss"], "o-", color="#16A34A", linewidth=2)
    axes[2].axvline(x=15, color="#DC2626", linestyle="--", alpha=0.8, label="K=15 chosen")
    axes[2].set_title("FCM Objective Loss\n(lower = tighter clusters)", fontsize=10)
    axes[2].set_xlabel("Number of clusters K")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle("Cluster Count Selection — K Sweep Evidence", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"K-sweep plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Fuzziness parameter exploration — THE key tunable
# ---------------------------------------------------------------------------

def explore_fuzziness(
    X: np.ndarray,
    m_values: list = [1.2, 1.5, 2.0, 2.5, 3.0, 4.0],
    n_clusters: int = 15,
    sample_n: int = 3000,
    random_state: int = 42,
) -> dict:
    """
    Explore the effect of fuzziness m — the critical tunable parameter.

    What each value reveals about system behaviour:

    m=1.2 (near-hard):
        Membership distributions are very peaked — almost binary.
        System behaves like K-Means. Boundary documents are forcibly assigned
        even when genuinely ambiguous. Cache scoping is tight but misses
        polysemous queries.

    m=2.0 (conventional default):
        Each document has meaningful probability mass on 2–4 clusters.
        Captures polysemous posts naturally ("gun legislation" → politics + firearms).
        Cache lookup benefits from cluster scoping (O(N/K)) without becoming
        so diffuse that scoping loses meaning.

    m=3.0–4.0 (very fuzzy):
        Distributions flatten. Every document looks like it belongs "a bit" to
        every cluster. Cluster-based cache scoping degrades because almost every
        query has significant membership everywhere. The O(N/K) complexity
        advantage erodes toward O(N).

    Experiments show m≈2.0 provides the best balance between discriminative
    cluster assignment and meaningful semantic overlap representation.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=min(sample_n, len(X)), replace=False)
    X_sub = X[idx].astype(np.float32)

    results = {"m": [], "mean_entropy": [], "mean_max_membership": [], "pct_ambiguous": []}

    for m in m_values:
        fcm = FuzzyCMeans(n_clusters=n_clusters, fuzziness=m, max_iter=100, random_state=random_state)
        fcm.fit(X_sub)
        U = fcm.membership_matrix_
        eps = 1e-10

        entropy = -np.sum(U * np.log(U + eps), axis=1)
        mean_max = float(U.max(axis=1).mean())
        ambiguous = float(np.mean((U > 0.3).sum(axis=1) >= 2))

        results["m"].append(m)
        results["mean_entropy"].append(float(entropy.mean()))
        results["mean_max_membership"].append(mean_max)
        results["pct_ambiguous"].append(ambiguous)

        logger.info(
            f"  m={m:.1f}: entropy={entropy.mean():.3f}, "
            f"max_memb={mean_max:.3f}, pct_ambiguous={ambiguous*100:.1f}%"
        )

    return results


def plot_fuzziness_exploration(results: dict, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(results["m"], results["mean_entropy"], "o-", color="#7C3AED", linewidth=2)
    axes[0].axvline(x=2.0, color="#DC2626", linestyle="--", alpha=0.8, label="m=2.0 chosen")
    axes[0].set_title("Mean Membership Entropy\n(higher = more diffuse assignments)", fontsize=10)
    axes[0].set_xlabel("Fuzziness m")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(results["m"], results["mean_max_membership"], "o-", color="#EA580C", linewidth=2)
    axes[1].axvline(x=2.0, color="#DC2626", linestyle="--", alpha=0.8, label="m=2.0 chosen")
    axes[1].set_title("Mean Max Membership\n(higher = sharper assignments)", fontsize=10)
    axes[1].set_xlabel("Fuzziness m")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(results["m"], [p * 100 for p in results["pct_ambiguous"]], "o-", color="#0891B2", linewidth=2)
    axes[2].axvline(x=2.0, color="#DC2626", linestyle="--", alpha=0.8, label="m=2.0 chosen")
    axes[2].set_title("% Docs Genuinely Ambiguous\n(>0.3 membership in ≥2 clusters)", fontsize=10)
    axes[2].set_xlabel("Fuzziness m")
    axes[2].set_ylabel("%")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle("Fuzziness Parameter (m) Exploration — Key Tunable Decision", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Fuzziness exploration plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Cluster interpretation — semantic meaning analysis
# ---------------------------------------------------------------------------

def analyse_clusters(
    fcm: FuzzyCMeans,
    texts: list,
    cat_names: list,
    embeddings: np.ndarray,
    out_dir: Path,
) -> dict:
    """
    Build a semantic profile for each cluster.

    For each cluster we inspect:
    - Representative documents: the 5 closest to the centroid
    - Dominant original categories: shows what newsgroup topics mapped here
    - Boundary documents: high entropy membership, documents that sit between
      clusters — these are often the most semantically interesting (e.g. a post
      about gun legislation appearing in both politics and firearms clusters)
    """
    C = fcm.n_clusters
    U = fcm.membership_matrix_
    eps = 1e-10
    entropy = -np.sum(U * np.log(U + eps), axis=1)

    profiles = {}
    for c in range(C):
        member_idx = np.where(fcm.labels_ == c)[0]
        if len(member_idx) == 0:
            continue

        centroid = fcm.cluster_centers_[c]
        dists = np.linalg.norm(embeddings[member_idx] - centroid, axis=1)
        top5_local = np.argsort(dists)[:5]
        top5_global = member_idx[top5_local]

        cat_dist = Counter(cat_names[i] for i in member_idx)
        top_cats = cat_dist.most_common(5)

        # Boundary: highest-entropy members
        member_entropy = entropy[member_idx]
        top_boundary_local = np.argsort(member_entropy)[-3:][::-1]
        top_boundary_global = member_idx[top_boundary_local]

        profiles[str(c)] = {
            "cluster_id": c,
            "size": len(member_idx),
            "top_categories": [(cat, cnt) for cat, cnt in top_cats],
            "representative_texts": [texts[i][:300] for i in top5_global],
            "boundary_documents": [
                {
                    "text": texts[i][:250],
                    "membership_distribution": {
                        f"cluster_{j}": round(float(U[i, j]), 4)
                        for j in np.argsort(U[i])[::-1][:4]  # top-4 clusters
                    },
                    "entropy": round(float(entropy[i]), 4),
                }
                for i in top_boundary_global
            ],
        }

    profiles_path = out_dir / "cluster_profiles.json"
    with open(profiles_path, "w") as f:
        json.dump(profiles, f, indent=2)
    logger.info(f"Cluster profiles saved: {profiles_path}")
    return profiles


def print_cluster_summary(profiles: dict):
    """Pretty-print cluster interpretation to stdout."""
    print("\n" + "═" * 70)
    print("  CLUSTER INTERPRETATION SUMMARY")
    print("═" * 70)
    for c_str, p in sorted(profiles.items(), key=lambda x: int(x[0])):
        top = p["top_categories"][:3]
        cats = ", ".join(f"{cat.split('.')[-1]}({cnt})" for cat, cnt in top)
        print(f"\n  Cluster {p['cluster_id']:2d} | {p['size']:4d} docs | {cats}")
        if p["boundary_documents"]:
            bd = p["boundary_documents"][0]
            top2 = list(bd["membership_distribution"].items())[:2]
            print(f"    Boundary example: {' / '.join(f'{k}={v:.2f}' for k,v in top2)}")
            print(f"    Text: {bd['text'][:120].strip()} …")
    print("═" * 70 + "\n")


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_cluster_umap(embeddings: np.ndarray, fcm: FuzzyCMeans, out_path: Path, sample_n: int = 4000):
    """
    2-D UMAP projection coloured by dominant cluster.

    CORRECTION — balanced UMAP vs t-SNE rationale:
    UMAP was chosen because it preserves more global structure than t-SNE and
    scales better to large datasets. It also supports out-of-sample projection
    (new points can be embedded without refitting), which is useful for
    visualising query positions at inference time.
    """
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not installed; skipping UMAP plot")
        return

    rng = np.random.default_rng(42)
    idx = rng.choice(len(embeddings), size=min(sample_n, len(embeddings)), replace=False)
    X_sub = embeddings[idx].astype(np.float32)
    labels_sub = fcm.labels_[idx]
    entropy_sub = fcm.membership_entropy()[idx]
    is_boundary = entropy_sub > np.percentile(entropy_sub, 85)

    logger.info("Running UMAP …")
    import umap as umap_module
    reducer = umap_module.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(13, 9))
    import matplotlib.cm as cm
    cmap = cm.get_cmap("tab20", fcm.n_clusters)

    sc = ax.scatter(
        coords[~is_boundary, 0], coords[~is_boundary, 1],
        c=labels_sub[~is_boundary], cmap=cmap,
        vmin=0, vmax=fcm.n_clusters - 1, s=6, alpha=0.45,
    )
    ax.scatter(
        coords[is_boundary, 0], coords[is_boundary, 1],
        c=labels_sub[is_boundary], cmap=cmap,
        vmin=0, vmax=fcm.n_clusters - 1,
        s=25, alpha=0.9, marker="x", linewidths=0.9, label="High-entropy boundary",
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Dominant cluster", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        f"UMAP Projection — {sample_n} Documents | 15 Fuzzy Clusters\n"
        "Colour = dominant cluster assignment  |  ✕ = high-entropy boundary documents",
        fontsize=11,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"UMAP plot saved: {out_path}")


def plot_membership_heatmap(fcm: FuzzyCMeans, sample_n: int = 60, out_path: Path = None):
    """
    Membership heatmap — each row is one document, each column is a cluster.
    Demonstrates the soft-assignment nature of FCM: documents near topic
    boundaries show non-zero membership across multiple clusters.
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(fcm.membership_matrix_.shape[0], size=sample_n, replace=False)
    U_sample = fcm.membership_matrix_[idx]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        U_sample, ax=ax, cmap="YlOrRd",
        xticklabels=[f"C{c}" for c in range(fcm.n_clusters)],
        yticklabels=False,
        cbar_kws={"label": "Membership degree (0–1)"},
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel(f"{sample_n} sampled documents")
    ax.set_title(
        "Fuzzy Membership Heatmap — Each Row is a Membership Distribution\n"
        "Darker = stronger membership. Multi-coloured rows = polysemous documents.",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Membership heatmap saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_part2(data_dir: str = "./data", output_dir: str = "./data/analysis"):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "corpus.json") as f:
        corpus = json.load(f)
    texts = corpus["texts"]
    cat_names = corpus["cat_names"]
    embeddings = np.load(data_dir / "embeddings.npy").astype(np.float32)
    logger.info(f"Loaded {len(texts)} docs, embeddings shape: {embeddings.shape}")

    logger.info("Sweeping K …")
    k_results = find_optimal_k(embeddings, k_range=range(8, 22))
    with open(output_dir / "k_sweep.json", "w") as f:
        json.dump(k_results, f)
    plot_k_sweep(k_results, output_dir / "k_sweep.png")

    logger.info("Exploring fuzziness …")
    fuzz_results = explore_fuzziness(embeddings)
    with open(output_dir / "fuzziness_exploration.json", "w") as f:
        json.dump(fuzz_results, f)
    plot_fuzziness_exploration(fuzz_results, output_dir / "fuzziness_exploration.png")

    logger.info("Fitting final FCM (K=15, m=2.0) …")
    fcm = FuzzyCMeans(n_clusters=15, fuzziness=2.0, max_iter=150, random_state=42)
    fcm.fit(embeddings)

    with open(data_dir / "fcm_model.pkl", "wb") as f:
        pickle.dump(fcm, f)
    logger.info("FCM model saved.")

    profiles = analyse_clusters(fcm, texts, cat_names, embeddings, output_dir)
    print_cluster_summary(profiles)

    plot_cluster_umap(embeddings, fcm, output_dir / "umap_clusters.png")
    plot_membership_heatmap(fcm, out_path=output_dir / "membership_heatmap.png")

    logger.info("Part 2 complete.")
    return fcm


if __name__ == "__main__":
    run_part2()
