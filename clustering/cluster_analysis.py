"""
clustering/cluster_analysis.py
================================
Cluster profiling, interpretation, and visualisation.
Imports FuzzyCMeans from fuzzy_cmeans and produces per-cluster profiles.
"""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def analyse_clusters(fcm, texts: list, cat_names: list, embeddings: np.ndarray, out_dir: Path) -> dict:
    """
    Build a semantic profile for each cluster.

    For each cluster we inspect:
    - Representative documents: 5 closest to the centroid
    - Dominant original categories: which newsgroup topics mapped here
    - Boundary documents: high entropy — posts genuinely belonging to
      multiple topics (e.g. gun legislation → politics + firearms)
    """
    U = fcm.membership_matrix_
    eps = 1e-10
    entropy = -np.sum(U * np.log(U + eps), axis=1)

    profiles = {}
    for c in range(fcm.n_clusters):
        member_idx = np.where(fcm.labels_ == c)[0]
        if len(member_idx) == 0:
            continue

        dists = np.linalg.norm(embeddings[member_idx] - fcm.cluster_centers_[c], axis=1)
        top5_global = member_idx[np.argsort(dists)[:5]]
        cat_dist = Counter(cat_names[i] for i in member_idx)
        top_boundary = member_idx[np.argsort(entropy[member_idx])[-3:][::-1]]

        profiles[str(c)] = {
            "cluster_id": c,
            "size": len(member_idx),
            "top_categories": cat_dist.most_common(5),
            "representative_texts": [texts[i][:300] for i in top5_global],
            "boundary_documents": [
                {
                    "text": texts[i][:250],
                    "membership_distribution": {
                        f"cluster_{j}": round(float(U[i, j]), 4)
                        for j in np.argsort(U[i])[::-1][:4]
                    },
                    "entropy": round(float(entropy[i]), 4),
                }
                for i in top_boundary
            ],
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cluster_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)
    logger.info(f"Cluster profiles saved.")
    return profiles


def print_cluster_summary(profiles: dict):
    print("\n" + "=" * 70)
    print("  CLUSTER INTERPRETATION SUMMARY")
    print("=" * 70)
    for c_str, p in sorted(profiles.items(), key=lambda x: int(x[0])):
        top = p["top_categories"][:3]
        cats = ", ".join(f"{cat.split('.')[-1]}({cnt})" for cat, cnt in top)
        print(f"\n  Cluster {p['cluster_id']:2d} | {p['size']:4d} docs | {cats}")
        if p["boundary_documents"]:
            bd = p["boundary_documents"][0]
            top2 = list(bd["membership_distribution"].items())[:2]
            print(f"    Boundary: {' / '.join(f'{k}={v:.2f}' for k, v in top2)}")
    print("=" * 70 + "\n")


def plot_cluster_umap(embeddings: np.ndarray, fcm, out_path: Path, sample_n: int = 4000):
    """
    2D UMAP projection. UMAP was chosen because it preserves more global
    structure than t-SNE and scales better to large datasets.
    """
    try:
        import umap as umap_module
        import matplotlib.cm as cm
    except ImportError:
        logger.warning("umap-learn not installed; skipping UMAP plot")
        return

    rng = np.random.default_rng(42)
    idx = rng.choice(len(embeddings), size=min(sample_n, len(embeddings)), replace=False)
    X_sub = embeddings[idx].astype(np.float32)
    labels_sub = fcm.labels_[idx]
    U_sub = fcm.membership_matrix_[idx]
    eps = 1e-10
    entropy = -np.sum(U_sub * np.log(U_sub + eps), axis=1)
    is_boundary = entropy > np.percentile(entropy, 85)

    logger.info("Running UMAP ...")
    coords = umap_module.UMAP(n_components=2, random_state=42).fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(13, 9))
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
    plt.colorbar(sc, ax=ax, label="Dominant cluster")
    ax.legend(loc="upper right")
    ax.set_title(
        f"UMAP Projection | {sample_n} Documents | 15 Fuzzy Clusters\n"
        "Colour = dominant cluster  |  x = high-entropy boundary documents", fontsize=11,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"UMAP plot saved: {out_path}")


def plot_membership_heatmap(fcm, sample_n: int = 60, out_path: Path = None):
    """Each row = one document. Multi-coloured rows = polysemous documents."""
    rng = np.random.default_rng(0)
    idx = rng.choice(fcm.membership_matrix_.shape[0], size=sample_n, replace=False)
    U_sample = fcm.membership_matrix_[idx]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        U_sample, ax=ax, cmap="YlOrRd",
        xticklabels=[f"C{c}" for c in range(fcm.n_clusters)],
        yticklabels=False,
        cbar_kws={"label": "Membership degree (0-1)"},
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel(f"{sample_n} sampled documents")
    ax.set_title(
        "Fuzzy Membership Heatmap\n"
        "Multi-coloured rows = polysemous documents belonging to multiple clusters",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Membership heatmap saved: {out_path}")
