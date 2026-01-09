#!/usr/bin/env python
"""
Task 1 analysis script for Seminar Session 3.

This script:
1. Loads the provided data_Kclusters.txt dataset.
2. Computes the Elbow curve for a range of k values.
3. Estimates the optimal k via a simple "farthest point" elbow heuristic.
4. Fits a final K-Means model with the selected k and stores the cluster labels.
5. Produces visualisations with clear headings and stores them in the answers folder.

All generated artefacts stay inside week3/answers as requested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Paths
PROJECT_ROOT = Path("/Users/abdulaaqib/Developer/ML CW Code")
DATA_PATH = PROJECT_ROOT / "week3" / "data_Kclusters.txt"
OUTPUT_DIR = PROJECT_ROOT / "week3" / "answers"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> np.ndarray:
    """Load the 2D dataset."""
    return np.loadtxt(path)


def compute_wcss(
    data: np.ndarray, k_values: Iterable[int], random_state: int = 42
) -> List[float]:
    """Calculate within-cluster sums of squares for each k."""
    wcss = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        model.fit(data)
        wcss.append(model.inertia_)
    return wcss


def compute_silhouette_scores(
    data: np.ndarray, k_values: Iterable[int], random_state: int = 42
) -> List[float]:
    """Silhouette scores offer supporting evidence for the best k."""
    scores = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append(score)
    return scores


def estimate_elbow_k(k_values: List[int], wcss: List[float]) -> int:
    """
    Estimate the elbow by finding the point farthest from the line connecting
    (k_min, wcss_min) and (k_max, wcss_max).
    """
    if len(k_values) != len(wcss):
        raise ValueError("k_values and wcss must be the same length.")

    ks = np.array(k_values, dtype=float)
    inertias = np.array(wcss, dtype=float)

    line_start = np.array([ks[0], inertias[0]])
    line_end = np.array([ks[-1], inertias[-1]])
    line_vec = line_end - line_start
    line_vec /= np.linalg.norm(line_vec)

    distances = []
    for k, inertia in zip(ks, inertias):
        point = np.array([k, inertia])
        vec = point - line_start
        # Equivalent to magnitude of 2D cross-product via determinant.
        distance = abs(line_vec[0] * vec[1] - line_vec[1] * vec[0])
        distances.append(distance)

    best_index = int(np.argmax(distances))
    return int(ks[best_index])


def save_elbow_plot(k_values: List[int], wcss: List[float], best_k: int) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, wcss, "bo-", linewidth=2, markersize=6)
    plt.axvline(best_k, color="orange", linestyle="--", label=f"Selected k = {best_k}")
    plt.title("Task 1 – Elbow Method", fontsize=14)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Within-cluster sum of squares (WCSS)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task1_elbow_plot.png", dpi=300)
    plt.close()


def save_silhouette_plot(k_values: List[int], scores: List[float]) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, scores, "gs-", linewidth=2, markersize=6)
    plt.title("Task 1 – Silhouette Scores", fontsize=14)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Mean silhouette score")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task1_silhouette_plot.png", dpi=300)
    plt.close()


def save_cluster_plot(data: np.ndarray, labels: np.ndarray, centers: np.ndarray, k: int):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10", s=20, alpha=0.8)
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="*",
        c="black",
        s=400,
        edgecolor="white",
        linewidth=1.5,
        label="Centroids",
    )
    plt.title(f"Task 1 – K-Means Clusters (k = {k})", fontsize=14)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task1_cluster_plot.png", dpi=300)
    plt.close()


def main() -> None:
    data = load_data(DATA_PATH)

    k_values = list(range(2, 13))
    wcss = compute_wcss(data, k_values)
    best_k = estimate_elbow_k(k_values, wcss)

    silhouette_scores = compute_silhouette_scores(data, k_values)

    save_elbow_plot(k_values, wcss, best_k)
    save_silhouette_plot(k_values, silhouette_scores)

    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels = final_model.fit_predict(data)
    centers = final_model.cluster_centers_
    save_cluster_plot(data, labels, centers, best_k)

    # Persist assignments for traceability.
    assignment_path = OUTPUT_DIR / "task1_cluster_assignments.csv"
    np.savetxt(
        assignment_path,
        np.column_stack((data, labels)),
        delimiter=",",
        fmt="%.6f",
        header="feature_1,feature_2,cluster",
        comments="",
    )

    # Store metrics to reference in the report.
    metrics_path = OUTPUT_DIR / "task1_metrics.txt"
    with metrics_path.open("w") as f:
        f.write(f"Selected k (elbow heuristic): {best_k}\n")
        f.write("WCSS values by k:\n")
        for k, val in zip(k_values, wcss):
            f.write(f"  k={k}: {val:.2f}\n")
        f.write("\nSilhouette scores by k:\n")
        for k, val in zip(k_values, silhouette_scores):
            f.write(f"  k={k}: {val:.4f}\n")


if __name__ == "__main__":
    main()
