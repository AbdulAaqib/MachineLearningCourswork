"""
Random Forest experiments for Seminar Session-05.

Reuses the workflow from `week5/Random_Forest_Classifier.ipynb`:
    - loads the iris dataset
    - creates the same random train/test split
    - trains RandomForestClassifier models with varying `max_features`
    - writes confusion-matrix plots (with headings) plus an accuracy summary
Outputs live under week5/answer as requested.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

# Headless backend so matplotlib can write files without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def prepare_data() -> Dict[str, pd.DataFrame]:
    """Load iris data, mimic the seminar train/test split, and encode labels."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

    np.random.seed(0)
    df["is_train"] = np.random.uniform(0, 1, len(df)) <= 0.75
    train = df[df["is_train"]].copy()
    test = df[~df["is_train"]].copy()

    features = df.columns[:4]
    y_train = pd.factorize(train["species"])[0]
    y_test = pd.Categorical(
        test["species"], categories=iris.target_names, ordered=True
    ).codes

    return {
        "iris": iris,
        "train": train,
        "test": test,
        "features": features,
        "y_train": y_train,
        "y_test": y_test,
    }


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: List[str],
    output_path: Path,
    max_features: int,
) -> None:
    """Write a confusion-matrix heatmap with a simple heading."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(f"Confusion Matrix (max_features = {max_features})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    assets = prepare_data()
    iris = assets["iris"]
    train = assets["train"]
    test = assets["test"]
    features = assets["features"]
    y_train = assets["y_train"]
    y_test = assets["y_test"]

    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_values = [2, 3, 4]
    experiment_results = []

    for max_feat in experiment_values:
        clf = RandomForestClassifier(
            max_features=max_feat,
            random_state=0,
        )
        clf.fit(train[features], y_train)
        test_preds = clf.predict(test[features])

        accuracy = accuracy_score(y_test, test_preds)
        cm = confusion_matrix(y_test, test_preds, labels=[0, 1, 2])

        plot_path = output_dir / f"confusion_matrix_max_features_{max_feat}.png"
        plot_confusion_matrix(
            cm,
            labels=list(iris.target_names),
            output_path=plot_path,
            max_features=max_feat,
        )

        experiment_results.append(
            {
                "max_features": max_feat,
                "accuracy": round(float(accuracy), 4),
                "confusion_matrix": cm.tolist(),
                "plot_path": plot_path.name,
            }
        )

    # Improved bar chart: start y-axis at 0.92, annotate bars with accuracy.
    plt.figure(figsize=(6, 4))
    bars = plt.bar(
        [str(res["max_features"]) for res in experiment_results],
        [res["accuracy"] for res in experiment_results],
        color="#1f77b4",
    )
    plt.ylim(0.92, 1.00)
    plt.title("Accuracy vs. Selected Feature Count")
    plt.xlabel("max_features value")
    plt.ylabel("Accuracy on held-out test split")
    # Add text labels (accuracies) above each bar
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.002,
                 f"{height:.4f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    accuracy_chart_path = output_dir / "accuracy_comparison.png"
    plt.savefig(accuracy_chart_path)
    plt.close()

    summary_table = pd.DataFrame(experiment_results)
    summary_csv_path = output_dir / "random_forest_accuracy_summary.csv"
    summary_table.to_csv(summary_csv_path, index=False)

    summary_json_path = output_dir / "random_forest_accuracy_summary.json"
    summary_json_path.write_text(json.dumps(experiment_results, indent=2))

    print("Experiments complete.")
    print(summary_table.to_string(index=False))
    print(f"\nSaved summary CSV to {summary_csv_path.name}")
    print(f"Saved accuracy chart to {accuracy_chart_path.name}")


if __name__ == "__main__":
    main()
