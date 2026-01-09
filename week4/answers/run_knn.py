"""
Week4 Seminar: K-NN Classification and Performance Metrics
-----------------------------------------------------------
This script loads the classification dataset, trains K-NN classifiers for
different values of K (3, 5, 7), computes performance metrics, and generates
confusion matrix and ROC curve plots.
"""

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

# =============================================================================
# Configuration
# =============================================================================
DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data_Seminar-Supervised-Classification-K-NN-Performance-Metrics.txt",
)
OUTPUT_DIR = os.path.dirname(__file__)

K_VALUES = [3, 5, 7]  # Different values of K to evaluate
TEST_SIZE = 0.2
RANDOM_STATE = 2

# =============================================================================
# Helpers
# =============================================================================


def plot_confusion_matrix(
    cm,
    classes,
    title="Confusion Matrix",
    cmap=plt.cm.Blues,
    save_path=None,
):
    """Plot and optionally save a confusion matrix."""
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_roc_curve(fpr, tpr, auc_value, k, save_path=None):
    """Plot and optionally save ROC curve."""
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (AUC = {auc_value:.2f})",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (K={k})")
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_data_scatter(X, y, save_path=None):
    """Plot scatter of features colored by class."""
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, marker="*", cmap="viridis")
    plt.title("Dataset Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main workflow
# =============================================================================


def run_knn_analysis():
    # Load data
    DATA = np.loadtxt(DATA_FILE)
    X = DATA[:, 0:2]
    y = DATA[:, 2]

    # Visualize data
    plot_data_scatter(X, y, save_path=os.path.join(OUTPUT_DIR, "data_scatter.png"))

    # Split into train/test
    trainX, testX, trainy, testy = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = []
    class_names = ["Negative", "Positive"]

    for k in K_VALUES:
        print(f"\n{'='*50}")
        print(f"K-NN with K = {k}")
        print("=" * 50)

        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(trainX, trainy)

        y_pred = model.predict(testX)
        cnf_matrix = confusion_matrix(testy, y_pred)

        TP = cnf_matrix[0, 0]
        FP = cnf_matrix[0, 1]
        FN = cnf_matrix[1, 0]
        TN = cnf_matrix[1, 1]

        # Metrics
        sensitivity = TP / (TP + FN)  # TPR / Recall for class 0
        specificity = TN / (TN + FP)  # TNR
        precision = TP / (TP + FP)  # PPV
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        f1 = f1_score(testy, y_pred, pos_label=1)

        # Probabilities for ROC
        probs = model.predict_proba(testX)[:, 1]
        fpr, tpr, _ = roc_curve(testy, probs)
        auc_value = roc_auc_score(testy, probs)

        print(f"TP = {TP}, FP = {FP}, FN = {FN}, TN = {TN}")
        print(f"Sensitivity (Recall) = {sensitivity:.2f}")
        print(f"Specificity = {specificity:.2f}")
        print(f"Precision = {precision:.2f}")
        print(f"Accuracy = {accuracy:.2f}")
        print(f"F1-Score = {f1:.2f}")
        print(f"AUC = {auc_value:.3f}")

        # Save confusion matrix plot
        cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_k{k}.png")
        plot_confusion_matrix(
            cnf_matrix,
            classes=class_names,
            title=f"Confusion Matrix (K={k})",
            save_path=cm_path,
        )

        # Save ROC curve plot
        roc_path = os.path.join(OUTPUT_DIR, f"roc_curve_k{k}.png")
        plot_roc_curve(fpr, tpr, auc_value, k, save_path=roc_path)

        results.append(
            {
                "k": k,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "TN": TN,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "precision": precision,
                "accuracy": accuracy,
                "f1_score": f1,
                "auc": auc_value,
            }
        )

    return results


if __name__ == "__main__":
    results = run_knn_analysis()
    print("\n\n===== SUMMARY =====")
    for r in results:
        print(
            f"K={r['k']}: AUC={r['auc']:.3f}, F1={r['f1_score']:.2f}, "
            f"Accuracy={r['accuracy']:.2f}, Sensitivity={r['sensitivity']:.2f}, "
            f"Specificity={r['specificity']:.2f}"
        )
