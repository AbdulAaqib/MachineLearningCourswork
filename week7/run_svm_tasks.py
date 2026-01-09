import os
# Ensure a writable MPL config dir and non-interactive backend
from pathlib import Path
MPL_DIR = Path(__file__).resolve().parent / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
import matplotlib
matplotlib.use("Agg")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict

from sklearn.svm import SVC

# Control whether to write numeric data results (CSV/JSON)
SAVE_DATA_RESULTS = False

# -------------------------
# Utils
# -------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_data_svm(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    # data_SVM.txt contains 3 columns: x1, x2, y (0/1)
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["x1", "x2", "y"])
    X = df[["x1", "x2"]].values
    y = df["y"].astype(int).values
    return X, y


def load_task2_data(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    # task2data.txt has header and y in {-1, +1}
    df = pd.read_csv(path, sep=r"\s+", header=0)
    # Normalize y to integers -1/+1
    y = df["y"].replace({"+1": 1, "+": 1}).astype(int).values if df["y"].dtype == object else df["y"].astype(int).values
    X = df[["x1", "x2"]].values
    return X, y


# -------------------------
# Notebook-style decision function plotting
# -------------------------

def plot_svc_decision_function(model: SVC, ax=None, plot_support: bool = False):
    """Plot the decision function for a 2D SVC (notebook-style)."""
    if ax is None:
        ax = plt.gca()

    # get current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 200)
    yy = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(xx, yy)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support and hasattr(model, 'support_vectors_'):
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
                   linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# -------------------------
# Task 1
# -------------------------

def run_task1(data_path: Path, out_dir: Path, Cs: List[float]) -> Dict[str, float]:
    X, y = load_data_svm(data_path)

    results = {}
    for C in Cs:
        model = SVC(kernel='linear', C=C, random_state=42)
        model.fit(X, y)
        yhat = model.predict(X)
        err = float(np.mean(y != yhat)) * 100.0
        results[str(C)] = err

        # Notebook-style combined plot: decision boundary and misclassified points
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        # Left: data colored by class
        ax[0].scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
        plot_svc_decision_function(model, ax=ax[0], plot_support=True)
        ax[0].set_title('C = {0:.3f}'.format(C), size=14)
        ax[0].set_xlabel('x1')
        ax[0].set_ylabel('x2')

        # Right: misclassified points highlighted via color map
        miscl = (y != yhat).astype(int)
        ax[1].scatter(X[:, 0], X[:, 1], c=miscl, s=50, cmap='autumn')
        plot_svc_decision_function(model, ax=ax[1], plot_support=True)
        ax[1].set_title('misclassified points', fontsize=14)
        ax[1].set_xlabel('x1')
        ax[1].set_ylabel('x2')

        plt.tight_layout()
        combined_path = out_dir / f"task1_svm_C_{C}_notebook_style.png"
        plt.savefig(combined_path, dpi=150)
        plt.close(fig)

    # Optionally save results json and csv
    if SAVE_DATA_RESULTS:
        with open(out_dir / "task1_svm_results.json", "w") as f:
            json.dump(results, f, indent=2)
        pd.DataFrame([
            {"C": float(c), "misclassification_percent": results[c]} for c in results
        ]).sort_values("C").to_csv(out_dir / "task1_svm_results.csv", index=False)

    return results


# -------------------------
# Task 2
# -------------------------

def run_task2(data_path: Path, out_dir: Path) -> Dict[str, float]:
    X, y = load_task2_data(data_path)

    model = SVC(kernel='linear', C=1.0, random_state=42)
    model.fit(X, y)
    yhat = model.predict(X)
    err = float(np.mean(y != yhat)) * 100.0

    w = model.coef_[0]
    b = model.intercept_[0]

    # Notebook-style single plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(model, ax=ax, plot_support=True)
    ax.set_title("Task 2: Linear SVM", fontsize=14)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.tight_layout()
    plt.savefig(out_dir / "task2_linear_svm.png", dpi=150)
    plt.close(fig)

    return {
        "w1": float(w[0]),
        "w2": float(w[1]),
        "b": float(b),
        "train_misclassification_percent": float(err)
    }


# -------------------------
# Write answers
# -------------------------

def write_answers(answer_path: Path, task1_results: Dict[str, float], task2_params: Dict[str, float], perceptron_miscls: float) -> None:
    lines = []
    lines.append("Support Vector Machines - Week 7 Answers")
    lines.append("==================================================")
    lines.append("")

    # Task 1
    lines.append("Task 1: SVM classification on data_SVM.txt")
    lines.append("")
    for c, err in sorted(((float(k), v) for k, v in task1_results.items()), key=lambda t: t[0]):
        lines.append(f"- C={c:.2f}: misclassification = {err:.2f}%")
    lines.append("")
    lines.append(f"Comparison to Perceptron (Week 6): {perceptron_miscls:.2f}% misclassified")
    lines.append("")
    lines.append("Brief discussion of C:")
    lines.append("- Smaller C (stronger regularization) -> wider margin, more tolerance for misclassification, usually a higher training error.")
    lines.append("- Larger C (weaker regularization) -> narrower margin, penalizes misclassifications more, often lower training error but potential overfitting.")
    lines.append("")

    # Task 2
    lines.append("Task 2: Linear SVM parameters and decision boundary")
    lines.append("")
    lines.append(f"- Learned parameters: w1 = {task2_params['w1']:.4f}, w2 = {task2_params['w2']:.4f}, b = {task2_params['b']:.4f}")
    lines.append(f"- Training misclassification: {task2_params['train_misclassification_percent']:.2f}%")
    lines.append("- Decision boundary and support vectors are shown in 'task2_linear_svm.png'.")
    lines.append("")

    ensure_dir(answer_path.parent)
    answer_path.write_text("\n".join(lines))


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    out_dir = root

    # Paths
    data1 = root / "data_SVM.txt"
    data2 = root / "task2data.txt"
    answer_path = root / "answer" / "answer.txt"

    ensure_dir(root)
    ensure_dir(answer_path.parent)

    # Task 1
    Cs = [0.01, 0.1, 1.0]
    task1_results = run_task1(data1, out_dir, Cs)

    # Task 2
    task2_params = run_task2(data2, out_dir)

    # Perceptron reference from week6 (given in brief)
    perceptron_miscls = 6.00

    # Write answers
    write_answers(answer_path, task1_results, task2_params, perceptron_miscls)

    print("Done. Outputs written to:")
    print(f"- {out_dir}")
    print(f"- {answer_path}")
