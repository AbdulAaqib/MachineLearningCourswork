#!/usr/bin/env python3
import os
import json
import argparse
from datetime import datetime
from typing import List

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # for headless save
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from data_utils import load_cifar10
from knn import KNearestNeighbor, cross_validate_k, choose_best_k


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_cv_plot(k_to_accuracies: dict, out_path: str):
    if not HAS_MPL:
        print("matplotlib not available; skipping CV plot save")
        return
    k_choices = sorted(k_to_accuracies.keys())
    # scatter each fold point
    for k in k_choices:
        accs = k_to_accuracies[k]
        plt.scatter([k] * len(accs), accs, label=f"k={k}")
    # trend line with std error bars
    means = np.array([np.mean(k_to_accuracies[k]) for k in k_choices])
    stds = np.array([np.std(k_to_accuracies[k]) for k in k_choices])
    plt.errorbar(k_choices, means, yerr=stds, fmt='-o', color='black')
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run week9 kNN tasks: CV to pick k, train on half data, test on all test.")
    parser.add_argument('--data-dir', type=str, default=os.path.join('week9', 'answers', 'data'), help='Directory to download/extract CIFAR-10')
    parser.add_argument('--download', action='store_true', help='Download CIFAR-10 if missing')
    parser.add_argument('--cv-num-folds', type=int, default=5)
    parser.add_argument('--cv-k-choices', type=int, nargs='+', default=[1, 3, 5, 7, 9, 11, 13, 15])
    parser.add_argument('--cv-batch-size', type=int, default=200)
    parser.add_argument('--cv-use-train-prefix', type=int, default=5000, help='Use only first N train samples for CV to keep it fast (set to 50000 for full)')
    parser.add_argument('--final-batch-size', type=int, default=100, help='Batch size for final test prediction')
    parser.add_argument('--final-seed', type=int, default=42, help='Random seed for selecting half training data')
    parser.add_argument('--answers-file', type=str, default=os.path.join('week9', 'answers', 'answers.txt'))
    parser.add_argument('--artifacts-dir', type=str, default=os.path.join('week9', 'answers'))

    args = parser.parse_args()

    ensure_dir(args.data_dir)
    ensure_dir(args.artifacts_dir)
    plots_dir = os.path.join(args.artifacts_dir, 'plots')
    ensure_dir(plots_dir)

    print("Loading CIFAR-10 ...")
    X_train, y_train, X_test, y_test = load_cifar10(args.data_dir, download_if_missing=args.download)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Optional: use a subset for CV to keep it quick
    if args.cv_use_train_prefix and args.cv_use_train_prefix < X_train.shape[0]:
        N = args.cv_use_train_prefix
        # Ensure divisibility for folds
        num_folds = args.cv_num_folds
        N_fold = N // num_folds
        N = N_fold * num_folds
        X_cv = X_train[:N]
        y_cv = y_train[:N]
        print(f"Cross-validation on subset: {X_cv.shape[0]} samples across {num_folds} folds ...")
    else:
        # Ensure divisibility by trimming the end if needed
        N = (X_train.shape[0] // args.cv_num_folds) * args.cv_num_folds
        X_cv = X_train[:N]
        y_cv = y_train[:N]
        print(f"Cross-validation on full-train subset of size {N} (divisible by folds)")

    k_to_accuracies = cross_validate_k(
        X_cv, y_cv,
        k_choices=args.cv_k_choices,
        num_folds=args.cv_num_folds,
        batch_size=args.cv_batch_size,
    )

    best_k, best_mean, best_std = choose_best_k(k_to_accuracies)
    print(f"Best k from CV: k={best_k} with mean acc={best_mean:.4f} +/- {best_std:.4f}")

    cv_plot_path = os.path.join(plots_dir, 'cv_k_plot.png')
    save_cv_plot(k_to_accuracies, cv_plot_path)

    # Final evaluation: train on half the training data, test on all test data
    rng = np.random.default_rng(args.final_seed)
    n_train = X_train.shape[0]
    half_n = n_train // 2
    half_indices = rng.choice(n_train, size=half_n, replace=False)
    X_half = X_train[half_indices]
    y_half = y_train[half_indices]

    clf = KNearestNeighbor()
    clf.train(X_half, y_half)
    print(f"Predicting on full test set with k={best_k}, train size={X_half.shape[0]} ...")
    y_pred = clf.predict(X_test, k=best_k, batch_size=args.final_batch_size)
    test_acc = float(np.mean(y_pred == y_test))
    print(f"Test accuracy (half train, full test): {test_acc:.4f}")

    # Save artifacts
    results = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'cv': {
            'num_folds': args.cv_num_folds,
            'k_choices': args.cv_k_choices,
            'k_to_accuracies': {int(k): [float(a) for a in accs] for k, accs in k_to_accuracies.items()},
            'best_k': int(best_k),
            'best_mean_acc': float(best_mean),
            'best_std_acc': float(best_std),
            'cv_plot_path': cv_plot_path,
            'cv_train_size': int(X_cv.shape[0]),
        },
        'final_eval': {
            'train_size': int(X_half.shape[0]),
            'test_size': int(X_test.shape[0]),
            'k': int(best_k),
            'test_accuracy': float(test_acc),
        }
    }

    results_path_json = os.path.join(args.artifacts_dir, 'results.json')
    with open(results_path_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path_json}")

    # Write/overwrite answers.txt per instructions
    answers_lines: List[str] = []
    answers_lines.append("University of West London - Machine Learning - Week 9")
    answers_lines.append("Task: Image Classification using k-NN (CIFAR-10)")
    answers_lines.append("")
    answers_lines.append("Task 1")
    answers_lines.append("Based on the cross-validation results, we select the value of k that maximizes the mean cross-validation accuracy across folds (ties broken by the smaller k). We then retrain (store) the classifier using half of the CIFAR-10 training data and evaluate on the full test set.")
    answers_lines.append("")
    answers_lines.append(f"Chosen k: {best_k}")
    answers_lines.append(f"Cross-validation mean accuracy: {best_mean:.4f} +/- {best_std:.4f}")
    answers_lines.append(f"Train size for final evaluation: {X_half.shape[0]} (half of {n_train})")
    answers_lines.append(f"Test size: {X_test.shape[0]}")
    answers_lines.append(f"Test accuracy with k={best_k}: {test_acc:.4f}")
    answers_lines.append("")
    answers_lines.append("Justification:")
    answers_lines.append("We evaluated a range of k values using k-fold cross-validation, following the provided notebook structure (splitting data into folds, training on k-1 folds and validating on the remaining fold). The selected k achieved the highest mean validation accuracy, indicating the best generalization among the tested values. Finally, we trained on half of the available training data to match the task requirement and reported the accuracy on the entire test set.")
    answers_text = "\n".join(answers_lines) + "\n"

    with open(args.answers_file, 'w') as f:
        f.write(answers_text)
    print(f"Wrote answers to {args.answers_file}")


if __name__ == '__main__':
    main()
