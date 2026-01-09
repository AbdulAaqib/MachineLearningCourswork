import json
import csv
from pathlib import Path
from itertools import product

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# Follow the notebook: same split and parameters where applicable
TEST_SIZE = 0.25
SPLIT_RANDOM_STATE = 3
MLP_RANDOM_STATE = 1
MAX_ITER = 10000
LEARNING_RATE = 'adaptive'


def arch_to_str(arch):
    if isinstance(arch, int):
        return f"({arch},)"
    return str(tuple(arch))


def run_experiment(architectures):
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_RANDOM_STATE
    )

    results = []

    for arch in architectures:
        mlp = MLPClassifier(
            hidden_layer_sizes=arch,
            max_iter=MAX_ITER,
            learning_rate=LEARNING_RATE,
            random_state=MLP_RANDOM_STATE,
        )
        mlp.fit(X_train, y_train)

        y_pred_train = mlp.predict(X_train)
        y_pred_test = mlp.predict(X_test)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)

        results.append({
            'hidden_layer_sizes': arch_to_str(arch),
            'n_hidden_layers': len(arch) if not isinstance(arch, int) else 1,
            'total_hidden_neurons': int(np.sum(arch if not isinstance(arch, int) else (arch,))),
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'n_iter_': getattr(mlp, 'n_iter_', None),
            'loss_': getattr(mlp, 'loss_', None),
        })

    return results


def main():
    # A curated set of architectures to try
    single = [(2,), (4,), (8,), (16,), (32,), (64,)]
    double = [(4, 4), (8, 4), (8, 8), (16, 8), (16, 16), (32, 16)]
    triple = [(4, 4, 2), (8, 4, 2), (16, 8, 4), (32, 16, 8)]

    architectures = single + double + triple

    results = run_experiment(architectures)

    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / 'mlp_results.csv'
    json_path = out_dir / 'mlp_results.json'

    fieldnames = list(results[0].keys()) if results else [
        'hidden_layer_sizes', 'n_hidden_layers', 'total_hidden_neurons', 'train_accuracy', 'test_accuracy', 'n_iter_', 'loss_'
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Also print a concise summary
    best = max(results, key=lambda r: r['test_accuracy']) if results else None
    if best:
        print('Best architecture by test accuracy:', best['hidden_layer_sizes'])
        print('Test accuracy:', best['test_accuracy'])
        print('Train accuracy:', best['train_accuracy'])

    # Check if any achieved 1.0 on test
    any_perfect = any(abs(r['test_accuracy'] - 1.0) < 1e-12 for r in results)
    print('Any architecture achieved 100% test accuracy:', any_perfect)


if __name__ == '__main__':
    main()
