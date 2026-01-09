import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless backend for file output
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def load_results(csv_path: Path):
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # cast numeric fields
            r['test_accuracy'] = float(r['test_accuracy'])
            r['train_accuracy'] = float(r['train_accuracy'])
            r['n_hidden_layers'] = int(r['n_hidden_layers'])
            r['total_hidden_neurons'] = int(r['total_hidden_neurons'])
            rows.append(r)
    return rows


def plot_arch_vs_test_accuracy(rows, out_path: Path):
    # Keep the order as in the CSV
    labels = [r['hidden_layer_sizes'] for r in rows]
    test_accs = [r['test_accuracy'] for r in rows]
    n_layers = [r['n_hidden_layers'] for r in rows]

    # Color by number of hidden layers
    layer_colors = {1: '#1f77b4', 2: '#2ca02c', 3: '#ff7f0e'}
    colors = [layer_colors.get(nl, '#7f7f7f') for nl in n_layers]

    x = range(len(labels))
    plt.figure(figsize=(14, 6))
    bars = plt.bar(x, test_accs, color=colors, edgecolor='black', linewidth=0.6)

    plt.xticks(x, labels, rotation=45, ha='right', fontsize=9)
    plt.ylim(0, max(1.0, max(test_accs) + 0.10))
    plt.ylabel('Test accuracy')
    plt.xlabel('hidden_layer_sizes')
    plt.title('MLP architectures vs Test accuracy (Iris, test_size=0.25, random_state=3)')

    # Annotate bars
    for rect, acc in zip(bars, test_accs):
        height = rect.get_height()
        y = min(height + 0.04, plt.gca().get_ylim()[1] - 0.02)
        plt.text(rect.get_x() + rect.get_width()/2.0, y, f"{acc:.3f}", ha='center', va='bottom', fontsize=8, clip_on=False)

    # Legend for layers
    legend_elements = [
        Patch(facecolor=layer_colors[1], edgecolor='black', label='1 hidden layer'),
        Patch(facecolor=layer_colors[2], edgecolor='black', label='2 hidden layers'),
        Patch(facecolor=layer_colors[3], edgecolor='black', label='3 hidden layers'),
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    here = Path(__file__).resolve().parent
    csv_path = here / 'mlp_results.csv'
    out_path = here / 'answer' / 'mlp_test_accuracy_by_architecture.png'

    rows = load_results(csv_path)
    plot_arch_vs_test_accuracy(rows, out_path)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
