import os
import argparse
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, models, layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd

# Ensure reproducibility (best effort)
np.random.seed(42)
tf.random.set_seed(42)


def build_model(kernel_size=(3, 3)):
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size, activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, kernel_size, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, kernel_size, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))  # logits

    model.compile(optimizer=Adam(),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def load_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    # Normalize to [0, 1]
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    # Flatten labels shape from (n,1) to (n,)
    train_labels = train_labels.reshape(-1)
    test_labels = test_labels.reshape(-1)
    return (train_images, train_labels), (test_images, test_labels)


def run_experiment(epochs, kernel_size, out_dir):
    (train_images, train_labels), (test_images, test_labels) = load_data()
    model = build_model(kernel_size=kernel_size)

    print(f"Training with epochs={epochs}, kernel_size={kernel_size}...")
    start = time.time()
    history = model.fit(train_images, train_labels,
                        epochs=epochs,
                        validation_data=(test_images, test_labels),
                        batch_size=128,
                        verbose=1)
    duration = time.time() - start

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

    # Save history plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy (epochs={epochs}, kernel={kernel_size})\nTest acc={test_acc:.4f}, time={duration/60:.1f} min')
    plt.legend()
    plot_path = os.path.join(out_dir, f"cnn_task_acc_epochs_{epochs}_kernel_{kernel_size[0]}x{kernel_size[1]}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # Return results
    return {
        'epochs': epochs,
        'kernel_size': f"{kernel_size[0]}x{kernel_size[1]}",
        'test_acc': float(test_acc),
        'test_loss': float(test_loss),
        'duration_sec': float(duration),
        'plot': os.path.basename(plot_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs_list', type=int, nargs='+', default=[10, 20])
    parser.add_argument('--kernel_size', type=int, nargs=2, default=[3, 3])
    parser.add_argument('--out_dir', type=str, default='.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for e in args.epochs_list:
        res = run_experiment(e, tuple(args.kernel_size), args.out_dir)
        results.append(res)

    # Save CSV and JSON summary
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.out_dir, 'cnn_task_results.csv')
    json_path = os.path.join(args.out_dir, 'cnn_task_results.json')
    df.to_csv(csv_path, index=False)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("Saved:", csv_path, json_path)


if __name__ == '__main__':
    main()
