import os
import sys

# Ensure Matplotlib uses a non-GUI backend and has a writable config dir
MPL_DIR = os.path.join(os.path.dirname(__file__), ".mplconfig")
os.environ.setdefault("MPLCONFIGDIR", MPL_DIR)
os.makedirs(MPL_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')  # Use headless backend to avoid macOS AppKit issues

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Usage:
# python composite_collages.py [base_dir]
# Default base_dir is the folder containing this script.

def load_img_safe(path):
    if not os.path.exists(path):
        print(f"Warning: missing {path}")
        return None
    try:
        return mpimg.imread(path)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None


def save_composite_3x3(base_dir):
    # 3x3 kernel: epochs 10, 20, 50
    epochs = [10, 20, 50]
    rows = [
        ("Accuracy", "acc"),
        ("Loss", "loss"),
        ("Confusion Matrix", "cm"),
    ]

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
    for col, e in enumerate(epochs):
        for row, (title, kind) in enumerate(rows):
            fname = {
                "acc": f"cnn_task_acc_epochs_{e}_kernel_3x3.png",
                "loss": f"cnn_task_loss_epochs_{e}_kernel_3x3.png",
                "cm": f"cnn_task_cm_epochs_{e}_kernel_3x3.png",
            }[kind]
            fpath = os.path.join(base_dir, fname)
            img = load_img_safe(fpath)
            ax = axes[row, col]
            ax.axis('off')
            if img is not None:
                ax.imshow(img)
            if row == 0:
                ax.set_title(f"Epoch {e}", fontsize=16)
            if col == 0:
                ax.set_ylabel(title, fontsize=14)

    plt.tight_layout()
    out_path = os.path.join(base_dir, "composite_kernel_3x3.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved:", out_path)


def save_composite_5x5(base_dir):
    # 5x5 kernel: epochs 10, 20, 50 (now includes 50)
    epochs = [10, 20, 50]
    rows = [
        ("Accuracy", "acc"),
        ("Loss", "loss"),
        ("Confusion Matrix", "cm"),
    ]

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
    for col, e in enumerate(epochs):
        for row, (title, kind) in enumerate(rows):
            fname = {
                "acc": f"cnn_task_acc_epochs_{e}_kernel_5x5.png",
                "loss": f"cnn_task_loss_epochs_{e}_kernel_5x5.png",
                "cm": f"cnn_task_cm_epochs_{e}_kernel_5x5.png",
            }[kind]
            fpath = os.path.join(base_dir, fname)
            img = load_img_safe(fpath)
            ax = axes[row, col]
            ax.axis('off')
            if img is not None:
                ax.imshow(img)
            if row == 0:
                ax.set_title(f"Epoch {e}", fontsize=16)
            if col == 0:
                ax.set_ylabel(title, fontsize=14)

    plt.tight_layout()
    out_path = os.path.join(base_dir, "composite_kernel_5x5.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved:", out_path)


def main():
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    save_composite_3x3(base_dir)
    save_composite_5x5(base_dir)


if __name__ == "__main__":
    main()
