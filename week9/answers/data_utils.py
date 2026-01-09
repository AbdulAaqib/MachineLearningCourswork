import os
import tarfile
import urllib.request
import pickle
import numpy as np
from typing import Tuple

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_TARBALL = "cifar-10-python.tar.gz"
CIFAR10_FOLDER = "cifar-10-batches-py"


def _download_cifar10(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    tarball_path = os.path.join(data_dir, CIFAR10_TARBALL)
    if not os.path.exists(tarball_path):
        print(f"Downloading CIFAR-10 from {CIFAR10_URL} ...")
        urllib.request.urlretrieve(CIFAR10_URL, tarball_path)
        print("Download complete.")
    else:
        print("CIFAR-10 tarball already exists, skipping download.")

    extract_dir = os.path.join(data_dir, CIFAR10_FOLDER)
    if not os.path.exists(extract_dir):
        print("Extracting CIFAR-10 tarball ...")
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("Extraction complete.")
    else:
        print("CIFAR-10 already extracted, skipping extraction.")


def _load_batch(file: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
        X = dict_[b'data']  # shape (N, 3072)
        y = dict_[b'labels']
        X = np.asarray(X, dtype=np.uint8)
        y = np.asarray(y, dtype=np.int64)
        return X, y


def load_cifar10(data_dir: str, download_if_missing: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CIFAR-10 dataset (python version) into NumPy arrays following the typical
    CS231n notebook structure. Returns X_train, y_train, X_test, y_test where X_* are
    float32 arrays of shape (N, 3072) and y_* are int64 arrays of shape (N,).
    """
    if download_if_missing:
        _download_cifar10(data_dir)

    base_dir = os.path.join(data_dir, CIFAR10_FOLDER)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(
            f"CIFAR-10 directory not found at {base_dir}. Set download_if_missing=True to download.")

    # Load training batches
    Xs = []
    ys = []
    for i in range(1, 6):
        batch_file = os.path.join(base_dir, f"data_batch_{i}")
        Xb, yb = _load_batch(batch_file)
        Xs.append(Xb)
        ys.append(yb)
    X_train = np.concatenate(Xs, axis=0)
    y_train = np.concatenate(ys, axis=0)

    # Load test batch
    X_test, y_test = _load_batch(os.path.join(base_dir, "test_batch"))

    # Convert to float32 and flatten (already flattened as 3072)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # No mean subtraction or normalization is applied here to mirror the simple kNN baseline
    return X_train, y_train, X_test, y_test
