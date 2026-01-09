from typing import Dict, List, Tuple
import numpy as np


class KNearestNeighbor:
    """
    A kNN classifier with L2 distance, implemented in NumPy and structured to
    mirror the CS231n-style notebooks.
    """

    def __init__(self):
        self.X_train: np.ndarray = None
        self.y_train: np.ndarray = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        For kNN, "training" is just memorizing the training data.
        X: (N_train, D) float32
        y: (N_train,) int64
        """
        assert X.ndim == 2 and y.ndim == 1
        assert X.shape[0] == y.shape[0]
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray, k: int = 5, batch_size: int = 200) -> np.ndarray:
        """
        Predict labels for test data X using k nearest neighbors.
        Computes distances in batches to control memory usage.
        Returns y_pred of shape (N_test,)
        """
        assert self.X_train is not None and self.y_train is not None, "Call train() first"
        assert k >= 1 and k <= self.X_train.shape[0]

        N_test = X.shape[0]
        y_pred = np.empty(N_test, dtype=self.y_train.dtype)

        # Precompute train norms for efficiency
        train_sq_norms = np.sum(self.X_train ** 2, axis=1)

        for start in range(0, N_test, batch_size):
            end = min(start + batch_size, N_test)
            X_batch = X[start:end]
            # compute squared L2 distances: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
            test_sq_norms = np.sum(X_batch ** 2, axis=1, keepdims=True)  # (B,1)
            cross = X_batch @ self.X_train.T  # (B, N_train)
            dists2 = test_sq_norms + train_sq_norms[None, :] - 2.0 * cross

            # For numerical issues, ensure non-negative
            np.maximum(dists2, 0.0, out=dists2)

            # Find k nearest neighbors using argpartition for efficiency
            knn_idx = np.argpartition(dists2, kth=k-1, axis=1)[:, :k]  # (B,k) unordered
            # Order the k neighbors by actual distance
            row_indices = np.arange(knn_idx.shape[0])[:, None]
            knn_sorted = knn_idx[row_indices, np.argsort(dists2[row_indices, knn_idx])]

            # Majority vote
            knn_labels = self.y_train[knn_sorted]
            # bincount per row
            preds = []
            for row in knn_labels:
                counts = np.bincount(row, minlength=int(self.y_train.max()) + 1)
                preds.append(np.argmax(counts))
            y_pred[start:end] = np.array(preds, dtype=self.y_train.dtype)

        return y_pred


def cross_validate_k(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_choices: List[int],
    num_folds: int = 5,
    batch_size: int = 200,
) -> Dict[int, np.ndarray]:
    """
    Perform k-fold cross-validation to find the best value of k.
    Returns a dict mapping k -> accuracies per fold (shape (num_folds,)).
    Mirrors the notebook's np.split-based fold creation.
    """
    assert X_train.shape[0] % num_folds == 0, "For simplicity, require divisible folds."

    X_folds = np.split(X_train, num_folds)
    y_folds = np.split(y_train, num_folds)

    k_to_accuracies: Dict[int, List[float]] = {k: [] for k in k_choices}

    for ik, k in enumerate(k_choices):
        for i in range(num_folds):
            # Validation fold i, training on all others
            X_val = X_folds[i]
            y_val = y_folds[i]
            X_tr = np.concatenate(X_folds[:i] + X_folds[i+1:], axis=0)
            y_tr = np.concatenate(y_folds[:i] + y_folds[i+1:], axis=0)

            clf = KNearestNeighbor()
            clf.train(X_tr, y_tr)
            y_pred = clf.predict(X_val, k=k, batch_size=batch_size)
            acc = float(np.mean(y_pred == y_val))
            k_to_accuracies[k].append(acc)

    # Convert lists to numpy arrays
    return {k: np.array(v, dtype=np.float32) for k, v in k_to_accuracies.items()}


def choose_best_k(k_to_accuracies: Dict[int, np.ndarray]) -> Tuple[int, float, float]:
    """
    Choose best k by highest mean CV accuracy. Return (best_k, mean_acc, std_acc).
    Ties are broken by selecting the smallest k.
    """
    best_k = None
    best_mean = -1.0
    best_std = 0.0
    for k, accs in k_to_accuracies.items():
        mean = float(np.mean(accs))
        std = float(np.std(accs))
        if (mean > best_mean) or (np.isclose(mean, best_mean) and (best_k is None or k < best_k)):
            best_k = k
            best_mean = mean
            best_std = std
    return best_k, best_mean, best_std
