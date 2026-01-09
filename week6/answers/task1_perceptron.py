#!/usr/bin/env python3
"""
Perceptron Algorithm Implementation for Task 1
Week 6 - Machine Learning
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Load data from data_Perceptron.txt
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_Perceptron.txt')
data = np.loadtxt(data_path)

# Extract features (first two columns) and labels (last column)
X = data[:, :2]
Y = data[:, 2].astype(int)

print(f"Data shape: {X.shape}")
print(f"Number of samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"Class distribution: Class 0: {np.sum(Y == 0)}, Class 1: {np.sum(Y == 1)}")

# Define the activation function that returns either 1 or 0
# Standard Perceptron: any value > 0 becomes 1, any value <= 0 becomes 0
def activation(x):
    return 1 if x > 0 else 0

# A function to calculate the unit vector of our weights vector
def calc_unit_vector(x):
    return x.transpose() / np.sqrt(x.transpose().dot(x))

# A function that returns values that lay on the hyperplane
def calc_hyperplane(X, w):
    return np.ravel([-(w[0] + x * w[1]) / w[2] for x in X])

# Visualize the original dataset
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis', alpha=0.6)
ax.set_title('Original Dataset', fontsize=16)
ax.set_xlabel('X1', fontsize=12)
ax.set_ylabel('X2', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'original_data.png'), dpi=150)
plt.close()

# Add a bias to the X vector
X_bias = np.ones([X.shape[0], 3])
X_bias[:, 1:3] = X

# Initialize weight vector with zeros
w = np.zeros([3, 1])

print("\nTraining Perceptron...")
print("Initial weights: w0 = {:.4f}, w1 = {:.4f}, w2 = {:.4f}".format(w[0][0], w[1][0], w[2][0]))

# Apply Perceptron learning rule (10 iterations)
for iteration in range(10):
    for i in range(X_bias.shape[0]):
        y = activation(w.transpose().dot(X_bias[i, :]))
        # Update weights
        w = w + ((Y[i] - y) * X_bias[i, :]).reshape(w.shape[0], 1)

print("\nFinal weights after training:")
print('w0 = {:.4f}'.format(w[0][0]))
print('w1 = {:.4f}'.format(w[1][0]))
print('w2 = {:.4f}'.format(w[2][0]))

# Calculate the class of the data points with the weight vector
result = [w.transpose().dot(x) for x in X_bias]
result_class = [activation(w.transpose().dot(x)) for x in X_bias]
result_class = np.array(result_class)

# Calculate unit vector for visualization
w_unit = calc_unit_vector(w).transpose()

# Calculate misclassified points
misclassified = (result_class != Y).astype(int)
num_misclassified = np.sum(misclassified)
misclassification_rate = (num_misclassified / len(Y)) * 100

print(f"\nMisclassification Analysis:")
print(f"Total points: {len(Y)}")
print(f"Misclassified points: {num_misclassified}")
print(f"Misclassification rate: {misclassification_rate:.2f}%")

# Check if data is linearly separable
is_linearly_separable = (num_misclassified == 0)

print(f"\nLinearly separable: {is_linearly_separable}")

# Visualize results
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Ground truth
ax[0].scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis', alpha=0.6)
ax[0].set_title('Ground Truth', fontsize=16)
ax[0].set_xlabel('X1', fontsize=12)
ax[0].set_ylabel('X2', fontsize=12)

# Perceptron classification with hyperplane
ax[1].scatter(X[:, 0], X[:, 1], c=result_class, cmap='viridis', alpha=0.6)
x_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
hyperplane = calc_hyperplane(x_range, w_unit)
ax[1].plot(x_range, hyperplane, lw=3, c='red', label='Decision Boundary')
ax[1].set_xlim(ax[0].get_xlim())
ax[1].set_ylim(ax[0].get_ylim())
ax[1].set_xlabel('X1', fontsize=12)
ax[1].set_ylabel('X2', fontsize=12)
ax[1].set_title('Perceptron classification with hyperplane', fontsize=16)
ax[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'perceptron_classification.png'), dpi=150)
plt.close()

# Visualize misclassified points
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
colors = ['green' if m == 0 else 'red' for m in misclassified]
ax.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6)
x_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
hyperplane = calc_hyperplane(x_range, w_unit)
ax.plot(x_range, hyperplane, lw=3, c='blue', label='Decision Boundary')
ax.set_xlabel('X1', fontsize=12)
ax.set_ylabel('X2', fontsize=12)
ax.set_title('Misclassified Points', fontsize=16)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'misclassified_points.png'), dpi=150)
plt.close()

print("\nPlots saved successfully!")
print("Files created:")
print("  - original_data.png")
print("  - perceptron_classification.png")
print("  - misclassified_points.png")
