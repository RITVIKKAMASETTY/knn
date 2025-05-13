# K-Nearest Neighbors (KNN) Implementation

## Overview

This repository provides a comprehensive implementation of the K-Nearest Neighbors (KNN) algorithm, a versatile non-parametric method used for classification and regression tasks in machine learning. KNN makes predictions based on the similarity between input samples and training data points.

## Table of Contents

- [Theory](#theory)
- [Installation](#installation)
- [Usage](#usage)
  - [Classification Example](#classification-example)
  - [Regression Example](#regression-example)
  - [Advanced Usage](#advanced-usage)
- [API Reference](#api-reference)
- [Distance Metrics](#distance-metrics)
- [Parameter Tuning](#parameter-tuning)
- [Performance Optimization](#performance-optimization)
- [Contributing](#contributing)
- [License](#license)

## Theory

K-Nearest Neighbors (KNN) is a lazy learning algorithm that stores all available cases and classifies new cases based on a similarity measure.

### Key Concepts

1. **Instance-Based Learning**: Unlike eager learners that build a generalized model, KNN memorizes the training instances and uses them directly for prediction.

2. **Non-parametric**: KNN makes no assumptions about the underlying data distribution.

3. **Locality-Based Classification**: Predictions are made based on the k closest training examples in the feature space.

4. **Majority Voting/Averaging**: For classification, the majority class among the k neighbors is assigned. For regression, the average of the k neighbors' values is computed.

### Algorithm Steps

1. **Choose the value of K**: Select the number of neighbors to consider.

2. **Find the K-Nearest Neighbors**: Calculate the distance between the query instance and all training samples.

3. **Voting/Averaging**: For classification, use majority voting among neighbors. For regression, average the values of neighbors.

4. **Make Prediction**: Assign the class label (classification) or predicted value (regression).

## Installation

```bash
# Using pip
pip install knn-implementation

# From source
git clone https://github.com/username/knn-implementation.git
cd knn-implementation
pip install -e .
```

## Usage

### Classification Example

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from knn_implementation import KNNClassifier

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize and train KNN classifier
knn = KNNClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.4f}")

# Get probability estimates
probabilities = knn.predict_proba(X_test)
print(f"Probability for first sample: {probabilities[0]}")
```

### Regression Example

```python
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from knn_implementation import KNNRegressor

# Load dataset
boston = load_boston()
X, y = boston.data, boston.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Initialize and train KNN regressor
knn = KNNRegressor(n_neighbors=7, metric='manhattan', weights='distance')
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
mse = np.mean((y_pred - y_test) ** 2)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
```

### Advanced Usage

```python
from knn_implementation import KNNClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create a pipeline with preprocessing and KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNNClassifier())
])

# Define parameter grid for tuning
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski'],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2, 3]  # Only relevant for Minkowski distance
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Get best parameters and results
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Test accuracy with best model: {accuracy:.4f}")
```

## API Reference

### `KNNClassifier` Class

```python
KNNClassifier(n_neighbors=5, weights='uniform', algorithm='auto', 
              leaf_size=30, p=2, metric='minkowski', 
              metric_params=None, n_jobs=None)
```

#### Parameters

- `n_neighbors` : int, default=5
  - Number of neighbors to use for prediction.

- `weights` : {'uniform', 'distance'} or callable, default='uniform'
  - Weight function used in prediction:
  - 'uniform' : All neighbors weighted equally.
  - 'distance' : Weight points by inverse of distance.
  - callable : Custom weight function.

- `algorithm` : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
  - Algorithm to compute nearest neighbors:
  - 'ball_tree' : Use BallTree.
  - 'kd_tree' : Use KDTree.
  - 'brute' : Use brute-force search.
  - 'auto' : Auto-select best algorithm.

- `leaf_size` : int, default=30
  - Leaf size for BallTree or KDTree.

- `p` : int, default=2
  - Power parameter for Minkowski metric.
  - p=1 is Manhattan distance, p=2 is Euclidean distance.

- `metric` : str or callable, default='minkowski'
  - Distance metric for the tree:
  - See `Distance Metrics` section for available metrics.

- `metric_params` : dict, default=None
  - Additional parameters for metric function.

- `n_jobs` : int, default=None
  - Number of parallel jobs for neighbor search.
  - None means 1, -1 means using all processors.

#### Methods

- `fit(X, y)` : Fit the model.
- `predict(X)` : Predict class labels for samples in X.
- `predict_proba(X)` : Return probability estimates for samples in X.
- `kneighbors(X, n_neighbors, return_distance)` : Find the K-neighbors of points.
- `score(X, y)` : Return the mean accuracy on the given test data and labels.

### `KNNRegressor` Class

```python
KNNRegressor(n_neighbors=5, weights='uniform', algorithm='auto', 
            leaf_size=30, p=2, metric='minkowski', 
            metric_params=None, n_jobs=None)
```

Parameters and methods are similar to `KNNClassifier`, with key differences:

- `predict` returns continuous values instead of class labels.
- `predict_proba` is not available.
- `score` returns the coefficient of determination R².

## Distance Metrics

KNN performance heavily depends on the distance metric used. This implementation supports the following distance metrics:

### Common Metrics

- **Euclidean** (`metric='euclidean'`): \\( \sqrt{\sum_i (x_i - y_i)^2} \\)
- **Manhattan** (`metric='manhattan'`): \\( \sum_i |x_i - y_i| \\)
- **Minkowski** (`metric='minkowski'`): \\( (\sum_i |x_i - y_i|^p)^{1/p} \\)
- **Chebyshev** (`metric='chebyshev'`): \\( \max_i |x_i - y_i| \\)

### Specialized Metrics

- **Mahalanobis** distance: Accounts for covariance structure in the data.
- **Cosine** similarity: Measures the angle between vectors, useful for text data.
- **Hamming** distance: For categorical features, counts attributes where values differ.
- **Jaccard** similarity: Used for binary features or sets.

### Custom Metrics

You can define your own distance function:

```python
def custom_distance(x, y):
    """Custom distance function between two points."""
    return np.sum(np.abs(x - y) ** 1.5)

knn = KNNClassifier(n_neighbors=5, metric=custom_distance)
```

## Parameter Tuning

Optimizing KNN hyperparameters is crucial for good performance:

### Cross-Validation Strategy

```python
from sklearn.model_selection import cross_val_score

k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
cv_scores = []

for k in k_values:
    knn = KNNClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, 'o-')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Accuracy vs. k Value')
plt.grid(True)
plt.savefig('knn_cv_results.png')
plt.show()
```

### Key Parameters to Tune

1. **Number of neighbors (k)**:
   - Smaller k: More sensitive to noise, more complex decision boundary
   - Larger k: Smoother decision boundary, but may miss patterns

2. **Distance metric**:
   - Try different metrics depending on your data
   - Scale features appropriately for distance-based metrics

3. **Weighting scheme**:
   - 'uniform': Equal weights for all neighbors
   - 'distance': Closer neighbors have more influence

## Performance Optimization

KNN can be computationally expensive, especially for large datasets. Here are optimization techniques:

### Dimensionality Reduction

Reduce feature dimensions before KNN:

```python
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

# Apply KNN on reduced data
knn = KNNClassifier(n_neighbors=5)
knn.fit(X_train_reduced, y_train)
```

### Approximate Nearest Neighbors

For very large datasets, consider approximate nearest neighbor algorithms:

```python
from knn_implementation import ApproxKNN

# Use LSH (Locality-Sensitive Hashing)
approx_knn = ApproxKNN(n_neighbors=5, algorithm='lsh', n_trees=10)
approx_knn.fit(X_train)
```

### Feature Scaling

Always scale features before applying KNN:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
