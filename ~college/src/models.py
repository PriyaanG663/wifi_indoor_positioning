"""
Machine Learning Models for WiFi Fingerprint Indoor Positioning
Course: UE24CS352A - Machine Learning Mini-Project
Includes:
- Weighted k-Nearest Neighbors (WKNN) Classifier & Regressor (Pure NumPy)
- Multi-Layer Perceptron (MLP) Neural Network for Coordinate Regression (Pure NumPy)
- Hierarchical Positioning System (Building -> Floor -> Coordinates)
- Probability Estimation (predict_proba) for Multi-Class ROC Analysis
"""

import numpy as np

class WeightedKNN:
    """
    Weighted k-Nearest Neighbors supporting both Multi-Output Regression,
    Multi-Class Classification, and Probability Estimation (predict_proba).
    Features selectable distance metrics: Euclidean, Manhattan, and Cosine.
    """
    def __init__(self, n_neighbors=5, metric='euclidean', weight_power=1.0, epsilon=1e-5):
        self.k = n_neighbors
        self.metric = metric
        self.weight_power = weight_power
        self.epsilon = epsilon
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        if len(self.y_train.shape) == 1 or self.y_train.shape[1] == 1:
            self.classes_ = np.unique(self.y_train)
        return self

    def _compute_distances(self, X_test):
        X_test = np.asarray(X_test, dtype=float)
        if self.metric == 'euclidean':
            # Vectorized Euclidean distance: sqrt(sum((x - y)^2))
            X_test_sq = np.sum(X_test**2, axis=1, keepdims=True)
            X_train_sq = np.sum(self.X_train**2, axis=1, keepdims=True).T
            cross = np.dot(X_test, self.X_train.T)
            dist_sq = np.maximum(X_test_sq + X_train_sq - 2 * cross, 0.0)
            return np.sqrt(dist_sq)
        elif self.metric == 'manhattan':
            return np.sum(np.abs(X_test[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]), axis=2)
        elif self.metric == 'cosine':
            dot = np.dot(X_test, self.X_train.T)
            norm_test = np.linalg.norm(X_test, axis=1, keepdims=True)
            norm_train = np.linalg.norm(self.X_train, axis=1, keepdims=True).T
            denom = np.maximum(norm_test * norm_train, self.epsilon)
            return 1.0 - (dot / denom)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def predict_regression(self, X_test):
        distances = self._compute_distances(X_test)
        k_indices = np.argsort(distances, axis=1)[:, :self.k]
        
        preds = []
        for i in range(len(X_test)):
            row_idx = k_indices[i]
            row_dists = distances[i, row_idx]
            
            # Inverse distance weighting
            weights = 1.0 / (np.power(row_dists, self.weight_power) + self.epsilon)
            weights /= np.sum(weights)
            
            y_k = self.y_train[row_idx]
            if len(y_k.shape) == 1:
                preds.append(np.sum(weights * y_k))
            else:
                preds.append(np.sum(weights[:, np.newaxis] * y_k, axis=0))
                
        return np.array(preds)

    def predict_classification(self, X_test):
        probs, classes = self.predict_proba(X_test)
        best_indices = np.argmax(probs, axis=1)
        return classes[best_indices]

    def predict_proba(self, X_test, classes=None):
        """
        Computes soft voting probabilities across the k nearest neighbors
        for ROC curve threshold sweeps and confidence metrics.
        """
        distances = self._compute_distances(X_test)
        k_indices = np.argsort(distances, axis=1)[:, :self.k]
        
        if classes is None:
            classes = self.classes_ if self.classes_ is not None else np.unique(self.y_train)
        classes = np.asarray(classes)
        class_to_idx = {c: idx for idx, c in enumerate(classes)}
        
        n_samples = len(X_test)
        n_classes = len(classes)
        probs = np.zeros((n_samples, n_classes), dtype=float)
        
        for i in range(n_samples):
            row_idx = k_indices[i]
            row_dists = distances[i, row_idx]
            weights = 1.0 / (np.power(row_dists, self.weight_power) + self.epsilon)
            
            for lbl, w in zip(self.y_train[row_idx], weights):
                if lbl in class_to_idx:
                    probs[i, class_to_idx[lbl]] += w
                    
            total = np.sum(probs[i])
            if total > 0:
                probs[i] /= total
            else:
                probs[i] = 1.0 / n_classes
                
        return probs, classes


class PureNumpyMLPRegressor:
    """
    Two-layer Multi-Layer Perceptron (MLP) for continuous multi-output coordinate regression.
    Implemented in pure NumPy with ReLU activations and mini-batch SGD with momentum.
    """
    def __init__(self, hidden_dim=128, lr=0.005, momentum=0.9, batch_size=32, epochs=40):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.epochs = epochs
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.mean_y = None
        self.std_y = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_y = np.mean(y, axis=0)
        self.std_y = np.std(y, axis=0) + 1e-6
        y_norm = (y - self.mean_y) / self.std_y

        n_samples, in_dim = X.shape
        out_dim = y.shape[1] if len(y.shape) > 1 else 1
        if len(y.shape) == 1:
            y_norm = y_norm[:, np.newaxis]

        np.random.seed(42)
        self.W1 = np.random.randn(in_dim, self.hidden_dim) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = np.random.randn(self.hidden_dim, out_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = np.zeros((1, out_dim))

        vW1, vb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        vW2, vb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)

        indices = np.arange(n_samples)
        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                bx = X[indices[start:end]]
                by = y_norm[indices[start:end]]

                # Forward pass
                z1 = np.dot(bx, self.W1) + self.b1
                a1 = np.maximum(0, z1) # ReLU
                preds = np.dot(a1, self.W2) + self.b2

                # Backward pass (MSE Loss)
                loss_grad = 2.0 * (preds - by) / bx.shape[0]
                dW2 = np.dot(a1.T, loss_grad)
                db2 = np.sum(loss_grad, axis=0, keepdims=True)

                da1 = np.dot(loss_grad, self.W2.T)
                dz1 = da1 * (z1 > 0)
                dW1 = np.dot(bx.T, dz1)
                db1 = np.sum(dz1, axis=0, keepdims=True)

                # Momentum updates
                vW1 = self.momentum * vW1 - self.lr * dW1
                vb1 = self.momentum * vb1 - self.lr * db1
                vW2 = self.momentum * vW2 - self.lr * dW2
                vb2 = self.momentum * vb2 - self.lr * db2

                self.W1 += vW1
                self.b1 += vb1
                self.W2 += vW2
                self.b2 += vb2

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        z1 = np.dot(X, self.W1) + self.b1
        a1 = np.maximum(0, z1)
        preds_norm = np.dot(a1, self.W2) + self.b2
        preds = preds_norm * self.std_y + self.mean_y
        return preds


class HierarchicalIndoorPositioningSystem:
    """
    Hierarchical Positioning Pipeline:
    1. Building Classifier: Predicts BuildingID in {0, 1, 2}
    2. Floor Classifier: Predicts Floor in {0, 1, 2, 3, 4}
    3. Coordinate Regressor: Predicts (Longitude, Latitude)
    """
    def __init__(self, k_building=3, k_floor=5, k_coords=7, coord_method='wknn'):
        self.building_model = WeightedKNN(n_neighbors=k_building, metric='euclidean', weight_power=1.0)
        self.floor_model = WeightedKNN(n_neighbors=k_floor, metric='euclidean', weight_power=1.5)
        self.coord_method = coord_method
        if coord_method == 'mlp':
            self.coord_model = PureNumpyMLPRegressor(hidden_dim=128, lr=0.005, epochs=30)
        else:
            self.coord_model = WeightedKNN(n_neighbors=k_coords, metric='euclidean', weight_power=2.0)

    def fit(self, X, y_building, y_floor, y_coords):
        self.building_model.fit(X, y_building)
        self.floor_model.fit(X, y_floor)
        if self.coord_method == 'mlp':
            self.coord_model.fit(X, y_coords)
        else:
            self.coord_model.fit(X, y_coords)
        return self

    def predict(self, X):
        pred_b = self.building_model.predict_classification(X)
        pred_f = self.floor_model.predict_classification(X)
        if self.coord_method == 'mlp':
            pred_coords = self.coord_model.predict(X)
        else:
            pred_coords = self.coord_model.predict_regression(X)
        return {
            'building': pred_b,
            'floor': pred_f,
            'coords': pred_coords
        }

    def predict_proba(self, X):
        """
        Returns prediction probabilities and class labels for both Building and Floor.
        """
        b_probs, b_classes = self.building_model.predict_proba(X)
        f_probs, f_classes = self.floor_model.predict_proba(X)
        return {
            'building_probs': b_probs,
            'building_classes': b_classes,
            'floor_probs': f_probs,
            'floor_classes': f_classes
        }
