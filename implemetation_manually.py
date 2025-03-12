import numpy as np
import pandas as pd
import random
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Step 1: Load Dataset
df = pd.read_csv("diabetes.csv")

# Handle missing values
df.fillna(df.median(), inplace=True)

# Define features and target
X = df.drop(columns=['Outcome']).values
y = df['Outcome'].values

# Feature Scaling (Standardization)
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Implement Decision Tree
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(set(y)) == 1 or len(y) < self.min_samples_split:
            return {'label': Counter(y).most_common(1)[0][0]}

        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return {'label': Counter(y).most_common(1)[0][0]}

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def _find_best_split(self, X, y):
        m, n = X.shape
        best_gini = float('inf')
        best_feature, best_threshold = None, None

        for feature in range(n):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                gini = self._gini_index(y[left_mask], y[right_mask])

                if gini < best_gini:
                    best_gini, best_feature, best_threshold = gini, feature, threshold

        return best_feature, best_threshold

    def _gini_index(self, left_y, right_y):
        def gini(y):
            classes, counts = np.unique(y, return_counts=True)
            p = counts / len(y)
            return 1 - np.sum(p ** 2)

        left_gini = gini(left_y)
        right_gini = gini(right_y)
        total_samples = len(left_y) + len(right_y)

        return (len(left_y) / total_samples) * left_gini + (len(right_y) / total_samples) * right_gini

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def _predict_sample(self, sample, tree):
        if 'label' in tree:
            return tree['label']
        if sample[tree['feature']] <= tree['threshold']:
            return self._predict_sample(sample, tree['left'])
        return self._predict_sample(sample, tree['right'])

# Step 3: Implement Random Forest
class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2, sample_size=0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.sample_size = sample_size
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = int(self.sample_size * len(y))
        indices = np.random.choice(len(y), size=n_samples, replace=True)
        return X[indices], y[indices]

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        final_predictions = [Counter(predictions[:, i]).most_common(1)[0][0] for i in range(len(X))]
        return np.array(final_predictions)

# Step 4: Train Random Forest and Evaluate
rf = RandomForest(n_trees=10, max_depth=5, min_samples_split=2)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Step 5: Calculate Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Random Forest Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
