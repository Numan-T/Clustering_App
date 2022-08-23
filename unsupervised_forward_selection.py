from sklearn.metrics import silhouette_score
import numpy as np
from heap import MaxHeap
import copy


class UnsupervisedForwardFeatureSelector:
    def __init__(self, algo, score_function=silhouette_score):
        self.algo = algo
        self.score_function = score_function
        self.score_selection_pairs = MaxHeap(list())
        self.best_selection = np.array([])
        self.best_score = float('-inf')

    def fit(self, X):
        """
        Apply the forward selection steps for the input data X.
        """
        n_features = len(X[0])
        current_selections = np.zeros(shape=n_features, dtype=bool)

        for _ in range(n_features):
            new_feature_idx, new_score = self.get_best_remaining_feature_and_score(
                X, current_selections)
            current_selections[new_feature_idx] = True
            self.score_selection_pairs.heappush(
                (new_score, tuple(current_selections.copy())))  # convert to tuple because MaxHeap is not implemented for np.ndarray
        self.set_best_score_and_selection()

    def get_best_remaining_feature_and_score(self, X, initial_selections):
        """
        For every remaining feature determine the score after adding it to the current feature selection. Return a tuple consisting of the highest score and the index of the feature whose addition has led to this score."""
        remaining_feature_indices = np.flatnonzero(~initial_selections)
        scores = MaxHeap(list())

        # For every remaining feature determine the score after adding it
        for feature_idx in remaining_feature_indices:
            # Append current selection and fit the model to it
            current_selections = initial_selections.copy()
            current_selections[feature_idx] = True
            filtered_X = X[:, current_selections]
            algo_copy = copy.deepcopy(self.algo)
            algo_copy.fit(filtered_X)

            # Get predicted labels
            y_pred = []
            if hasattr(algo_copy, "labels_"):
                y_pred = algo_copy.labels_.astype(int)
            else:
                y_pred = algo_copy.predict(filtered_X)

            # Get score for predicted labels
            label_count = len(np.unique(y_pred))
            if label_count < 2:  # No clusters found
                scores.heappush((float('-inf'), feature_idx))
            else:
                scores.heappush(
                    (self.score_function(filtered_X, y_pred), feature_idx))

        # Return best new feature index with score after adding it
        best_score, best_feature_idx = scores.heappop()
        return best_feature_idx, best_score

    def set_best_score_and_selection(self):
        """
        Set the best score and selection as the top heap element from the best_score_selection_pair-heap.
        """
        best_score_selection_pair = self.score_selection_pairs.heappop()
        self.best_score = best_score_selection_pair[0]
        self.best_selection = np.asarray(
            best_score_selection_pair[1])  # convert tuple to np.ndarray
        # push back on heap to maintain it
        self.score_selection_pairs.heappush(best_score_selection_pair)

    def transform(self, X):
        """
        Filter the input data X by the best feature selection found and return it.
        """
        X_best = X[:, self.best_selection]
        return X_best

    def fit_transform(self, X):
        """
        First apply sequential forward feature selection on the input data X and then return a filtered version of X that is filtered by the best feature selection found.
        """
        self.fit(X)
        return self.transform(X)
