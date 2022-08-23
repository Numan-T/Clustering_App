from sklearn.metrics import silhouette_score
import numpy as np
from unsupervised_forward_selection import UnsupervisedForwardFeatureSelector


class ClusteringAlgo():

    def __init__(self, algo, algo_name, X, feature_selection_mode="sequential"):
        self.algo = algo
        self.algo_name = algo_name
        self.X = X
        self.feature_selection_mode = feature_selection_mode
        self.feature_selection = np.array([True] * X.shape[1])
        self.clustering_result = self.clustering()

    def clustering(self):
        """
        Apply the feature selection method specified in self.feature_selection_mode. Then apply clustering and return the obtained cluster labels/assignments. 
        """
        # Apply Sequential Feature Selection with Silhouette Score
        if self.feature_selection_mode == "sequential":
            feature_selector = UnsupervisedForwardFeatureSelector(
                algo=self.algo)
            self.X = feature_selector.fit_transform(X=self.X)
            self.feature_selection = feature_selector.best_selection

        # Apply clustering method
        self.algo.fit(X=self.X)

        # Get cluster labels/assignments
        if hasattr(self.algo, "labels_"):
            y_pred = self.algo.labels_.astype(int)
        else:
            y_pred = self.algo.predict(self.X)

        label_count = len(np.unique(y_pred))
        if label_count < 2:  # No clusters found
            y_pred = ["No result"]

        return y_pred

    def get_result_score(self):
        """
        Return the silhouette score for the clustering result. If there is no result, return "No result" instead.
        """
        if self.clustering_result[0] == "No result":
            return "No result"
        return silhouette_score(self.X, self.clustering_result)

    def get_result_std(self):
        """
        Return the standard deviation of the cluster sizes. If there is no result, return 0.
        """
        cluster_info = self.get_cluster_info_dict()
        if self.clustering_result[0] == "No result":
            return 0
        result_std = np.std([counts for _, counts in cluster_info.items()])
        return result_std

    def get_cluster_info_dict(self):
        """
        Return a dictionary with the unique cluster labels and their according sizes.
        """
        label, counts = np.unique(
            self.clustering_result,
            return_counts=True)
        cluster_info_dict = dict(zip(label, counts))
        return cluster_info_dict

    def get_num_clusters_found(self):
        """
        Return the number of clusters found by the algorithm.
        """
        return len(self.get_cluster_info_dict())
