from sklearn import cluster, mixture
from yellowbrick.cluster import KElbowVisualizer
from clustering_algo import ClusteringAlgo
from tau_balanced_clustering import TBC
import copy
import numpy as np


class OptimizedClustering():

    def __init__(self, X, k_clusters_range, feature_selection_mode="sequential"):
        self.X = X
        self.k_clusters_range = k_clusters_range
        self.feature_selection_mode = feature_selection_mode
        self.num_algo_runs = 10
        self.clustering_algorithms = []
        self.result_infos = {}
        self.best_algo_name = None
        self.params = {}
        self.initialize_params()
        self.initialize_clustering_algorithms()

    def elbow_criterion(self, model):
        """
        Return the best value for k ("elbow") obtained with the elbow method. If no elbow was found, return the mid in the preferred range for k.
        """
        visualizer = KElbowVisualizer(
            model, k=self.k_clusters_range)
        visualizer.fit(self.X)
        k_elbow = visualizer.elbow_value_
        if k_elbow is None:
            k_elbow = round(
                (self.k_clusters_range[0] + self.k_clusters_range[1]) / 2)
        return k_elbow

    def initialize_params(self):
        """
        Initialize the parameter settings for the clustering algorithms.
        """
        self.params["random_state"] = 0
        self.params["n_jobs"] = -1  # use all cpu kernels for parallelization
        # run algorithm only once. Multiple runs are handled in self.fit() with self.num_algo_runs
        self.params["n_init"] = 1
        self.params["k_clusters"] = self.elbow_criterion(cluster.KMeans())

    def initialize_clustering_algorithms(self):
        """
        Initialize the clustering algorithms that are considered for the analysis.
        """
        kmeans = cluster.KMeans(
            n_clusters=self.params["k_clusters"],
            # random_state=self.params["random_state"],
            n_init=self.params["n_init"]
        )
        ms = cluster.MeanShift(
            n_jobs=self.params["n_jobs"]
        )
        mini_batch_kmeans = cluster.MiniBatchKMeans(
            n_clusters=self.params["k_clusters"],
            # random_state=self.params["random_state"],
            n_init=self.params["n_init"]
        )
        dbscan = cluster.DBSCAN(
            n_jobs=self.params["n_jobs"]
        )
        gmm = mixture.GaussianMixture(
            n_components=self.params["k_clusters"],
            # random_state=self.params["random_state"],
            n_init=self.params["n_init"]
        )
        spectral = cluster.SpectralClustering(
            n_clusters=self.params["k_clusters"],
            # random_state=self.params["random_state"],
            n_init=self.params["n_init"],
            n_jobs=self.params["n_jobs"],
        )
        optics = cluster.OPTICS(
            n_jobs=self.params["n_jobs"]
        )
        affinity_propagation = cluster.AffinityPropagation(
            # random_state=self.params["random_state"]
        )
        agglomerative_ward_linkage = cluster.AgglomerativeClustering(
            n_clusters=self.params["k_clusters"]
        )

        tbc = TBC(
            k_clusters=self.params["k_clusters"],
            tau=int(0.5 * len(self.X) / self.params["k_clusters"])
        )

        """
        # Not used because of Errors and possible bugs in the implementations.
        # A recent blog post also assuming this was found.

        birch = cluster.Birch(
            n_clusters=self.params["k_clusters"]
        )
        bisecting_kmeans = cluster.BisectingKMeans(
            n_clusters=self.params["k_clusters"],
            random_state=self.params["random_state"],
            n_init=self.params["n_init"],
        )
        spectral_biclustering = cluster.SpectralBiclustering(
            n_clusters=self.params["k_clusters"],
            random_state=self.params["random_state"],
            n_init=self.params["n_init"],
        )
        spectral_coclustering = cluster.SpectralBiclustering(
            n_clusters=self.params["k_clusters"],
            random_state=self.params["random_state"],
            n_init=self.params["n_init"],
        )
        """
        self.clustering_algorithms = [
            (kmeans, "k-means"),
            (mini_batch_kmeans, "Mini Batch k-means"),
            (ms, "Mean-Shift"),
            (dbscan, "DBSCAN"),
            (spectral, "Spectral Clustering"),
            (optics, "OPTICS"),
            (affinity_propagation, "Affinity Propagation"),
            (agglomerative_ward_linkage, "Agglomerative Clustering (ward)"),
            (gmm, "GMM"),
            (tbc, "TBC"),
        ]

    def fit(self):
        """
        Apply each clustering algorithm self.num_algo_runs times and get information about the results of their best runs.
        """

        for (algo, algo_name) in self.clustering_algorithms:
            best_score = float('-inf')
            for i in range(self.num_algo_runs):
                algo_copy = copy.deepcopy(algo)
                algo_init_i = ClusteringAlgo(
                    algo_copy, algo_name, self.X, feature_selection_mode=self.feature_selection_mode)
                algo_results_i = self.get_clustering_infos(algo_init_i)
                # Update result_infos with best run
                if algo_results_i["score"] != "No result" and algo_results_i["score"] > best_score:
                    best_score = algo_results_i["score"]
                    self.result_infos[algo_name] = algo_results_i
                # Exit loop if the cluster assignments didn't change
                elif algo_results_i["score"] == best_score:
                    if np.array_equal(algo_results_i["cluster_assignments"], self.result_infos[algo_name]["cluster_assignments"]):
                        break

    def get_clustering_infos(self, algo):
        """
        Get information about the results of a clustering algorithm and return them in form of a dictionary.
        """
        cluster_infos = algo.get_cluster_info_dict()
        result_infos = {
            "score":                algo.get_result_score(),
            "n_selected_features":  len(algo.X[0]),
            "feature_selection":    algo.feature_selection,
            "k_selected_clusters":  len(cluster_infos),
            "cluster_infos":        cluster_infos,
            "cluster_assignments":  algo.clustering_result,
            "result_std":           algo.get_result_std()
        }
        return result_infos

    def find_best_clustering_algo(self):
        """
        Determine the best clustering algo by its score.
        """
        best_score = float('-inf')

        for (_, algo_name) in self.clustering_algorithms:
            algo_score = self.result_infos[algo_name]["score"]
            if algo_score != "No result":
                if algo_score > best_score:
                    best_score = algo_score
                    self.best_algo_name = algo_name

    def get_all_results_string(self):
        """
        Return a string with information about all applied clustering algorithms and their corresponding results.
        """
        all_results_string = ""
        all_results_string += "-"*79 + "\n"
        all_results_string += "-"*34 + "ALL RESULTS" + "-"*34 + "\n"

        for algo_name, algo_info in self.result_infos.items():
            all_results_string += "-"*79 + "\n"
            all_results_string += "--- " + algo_name + "\n"

            if algo_info["cluster_assignments"][0] == "No result":
                all_results_string += "The Algorithm could not find a result." + "\n"
            else:
                all_results_string += "Score: " + \
                    str(algo_info["score"]) + "\n"
                all_results_string += "Features selected: " + \
                    str(algo_info["n_selected_features"]) + "\n"
                all_results_string += "Clusters found: " +\
                    str(algo_info["k_selected_clusters"]) + "\n"
                all_results_string += "Cluster Info: " +\
                    str(algo_info["cluster_infos"]) + "\n"
                all_results_string += "Result standard deviation: " +\
                    str(algo_info["result_std"]) + "\n"
                all_results_string += "\n" + "Cluster assignments: " + "\n"
                all_results_string += str(
                    algo_info["cluster_assignments"]) + "\n"
        return all_results_string

    def get_best_result_string(self):
        """
        Return a string with information about the best clustering algorithm and corresponding results.
        """
        best_result_string = ""
        best_result_string += "-"*79 + "\n"
        best_result_string += "-"*34 + "BEST RESULT" + "-"*34 + "\n"
        self.find_best_clustering_algo()
        if self.best_algo_name == None:
            best_result_string += "The Algorithm could not find a result."
        else:
            best_algo_result_infos = self.result_infos[self.best_algo_name]
            best_result_string += "--- " + self.best_algo_name + "\n"
            best_result_string += "Score: " + \
                str(best_algo_result_infos["score"]) + "\n"
            best_result_string += "Features selected: " + \
                str(best_algo_result_infos["n_selected_features"]) + "\n"
            best_result_string += "Clusters found: " + \
                str(best_algo_result_infos["k_selected_clusters"]) + "\n"
            best_result_string += "Cluster Info: " + \
                str(best_algo_result_infos["cluster_infos"]) + "\n"
            best_result_string += "Result standard deviation: " + \
                str(best_algo_result_infos["result_std"]) + "\n"
            best_result_string += "\n" + "Cluster assignments: " + "\n"
            best_result_string += str(
                best_algo_result_infos["cluster_assignments"]) + "\n"
        return best_result_string
