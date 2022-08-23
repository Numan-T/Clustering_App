import random
import heapq  # Min-Heap
from heap import MaxHeap
import numpy as np


class ClusterCenter:
    def __init__(self, label, value):
        self.label = label
        self.value = value  # feature vector that is the cluster center
        self.cluster = MaxHeap()  # top element is the biggest


class TBC:
    def __init__(self, k_clusters, tau=0, seed=random.seed(), max_iter=20):
        self.k_clusters = k_clusters  # number of clusters
        self.tau = tau  # max cluster size difference
        self.random_engine = random.Random(seed)
        self.max_iter = max_iter  # max iterations of cluster updates
        self.data = []  # input data
        self.n_samples = 0  # data size (number of observations)
        self.n_features = 0   # number of dimensions (features)
        # cluster assignments (labels) of all feature vectors
        self.labels_ = []
        self.cluster_centers = []
        self.max_num_largest_clusters = 0
        # max bound of cluster size
        self.bound = 0

    def assign_init_values(self, X):
        """
        Define initialization values and calculate the maximum number of clusters and the maximum size bound of a cluster.
        """
        self.data = X
        self.n_samples, self.n_features = X.shape
        self.max_num_largest_clusters = (self.n_samples -
                                         self.k_clusters*np.ceil(self.n_samples/self.k_clusters)) / self.tau + self.k_clusters
        # max bound of cluster size
        self.bound = int(
            ((self.n_samples - self.max_num_largest_clusters*self.tau) / self.k_clusters) + self.tau)

    def fit(self, X):
        """
        Start the TBC algorithm. First declare (random) initial cluster centers. Then assign the observations to the clusters and update the cluster centers until the clusters converge or self.max_iter iterations are reached.
        """
        self.assign_init_values(X)
        self.init_cluster_centers()

        i = 0
        while True:
            i += 1
            old_assignment = self.labels_
            self.reassign()
            # break if converging or after 20 iterations
            # if old_assignment == self.labels_ or i > self.max_iter:
            if np.array_equal(old_assignment, self.labels_) or i > self.max_iter:

                # for usibility with sklearn parts save as np.array
                self.labels_ = np.array(self.labels_)
                break
            self.update_cluster_centers()
        return 0

    def init_cluster_centers(self):
        """
        Initialize k randomly chosen data points as cluster centers.
        """
        indices = list(range(self.n_samples))
        self.random_engine.shuffle(indices)

        for i in range(self.k_clusters):
            ith = self.data[indices[i]]  # random data point
            self.cluster_centers.append(ClusterCenter(label=i, value=ith))

    def reassign(self):
        """
        Reset all cluster assignments and reassign the data points to the cluster with the nearest center.
        """
        self.labels_ = [-1] * self.n_samples
        for i in range(self.n_samples):
            self.assign_to_nearest_center(i, self.data[i])

    def update_cluster_centers(self):
        """
        Update all cluster centers to be a vector that contains the mean values of each dimension over all data points assigned to a cluster.
        """
        for cluster_center in self.cluster_centers:
            # 1.count items in the cluster
            cluster_size = len(cluster_center.cluster.heap_list)

            # 2. sum the values of each dimension in the cluster
            new_cluster_value = [0]*self.n_features
            while cluster_center.cluster.heap_list:  # while not empty
                _, data_idx = cluster_center.cluster.heappop()
                data_value = self.data[data_idx]
                for i in range(self.n_features):
                    new_cluster_value[i] += data_value[i]

            # 3. divide by the length of the data and update the value of the cluster center
            if (cluster_size > 0):
                cluster_center.value = [
                    i / cluster_size for i in new_cluster_value]

    def assign_to_nearest_center(self, data_idx, data_item):
        """
        Assign the input data point to the nearest cluster if it hasn't reached the size boundary. Else check if it is nearer to the cluster center then the boundary data point of this cluster and if so replace it and start the processs again for the replaced boundary data point. If not repeat all steps with the next nearest cluster until it could be assigned.

        Attributes:
            dataIndex (int): Index of an data point.
            dataItem (list(float)): A data point/feature vector.
        """
        # Initialize a min-heap for the distances to all cluster centers
        nearest_clusters = []
        heapq.heapify(nearest_clusters)

        # Get distances and push them into min-heap with cluster center indexes
        for i in range(self.k_clusters):
            dist = self.distance(self.cluster_centers[i].value, data_item)
            heapq.heappush(nearest_clusters, (dist, i))

        # Assign the data point to a cluster
        while nearest_clusters:  # while not empty
            nearest_center_dist, nearest_center_idx = heapq.heappop(
                nearest_clusters)
            nearest_center = self.cluster_centers[nearest_center_idx]

            # If the nearest cluster is not full, the data is assigned to the cluster
            if (len(nearest_center.cluster.heap_list) < self.bound):
                nearest_center.cluster.heappush(
                    (nearest_center_dist, data_idx))
                self.labels_[data_idx] = nearest_center.label
                return
            else:
                # If the cluster size reaches the upper limit of the boundary,
                # compare the distance between the boundary data and the center
                # to the distance between the current data and the center

                boundary_dist, boundary_idx = nearest_center.cluster.heap_list[0]

                # If the boundary data is farther than the current data
                if (boundary_dist > nearest_center_dist):
                    # replace the boundary data with the current data
                    nearest_center.cluster.heappop()
                    nearest_center.cluster.heappush(
                        (nearest_center_dist, data_idx))
                    self.labels_[data_idx] = nearest_center.label
                    # assign the kicked out boundary data to another center
                    self.assign_to_nearest_center(
                        boundary_idx, self.data[boundary_idx])
                    return

    def distance(self, data1, data2):
        """
        Calculate the sum of squared distances of two data points.
        """
        result = 0
        for i in range(self.n_features):
            result += (data1[i] - data2[i]) * (data1[i] - data2[i])
        return result

    def get_assignments(self):
        """
        Return the obtained cluster assignments.
        """
        return np.array(self.labels_)
