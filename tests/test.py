"""Testing."""
import os
import random
import unittest

import numpy as np
import wkmeans as wkm


class TestWKMeans(unittest.TestCase):

    def test_init(self):
        # Case 1: Do not give the data, give the counts, give the number of
        # data points you want it to auto-generate.
        K = 3
        N = 10
        random_counts = np.random.random_sample((N,)) * 100
        alpha = 3
        beta = 0.9
        max_runs = 10
        label = 'Test'
        verbose = True
        mu = [[0, 0], [1, 2], [6, 4]]
        max_diff = 0.005
        w = wkm.KPlusPlus(K, N=N, c=random_counts, alpha=alpha, beta=beta,
                          max_runs=max_runs, label=label, verbose=verbose,
                          mu=mu, max_diff=max_diff)
        self.assertTrue(w.K == K)
        self.assertTrue(w.N == N)
        self.assertTrue((w.counts_per_data_point == random_counts).all())
        self.assertTrue(w.alpha == alpha)
        self.assertTrue(w.beta == beta)
        self.assertTrue(w.max_runs == max_runs)
        self.assertTrue(w.label == label)
        self.assertTrue(w.verbose == verbose)
        self.assertTrue(w.mu == mu)
        self.assertTrue(w.method == 'manual')  # because we gave it mu
        self.assertTrue(w.clusters is None)
        self.assertTrue((w.cluster_indices == [None for i in w.X]).all())
        self.assertTrue(w.counts_per_cluster == [0 for x in range(w.K)])
        self.assertIsInstance(w.X, np.ndarray)
        self.assertTrue(len(w.X.shape) == 2)
        self.assertTrue((w.scaling_factor == np.ones((w.K)) / w.K).all())
        self.assertTrue(w.runs == 0)
        self.assertTrue(w.max_diff == max_diff)

        # Case 2: Do not give the data, do not give the counts, give the number
        # of datapoints you want it to generate.
        w = wkm.KPlusPlus(K, N=N)
        self.assertTrue(w.K == K)
        self.assertTrue(w.N == N)
        self.assertTrue(w.counts_per_data_point ==
                        [1 for x in w.X])
        self.assertTrue(w.alpha == 0)
        self.assertTrue(w.beta == 0)
        self.assertIsInstance(w.X, np.ndarray)
        self.assertTrue(len(w.X.shape) == 2)
        self.assertTrue(w.counts_per_cluster == [0 for x in range(w.K)])
        self.assertTrue(w.method == 'random')

        # Case 3: Give data, do not give counts, do not give number of
        # datapoints.
        X = [[1, 2], [3, 12], [8, 200]]
        N = len(X)
        w = wkm.KPlusPlus(K, X=X)
        self.assertTrue(w.K == K)
        self.assertTrue(w.N == N)

    def test__init_gauss(self):
        K = 3
        N = 10
        random_counts = np.random.random_sample((N,)) * 100
        w = wkm.KPlusPlus(K, N=N, c=random_counts)
        data = w._init_gauss(N)
        self.assertTrue(len(data) == N)

    def test_plot_clusters(self):
        K = 3
        N = 10
        random_counts = np.random.random_sample((N,)) * 100
        w = wkm.KPlusPlus(K, N=N, c=random_counts)
        w.plot_clusters()
        cwd = os.getcwd()
        filename = '/kpp_N_%i_K_%i_alpha_%i_beta_%i_%i.png' % (w.N, w.K,
                                                               w.alpha,
                                                               w.beta,
                                                               0)
        self.assertTrue(cwd + filename)
        os.remove(cwd + filename)

    def test__cluster_points(self):
        K = 3
        N = 10
        random_counts = np.random.random_sample((N,)) * 100
        w = wkm.KPlusPlus(K, N=N, c=random_counts, verbose=False)
        w.mu = random.sample(list(w.X), w.K)
        self.assertTrue(w.clusters == None)
        w._cluster_points()
        self.assertTrue(len(w.clusters) == K)

        K = 3
        X = [[0, 0], [3, 3], [300, 300]]
        # Because the data is so simplistic, I know that clusters will be:
        clusters = [[[0, 0]], [[3, 3]], [[300, 300]]]
        w = wkm.KPlusPlus(K, X=X, verbose=False)
        w.mu = random.sample(w.X, w.K)
        self.assertTrue(w.clusters == None)
        w._cluster_points()
        self.assertTrue(len(w.clusters) == K)
        # Confirm clusters computed inside the object are identical (different
        # order allowed) to those I know:
        for c in w.clusters:
            for t, test_c in enumerate(clusters):
                if c == test_c:
                    del clusters[t]
        # Assert it has been consumed:
        self.assertTrue(clusters == [])

        K = 2
        X = [[0, 0], [3, 3], [300, 300]]
        # Because the data is so simplistic, I know that clusters will be:
        clusters = [[[0, 0], [3, 3]], [[300, 300]]]
        w = wkm.KPlusPlus(K, X=X, verbose=False)
        w.mu = [[0, 0], [300, 300]]
        self.assertTrue(w.clusters == None)
        w._cluster_points()
        self.assertTrue(len(w.clusters) == K)

        # Confirm clusters computed inside the object are identical (different
        # order allowed) to those I know:
        for c in w.clusters:
            for t, test_c in enumerate(clusters):
                if c == test_c:
                    del clusters[t]
        # Assert it has been consumed:
        self.assertTrue(clusters == [])


if __name__ == '__main__':
    unittest.main()
