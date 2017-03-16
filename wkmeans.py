
################################################################################
#                                                                              #
# Original code for vanilla k-means and k-means++ found here:                  #
# https://datasciencelab.wordpress.com/2013/12/12/                             #
# clustering-with-k-means-in-python/                                           #
# https://datasciencelab.wordpress.com/2014/01/15/                             #
# improved-seeding-for-clustering-with-k-means/                                #
#                                                                              #
# Many modifications to allow for weighted k-means added by Olivia Guest.      #
#                                                                              #
# Pseudo-code by Bradley C Love:                                               #
#                                                                              #
# scaling_{t_0}=1/k where k is number of clusters                              #
#                                                                              #
# KITTY                                                                        #
# **find euclidean distance of each point to each centroid                     #
# **multiply by scaling factor for each centroid                               #
# **find closest scaled distance centroid for each item                        #
# **update cluster centroids                                                   #
# **computer luce_choice_share based on above centroid counts                  #
# **scaling_{next}=alpha*scaling_{current}+(1-alpha)*luce_choice_share         #
# if not stopping condition,                                                   #
#   GOTO KITTY                                                                 #
#                                                                              #
# stopping condition?                                                          #
#     -fixed number of iterations                                              #
#     -and/or measure of how much cluster assignments change.                  #
#                                                                              #
# idea if problem: make alpha increase over iterations                         #
#                                                                              #
################################################################################

from __future__ import division

import random
import warnings
import itertools
import sklearn.datasets

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

def counted(f):
    """Decorator for returning number of times a function has been called.
    Author: FogleBird; Source: http://stackoverflow.com/a/21717396;
    """
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)
    wrapped.calls = 0
    return wrapped

class KMeans():
    """Class for running weighted k-means.

    Required argument:
    K     -- the number of clusters.

    Keyword arguments:
    X     -- the data (default None; thus auto-generated, see below);
    N     -- number of unique data points to generate (default 0);
    c     -- number of non-unique points represented by a data point (default
             None; to mean every data point is unique);
    alpha -- the exponent used to calculate the scaling factor (default 0);
    beta  -- the stickiness parameter used during time-averaging (default 0).

    ***

    Original author: The Data Science Lab; Source: https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python

    Weighted k-means author: Olivia Guest.
    """
    def __init__(self, K, X=None, N=0, c=None, alpha=0, beta=0):

        self.K = K
        if X is None:
            if N == 0:
                raise Exception("If no data is provided, \
                                 a parameter N (number of points) is needed")
            else:
                self.N = N
                self.X = self._init_gauss(N)
        else:
            self.X = X
            self.N = len(X)

        # The coordinates of the centroids:
        self.mu = None

        # We need to keep track of previous centroids.
        self.old_mu = None

        # Numpy array of clusters containing their index and member items.
        self.clusters = None
        # self.cluster_indices = np.asarray([None for i in self.X])

        # What kind of initialisation we want, only vanilla or k++ available:
        self.method = None

        # For scaling distances as a function of cluster size:
        # the power the cardinalities will be raised to;
        self.alpha = alpha
        # and to scale the distances between points and centroids, which is
        # initialised to 1/k for all clusters.
        self.scaling_factor = np.ones((self.K)) / self.K

        # The stickiness used within the time-averaging.
        self.beta = beta

        # How many counts are represented by a single data point:
        if c is None:
            self.counts_per_data_point = [1 for x in self.X]
        else:
            self.counts_per_data_point = c

        # How many counts are in each cluster:
        self.counts_per_cluster = [0 for x in range(self.K)]

    def _init_gauss(self, N):
        """Create test data in which there are three bivariate Gaussians. Their
        centers are arranged in an equilateral triange, with the top Gaussian
        having double the density of the other two. This is tricky test data
        because the points in the top part is double that of the lower parts of
        the space, meaning that a typical k-means run will create unequal
        clusters, while a weighted k-means will attempt to balance data points
        betwen the clusters.
        """
        # Set up centers of bivariate Gaussians in a equilateral triangle.
        centers = [[0, 0], [1, 0], [0.5, np.sqrt(0.75)]]

        # The SDs:
        cluster_std = [0.3, 0.3, 0.3]

        # The number of Gaussians:
        n_clusters = len(centers)

        # The number of points, recall we need double at the top point hence 3/4
        # of points are being generated now.
        n_samples = int(0.75 * N)


        data, labels_true = sklearn.datasets.make_blobs(n_samples=n_samples,\
                            centers=centers, cluster_std=cluster_std)

        # Now to generate the extra data points for the top of the triangle:
        centers = [[0.5, np.sqrt(0.75)]]
        cluster_std = [0.3]
        n_clusters = len(centers)
        extra, labels_true = sklearn.datasets.make_blobs(n_samples=int(0.25*N),\
                             centers=centers, cluster_std=cluster_std)

        # Merge the points together to create the full dataset.
        data = np.concatenate((data, extra), axis=0)

        return data

    @counted
    def plot_clusters(self, snapshot = '0'):
        """Plot colour-coded clusters using a scatterplot."""
        X = self.X
        fig = plt.figure(figsize=(5,5))
        ax = plt.gca()
        palette = itertools.cycle(sns.color_palette())

        if self.mu and self.clusters:
            # If we have clusters and centroids, graph them.
            mu = self.mu
            clus = self.clusters
            K = self.K
            for m, clu in enumerate(clus):
                # For each cluster: a) get a colour from the palette;
                cs = cm.spectral(1.*m/K)
                cs = next(palette)
                # b) plot the data points in the cluster;
                plt.plot(zip(*clus[m])[0], zip(*clus[m])[1], '.',\
                         markersize=8, color=cs, alpha=0.5)
                # and c) plot the centroid of the cluster.
                plt.plot(mu[m][0], mu[m][1], 'o', marker='*',\
                         markersize=12, color=cs,  markeredgecolor='white',
                         markeredgewidth=1.0)
        else:
            # Otheriwse, just graph the data.
            plt.plot(zip(*X)[0], zip(*X)[1], '.', alpha=0.5)

        # Give the title as a function of the initialisation method:
        if self.method == '++':
            title = 'K-means++'
        else:
            title = 'K-means with random initialization'
        pars = r'$N=%s, K=%s, \alpha=%s$' % (str(self.N), str(self.K),\
               str(self.alpha))
        plt.title('\n'.join([pars, title]), fontsize=16)

        # Finally, save the figure as a PNG.
        plt.savefig('kpp_N%s_K%s_alpha%s_%s.png' % (str(self.N), str(self.K),\
                    str(self.alpha), str(snapshot)),\
                    bbox_inches='tight', dpi=200)

    @counted
    def _cluster_points(self):
        """Cluster the points."""

        # Initialise the values for the clusters and their counts
        clusters = [[] for i in range(self.K)]
        counts_per_cluster = [0 for i in range(self.K)]

        ########################################################################
        #Firstly perform classical k-means, weighting the Euclidean distances.
        ########################################################################
        for index, x in enumerate(self.X):
                # For each data point x, find the minimum weighted distance to
                # cluster i from point x.
                bestmukey = min([(i[0],\
                                self.scaling_factor[i[0]] *\
                                np.linalg.norm(x-self.mu[i[0]]))\
                                for i in enumerate(self.mu)],
                            key=lambda t:t[1])[0]
                # Add the data point x to the cluster it is closest to.
                clusters[bestmukey].append(x)
                counts_per_cluster[bestmukey] += self.counts_per_data_point[index]
                # self.cluster_indices[index]  = bestmukey

        # Update the clusters.
        self.clusters = clusters
        self.counts_per_cluster = counts_per_cluster
        print '\tNumber of unique items per cluster: ', [len(x) for x in self.clusters]
        print '\tNumber of items per cluster: ', self.counts_per_cluster

        # assert [np.count_nonzero(self.cluster_indices == i)\
        #         for i in range(self.K)] == [len(x) for x in self.clusters]
        # print '\tNumber of items per cluster: ',\
        #         [np.count_nonzero(self.cluster_indices == i) for i in range(self.K)]

        ########################################################################
        # Secondly, calculate the scaling factor for each cluster.
        ########################################################################
        # Now that we have clusters to work with (at initialisation we don't),
        # we can calculate the scaling_factor per cluster. This calculates the
        # cardinality of the cluster raised to the power alpha, so it is purely
        # a function of the number of items in each cluster and the value of
        # alpha.
        scaling_factor = np.asarray([self.counts_per_cluster[index]**self.alpha\
                                for index, cluster in enumerate(self.clusters)])

        # Now we have all the numerators, divide them by their sum. This is also
        # known as the Luce choice share.
        scaling_factor = scaling_factor/np.sum(scaling_factor)

        # The scaling factors should sum to one here.
        # print 'Sum of luce choice shares:', np.around(np.sum(scaling_factor))
        # assert np.around(np.sum(scaling_factor)) == 1

        # Now we want to employ time-averaging on the scaling factor.
        scaling_factor = (1 - self.beta) * scaling_factor +\
                         (self.beta) * self.scaling_factor

        # Update the scaling factors for the next time step.
        self.scaling_factor = scaling_factor

        # The scaling factors should sum to one here too.
        # print 'Sum of scaling factors:',\
        #         np.around(np.sum(self.scaling_factor))
        # assert np.around(np.sum(self.scaling_factor)) == 1
        print '\tScaling factors per cluster: ', self.scaling_factor

    def _reevaluate_centers(self):
        """Update the controids (aka mu) per cluster."""
        clusters = self.clusters
        new_mu = []
        for k in self.clusters:
            # For each key, add a new centroid (mu) by calculating the cluster
            #  mean.
            new_mu.append(np.mean(k, axis = 0))
        # Update the list of centroids that we just calculated.
        self.mu = new_mu

    def _has_converged(self):
        """Check if the items within each cluster have stabilised between two
        runs. This checks to see if the Euclidean distance between the centroids
        is lower than a fixed constant (0.001).
        """
        # Calculate the distance between previous and current centroids.
        dist = np.linalg.norm(np.asarray(self.mu)-np.asarray(self.old_mu))
        if self.clusters:
            for clu in self.clusters:
                # For each clusters, check the length. If zero, we have a
                # problem, we have lost clusters.
                if len(clu) is 0:
                    raise ValueError('One or more clusters disappeared because'\
                                     ' all points rushed away to other'\
                                     ' cluster(s). Try increasing the'\
                                     ' stickiness parameter (beta).')
            print '\tDistance between previous and current centroids: ', dist

        # Return true if the items in each cluster have not changed much since
        # the last time this was run
        return dist < 0.00001

    def find_centers(self, method='random'):
        """Find the centroids per cluster until equilibrium."""
        self.method = method
        X = self.X
        K = self.K
        # Use max_runs to stop running forever in cases of non-convergence:
        max_runs = 200
        # Previous centroids set to random values.
        self.old_mu = random.sample(X, K)

        if method != '++':
            # If method of initialisation is not k++, use random centeroids.
            self.mu = random.sample(X, K)

        while not self._has_converged() and\
              self._cluster_points.calls < max_runs:
            print 'Run: ', self._cluster_points.calls
            # While the algorithm has neither converged nor been run too many
            # times:
            # a) keep track of old centroids;
            self.old_mu = self.mu
            # b) assign all points in X to clusters;
            self._cluster_points()
            # c) recalculate the centers per cluster.
            self._reevaluate_centers()

class KPlusPlus(KMeans):
    """Augment the KMeans class with k-means++ capabilities.
    Author: The Data Science Lab; Source: https://datasciencelab.wordpress.com/
    2014/01/15/improved-seeding-for-clustering-with-k-means/.
    """
    def _dist_from_centers(self):
        """Calculate the distance of each point to the closest centroids."""
        cent = self.mu
        X = self.X
        D2 = np.array([min([np.linalg.norm(x-c)**2 for c in cent]) for x in X])
        self.D2 = D2

    def _choose_next_center(self):
        """Select the next center probabilistically."""
        self.probs = self.D2/self.D2.sum()
        self.cumprobs = self.probs.cumsum()
        r = random.random()
        ind = np.where(self.cumprobs >= r)[0][0]
        return(self.X[ind])

    def init_centers(self):
        """Initialise the centers."""
        self.mu = random.sample(self.X, 1)
        while len(self.mu) < self.K:
            self._dist_from_centers()
            self.mu.append(self._choose_next_center())

    def plot_init_centers(self):
        """Plot the centers using a scatterplot."""
        X = self.X
        fig = plt.figure(figsize=(5,5))
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.plot(zip(*X)[0], zip(*X)[1], '.', alpha=0.5)
        plt.plot(zip(*self.mu)[0], zip(*self.mu)[1], 'ro')
        plt.savefig('kpp_init_N%s_K%s.png' % (str(self.N),str(self.K)),\
                    bbox_inches='tight', dpi=200)
