# This code runs the k-means++ algorithm on some test data and saves the
# output to an image file in the same directory.

import numpy as np
import wkmeans as km
# If you base your code on this and put wkmeans in its own directory (called
# weighted_k_means, as would happen if you clone this repo) use the following
# to import instead:
# import weighted_k_means.wkmeans as km

from datetime import datetime
startTime = datetime.now()

# Number of data points we want to generate:
N = 5000
# Random counts because each data point is not unique and can even be "empty",
# i.e., have a count of zero:
random_counts = np.random.randint(100, size=(N))

# Initialise the class with some default values:
kmeans = km.KPlusPlus(3, N=N, c=random_counts, alpha=3, beta=0.9)

# If you have your own data use:
# kmeans = km.KPlusPlus(3, X=my_data, c=my_counts, alpha=3, beta=0.9)

# Initialise centroids using k-means++...
kmeans.init_centers()
# and run to find clusters:
kmeans.find_centers(method='++')

# Now plot the result:
kmeans.plot_clusters(kmeans.plot_clusters.calls)

# We're done so print some useful info:
print 'The End!'
print '\tRun time: ', datetime.now() - startTime
print '\tTotal runs: ', kmeans._cluster_points.calls
print '\tNumber of unique items per cluster: ', [len(x) for x in kmeans.clusters]
print '\tNumber of items per cluster: ', kmeans.counts_per_cluster
