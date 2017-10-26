"""Run k-means++ on test data and save output to image in same directory."""

import numpy as np
import wkmeans as wkm
# If you base your code on this and put wkmeans in its own directory (called
# weighted_k_means, as would happen if you clone this repo) use the following
# to import instead:
# import weighted_k_means.wkmeans as wkm

from datetime import datetime
startTime = datetime.now()

# Number of data points we want to generate:
N = 5000
# Random counts because each data point is not unique and can even be "empty",
# i.e., have a count of zero:
# random_counts = np.random.randint(100, size=(N))  # integers for the counts
random_counts = np.random.random_sample((N,)) * 100  # floats for the counts

# Initialise the class with some default values:
wkmeans = wkm.KPlusPlus(3, N=N, c=random_counts, alpha=3, beta=0.9)

# If you have your own data use:
# wkmeans = wkm.KPlusPlus(3, X=my_data, c=my_counts, alpha=3, beta=0.9)

# Initialise centroids using k-means++...
wkmeans.init_centers()
# and run to find clusters:
wkmeans.find_centers(method='++')

# Now plot the result:
wkmeans.plot_clusters(wkmeans.plot_clusters.calls)

# We're done so print some useful info:
print 'The End!'
print '\tRun time: ', datetime.now() - startTime
print '\tTotal runs: ', wkmeans._cluster_points.calls
print '\tNumber of unique items per cluster: ', [len(x) for x in
                                                 wkmeans.clusters]
print '\tNumber of items per cluster: ', wkmeans.counts_per_cluster
