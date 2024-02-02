from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt


tanimoto_matrix = np.load("tanimoto_matrix.npy")

db = DBSCAN(metric='precomputed', eps=0.7, min_samples=6).fit(X=tanimoto_matrix)

labels = list(db.labels_)


print ("labels len", len(labels))
print ("noise", list(labels).count(-1))

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

# show how big the clusters are
count = {unique_label: labels.count(unique_label) for unique_label in set(labels)}
print(sorted(count.items(), key=lambda x: x[1], reverse=True))

np.savetxt('labels.dat', labels, fmt='%d')

# plot on a 2d such that you see the cluster
