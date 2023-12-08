import numpy as np
# from matplotlib import pyplot as plt
# from sklearn.datasets.samples_generator import make_blobs

n=100
d=8
x = np.sum(np.around(np.random.rand(n,d)), axis=0)
# blobs, clust = m 0
path = "../../../../data"
np.savez("../../../data/points_100_8.npz", x)
