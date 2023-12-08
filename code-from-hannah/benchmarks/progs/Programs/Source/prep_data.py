import numpy as np

X = np.load("../../../data/points_16.npz")['arr_0']

for item in X:
    # for item in row:
    print(int(item),end=' ')
    # print()
