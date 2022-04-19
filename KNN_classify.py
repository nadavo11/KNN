import numpy as np   # check out how to install numpy
from utils import load, plot_sample
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

def KNN_classify(k,X,training_data):
    Xtrain = training_data[0]
    Ytrain = training_data[1]

    d = {}
    # Iterate for each training datapoint:
    # ------

    for i, digit in enumerate(Xtrain):

        # plot_sample(digit,[3])

        # calc auclidien distanse:
        # -then save in a HashTable
        distance = np.linalg.norm(X - digit)
        d[i] = distance

    # get K minimum distance neighbors
    K = sorted(d, key=d.get)[:k]
    arr = []
    # run over each of nearest n's, and count them.
    for index in K:
        neigh = Ytrain[index]
        arr.append(neigh[0])
    c = Counter(arr)

    return c.most_common()[0][0]  # return the most common digit among k neighbors
