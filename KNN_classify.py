import numpy as np   # check out how to install numpy
from utils import load, plot_sample
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

def KNN_classify(k,X,training_data):
    Xtrain = training_data[0]
    Ytrain = training_data[1]
    #open a dict to store the label of eachdatapoint (by distance)
    d = {}
    # Iterate for each training datapoint:
    # ------

    for i, digit in enumerate(Xtrain):
        # calc auclidien distanse:
        # -then save in a HashTable
        distance = np.linalg.norm(X - digit)
        d[i] = distance

    # get K minimum distance neighbors
    k_nearst_neighbors = sorted(d, key=d.get)[:k]

    arr = []
    # run over each of nearest n's, and count them.
    for index in k_nearst_neighbors:
        neighbor = Ytrain[index]
        arr.append(neighbor[0])

    # return the most common digit among k neighbors
    return Counter(arr).most_common()[0][0]