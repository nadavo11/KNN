import json
import numpy as np   # check out how to install numpy
from utils import load, plot_sample
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
# =========================================
#       Homework on K-Nearest Neighbors
# =========================================
# Course: Introduction to Information Theory
# Lecturer: Haim H. Permuter.
#
# NOTE:
# -----
# Please change the variable ID below to your ID number as a string.
# Please do it now and save this file before doing the assignment

ID = '312349509'

# Loading and plot a sample of the data
# ---------------------------------------
# The MNIST database contains in total of 60000 train images and 10000 test images
# of handwritten digits.
# This database is a benchmark for many classification algorithms, including neural networks.
# For further reading, visit http://yann.lecun.com/exdb/mnist/
#
# You will implement the KNN algorithm to classify two digits from the database: 3 and 5.
# First, we will load the data from .MAT file we have prepared for this assignment: MNIST_3_and_5.mat

Xtrain, Ytrain, Xvalid, Yvalid, Xtest = load('MNIST_3_and_5.mat')

# The data is divided into 2 pairs:
# (Xtrain, Ytrain) , (Xvalid, Yvalid)
# In addition, you have unlabeled test sample in Xtest.
#
# Each row of a X matrix is a sample (gray-scale picture) of dimension 784 = 28^2,
# and the digit (number) is in the corresponding row of the Y vector.
#
# To plot the digit, see attached function 'plot_sample.py'

sampleNum = 0
plot_sample(Xvalid[sampleNum, :], Yvalid[sampleNum, :])

# Build a KNN classifier based on what you've learned in class:
#
# 1. The classifier should be given a train dataset (Xtrain, Ytain),  and a test sample Xtest.
# 2. The classifier should return a class for each row in test sample in a column vector Ytest.
#
# Finally, your results will be saved into a <ID>.txt file, where each row is an element in Ytest.
#
# Note:
# ------
# For you conveniece (and ours), give you classifications in a 1902 x 1 vector named Ytest,
# and set the variable ID at the beginning of this file to your ID.

training_data = [Xtrain,Ytrain]
def KNN_classify(k,X,training_data):
    Xtrain = training_data[0]
    Ytrain = training_data[1]
    d = {}
    # Iterate for each training datapoint:
    # ------

    for i, digit in enumerate(Xtrain):

        #plot_sample(digit,[3])

        # calc auclidien distanse:
        # -then save in a HashTable
        distance = np.linalg.norm(X - digit)
        d[i] = distance

    #get K minimum distance neighbors
    K = sorted(d, key=d.get)[:k]
    arr = []
    #run over each of nearest n's, and count them.
    for index in K:

        neigh = Ytrain[index]
        arr.append(neigh[0])
    c = Counter(arr)

    return c.most_common()[0][0]  #return the most common digit among k neighbors

for X in Xtest:
    guess = KNN_classify(5,X)
    plot_sample(X,[guess])
    # Example submission array - comment/delete it before submitting:

Ytest = np.arange(0, Xtest.shape[0])

# save classification results
print('saving')
np.savetxt(f'{ID}.txt', Ytest, delimiter=", ", fmt='%i')
print('done')
