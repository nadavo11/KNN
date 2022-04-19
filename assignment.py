import json
import numpy as np   # check out how to install numpy
from utils import load, plot_sample
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from KNN_classify import KNN_classify

# Course: Machine Learning & Introduction to Information Theory
ID = '312349509'

# Loading and plot a sample of the data
# ---------------------------------------
# The MNIST database contains in total of 60000 train images and 10000 test images
# of handwritten digits.
# This database is a benchmark for many classification algorithms, including neural networks.
# For further reading, visit http://yann.lecun.com/exdb/mnist/
#
# Here we implement the KNN algorithm to classify two digits from the database: 3 and 5.
# First, we will load the data from .MAT file we have prepared for this assignment: MNIST_3_and_5.mat
#________________________________________

Xtrain, Ytrain, Xvalid, Yvalid, Xtest = load('MNIST_3_and_5.mat')

# The data is divided into 2 pairs:
# (Xtrain, Ytrain) , (Xvalid, Yvalid)
# In addition, you have unlabeled test sample in Xtest.
#
# Each row of a X matrix is a sample (gray-scale picture) of dimension 784 = 28^2,
# and the digit (number) is in the corresponding row of the Y vector.
#
# To plot the digit, see attached function 'plot_sample.py'
#____________________________________________________________
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

training_data = [Xtrain,Ytrain]
Ytest = np.array([])
for X in Xtest:
    guess = KNN_classify(5,X,training_data)
    Ytest = np.append(Ytest, guess)
    #plot_sample(X,[guess])
    # Example submission array - comment/delete it before submitting:



# save classification results
print('saving')
np.savetxt(f'{ID}.txt', Ytest, delimiter=", ", fmt='%i')
print('done')
