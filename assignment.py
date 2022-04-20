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
"""
______________________________________________________________________
to test your own KNN code, load your training data here:            
X is the observations array (ie. a pictures)
Y is the labels array
______________________________________________________________________
"""
training_data = [Xtrain,Ytrain]

"""
________________________________________________________________________
this section is here for us to varify our algorithm                     |
by running it on a valid dataset, NOT a part of the training data       |
it will point out the pictures and labels that the algorithm did not    |
identify correctly and show their labels                                |
________________________________________________________________________|
"""

# Yv = np.array([])
# n = 0
# for i, X in enumerate(Xvalid):
#     guess = KNN_classify(3,X,training_data)
#     if guess != Yvalid[i][0]:
#         print(i,":  ",guess, " - ",Yvalid[i], " = ", guess - Yvalid[i])
#         plot_sample(X, [guess])
#         n+=1
#
# print(n)

"""
________________________________________________________________________
this section is here for classification of an unlabeld set of data      |
it will NOT point out the pictures mistakes!                            |
so it is recommended to first validate your KNN model.                  |
when you are done with validation, its time to comment it               |
out (by selecting it and pressing CTRL + /) and move to this segment    |
________________________________________________________________________|
"""


Ytest = np.array([])
for X in Xtest:
    guess = KNN_classify(5, X, training_data)
    Ytest = np.append(Ytest, guess)



print('saving')
np.savetxt(f'{ID}.txt', Ytest, delimiter=", ", fmt='%i')
print('done')
