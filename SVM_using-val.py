# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:22:14 2020

@author: LAVEENA
"""

import scipy.io
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics

#importing data
data = scipy.io.loadmat('C://Users/LAVEENA/Desktop/machine-learning-ex6/ex6/ex6data3')
X = data['X']
y = data['y']
xval = data['Xval']
yval = data['yval']

#slicing X in to 2 separate feature vectors
a = X[:,0:1]
b = X[:,1:2]

plt.scatter(a, b, c=y)

#Create a svm Classifier with Gaussian kernel
clf = svm.SVC(kernel='rbf', C = 4, gamma = 0.25) # Gaussian Kernel
#Train the model using the training sets
clf.fit(X, y)
#Predict the response for test dataset
y_pred = clf.predict(xval)
print("Accuracy:",metrics.accuracy_score(yval, y_pred))

import numpy as np
y_pred = np.reshape(y_pred,(y_pred.shape[0],1))
plt.scatter(xval[:,0:1], xval[:,1:2], c=y_pred)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(yval, y_pred))
print(classification_report(yval, y_pred))