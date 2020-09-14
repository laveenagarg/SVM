# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:52:46 2020

@author: LAVEENA
"""
import scipy.io
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

#importing data
data = scipy.io.loadmat('C://Users/LAVEENA/Desktop/machine-learning-ex6/ex6/ex6data1')
X = data['X']
y = data['y']
#slicing X in to 2 separate feature vectors
a = X[:,0:1]
b = X[:,1:2]

plt.scatter(a, b, c=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#Create a svm Classifier with linear kernel
clf = svm.SVC(kernel='linear', C = 1) # Linear Kernel
#Train the model using the training sets
clf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


y_pred = np.reshape(y_pred,(y_pred.shape[0],1))
plt.scatter(X_test[:,0:1], X_test[:,1:2], c=y_pred)