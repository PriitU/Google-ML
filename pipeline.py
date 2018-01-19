# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 17:34:28 2018

@author: ulmas
"""
#import dataset
from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

#split dataset to train and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=.5)

"""
#Decision tree classifier
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
"""

"""
#K-nearest neighbors classifier
"""
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

#train the classifier
my_classifier.fit(X_train, y_train)

#test the classifier
predictions = my_classifier.predict(X_test)
#print(predictions)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))