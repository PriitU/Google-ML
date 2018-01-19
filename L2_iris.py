# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:07:32 2018

@author: ulmas

0-setosa
1-
2-

"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus

iris = load_iris()
test_idx = [0,50,100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx] 
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)


print(test_target)
print(clf.predict(test_data))

"""
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")
"""
dot_data = tree.export_graphviz(clf,
                     out_file=None,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True,
                     rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")
