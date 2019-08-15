# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 12:49:05 2019

@author: Dekel
"""

from sklearn.linear_model import LogisticRegression

def logreg_classifier(dataset_info,num_labels,graph_to=None, verbose=True):
    print('\nClassifier: Logistic Regression\n')
    (X_train, Y_train), (X_test, Y_test) = dataset_info.ds.get_dataset()
    logreg = LogisticRegression(solver='lbfgs')
    model = logreg.fit(X_train, Y_train)
    return model