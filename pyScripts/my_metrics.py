# -*- coding: utf-8 -*-
"""
Created on Sun May 26 13:49:21 2019

@author: Dekel
"""


import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc


def calc_roc_curve(dataset,model):  
    (X_train, Y_train), (X_test, Y_test) = dataset  
    pred_probab = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test[:,1], pred_probab[:,0],  pos_label = 0) # idx label 0 is the positive class (e.g. 'Save')
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc, thresholds


def plot_roc_curve(fpr, tpr, roc_auc):
    # Plot of a ROC curve for a specific class
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()