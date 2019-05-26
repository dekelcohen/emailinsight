# -*- coding: utf-8 -*-
"""
Created on Sun May 26 13:49:21 2019

@author: Dekel
"""

from keras import backend as K

def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    recl = recall(y_true, y_pred)
    return 2*((prec*recl)/(prec+recl+K.epsilon()))


#import numpy as np
#print(precision(K.constant([1, 1, 1]),K.constant([1, 1, 1])))