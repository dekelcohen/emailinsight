# -*- coding: utf-8 -*-
"""
Created on Sun May 26 13:49:21 2019

@author: Dekel
"""


import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from hpyutils import setattrs 

def calc_roc_curve(dataset,pred_probab):  
    (X_train, Y_train), (X_test, Y_test) = dataset      
    fpr, tpr, thresholds = roc_curve(Y_test[:,1], pred_probab[:,0],  pos_label = 0) # idx label 0 is the positive class (e.g. 'Save')
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc, thresholds

def get_threshold_by_fpr(fpr_thresh, pred_probab, fpr, tpr,thresholds):
    '''
    Given fpr_thresh (ex: 0.1), finds the last in fpr array where fpr[idx] <= fpr_thresh --> return the threshold[idx] (ex: 0.55)
    Assumes: There exist an element in fpr array with value 
    '''    
    idx = np.where(fpr <= fpr_thresh)[0][-1]
    return thresholds[idx],fpr[idx], tpr[idx]

def get_predictions_by_thresh(pred_probab,thresh):    
    '''
    Given a threshold and 2 class probabilites numpy 2d array, return numpy 1D array of 1 or 0 in each elem.
    Note: If col 0 has proba > thresh --> it should say 0 (to predict class 0), not 1
    '''
    return 1-(pred_probab[:,0] > thresh).astype('int32')

def get_roc_info(dataset,model,dataset_info):
    '''
    Main entry point function that predicts classes 0,1 with fpr based threshold, calc ROC and return all associated metrics
    '''
    (X_train, Y_train), (X_test, Y_test) = dataset  
    pred_probab = model.predict(X_test)
    fpr, tpr, roc_auc, thresholds = calc_roc_curve(dataset,pred_probab)
    sel_thres, sel_fpr,sel_tpr = get_threshold_by_fpr(dataset_info.fpr_thresh, pred_probab, fpr, tpr,thresholds)
    setattrs(dataset_info.new_metrics,
           fpr=fpr,
           tpr=tpr, 
           roc_auc=roc_auc, 
           thresholds=thresholds,
           sel_thres=sel_thres, 
           sel_fpr=sel_fpr, 
           sel_tpr=sel_tpr)    
    predictions = get_predictions_by_thresh(pred_probab,sel_thres)
    return predictions
    
def plot_roc_curve(new_metrics):    
    # Plot of a ROC curve for a specific class
    plt.figure()
    lw = 2
    plt.plot(new_metrics.fpr, new_metrics.tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % new_metrics.roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    