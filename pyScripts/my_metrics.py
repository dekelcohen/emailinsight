# -*- coding: utf-8 -*-
"""
Created on Sun May 26 13:49:21 2019

@author: Dekel
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score, confusion_matrix
from hpyutils import MyObj, setattrs

def calc_roc_curve(Y_test, pred_probab):
    fpr, tpr, thresholds = roc_curve(Y_test[:,1], pred_probab[:,0],  pos_label = 0,drop_intermediate=False) # idx label 0 is the positive class (e.g. 'Save')
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

def plot_confusion_matrix(cm, label_names, title='Confusion matrix', cmap=plt.cm.Blues, save_to = None):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation='vertical')
    plt.yticks(tick_marks, label_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_to is not None:
        plt.savefig(save_to,bbox_inches='tight')
    
def plot_metrics(nm,label_names):
    # Confusion Matrix
    plot_confusion_matrix(nm.confusion_mat, label_names)        
    # ROC curve
    plot_roc_curve(nm)
        
    
def print_metrics(nm,roc=False):
    print('\nConfusion matrix: (sel_thres=%f, sel_tpr %f, sel_fpr %f)' % (nm.sel_thres,nm.sel_tpr,nm.sel_fpr))
    print(nm.confusion_mat)
    print('\nOld Confusion matrix (thres=0.5):')        
    print(nm.confusion_mat_def)
    if roc:
        print('ROC Curve:')        
        print('sel_thres %f, sel_tpr %f, sel_fpr %f,thresholds: %s fpr: %s tpr: %s' % (nm.sel_thres,nm.sel_tpr,nm.sel_fpr,nm.thresholds,nm.fpr,nm.tpr))
    
    if not getattr(nm,'test_group_binned_train_count',None) is None:
        print(nm.test_group_binned_train_count)

def calc_test_group_stats(df_t,dataset_info,y_true):
    # Accuracy per group
    testgroup = getattr(dataset_info.metrics,'testgroupby',None)
    if testgroup is None:
        return None
    df_train = dataset_info.ds.get_X_train()
    df_t['correct_pred'] = y_true == df_t['predictions']

    df_stat = pd.DataFrame()
    df_stat[testgroup] = list(df_t.groupby(testgroup).groups.keys())
    df_stat['grp_avg_acc'] = list(df_t.groupby(testgroup)['correct_pred'].mean())
    df_stat['test_count'] = list(df_t.groupby(testgroup)[testgroup].count())
    df_train_groups = pd.DataFrame()
    df_train_groups[testgroup] = list(df_train.groupby(testgroup).groups.keys())
    df_train_groups['train_count'] = list(df_train.groupby(testgroup)[testgroup].count())
    
    df_group_stats =  pd.merge(df_stat,df_train_groups,on=testgroup,how='left')
    df_group_stats['train_count'].fillna(0,inplace=True)
    bins = pd.IntervalIndex.from_tuples([(0, 0), (1, 1),(2, 2),(3, 3),(4, 4),(5, 5),(6, 6),(7, 7),(8, 10),(11, 20),(21, 50),(51, 100),(101, max(df_group_stats['train_count']))], closed='both')
    df_group_stats['bin_train_count'] = pd.cut(df_group_stats['train_count'], bins=bins)    
    test_group_binned_train_count = df_group_stats.groupby('bin_train_count').agg({'grp_avg_acc': 'mean', testgroup : 'count', 'test_count' : 'sum'})
    return test_group_binned_train_count

def calc_metrics(num_labels, model,dataset_info):
    '''
    Main entry point function that predicts classes 0,1 with fpr based threshold, calc ROC and return all associated metrics
    '''
    (X_train, Y_train), (X_test, Y_test) = dataset_info.ds.get_dataset(to_categorical=True, num_labels=num_labels)
    pred_probab = model.predict_proba(X_test)
    X_test_indexes = []
    for index, row in enumerate(X_test):
        X_test_indexes.append(index)
        
    df_t = dataset_info.ds.get_X_test()    
    df_t['pred_probab'] = list(pred_probab)

    fpr, tpr, roc_auc, thresholds = calc_roc_curve(Y_test, pred_probab)
    sel_thres, sel_fpr,sel_tpr = get_threshold_by_fpr(dataset_info.metrics.fpr_thresh, pred_probab, fpr, tpr,thresholds)
    predictions = get_predictions_by_thresh(pred_probab,sel_thres)    
    df_t['predictions'] = predictions    
    y_true = Y_test[:,1]
    precision, recall, f_score,support = precision_recall_fscore_support(y_true,predictions, pos_label = 0, average = 'binary')
    accuracy = accuracy_score(y_true, predictions)
    confusion_mat = confusion_matrix(y_true,predictions)
    # Threshold 0.5 confusion matrix (def)
    predictions_def = get_predictions_by_thresh(pred_probab,0.5)
    confusion_mat_def = confusion_matrix(y_true,predictions_def)
    
    # Avg accuracy per group (sender)
    test_group_binned_train_count = calc_test_group_stats(df_t,dataset_info,y_true)
    
    
    new_metrics = MyObj()
    setattrs(new_metrics,
           fpr=fpr,
           tpr=tpr, 
           roc_auc=roc_auc,
           thresholds=thresholds,
           sel_thres=sel_thres, 
           sel_fpr=sel_fpr, 
           sel_tpr=sel_tpr,
           precision=precision,
           recall=recall,
           f_score=f_score,
           support=support,
           accuracy=accuracy,
           confusion_mat=confusion_mat,
           confusion_mat_def=confusion_mat_def,
           test_group_binned_train_count=test_group_binned_train_count) 
    
    return new_metrics,predictions