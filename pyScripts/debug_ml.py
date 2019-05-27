# -*- coding: utf-8 -*-
"""
Created on Sun May 26 13:49:21 2019

@author: Dekel
"""


import shap
import numpy as np
import matplotlib.pyplot as plt

def explain_predictions(dataset,predictions,model,feature_names,label_names):
     (X_train, Y_train), (X_test, Y_test) = dataset     
     # select a set of background examples to take an expectation over
     background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
     # explain predictions of the model on four images
     e = shap.DeepExplainer(model, background)
     test_samples = X_test[1:30]
     shap_values = e.shap_values(test_samples)
     # plot the feature attributions
     # summarize the effects of all the features
     plt.figure()
     shap.summary_plot(shap_values, test_samples,feature_names=feature_names)     