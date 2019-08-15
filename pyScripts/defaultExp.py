# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:12:42 2019

@author: Dekel
"""

from kerasExperiments import run_multi_exps_configs, BaseExp

exps = [ BaseExp() ]
df_results = run_multi_exps_configs(exps)

out_df = df_results[['accuracy','precision','recall','sel_tpr','roc_auc']]
print(out_df)
